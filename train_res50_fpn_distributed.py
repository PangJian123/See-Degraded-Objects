import os
import datetime

import torch

import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
import numpy as np
from AOD.model import AODnet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir

# import clip
from network_files.det_utils import Res5ROIHeads, DarkChannelPrior

def create_model(num_classes, load_pretrain_weights=True, fuzzy=None):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path="./weights/resnet50/resnet50-0676ba61.pth",
                                     norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    # model = FasterRCNN(backbone=backbone, num_classes=91)
    detectron2_roi = Res5ROIHeads()

    dark_channel_piror = DarkChannelPrior(kernel_size=15, top_candidates_ratio=0.0001,
                                          omega=0.95, radius=40, eps=1e-3, open_threshold=True, depth_est=True)

    model = FasterRCNN(backbone=backbone, num_classes=num_classes, detectron2_roi_res5=detectron2_roi, dark=dark_channel_piror, fuzzy=fuzzy)

    # if load_pretrain_weights:
    #     # 载入预训练模型权重
    #     # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    #     weights_dict = torch.load("./weights/fasterrcnn_resnet50/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", map_location='cpu')
    #     missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    #     if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    #         print("missing_keys: ", missing_keys)
    #         print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)


    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    aspect_ratio_group_factor = 3
    VOC_root = args.data_path
    batch_size = args.batch_size
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = args.num_workers
    fuzzy_a = args.fuzzy_a
    fuzzy_c = args.fuzzy_c
    p_ce = args.p_ce
    member_f = args.member_f

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    log_dir = './runs/' + args.name
    if not os.path.isdir(log_dir):
       os.makedirs(log_dir)

    # 检查保存权重文件夹是否存在，不存在则创建
    weights_dir = log_dir + '/' + 'weights/'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # 用来保存coco_info的文件
    results_file = log_dir + '/' + "{}.txt".format(args.name)
    results_file_RTTS = log_dir + '/' + "{}_RTTS.txt".format(args.name)

    # check voc root
    # if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
    #     raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt", args.json_name, 'train', args.prior)
    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt", args.json_name, 'val', args.prior)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    print('Using %g dataloader workers' % nw)

    # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_sampler=train_batch_sampler,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      sampler=test_sampler,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1, fuzzy=[fuzzy_a, fuzzy_c, p_ce, member_f])
    model.to(device)
    print('device',device)
    # create defog model
    # defog_model = AODnet()
    # load defog model
    # checkpoint = torch.load(args.defog_ckpt, map_location='cpu')
    # defog_model.load_state_dict(checkpoint['de_fog'])
    # defog_dict = torch.load(args.defog_ckpt)
    # defog_model.load_state_dict(defog_dict)

    # clip model
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root='./weights/Vit-B-32.pt')
    clip_model, clip_preprocess = None, None



    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # defog_model.to(device)
    # clip_model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # define defog optimizer
    # defog_optimizer = torch.optim.Adam(defog_model.parameters(), lr=args.defog_lr, weight_decay=0.0001)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))


    num_epochs = args.epochs
    train_loss = []
    learning_rate = []
    val_map = []
    train_clip_loss = []
    train_neg_loss = []
    current_map = torch.tensor([0.])

    for epoch in range(args.start_epoch, num_epochs, 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch, printing every 50 iterations
        detloss, lr, cliploss, neg_loss = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler,
                                              clip_model=clip_model, clip_preprocess=clip_preprocess)
        train_loss.append(detloss.item())
        train_clip_loss.append(cliploss.item())
        train_neg_loss.append(neg_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device, clip_model=clip_model, clip_preprocess=clip_preprocess)
        # coco_info_RTTS = utils.evaluate(model, val_data_loader_RTTS, device=device, )

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in
                               coco_info + [detloss.item()] + [cliploss.item()] + [neg_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP
        if current_map < coco_info[1]:
            if args.output_dir:
                # 只在主节点上执行保存权重操作
                current_map = coco_info[1]
                save_files = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch}
                save_on_master(save_files,
                               weights_dir + "mobile-model-best.pth")

        # save weights
        if (epoch+1) % 10 == 0:
            save_files = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch}
            save_on_master(save_files,
                           weights_dir + "mobile-model-best.pth")

    if args.rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate, log_dir, 'det_loss')
            plot_loss_and_lr(train_clip_loss, learning_rate, log_dir, 'clip_loss')
            plot_loss_and_lr(train_neg_loss, learning_rate, log_dir, 'train_neg_loss')

        # plot mAP curve
        if len(val_map) != 0:
            from plot_curve import plot_map
            plot_map(val_map, log_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    # log name
    parser.add_argument('--name', default='exp', help='device')
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # workers
    parser.add_argument('--num_workers', default=4, type=int, help='number workers')

    # 训练数据集的根目录(VOCdevkit)
    # parser.add_argument('--data-path', default="/media/pj/document/Datasets/RTTS_VOC/", help='dataset') # num-classes=9
    # parser.add_argument('--data-path', default="/media/pj/document/Datasets/VOCFOG/", help='dataset')
    # parser.add_argument('--data-path', default="/media/pj/document/Datasets/Foggy_Driving_VOC_DCP/", help='dataset')
    # parser.add_argument('--data-path', default="/media/pj/document/Datasets/Foggy_Cityscapes_beta_0.02_VOC", help='dataset')
    # parser.add_argument('--data-path', default="/media/pj/document/Datasets/RTTS_DCP/", help='dataset')
    parser.add_argument('--data-path', default="/media/pj/document/Datasets/RTTS_VOC/", help='dataset')

    parser.add_argument('--data-path-RTTS', default="/media/pj/document/Datasets/RTTS_DCP/", help='dataset')

    # 检测目标类别数(不包含背景)
    # parser.add_argument('--num-classes', default=8, type=int, help='num_classes')
    parser.add_argument('--num-classes', default=5, type=int, help='num_classes')
    # parser.add_argument('--json-name', default='pascal_voc_classes.json', type=str, help='num_classes')
    parser.add_argument('--json-name', default='voc_fog_classes.json', type=str, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    # parser.add_argument('--resume', default="./runs/fuzzy_math_pretrain_foggycity/weights/pretrain-res-model-19.pth", type=str, help='resume from checkpoint')  # './save_weights/pretrain.pth'
    parser.add_argument('--resume', default="", type=str, help='resume from checkpoint')  # './save_weights/pretrain.pth'


    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    # defog args
    parser.add_argument('--defog_lr', default=1e-4, type=float,)
    parser.add_argument('--defog-ckpt', default='./runs/train_defog_lre-4/weights/mobile-model-19.pth', help='path where to save')
    # parser.add_argument('--defog-ckpt', default='./AOD/model_pretrained/dict_aod.pt', help='path where to save')

    parser.add_argument("--fuzzy_a", default=2, type=float)
    parser.add_argument("--fuzzy_c", default=1, type=float)

    parser.add_argument("--prior", default='trans', type=str) # trans / ams / weight_fusion / inversedepth / voc_fusion

    parser.add_argument("--p_ce", default='1', type=float)

    parser.add_argument("--member_f", default='norm', type=str) # tri / norm

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
