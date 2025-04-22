import os
import time
import json

import numpy
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from my_dataset import VOCDataSet

import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs
from network_files.det_utils import Res5ROIHeads
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

    model = FasterRCNN(backbone=backbone, num_classes=91, detectron2_roi_res5=detectron2_roi, dark=dark_channel_piror, fuzzy=fuzzy)

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

# def create_model(num_classes):
#     # mobileNetv2+faster_RCNN
#     # backbone = MobileNetV2().features
#     # backbone.out_channels = 1280
#     #
#     # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
#     #                                     aspect_ratios=((0.5, 1.0, 2.0),))
#     #
#     # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
#     #                                                 output_size=[7, 7],
#     #                                                 sampling_ratio=2)
#     #
#     # model = FasterRCNN(backbone=backbone,
#     #                    num_classes=num_classes,
#     #                    rpn_anchor_generator=anchor_generator,
#     #                    box_roi_pool=roi_pooler)
#
#     # resNet50+fpn+faster_RCNN
#     # 注意，这里的norm_layer要和训练脚本中保持一致
#     backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
#     detectron2_roi = Res5ROIHeads()
#     # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
#     model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5, detectron2_roi_res5=detectron2_roi)
#
#     return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # val data loader
    dataset_path = '/media/pj/document/Datasets/Foggy_Cityscapes_beta_0.02_VOC/'
    # dataset_path = '/media/pj/document/Datasets/RTTS_VOC/'
    # dataset_path = '/media/pj/document/Datasets/DUO_VOC/'
    # read class_indict
    label_json_path = 'voc_fog_classes.json'
    num_classes = 6
    # label_json_path = 'DUO_classes.json'
    # num_classes = 5



    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    val_dataset = VOCDataSet(dataset_path, "2012", data_transform["val"], "val.txt", label_json_path, 'val', prior='weight_fusion')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=4,
                                                  collate_fn=val_dataset.collate_fn)

    # create model
    model = create_model(num_classes=num_classes, fuzzy=[2., 1, 1, 'weight_fusion'])
    weights_path = "./runs/FOC_weightfusion_fuzzya_2_mem_tri_ce1/weights/mobile-model-best.pth"


    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)


    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    model.eval()  # 进入验证模式
    for i, (image, targets, _) in enumerate(val_data_loader):
        with torch.no_grad():
            # init
            img_height, img_width = image[0].shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img, )

            t_start = time_synchronized()
            predictions = model(image[0].unsqueeze(0).to(device), )[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            img_name = targets[0]['filename']
            original_img = Image.open(dataset_path + 'JPEGImages' + '/' + img_name)

            # # ------------------------------visualize degree estimation
            # # load dict
            # original_img = Image.open(dataset_path + 'JPEGImages' + '/' + 'aachen_000000_000019_leftImg8bit_foggy_beta_0.02.png')
            # import pickle
            # filename = 'tri_aachen_000000_000019_leftImg8bit.pickle'
            # # 使用pickle.load加载文件中的字典
            # with open(filename, 'rb') as file:
            #     loaded_data = pickle.load(file)
            # index_bbox = [0,2,15,16,67,74]
            # am_mean_np = numpy.array(loaded_data['am_mean_np'])
            # trans_mean_np = numpy.array(loaded_data['trans_mean_np'])
            # am_weight_np = numpy.array(loaded_data['am_weight_np'])
            # trans_weight_np = numpy.array(loaded_data['trans_weight_np'])
            # box_numpy = loaded_data['bbox']
            # box_numpy[:, 0] = box_numpy[:, 0] * img_width
            # box_numpy[:, 1] = box_numpy[:, 1] * img_height
            # box_numpy[:, 2] = box_numpy[:, 2] * img_width
            # box_numpy[:, 3] = box_numpy[:, 3] * img_height
            # plot_img = draw_objs(original_img,
            #                      box_numpy[index_bbox],
            #                      numpy.arange(am_weight_np.shape[0])[index_bbox],
            #                      am_weight_np[index_bbox],
            #                      category_index=category_index,
            #                      box_thresh=0.,  # for foc and rtts is 0.8, for uw is 0.5
            #                      line_thickness=3,
            #                      font='arial.ttf',
            #                      font_size=24,
            #
            #                      )
            #
            # # plt.imshow(plot_img)
            # # plt.show()
            # # 保存预测的图片结果
            # plot_img.save(f"./visualization/degree_estimation/aachen_000000_000019_leftImg8bit.png")
            # plt.close()
            # # ------------------------------visualize degree estimation

            plot_img = draw_objs(original_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.8,  # for foc and rtts is 0.8, for uw is 0.5
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)

            plt.imshow(plot_img)
            # plt.show()
            # 保存预测的图片结果
            plot_img.save(f"./visualization/weight_fusion/{img_name}")
            plt.close()

            # # save gt bboxes
            # img_name = targets[0]['filename']
            # original_img = Image.open(dataset_path + 'JPEGImages' + '/' + img_name)
            #
            # target_boxes = targets[0]["boxes"].to("cpu").numpy()
            # target_classes = targets[0]["labels"].to("cpu").numpy()
            # target_scores = numpy.ones_like(target_classes)
            #
            # plot_img = draw_objs(original_img,
            #                      target_boxes,
            #                      target_classes,
            #                      target_scores,
            #                      category_index=category_index,
            #                      box_thresh=0.5,
            #                      line_thickness=3,
            #                      font='arial.ttf',
            #                      font_size=20)
            # # plt.imshow(plot_img)
            # # plt.show()
            # # 保存预测的图片结果
            #
            # plot_img.save(f"./visualization/gt/{img_name}")

if __name__ == '__main__':
    main()

