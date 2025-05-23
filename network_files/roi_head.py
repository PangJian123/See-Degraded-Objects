from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import math
from . import det_utils
from . import boxes as box_ops
# from det_utils import irm_loss

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, exp_weight=None, exp_trans_l1=None, p_ce=None):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor], Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息，shape=[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]

    # generate trans weight
    weight_for_ce = torch.ones_like(labels).float()
    weight_for_ce[sampled_pos_inds_subset] = exp_weight # used for boxes

    # physics-guided loss
    tmp_ce = F.cross_entropy(class_logits, labels, reduction='none') # class_logits[1024,6], labels[1024]
    classification_loss = torch.mean(tmp_ce * weight_for_ce) * p_ce

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False, trans_box=exp_trans_l1
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img,
                 clip_mlp,
                 detectron2_roi_res5=None,
                 fusion_mlp=None,
                 fuzzy=None):  # default: 100
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

        self.clip_mlp = clip_mlp  # clip parameters
        self.fusion_mlp = fusion_mlp
        self.detectron2_roi_res5 = detectron2_roi_res5
        if fuzzy is not None:
            self.fuzzy_a = fuzzy[0]
            self.fuzzy_c = fuzzy[1]  # not used
            self.p_ce = fuzzy[2]
            self.member_f = fuzzy[3]


    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应标签
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        pos_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]  # pos_inds_img[1471]:positive is 1, totally 21 samples. neg_inds_img[1471], negative is 1, totally 491 samples.
            sampled_inds.append(img_sampled_inds)
            pos_inds.append(torch.where(pos_inds_img)[0])
        return sampled_inds, pos_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        list元素个数为batch_size
        Args:
            proposals: rpn预测的boxes
            targets:

        Returns:

        """

        # 检查target数据是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取标注好的boxes以及labels信息
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本
        sampled_inds, pos_inds = self.subsample(labels)  # sample_inds [512], ascend order. pos_inds [21], proposal[pos_inds] will select the positive samples

        pos_proposal_idx = []
        pos_proposal_matched_idxs = []
        for img_id in range(len(proposals)):
            pos_proposal_matched_idxs.append(matched_idxs[img_id][pos_inds[img_id]])

        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id] # 512 index which index in the 1471 proposals
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

            # generate the postive proposal index in 512 proposals.
            equal_mask = torch.eq(img_sampled_inds[:, None], pos_inds[img_id])
            pos_index_in_img_sampled_inds = torch.nonzero(equal_mask, as_tuple=False)
            pos_proposal_idx.append(pos_index_in_img_sampled_inds[:, 0])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets, pos_proposal_idx, pos_proposal_matched_idxs

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        （1）根据proposal以及预测的回归参数计算出最终bbox坐标
        （2）对预测类别结果进行softmax处理
        （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        （4）移除所有背景信息
        （5）移除低概率目标
        （6）移除小尺寸目标
        （7）执行nms处理，并按scores进行排序
        （8）根据scores排序返回前topk个目标
        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal
            image_shapes: 打包成batch前每张图像的宽高

        Returns:

        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None,    # type: Optional[List[Dict[str, Tensor]]]
                trans_maps=None,
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            proposals, labels, regression_targets, pos_proposal_idx, pos_proposal_matched_idxs = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        box_features = self.box_roi_pool(features, proposals, image_shapes)  # feature[2,512,42,84], proposals[0]=[512,4]
        box_features = self.detectron2_roi_res5(box_features)

        # 通过roi_pooling后的两层全连接层
        box_features = self.box_head(box_features)  # [1024,1024]

        # --------------------use membership function to scale weight
        if self.training:
            if isinstance(trans_maps, list):
                am_prior = trans_maps[0]
                trans_prior = trans_maps[1]
                tmp_am_weight, tmp_trans_weight = [], []
                for i in range(len(proposals)):
                    proposal_each_img = proposals[i][pos_proposal_idx[i]]
                    for j, bbox in enumerate(proposal_each_img):
                        xmin, ymin, xmax, ymax = bbox.int()
                        max_size_x = am_prior[i].shape[2]
                        max_size_y = am_prior[i].shape[1]
                        # ---------------narrow the bbox to the proper size----------------
                        if xmin < 0 or ymin < 0:
                            # print(f"xmin<0 or ymin<0 or xmax<0 or ymax<0, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                            if xmin < 0:
                                xmin = torch.zeros_like(xmin).cuda()
                            if ymin < 0:
                                ymin = torch.zeros_like(ymin).cuda()
                        if xmax > max_size_x or ymax > max_size_y:
                            # print(f"xmax > max_size_x or ymax > max_size_y, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                            if xmax > max_size_x:
                                xmax = torch.tensor(max_size_x).cuda()
                            if ymax > max_size_y:
                                ymax = torch.tensor(max_size_y).cuda()
                        if xmax <= xmin or ymax <= ymin:
                            # print(f"xmax <= xmin or ymax <= ymin, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                            tmp_am_weight.append(torch.tensor(0).unsqueeze(0).cuda())
                            tmp_trans_weight.append(torch.tensor(0).unsqueeze(0).cuda())
                            continue
                        # AM
                        am_bbox = am_prior[i][:, ymin:ymax, xmin:xmax]
                        am_c = am_prior[i].max()
                        am_a = 0
                        am_x = am_bbox.mean()
                        if torch.isnan(am_x):
                            print(f"mean trans box is nan, c:{am_c}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                        if self.member_f == 'tri':
                            am_weight = 1 - (am_c - am_x) / (am_c - am_a)  # triangle
                            if am_c < 0:
                                am_weight = torch.tensor(0).unsqueeze(0).cuda()
                                print("c < 0 !")
                        elif self.member_f == 'norm':
                            std_dev = 1  # normalization
                            am_weight = (1 / (std_dev * math.sqrt(2 * math.pi))) * torch.exp(
                                -0.5 * ((am_x - am_c) / std_dev) ** 2)
                        else:
                            assert print('no membership function in args')
                        if torch.isnan(am_weight):
                            am_weight = torch.ones_like(am_weight)
                        tmp_am_weight.append(am_weight.unsqueeze(0))
                        # Trans
                        trans_bbox = trans_prior[i][:, ymin:ymax, xmin:xmax]
                        trans_c = trans_prior[i].max()
                        trans_a = 0
                        trans_x = trans_bbox.mean()
                        if torch.isnan(trans_x):
                            print(f"mean trans box is nan, c:{trans_c}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                        if self.member_f == 'tri':
                            trans_weight = 1 - (trans_c - trans_x) / (trans_c - trans_a)  # triangle
                            if trans_c < 0:
                                trans_weight = torch.tensor(0).unsqueeze(0).cuda()
                                print("c < 0 !")
                        elif self.member_f == 'norm':
                            std_dev = 1  # normalization
                            trans_weight = (1 / (std_dev * math.sqrt(2 * math.pi))) * torch.exp(
                                -0.5 * ((trans_x - trans_c) / std_dev) ** 2)
                        else:
                            assert print('no membership function in args')
                        if torch.isnan(trans_weight):
                            trans_weight = torch.ones_like(trans_weight)
                        tmp_trans_weight.append(trans_weight.unsqueeze(0))

                am_weight = torch.cat(tmp_am_weight, dim=0)
                trans_weight = torch.cat(tmp_trans_weight, dim=0)
                mean_weight = (am_weight + trans_weight) / 2
                exp_weight = torch.exp(self.fuzzy_a * mean_weight)
            else:
                tmp_trans_pos, tmp_trans_neg = [], []
                for i in range(len(proposals)):
                    proposal_each_img = proposals[i][pos_proposal_idx[i]]
                    for j, bbox in enumerate(proposal_each_img):
                        xmin, ymin, xmax, ymax = bbox.int()
                        max_size_x = trans_maps[i].shape[2]
                        max_size_y = trans_maps[i].shape[1]
                        if xmin < 0 or ymin < 0:
                            # print(f"xmin<0 or ymin<0 or xmax<0 or ymax<0, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                            if xmin < 0:
                                xmin = torch.zeros_like(xmin).cuda()
                            if ymin < 0:
                                ymin = torch.zeros_like(ymin).cuda()
                        if xmax > max_size_x or ymax > max_size_y:
                            # print(f"xmax > max_size_x or ymax > max_size_y, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                            if xmax > max_size_x:
                                xmax = torch.tensor(max_size_x).cuda()
                            if ymax > max_size_y:
                                ymax = torch.tensor(max_size_y).cuda()
                        if xmax <= xmin or ymax <= ymin:
                            # print(f"xmax <= xmin or ymax <= ymin, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                            tmp_trans_pos.append(torch.tensor(0).unsqueeze(0).cuda())
                            continue
                        trans_bbox = trans_maps[i][:, ymin:ymax, xmin:xmax]
                        c = trans_maps[i].max()
                        mean_c = trans_maps[i].mean()
                        a = 0
                        x = trans_bbox.mean()
                        if torch.isnan(x):
                            print(f"mean trans box is nan, c:{c}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}")
                        if self.member_f == 'tri':
                            if c < 0:
                                tmp_trans_pos.append(torch.tensor(0).unsqueeze(0).cuda())
                                print("c < 0 !")
                                continue  # 1,
                            mu_trans = 1 - (c - x) / (c - a)  # triangle
                        elif self.member_f == 'norm':
                            std_dev = 1 # normalization
                            mu_trans = (1 / (std_dev * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - c) / std_dev) ** 2)
                        else:
                            assert print('no membership function in args')

                        if torch.isnan(mu_trans):
                            mu_trans = torch.ones_like(mu_trans)
                        tmp_trans_pos.append(mu_trans.unsqueeze(0))
                    # # ----------------------------------------------draw_box
                    # if i == 0 :
                    #     from draw_box_utils import draw_objs
                    #     from torchvision.transforms import transforms
                    #     from PIL import Image
                    #     image_tensor = trans_maps[i]
                    #     # mean = [0.485, 0.456, 0.406]
                    #     # std = [0.229, 0.224, 0.225]
                    #     # mean = torch.as_tensor(mean)
                    #     # std = torch.as_tensor(std)
                    #     # ori_clear_image = image_tensor.cpu() * std[:, None, None] + mean[:, None, None]
                    #     np_clear_images = image_tensor.cpu().detach().numpy().transpose((1, 2, 0)).repeat(3, 2)
                    #     draw_image = Image.fromarray((np_clear_images * 255 / np_clear_images.max()).astype('uint8'))
                    #     pos_boxes = proposal_each_img.cpu()
                    #     pos_classes = torch.ones(proposal_each_img.shape[0]).int()
                    #     pos_scores = torch.cat(tmp_trans_pos, dim=0).cpu().numpy()
                    #     cate = {
                    #         "1": 'person',
                    #         "2": 'car',
                    #         "3": 'bus',
                    #         "4": 'bicycle',
                    #         "5": 'motorbike',
                    #     }
                    #     pos_img = draw_objs(draw_image, pos_boxes, pos_classes, pos_scores, category_index=cate)
                    #     pos_img.save('./visualization/pos_trans.jpg')
                    #     # neg_img = draw_objs(draw_image, neg_boxes, neg_classes, neg_scores, category_index=cate)
                    #     # neg_img.save('./visualization/neg_trans.jpg')
                trans_pos = torch.cat(tmp_trans_pos, dim=0)
                exp_weight = torch.exp(self.fuzzy_a*trans_pos)

        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets, exp_weight, None, self.p_ce)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals,
                                                                image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses
