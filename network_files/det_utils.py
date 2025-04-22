import torch
import math
from typing import List, Tuple
from torch import Tensor
from detectron2.layers import ROIAlign
from torch import nn
from detectron2.modeling.backbone.resnet import ResNet
from detectron2.modeling.backbone.resnet import BottleneckBlock
from torchvision.ops import roi_align
from torch.nn.functional import normalize
from torchvision.ops import MultiScaleRoIAlign
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import numpy as np
import sys

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs
        for matched_idxs_per_image in matched_idxs:
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos)
            # 指定负样本数量
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        Args:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
            proposals: List[Tensor] anchors/proposals

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """

        Args:
            rel_codes: bbox regression parameters
            boxes: anchors/proposals

        Returns:

        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        dx = rel_codes[:, 0::4] / wx   # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy   # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的iou最大值，并记录索引，
        iou<low_threshold索引值为-1， low_threshold<=iou<high_threshold索引值为-2
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # M x N 的每一列代表一个anchors与所有gt的匹配iou值
        # matched_vals代表每列的最大值，即每个anchors与所有gt匹配的最大iou值
        # matches对应最大值所在的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # 对于每个gt boxes寻找与其iou最大的anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        # Find highest quality match available, even if it is low, including ties
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        # gt_pred_pairs_of_highest_quality = torch.nonzero(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True, trans_box=None):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    # loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if trans_box is not None:
        # loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta) * trans_box[:,None]
        loss = n * trans_box[:,None]
    else:
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class BachDistance_loss(object):
    def __init__(self):
        self.loss = torch.nn.L1Loss(reduction='mean')
        # self.loss = torch.nn.MSELoss()
        # self.param = cfg.MODEL.PARAM.BD_param
        # self.metric = cfg.MODEL.PARAM.METRIC

    def __call__(self, f, hazy_f):
        m = f.size(0)
        xx = torch.pow(f, 2).sum(1, keepdim=True).expand(m, m)
        yy = torch.pow(f, 2).sum(1, keepdim=True).expand(m, m).t()
        dist_f = xx + yy
        dist_f.addmm_(1, -2, f, f.t())
        dist_f = dist_f.clamp(min=1e-12).sqrt()  # for numerical stability


        m = hazy_f.size(0)
        xx = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m)
        yy = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m).t()
        dist_hf = xx + yy
        dist_hf.addmm_(1, -2, hazy_f, hazy_f.t())
        dist_hf = dist_hf.clamp(min=1e-12).sqrt()  # for numerical stability
        loss = self.loss(dist_f, dist_hf)
        return loss


def irm_loss(image_features, clip_features):  # image_features[5,512], clip_features [5,512]
    weight = 15 # 15
    g_img = normalize(torch.matmul(image_features, torch.t(image_features)), 1)
    g_clip = normalize(torch.matmul(clip_features, torch.t(clip_features)), 1)
    irm_loss = torch.norm(g_img - g_clip) ** 2
    return irm_loss * weight


class Res5ROIHeads(nn.Module):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """
    def __init__(
        self,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super(Res5ROIHeads, self).__init__()

        in_features = ['-1']
        self.output_size = (14, 14)
        self.pooler_scales     = 0.0625
        self.sampling_ratio    = 0

        # self.poolers = ROIAlign(
        #         self.output_size, spatial_scale=pooler_scales, sampling_ratio=sampling_ratio, aligned=True
        #     )

        # res5 block
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = 1
        width_per_group      = 64
        bottleneck_channels  = 512
        out_channels         = 1024 # 2048
        stride_in_1x1        = True
        norm                 = 'FrozenBN'

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 4,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        self.res5 = nn.Sequential(*blocks)

    # def forward(self, fea, rois, image_shape):
    #     # for VGG
    #     features = fea['0']
    #     pool_fea = roi_align(
    #         features,
    #         rois,
    #         self.output_size,
    #         self.pooler_scales,
    #         self.sampling_ratio,
    #         aligned=True,
    #     )
    #     out = self.res5(pool_fea)
    #     return out
    def forward(self, box_features):
        # for resnet
        return self.res5(box_features)


class DarkChannelPrior(nn.Module):
    def __init__(self, kernel_size, top_candidates_ratio, omega,
                 radius, eps,
                 open_threshold=True,
                 depth_est=False):
        super().__init__()

        # dark channel piror
        self.kernel_size = kernel_size
        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), padding=0)

        # airlight estimation.
        self.top_candidates_ratio = top_candidates_ratio
        self.open_threshold = open_threshold

        # raw transmission estimation
        self.omega = omega

        # image guided filtering
        self.radius = radius
        self.eps = eps
        self.guide_filter = GuidedFilter2d(radius=self.radius, eps=self.eps)

        self.depth_est = depth_est

    def forward(self, image):

        # compute the dark channel piror of given image.
        b, c, h, w = image.shape
        image_pad = self.pad(image)
        local_patches = self.unfold(image_pad) # 1，675，129060： 15*15 kernel sliding on the image generates patches, each patch contain 675=15*15*3, totally 129060 patches
        dc, dc_index = torch.min(local_patches, dim=1, keepdim=True)
        dc = dc.view(b, 1, h, w)
        dc_vis = dc
        # airlight estimation.
        top_candidates_nums = int(h * w * self.top_candidates_ratio)
        dc = dc.view(b, 1, -1)  # dark channels
        searchidx = torch.argsort(-dc, dim=-1)[:, :, :top_candidates_nums]
        searchidx = searchidx.repeat(1, 3, 1)
        image_ravel = image.view(b, 3, -1)
        value = torch.gather(image_ravel, dim=2, index=searchidx)
        airlight, image_index = torch.max(value, dim=-1, keepdim=True)
        airlight = airlight.squeeze(-1)
        if self.open_threshold:
            airlight = torch.clamp(airlight, max=220)

        # get the raw transmission
        airlight = airlight.unsqueeze(-1).unsqueeze(-1)
        processed = image / airlight

        processed_pad = self.pad(processed)
        local_patches_processed = self.unfold(processed_pad)
        dc_processed, dc_index_processed = torch.min(local_patches_processed, dim=1, keepdim=True)
        dc_processed = dc_processed.view(b, 1, h, w)

        raw_t = 1.0 - self.omega * dc_processed
        if self.open_threshold:
            raw_t = torch.clamp(raw_t, min=0.2)

        # raw transmission guided filtering.
        # refined_tranmission = soft_matting(image_data_tensor,raw_transmission,r=40,eps=1e-3)
        normalized_img = simple_image_normalization(image)
        refined_transmission = self.guide_filter(raw_t, normalized_img)

        # recover image: get radiance.
        image = image.float()
        tiledt = refined_transmission.repeat(1, 3, 1, 1)

        dehaze_images = (image - airlight) * 1.0 / tiledt + airlight

        # recover scaled depth or not
        if self.depth_est:
            depth = recover_depth(refined_transmission)
            return dehaze_images, dc_vis, airlight, raw_t, refined_transmission, depth

        return dehaze_images, dc_vis, airlight, raw_t, refined_transmission


def simple_image_normalization(tensor):
    b, c, h, w = tensor.shape
    tensor_ravel = tensor.view(b, 3, -1)
    image_min, _ = torch.min(tensor_ravel, dim=-1, keepdim=True)
    image_max, _ = torch.max(tensor_ravel, dim=-1, keepdim=True)
    image_min = image_min.unsqueeze(-1)
    image_max = image_max.unsqueeze(-1)

    normalized_image = (tensor - image_min) / (image_max - image_min)
    return normalized_image


def recover_depth(transmission, beta=0.001):
    negative_depth = torch.log(transmission)
    return (-negative_depth) / beta


class GuidedFilter2d(nn.Module):
    def __init__(self, radius: int, eps: float):
        super().__init__()
        self.r = radius
        self.eps = eps

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            return guidedfilter2d_color(guide, x, self.r, self.eps)
        elif guide.shape[1] == 1:
            return guidedfilter2d_gray(guide, x, self.r, self.eps)
        else:
            raise NotImplementedError


def guidedfilter2d_gray(guide, src, radius, eps, scale=None):
    """guided filter for a gray scale guide image

    Parameters
    -----
    guide: (B, 1, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1. / scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1. / scale, mode="nearest")
        radius = radius // scale

    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N
    mean_p = boxfilter2d(src, radius) / N
    mean_Ip = boxfilter2d(guide * src, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter2d(guide * guide, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter2d(a, radius) / N
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = mean_a * guide + mean_b
    return q

def guidedfilter2d_color(guide, src, radius, eps, scale=None):
    """guided filter for a color guide image

    Parameters
    -----
    guide: (B, 3, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    assert guide.shape[1] == 3
    if src.ndim == 3:
        src = src[:, None]
    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1. / scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1. / scale, mode="nearest")
        radius = radius // scale

    guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1)  # b x 1 x H x W
    ones = torch.ones_like(guide_r)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N  # b x 3 x H x W
    mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1)  # b x 1 x H x W

    mean_p = boxfilter2d(src, radius) / N  # b x C x H x W

    mean_Ip_r = boxfilter2d(guide_r * src, radius) / N  # b x C x H x W
    mean_Ip_g = boxfilter2d(guide_g * src, radius) / N  # b x C x H x W
    mean_Ip_b = boxfilter2d(guide_b * src, radius) / N  # b x C x H x W

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p  # b x C x H x W
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p  # b x C x H x W
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p  # b x C x H x W

    var_I_rr = boxfilter2d(guide_r * guide_r, radius) / N - mean_I_r * mean_I_r + eps  # b x 1 x H x W
    var_I_rg = boxfilter2d(guide_r * guide_g, radius) / N - mean_I_r * mean_I_g  # b x 1 x H x W
    var_I_rb = boxfilter2d(guide_r * guide_b, radius) / N - mean_I_r * mean_I_b  # b x 1 x H x W
    var_I_gg = boxfilter2d(guide_g * guide_g, radius) / N - mean_I_g * mean_I_g + eps  # b x 1 x H x W
    var_I_gb = boxfilter2d(guide_g * guide_b, radius) / N - mean_I_g * mean_I_b  # b x 1 x H x W
    var_I_bb = boxfilter2d(guide_b * guide_b, radius) / N - mean_I_b * mean_I_b + eps  # b x 1 x H x W

    # determinant
    cov_det = var_I_rr * var_I_gg * var_I_bb \
              + var_I_rg * var_I_gb * var_I_rb \
              + var_I_rb * var_I_rg * var_I_gb \
              - var_I_rb * var_I_gg * var_I_rb \
              - var_I_rg * var_I_rg * var_I_bb \
              - var_I_rr * var_I_gb * var_I_gb  # b x 1 x H x W

    # inverse
    inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det  # b x 1 x H x W
    inv_var_I_rg = - (var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det  # b x 1 x H x W
    inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det  # b x 1 x H x W
    inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det  # b x 1 x H x W
    inv_var_I_gb = - (var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det  # b x 1 x H x W
    inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det  # b x 1 x H x W

    inv_sigma = torch.stack([
        torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
        torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
        torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1)
    ], 1).squeeze(-3)  # b x 3 x 3 x H x W

    cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1)  # b x 3 x C x H x W

    a = torch.einsum("bichw,bijhw->bjchw", (cov_Ip, inv_sigma))
    b = mean_p - a[:, 0] * mean_I_r - a[:, 1] * mean_I_g - a[:, 2] * mean_I_b  # b x C x H x W

    mean_a = torch.stack([boxfilter2d(a[:, i], radius) / N for i in range(3)], 1)
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = torch.stack([F.interpolate(mean_a[:, i], guide.shape[-2:], mode='bilinear') for i in range(3)], 1)
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = torch.einsum("bichw,bihw->bchw", (mean_a, guide)) + mean_b

    return q

def boxfilter2d(src, radius):
    return _diff_y(_diff_x(src, radius), radius)

def _diff_x(src, r):
    cum_src = src.cumsum(-2)

    left = cum_src[..., r:2*r + 1, :]
    middle = cum_src[..., 2*r + 1:, :] - cum_src[..., :-2*r - 1, :]
    right = cum_src[..., -1:, :] - cum_src[..., -2*r - 1:-r - 1, :]

    output = torch.cat([left, middle, right], -2)

    return output

def _diff_y(src, r):
    cum_src = src.cumsum(-1)

    left = cum_src[..., r:2*r + 1]
    middle = cum_src[..., 2*r + 1:] - cum_src[..., :-2*r - 1]
    right = cum_src[..., -1:] - cum_src[..., -2*r - 1:-r - 1]

    output = torch.cat([left, middle, right], -1)

    return output