import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import BuildDataLoader, BuildDataset, visual_bbox_mask, cv2
from functools import partial
import logging
from sklearn.metrics import auc

import time

import sys

IN_COLAB = 'google' in sys.modules

if not IN_COLAB:
    import coloredlogs


class SOLOHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768),
                               (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2, alpha=0.25, weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
    # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
    # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
    # self.cate_out is 1 out-layer of conv2d
    # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d,
    #   one for each fpn_feat
    def _init_layers(self):
        # initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList()
        self.ins_head = nn.ModuleList()
        self.cate_out = None
        self.ins_out_list = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        # Initialize category branch
        self.cate_head.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=256,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=256),
                nn.ReLU()))
        #   add 6 intermediate convolution layers
        for _ in range(6):
            self.cate_head.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=256,
                              out_channels=256,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=256),
                    nn.ReLU()))
        # add output convolution layer
        self.cate_out = nn.Conv2d(in_channels=256,
                                  out_channels=self.cate_out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=True)

        # Initialize mask branch
        self.ins_head.append(
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels + 2,
                          out_channels=256,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=256),
                nn.ReLU()))
        # add 6 intermediate convolution layers
        for _ in range(6):
            self.ins_head.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=256,
                              out_channels=256,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=256),
                    nn.ReLU()))
        # add output convolution layer
        for num_grid in self.seg_num_grids:
            self.ins_out_list.append(
                nn.Conv2d(in_channels=256,
                          out_channels=num_grid**2,
                          kernel_size=1,
                          bias=True))

        # Initialize weights
        self.cate_head.apply(self._init_weights)
        self.ins_head.apply(self._init_weights)
        self.cate_out.apply(self._init_weights)
        self.ins_out_list.apply(self._init_weights)

    # This function initialize weights for head network
    def _init_weights(self, m):
        # initialize the weights
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
    # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
    # if eval = False
    # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # if eval==True
    # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward(self, fpn_feat_list, device, eval=False, ori_size=None):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        del fpn_feat_list

        assert new_fpn_list[0].shape[1:] == (256, 100, 136)
        quart_shape = [
            new_fpn_list[0].shape[-2] * 2, new_fpn_list[0].shape[-1] * 2
        ]  # stride: 4
        # use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(
            self.forward_single_level,
            new_fpn_list,
            list(range(len(new_fpn_list))),
            device=device,
            eval=eval,
            upsample_shape=quart_shape,
            ori_size=ori_size)

        assert len(new_fpn_list) == len(self.seg_num_grids)
        del quart_shape, new_fpn_list

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
    # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        # finish this function
        new_fpn_list = fpn_feat_list
        del fpn_feat_list
        # resize first level to 1/2
        _, _, H, W = new_fpn_list[0].size()
        new_fpn_list[0] = nn.functional.interpolate(
            new_fpn_list[0], torch.Size([H // 2, W // 2]))
        # resize last level to 2
        _, _, H, W = new_fpn_list[-1].size()
        new_fpn_list[-1] = nn.functional.interpolate(
            new_fpn_list[-1], torch.Size([2 * H, 2 * W]))
        return new_fpn_list

    def _append_xy_coordinates(self, fpn_feat, device):
        """
        ---- ZHIHAO RUAN 2020-10-10 02:56
        This function apopends the normalized x/y coordinates to fpn_featmap
        according to its dimensions

        Input:
            fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        Output:
            fpn_feat: (bz, fpn_channels(256) + 2, H_feat, W_feat)

        """
        # construct normalized x,y coordinates
        feat_bz, _, feat_H, feat_W = fpn_feat.size()
        normalized_y_feat, normalized_x_feat = torch.meshgrid(
            torch.arange(feat_H), torch.arange(feat_W))
        normalized_x_feat = normalized_x_feat.type(torch.float) / feat_W
        normalized_y_feat = normalized_y_feat.type(torch.float) / feat_H
        # expand coordinates and repeats along batch & channel dimension
        normalized_x_feat = normalized_x_feat.repeat(feat_bz, 1, 1,
                                                     1).to(device)
        normalized_y_feat = normalized_y_feat.repeat(feat_bz, 1, 1,
                                                     1).to(device)
        # concatenate them along the channel dimension
        fpn_feat = torch.cat([fpn_feat, normalized_x_feat, normalized_y_feat],
                             dim=1)

        del normalized_x_feat, normalized_y_feat

        return fpn_feat

    # This function forward a single level of fpn_featmap through the network
    # Input:
    # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
    # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # ori_size: [ori_H, ori_W]    (for eval=True upsampling)
    # Output:
    # if eval==False
    # cate_pred: (bz,C-1,S,S)
    # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
    # if eval==True
    # cate_pred: (bz,S,S,C-1) / after point_NMS
    # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self,
                             fpn_feat,
                             idx,
                             device="cpu",
                             eval=False,
                             upsample_shape=None,
                             ori_size=None):
        # upsample_shape is used in eval mode
        # Notice, we distinguish the training and inference.
        _, _, fpn_H_feat, fpn_W_feat = fpn_feat.shape
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        del fpn_feat

        num_grid = self.seg_num_grids[idx]  # current level grid

        # Forward category head
        for i in range(len(self.cate_head)):
            cate_pred = self.cate_head[i](cate_pred)
        cate_pred = self.sigmoid(self.cate_out(cate_pred))
        # resize category branch output to SxS
        cate_pred = nn.functional.interpolate(cate_pred,
                                              torch.Size([num_grid, num_grid]),
                                              mode='bilinear',
                                              align_corners=False)

        # Forward mask head
        # append normalized x, y coordinates to current input
        ins_pred = self._append_xy_coordinates(ins_pred, device)
        # forward the constructed layer
        for i in range(len(self.ins_head)):
            ins_pred = self.ins_head[i](ins_pred)
        ins_pred = self.sigmoid(self.ins_out_list[idx](ins_pred))

        # upsampling to 2*H_feat, 2*W_feat
        _, _, H_feat, W_feat = ins_pred.size()
        ins_pred = nn.functional.interpolate(ins_pred,
                                             torch.Size(
                                                 [2 * H_feat, 2 * W_feat]),
                                             mode='bilinear',
                                             align_corners=False)

        # in inference time, upsample the pred to (ori image size/4)
        if eval:
            ori_H, ori_W = ori_size
            # resize ins_pred
            ins_pred = nn.functional.interpolate(ins_pred,
                                                 torch.Size(
                                                     [ori_H // 4, ori_W // 4]),
                                                 mode='bilinear',
                                                 align_corners=False)
            cate_pred = self.points_nms(cate_pred).permute(0, 2, 3, 1)

        # check flag
        if not eval:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_H_feat * 2,
                                          fpn_W_feat * 2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
    # heat: (bz,C-1, S, S)
    # Output:
    # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel),
                                        stride=1,
                                        padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
    # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
    # cate_loss, mask_loss, total_loss
    def loss(self, cate_pred_list, ins_pred_list, ins_gts_list,
             ins_ind_gts_list, cate_gts_list, device):
        # compute loss, vectorize this part will help a lot. To avoid potential
        # ill-conditioning, if necessary, add a very small number to denominator for
        # focalloss and diceloss computation.

        # uniform the expression for ins_gts & ins_preds
        # ins_gts: list, len(fpn), (active_across_batch, 2H_feat, 2W_feat)
        # ins_preds: list, len(fpn), (active_across_batch, 2H_feat, 2W_feat)
        ins_gts = [
            torch.cat([
                ins_labels_level_img[ins_ind_labels_level_img, ...]
                for ins_labels_level_img, ins_ind_labels_level_img in zip(
                    ins_labels_level, ins_ind_labels_level)
            ], 0).to(device) for ins_labels_level, ins_ind_labels_level in zip(
                zip(*ins_gts_list), zip(*ins_ind_gts_list))
        ]
        ins_preds = [
            torch.cat([
                ins_preds_level_img[ins_ind_labels_level_img, ...]
                for ins_preds_level_img, ins_ind_labels_level_img in zip(
                    ins_preds_level, ins_ind_labels_level)
            ], 0).to(device) for ins_preds_level, ins_ind_labels_level in zip(
                ins_pred_list, zip(*ins_ind_gts_list))
        ]

        del ins_pred_list, ins_gts_list, ins_ind_gts_list

        # TODO: consider changing this to MultiApply()
        L_mask = 0.0
        count = 0
        for fpn in range(len(ins_gts)):
            for i in range(ins_gts[fpn].shape[0]):
                L_mask += self.DiceLoss(ins_preds[fpn][i], ins_gts[fpn][i],
                                        device)
                count += 1
        L_mask /= count

        del ins_gts, ins_preds

        # uniform the expression for cate_gts & cate_preds
        # cate_gts: (bz*fpn*S^2,), img, fpn, grids
        # cate_preds: (bz*fpn*S^2, C-1), ([img, fpn, grids], C-1)
        cate_gts = [
            torch.cat([
                cate_gts_level_img.flatten()
                for cate_gts_level_img in cate_gts_level
            ]) for cate_gts_level in zip(*cate_gts_list)
        ]
        cate_gts = torch.cat(cate_gts)
        cate_preds = [
            cate_pred_level.permute(0, 2, 3,
                                    1).reshape(-1, self.cate_out_channels)
            for cate_pred_level in cate_pred_list
        ]
        cate_preds = torch.cat(cate_preds, 0)

        del cate_pred_list, cate_gts_list

        L_cate = self.FocalLoss(cate_preds, cate_gts,
                                device) / cate_gts.shape[0]  # normalize?

        del cate_gts, cate_preds

        mask_lambda = 3
        return L_cate + mask_lambda * L_mask, L_cate, L_mask

    # This function compute the DiceLoss
    # Input:
    # mask_pred: (2H_feat, 2W_feat)
    # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar

    def DiceLoss(self, mask_pred, mask_gt, device):
        # Inputs are torch ndarrays
        numerator = torch.sum(2 * mask_gt * mask_pred).to(device)
        denominator = torch.sum(mask_gt**2 + mask_pred**2).to(device) + 1e-10

        del mask_pred, mask_gt

        return 1 - numerator / denominator

    # This function compute the cate loss
    # Input:
    # cate_preds: (num_entry, C-1)
    # cate_gts: (num_entry,)
    # Output: focal_loss, scalar

    def FocalLoss(self, cate_preds, cate_gts, device):
        # Inputs are torch ndarrays
        n, c = cate_preds.shape
        cate_gts_onehot = torch.zeros((n, c + 1), dtype=int).to(device)
        cate_gts_onehot[torch.arange(n), cate_gts] = 1
        cate_gts_onehot = cate_gts_onehot[:, 1:].flatten()
        p = cate_preds.flatten()

        del cate_preds, cate_gts

        alphat = abs(1 - cate_gts_onehot - self.cate_loss_cfg['alpha'])

        pt = abs(1 - cate_gts_onehot - p) + 1e-10

        fl = -alphat * (1 - pt)**self.cate_loss_cfg['gamma'] * torch.log(pt)
        return torch.sum(fl).to(device)

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # / ins_pred_list is only used to record feature map
    # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
    # label_list: list, len(batch_size), each (n_object, )
    # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list,
               device=torch.device("cpu")):
        # use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list.
        #       Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output
        #   in training

        featmap_size_list = [
            ins_pred_list[i].size() for i in range(len(ins_pred_list))
        ]

        del ins_pred_list

        ins_gts_list, ins_ind_gts_list, cate_gts_list = \
            self.MultiApply(self.target_single_img,
                            bbox_list,
                            label_list,
                            mask_list,
                            featmap_sizes=featmap_size_list,
                            device=device)

        del featmap_size_list, bbox_list, label_list, mask_list

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2, )
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1],
                                             self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list

    # -----------------------------------
    # process single image in one batch
    # -----------------------------------
    # input:
    # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
    # gt_labels_raw: n_obj,
    # gt_masks_raw: n_obj, H_ori, W_ori
    # featmap_sizes: list of shapes of featmap
    # output:
    # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
    # cate_label_list: list, len: len(FPN), (S, S)
    # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None,
                          device=torch.device("cpu")):
        # finish single image target build
        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        # compute the area of every object in this single image
        bbox_w = gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]
        bbox_h = gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]
        scales = torch.sqrt(bbox_w * bbox_h)

        del gt_bboxes_raw

        # group objects in the same level together
        fpn_objects = [[] for _ in range(len(featmap_sizes))]
        for i, scale in enumerate(scales):
            if scale < 96:
                fpn_objects[0].append(i)
            if scale >= 48 and scale < 192:
                fpn_objects[1].append(i)
            if scale >= 96 and scale < 384:
                fpn_objects[2].append(i)
            if scale >= 192 and scale < 768:
                fpn_objects[3].append(i)
            if scale >= 384:
                fpn_objects[4].append(i)

        del scale

        #  in each level, for all belonging objects, construct true labels
        for i, group in enumerate(fpn_objects):
            # extract dimensions
            _, num_grid_2, H_feat, W_feat = featmap_sizes[i]
            num_grid = int(np.sqrt(num_grid_2))
            # initialize outputs
            ins_label = torch.zeros(num_grid_2, H_feat, W_feat, device=device)
            cate_label = torch.zeros(num_grid, num_grid, device=device)
            ins_ind_label = torch.zeros(num_grid_2,
                                        dtype=torch.bool,
                                        device=device)

            # iterate over each object in the level
            for object_idx in group:
                # find their center point and center region
                mask_rescaled = gt_masks_raw[object_idx].unsqueeze(
                    0).unsqueeze(1).type(torch.float)
                mask_rescaled = F.interpolate(mask_rescaled,
                                              torch.Size([H_feat, W_feat]),
                                              mode='bilinear',
                                              align_corners=False)
                mask_rescaled = torch.squeeze(mask_rescaled)

                c_y, c_x = ndimage.center_of_mass(mask_rescaled.cpu().numpy())
                w = 0.2 * bbox_w[object_idx]
                h = 0.2 * bbox_h[object_idx]

                # Find their active grids
                # compute center grid
                center_grid_w = int(c_x / W_feat * num_grid)
                center_grid_h = int(c_y / H_feat * num_grid)
                # compute top, bottom, left right grid index
                top_grid_idx = max(0, int((c_y - h / 2) / H_feat * num_grid))
                bottom_grid_idx = min(num_grid - 1,
                                      int((c_y + h / 2) / H_feat * num_grid))
                left_grid_idx = max(0, int((c_x - w / 2) / W_feat * num_grid))
                right_grid_idx = min(num_grid - 1,
                                     int((c_x + w / 2) / W_feat * num_grid))
                # constrain the region ob active grid within 3x3
                top_idx = max(top_grid_idx, center_grid_h - 1)
                bottom_idx = min(bottom_grid_idx, center_grid_h + 1)
                left_idx = max(left_grid_idx, center_grid_w - 1)
                right_idx = min(right_grid_idx, center_grid_w + 1)

                # Activate the cate_label
                cate_label[top_idx:bottom_idx + 1, left_idx:right_idx + 1] = \
                    gt_labels_raw[object_idx]

                # Activate the ins_ind_label
                for idx in range(top_idx, bottom_idx + 1):
                    ins_ind_label[num_grid * idx + left_idx:num_grid * idx +
                                  right_idx + 1] = True

                # Assign the ins_label
                ins_label[ins_ind_label] = mask_rescaled

                del mask_rescaled

            # append the output labels to list
            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)
            cate_label_list.append(cate_label.type(torch.long))

            del ins_label, ins_ind_label, cate_label

        del group, gt_masks_raw, gt_labels_raw

        # check flag
        assert ins_label_list[1].shape == (1296, 200, 272)
        assert ins_ind_label_list[1].shape == (1296, )
        assert cate_label_list[1].shape == (36, 36)

        # make ins_ind_label_list bool, cate_label_list long,
        #   and transfer all of them to device
        return ins_label_list, ins_ind_label_list, cate_label_list

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size,
                    device,
                    cate_thresh=0.5,
                    ins_thresh=0.5):
        """
        This function receive pred list from forward and post-process

        Input:
        -----
            ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
            cate_pred_list: list, len(fpn), (bz,S,S,C-1)
            ori_size: [ori_H, ori_W]

        Output:
        -----
            NMS_sorted_scores_list, list, len(bz), (keep_instance,)
            NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
            NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        """

        # finish PostProcess
        ori_H, ori_W = ori_size

        # For each FPN level we convert each category prediction tensor
        #   with size (S, S, C-1) to a tensor with size (S2,C-1)
        #   (similar, to indexing of the mask prediction)
        #
        for i in range(len(cate_pred_list)):
            bz, _, _, C = cate_pred_list[i].size()
            cate_pred_list[i] = cate_pred_list[i].view(bz, -1, C)

        # For each image we concatenate the reshaped category tensors,
        #   and the mask tensors from all the levels to get:
        #   ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        #   cate_pred_img: (all_level_S^2, C-1)
        #
        cate_pred_img = cate_pred_list[0]
        ins_pred_img = ins_pred_list[0]
        for i_fpn in range(1, len(cate_pred_list)):
            cate_pred_img = torch.cat([cate_pred_img, cate_pred_list[i_fpn]],
                                      dim=1)
            ins_pred_img = torch.cat([ins_pred_img, ins_pred_list[i_fpn]],
                                     dim=1)

        del ins_pred_list, cate_pred_list

        ins_pred_img_list = [ins.to(device) for ins in ins_pred_img]
        cate_pred_img_list = [cate.to(device) for cate in cate_pred_img]

        del cate_pred_img, ins_pred_img

        NMS_sorted_score_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = \
            self.MultiApply(self.PostProcessImg,
                            ins_pred_img_list,
                            cate_pred_img_list,
                            ori_size=ori_size,
                            cate_thresh=self.postprocess_cfg['cate_thresh'],
                            ins_thresh=self.postprocess_cfg['ins_thresh'],
                            device=device)

        return NMS_sorted_score_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list

    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size=None,
                       cate_thresh=0.5,
                       ins_thresh=0.5,
                       device=torch.device("cpu")):
        """
        This function Postprocess on single img

        Input:
        -----
            ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
            cate_pred_img: (all_level_S^2, C-1)

        Output:
        -----
            NMS_sorted_scores, (keep_instance,)
            NMS_sorted_cate_label, (keep_instance,)
            NMS_sorted_ins, (keep_instance, ori_H, ori_W)
        """

        # PostProcess on single image.

        # Find the grid cells which have maximum category prediction c_max >cate_thresh
        cate_max = torch.max(cate_pred_img, dim=1).values
        cate_max_idx = (cate_max > cate_thresh).nonzero(as_tuple=True)[0]

        # For each one of the grid cell with c_max > cate_thresh, and mask m, find their score
        scores = torch.zeros(ins_pred_img.size()[0]).to(device)
        for i in cate_max_idx:
            c_max = cate_max[i]

            # avoid divide-by-0 error
            if torch.sum(ins_pred_img[i] > ins_thresh).item() > 0:
                scores[i] = c_max * torch.sum(ins_pred_img[i] * (ins_pred_img[i] > ins_thresh)) / \
                    torch.sum(ins_pred_img[i] > ins_thresh).item()

        del cate_max, cate_max_idx

        # sort the masks according to their scores
        scores_sorted, scores_sorted_idx = torch.sort(scores, descending=True)
        idx_nonzero = torch.nonzero(scores_sorted, as_tuple=False).size()[0]
        ins_sorted_pred_img = ins_pred_img[scores_sorted_idx][:idx_nonzero]
        cate_sorted_pred_img = cate_pred_img[scores_sorted_idx][:idx_nonzero]

        del scores, scores_sorted_idx
        del ins_pred_img, cate_pred_img

        # apply matrix NMS to get NMS scores
        matrix_NMS_scores = self.MatrixNMS(ins_sorted_pred_img,
                                           scores_sorted[:idx_nonzero],
                                           device,
                                           ins_thresh=ins_thresh)

        del scores_sorted

        # only keep the masks with the k highest NMS scores
        nms_sorted_scores, nms_sorted_idx = torch.sort(matrix_NMS_scores,
                                                       descending=True)
        keep_instance = self.postprocess_cfg['keep_instance']
        nms_sorted_cate_label = \
            cate_sorted_pred_img[nms_sorted_idx][:keep_instance]
        nms_sorted_ins = \
            ins_sorted_pred_img[nms_sorted_idx][:keep_instance]

        del ins_sorted_pred_img, cate_sorted_pred_img, matrix_NMS_scores
        del nms_sorted_idx

        # resize the images back to (H, W)
        nms_sorted_ins = F.interpolate(nms_sorted_ins.unsqueeze(0), ori_size)
        nms_sorted_ins = torch.squeeze(nms_sorted_ins)

        return nms_sorted_scores, nms_sorted_cate_label, nms_sorted_ins

    def MatrixNMS(self,
                  sorted_ins,
                  sorted_scores,
                  device,
                  method='gauss',
                  gauss_sigma=0.5,
                  ins_thresh=0.5):
        """
        This function perform matrix NMS

        Input:
        -----
            sorted_ins: (n_act, ori_H/4, ori_W/4)
            sorted_scores: (n_act,)
        Output:
        -----
            decay_scores: (n_act,)
        """
        # finish MatrixNMS
        n_act = len(sorted_scores)

        # make the mask hard for sorted_ins
        sorted_ins = (sorted_ins > ins_thresh).type(torch.float)

        # reshape mask for computation
        #   (n_act, H, W) --> (n_act, H*W)
        sorted_ins = sorted_ins.view(n_act, -1)

        # compute the IOU for all masks
        intersection = sorted_ins @ sorted_ins.T
        areas = torch.sum(sorted_ins, dim=1).expand(n_act, n_act)
        union = areas + areas.T - intersection
        # only keeps the upper right triangle, and eliminates the main diagonal
        #   (IOU(i, i) = 0)
        ious = (intersection / union).triu(diagonal=1)

        del sorted_ins

        # IOU(*, i) for (n_act, n_act)
        #   take i as row and k as column
        #   for each i, all non-zero (i, k) represents IOU(i, k) where s_k > s_i
        #
        # ious_i == IOU(*, i)
        #
        #   since we would like to minimize f(IOU(*, i)), we wish to find max(IOU(*, i))
        #
        ious_i = torch.max(ious, dim=0).values
        ious_i = ious_i.expand(n_act, n_act).T

        # Matrix NMS
        if method == 'gauss':
            decay = torch.exp(-(ious**2 - ious_i**2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_i)

        # min of f(IOU(i, j)) / f(IOU(*, i))
        decay = torch.min(decay, dim=0).values
        return decay * sorted_scores

    # -----------------------------------
    # The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # color_list: list, len(C-1)
    # img: (bz,3,Ori_H, Ori_W)
    # self.strides: [8,8,16,32,32]

    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img,
               iter=None):
        # target image recover, for each image, recover their segmentation in 5 FPN levels.
        # This is an important visual check flag.
        bz, _, ori_h, ori_w = img.size()
        for batch_id in range(bz):
            for fpn_id in range(len(ins_gts_list[batch_id])):
                # retrive num_grid in this level of FPN
                num_grid = cate_gts_list[batch_id][fpn_id].size()[0]
                # get activated category grid cell coordinates
                activated_grid_idx_list = cate_gts_list[batch_id][
                    fpn_id].nonzero(as_tuple=False)
                # iterate through each grid cell coordinates
                classes = [0]
                masks = []
                for grid_idx in activated_grid_idx_list:
                    if cate_gts_list[batch_id][fpn_id][
                            grid_idx[0], grid_idx[1]] not in classes:
                        classes = cate_gts_list[batch_id][fpn_id][grid_idx[0],
                                                                  grid_idx[1]]
                        mask = ins_gts_list[batch_id][fpn_id][grid_idx[0] *
                                                              num_grid +
                                                              grid_idx[1]]
                        mask = torch.squeeze(
                            F.interpolate(
                                mask.unsqueeze(0).unsqueeze(1),
                                torch.Size([ori_h, ori_w])))
                        masks.append(mask)

                if masks:
                    masks = torch.stack(masks)
                    outim = visual_bbox_mask(img[batch_id], masks)
                else:
                    outim = visual_bbox_mask(img[batch_id])

                cv2.imwrite(
                    "./testfig/visual_plotGT_" + str(iter) + "_" +
                    str(batch_id) + "_" + str(fpn_id) + ".png", outim)
                cv2.imshow("visualize target", outim)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # This function plot the inference segmentation in img
    # Input:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    # color_list: ["jet", "ocean", "Spectral"]
    # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self, NMS_sorted_scores_list, NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list, color_list, img, iter_ind):
        # Plot predictions
        # target image recover, for each image, recover their segmentation in 5 FPN levels.
        # This is an important visual check flag.
        bz, _, ori_h, ori_w = img.size()
        for batch_id in range(bz):
            masks = []
            labels = []
            if NMS_sorted_ins_list[batch_id].shape[0] > 0:
                for ins_id in range(NMS_sorted_ins_list[batch_id].shape[0]):
                    class_id = NMS_sorted_cate_label_list[batch_id][ins_id]
                    labels.append(int(torch.argmax(class_id)))
                    mask = NMS_sorted_ins_list[batch_id][ins_id]
                    mask = (mask > self.postprocess_cfg['ins_thresh']).type(
                        torch.float)
                    masks.append(mask)

            masks = torch.stack(masks)
            outim = visual_bbox_mask(img[batch_id], masks, labels=labels)

            cv2.imwrite(
                "./testfig/visual_plotInfer_" + str(iter_ind) + "_" +
                str(batch_id) + "_" + str(ins_id) + ".png", outim)
            # cv2.imshow("visualize infer", outim)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def solo_evaluation(self, NMS_sorted_scores_list,
                        NMS_sorted_cate_label_list, NMS_sorted_ins_list,
                        label_list, mask_list):
        """
        Constructs matches list & scores list for every class

        Input:
        -----
            NMS_sorted_scores_list, list, len(bz), (keep_instance,)
            NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
            NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
            label_list: list, len(batch_size), each (n_object, )
            mask_list: list, len(batch_size), each (n_object, 800, 1088)

        Output:
        -----
            match       - shape (N, 5, 3)   - matches with respect to true labels for all classes
            score       - shape (N, 5, 3)   - scores with respect to true labels for all classes
            num_true        - shape (3,)    - total trues per class
            num_positive    - shape (3,)    - total positives
        """
        match = torch.zeros(len(NMS_sorted_cate_label_list),
                            self.postprocess_cfg['keep_instance'],
                            self.cate_out_channels)
        score = torch.zeros(match.size())
        num_true = torch.zeros(self.postprocess_cfg['keep_instance'])
        num_positive = torch.zeros(self.postprocess_cfg['keep_instance'])

        batch_size = len(label_list)

        for bz in range(batch_size):
            # calculate trues
            label = label_list[bz]
            num_true[label[label > 0].type(torch.long) - 1] += 1

            # calculate positives
            cate = torch.argmax(NMS_sorted_cate_label_list[bz], dim=1)
            num_positive[cate.type(torch.long)] += 1

            for i_pred, class_pred in enumerate(cate):
                if class_pred + 1 in label:

                    # retrieve class gt label
                    i_gt_list = (label == class_pred +
                                 1).nonzero(as_tuple=False)

                    for i_gt in i_gt_list:
                        # retrieve masks
                        mask_pred = NMS_sorted_ins_list[bz][i_pred]
                        # create a hard mask for mask_pred
                        mask_pred = (
                            mask_pred
                            > self.postprocess_cfg['ins_thresh']).type(
                                torch.float)
                        mask_gt = mask_list[bz][i_gt]

                        # compute IOU
                        intersection = torch.sum(mask_pred * mask_gt)
                        iou = intersection / (torch.sum(mask_pred) +
                                              torch.sum(mask_gt) -
                                              intersection)

                        if iou > self.postprocess_cfg['IoU_thresh']:
                            match[bz, i_pred, class_pred] = 1
                            score[bz, i_pred,
                                  class_pred] = NMS_sorted_scores_list[bz][
                                      i_pred]

        return match, score, num_true, num_positive

    def average_precision(self,
                          match_values,
                          score_values,
                          total_trues,
                          total_positives,
                          threshold=0.6):
        """
        Input:
        -----
            match_values - shape (N,5) - matches with respect to true labels for a single class
            score_values - shape (N,5) - objectness for a single class
            total_trues     - int      - total number of true labels for a single class in the
                                         entire dataset
            total_positives - int      - total number of positive labels for a single class in the
                                        entire dataset

        Output:
        -----
            area, sorted_recall, sorted_precision
        """
        # please fill in the appropriate arguments
        # compute the average precision as mentioned in the PDF.
        # it might be helpful to use - from sklearn.metrics import auc
        #   to compute area under the curve.
        area, sorted_recall, sorted_precision = None, None, None

        max_score = torch.max(score_values).item()
        ln = torch.linspace(threshold, max_score, steps=100)
        precision_mat = torch.zeros(101)
        recall_mat = torch.zeros(101)

        # iterate through the linspace
        for i, th in enumerate(ln):
            matches = match_values[score_values > th]
            TP = torch.sum(matches)  # true positives
            precision = 1
            if total_positives > 0:
                precision = TP / total_positives

            recall = 1
            if total_trues > 0:
                recall = TP / total_trues

            precision_mat[i] = precision
            recall_mat[i] = recall

        recall_mat[100] = 0
        precision_mat[100] = 1
        sorted_idx = torch.argsort(recall_mat)
        sorted_recall = recall_mat[sorted_idx]
        sorted_precision = precision_mat[sorted_idx]
        area = auc(sorted_recall, sorted_precision)

        return area, sorted_recall, sorted_precision


from backbone import Resnet50Backbone
from tqdm import tqdm
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0)
    test_loader = test_build_loader.loader()

    del train_dataset, test_dataset

    # detect device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    resnet50_fpn = Resnet50Backbone().to(device)
    # class number is 4, because consider the background as one category.
    solo_head = SOLOHead(num_classes=4).to("cpu")
    # loop the image
    for iter, data in tqdm(enumerate(train_loader, 0)):
        img, label_list, mask_list, bbox_list = [
            data[i] for i in range(len(data))
        ]
        img = img.to(device)

        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = [val.cpu() for val in list(backout.values())]
        # make the target

        # demo
        cate_pred_list, ins_pred_list = solo_head.forward(
            fpn_feat_list,
            torch.device("cpu"),
            eval=True,
            ori_size=img.size()[-2:])
        solo_head.PostProcess(ins_pred_list,
                              cate_pred_list,
                              ori_size=img.size()[-2:])

        del img, label_list, mask_list, bbox_list
        del backout, fpn_feat_list, cate_pred_list, ins_pred_list

        # ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
        #                                                                  bbox_list,
        #                                                                  label_list,
        #                                                                  mask_list)
        # mask_color_list = ["jet", "ocean", "Spectral"]
        # solo_head.PlotGT(ins_gts_list, ins_ind_gts_list,
        #                  cate_gts_list, mask_color_list, img, iter=iter)
