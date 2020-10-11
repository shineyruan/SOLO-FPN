import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import BuildDataLoader, BuildDataset
from functools import partial


class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384),
                               (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                    alpha=0.25,
                                    weight=1),
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
        self.cate_head.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                      out_channels=256,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      bias=False),
                                            nn.GroupNorm(num_groups=num_groups, num_channels=256),
                                            nn.ReLU()))
        #   add 6 intermediate convolution layers
        for _ in range(6):
            self.cate_head.append(nn.Sequential(nn.Conv2d(in_channels=256,
                                                          out_channels=256,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=1,
                                                          bias=False),
                                                nn.GroupNorm(num_groups=num_groups,
                                                             num_channels=256),
                                                nn.ReLU()))
        # add output convolution layer
        self.cate_out = nn.Conv2d(in_channels=256,
                                  out_channels=self.cate_out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=True)

        # Initialize mask branch
        self.ins_head.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels + 2,
                                                     out_channels=256,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False),
                                           nn.GroupNorm(num_groups=num_groups, num_channels=256),
                                           nn.ReLU()))
        # add 6 intermediate convolution layers
        for _ in range(6):
            self.ins_head.append(nn.Sequential(nn.Conv2d(in_channels=256,
                                                         out_channels=256,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1,
                                                         bias=False),
                                               nn.GroupNorm(num_groups=num_groups,
                                                            num_channels=256),
                                               nn.ReLU()))
        # add output convolution layer
        for num_grid in self.seg_num_grids:
            self.ins_out_list.append(nn.Conv2d(in_channels=256,
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
    def forward(self, fpn_feat_list, device, eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256, 100, 136)
        quart_shape = [new_fpn_list[0].shape[-2] * 2,
                       new_fpn_list[0].shape[-1] * 2]  # stride: 4
        # use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level,
                                                        new_fpn_list,
                                                        list(range(len(new_fpn_list))),
                                                        device=device,
                                                        eval=eval,
                                                        upsample_shape=quart_shape)

        assert len(new_fpn_list) == len(self.seg_num_grids)

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
        # resize first level to 1/2
        _, _, H, W = new_fpn_list[0].size()
        new_fpn_list[0] = nn.functional.interpolate(new_fpn_list[0], torch.Size([H // 2, W // 2]))
        # resize last level to 2
        _, _, H, W = new_fpn_list[-1].size()
        new_fpn_list[-1] = nn.functional.interpolate(new_fpn_list[-1], torch.Size([2 * H, 2 * W]))
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
        normalized_x_feat = torch.stack([torch.Tensor(range(feat_W))
                                         for _ in range(feat_H)]).to(device)
        normalized_x_feat /= feat_W
        normalized_y_feat = torch.stack([torch.Tensor([i for _ in range(feat_W)])
                                         for i in range(feat_H)]).to(device)
        normalized_y_feat /= feat_H
        # expand coordinates and repeats along batch & channel dimension
        normalized_x_feat = normalized_x_feat.repeat(feat_bz, 1, 1, 1)
        normalized_y_feat = normalized_y_feat.repeat(feat_bz, 1, 1, 1)
        # concatenate them along the channel dimension
        fpn_feat = torch.cat([fpn_feat, normalized_x_feat, normalized_y_feat], dim=1)

        return fpn_feat

    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
        # cate_pred: (bz,C-1,S,S)
        # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
        # cate_pred: (bz,S,S,C-1) / after point_NMS
        # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, device="cpu", eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        # Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid

        # Forward category head
        for i in range(len(self.cate_head)):
            cate_pred = self.cate_head[i](cate_pred)
        cate_pred = self.sigmoid(self.cate_out(cate_pred))
        # resize category branch output to SxS
        cate_pred = nn.functional.interpolate(cate_pred, torch.Size([num_grid, num_grid]),
                                              mode='bilinear')

        # Forward mask head
        # append normalized x, y coordinates to current input
        ins_pred = self._append_xy_coordinates(ins_pred, device)
        # forward the constructed layer
        for i in range(len(self.ins_head)):
            ins_pred = self.ins_head[i](ins_pred)
        ins_pred = self.sigmoid(self.ins_out_list[idx](ins_pred))

        # upsampling to 2*H_feat, 2*W_feat
        _, _, H_feat, W_feat = ins_pred.size()
        ins_pred = nn.functional.interpolate(ins_pred, torch.Size([2 * H_feat, 2 * W_feat]))

        # in inference time, upsample the pred to (ori image size/4)
        if eval:
            # resize ins_pred
            ins_pred = nn.functional.interpolate(ins_pred, torch.Size([H_feat // 4, W_feat // 4]))
            cate_pred = self.points_nms(cate_pred).permute(0, 2, 3, 1)

        # check flag
        if not eval:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (
                num_grid**2, fpn_feat.shape[2] * 2, fpn_feat.shape[3] * 2)
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
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
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
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        # TODO: compute loss, vectorize this part will help a lot. To avoid potential
        #   ill-conditioning, if necessary, add a very small number to denominator for
        #   focalloss and diceloss computation.
        pass

    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar

    def DiceLoss(self, mask_pred, mask_gt):
        # TODO: compute DiceLoss
        pass

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        # TODO: compute focalloss
        pass

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
    def target(self, ins_pred_list, bbox_list, label_list, mask_list):
        # use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list.
        #       Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output
        #   in training

        featmap_size_list = [ins_pred_list[i].size() for i in range(len(ins_pred_list))]

        ins_gts_list, ins_ind_gts_list, cate_gts_list = \
            self.MultiApply(self.target_single_img,
                            bbox_list,
                            label_list,
                            mask_list,
                            featmap_sizes=featmap_size_list)

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

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
                          featmap_sizes=None):
        # finish single image target build
        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        # compute the area of every object in this single image
        bbox_w = gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]
        bbox_h = gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]
        scales = torch.sqrt(bbox_w * bbox_h)

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

        #  in each level, for all belonging objects, construct true labels
        for i, group in enumerate(fpn_objects):
            # extract dimensions
            _, num_grid_2, H_feat, W_feat = featmap_sizes[i]
            num_grid = int(np.sqrt(num_grid_2))
            # initialize outputs
            ins_label = torch.zeros(num_grid_2, H_feat, W_feat)
            cate_label = torch.zeros(num_grid, num_grid)
            ins_ind_label = torch.zeros(num_grid_2)

            # iterate over each object in the level
            for object_idx in group:
                # find their center point and center region
                c_y, c_x = ndimage.center_of_mass(gt_masks_raw[object_idx].numpy())
                w = 0.2 * bbox_w[object_idx]
                h = 0.2 * bbox_h[object_idx]

                # Find their active grids
                # compute center grid
                center_grid_w = int(c_x / W_feat * num_grid)
                center_grid_h = int(c_y / H_feat * num_grid)
                # compute top, bottom, left right grid index
                top_grid_idx = max(0, int((c_y - h / 2) / H_feat * num_grid))
                bottom_grid_idx = min(num_grid - 1, int((c_y + h / 2) / H_feat * num_grid))
                left_grid_idx = max(0, int((c_x - w / 2) / W_feat * num_grid))
                right_grid_idx = min(num_grid - 1, int((c_x + w / 2) / W_feat * num_grid))
                # constrain the region ob active grid within 3x3
                top_idx = max(top_grid_idx, center_grid_h - 1)
                bottom_idx = min(bottom_grid_idx, center_grid_h + 1)
                left_idx = max(left_grid_idx, center_grid_w - 1)
                right_idx = min(right_grid_idx, center_grid_w + 1)

                # Activate the cate_label
                cate_label[top_idx:bottom_idx + 1, left_idx:right_idx + 1] = \
                    gt_labels_raw[object_idx]

                # Assign the ins_label
                for idx in range(top_idx, bottom_idx + 1):
                    ins_label[num_grid * idx + left_idx:num_grid * idx + right_idx + 1] = \
                        nn.functional.interpolate(gt_masks_raw[object_idx],
                                                  torch.Size([H_feat, W_feat]))

                # Activate the ins_ind_label
                for idx in range(top_idx, bottom_idx + 1):
                    ins_ind_label[num_grid * idx + left_idx:num_grid * idx + right_idx + 1] = 1

            # append the output labels to list
            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)
            cate_label_list.append(cate_label)

        # check flag
        assert ins_label_list[1].shape == (1296, 200, 272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        # TODO: finish PostProcess
        pass

    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        # TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        # TODO: finish MatrixNMS
        pass

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
               img):
        # TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        # This is an important visual check flag.
        pass

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        # TODO: Plot predictions
        pass


from backbone import Resnet50Backbone
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
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # detect device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet50_fpn = Resnet50Backbone().to(device)
    # class number is 4, because consider the background as one category.
    solo_head = SOLOHead(num_classes=4).to("cpu")
    # loop the image
    for iter, data in enumerate(train_loader, 0):
        print("Iteration: %d", iter)
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)

        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = []
        for val in list(backout.values()):
            fpn_feat_list.append(val.cpu())
        # make the target

        # demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list,
                                                          torch.device("cpu"),
                                                          eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list, ins_ind_gts_list,
                         cate_gts_list, mask_color_list, img)
