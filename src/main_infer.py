import torch
from backbone import Resnet50Backbone
from tqdm import tqdm
from dataset import BuildDataLoader, BuildDataset, visual_bbox_mask
from solo_head import SOLOHead, logging, sys, IN_COLAB
import os

import time

if not IN_COLAB:
    from solo_head import coloredlogs

COLAB_ROOT = "/content/drive/My Drive/CIS680_2019/SOLO-FPN"

if __name__ == '__main__':
    if not IN_COLAB:
        coloredlogs.install(level='INFO')
    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "data/hw3_mycocodata_bboxes_comp_zlib.npy"
    checkpoints_path = "checkpoints/"
    mAP_path = "mAP/"

    if IN_COLAB:
        imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        masks_path = os.path.join(COLAB_ROOT, masks_path)
        labels_path = os.path.join(COLAB_ROOT, labels_path)
        bboxes_path = os.path.join(COLAB_ROOT, bboxes_path)
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)
        mAP_path = os.path.join(COLAB_ROOT, mAP_path)

    os.makedirs(checkpoints_path, exist_ok=True)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset; set seed
    train_dataset, test_dataset = \
        torch.utils.data.random_split(dataset, [train_size, test_size],
                                      generator=torch.random.manual_seed(1))
    # push the randomized training data into the dataloader

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
    resnet_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    solo_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    if not IN_COLAB:
        # if not in Google Colab, save GPU memory for SOLO head
        # solo_device = torch.device("cpu")
        pass

    torch.cuda.empty_cache()

    resnet50_fpn = Resnet50Backbone(device=resnet_device, eval=True)
    # class number is 4, because consider the background as one category.
    solo_head = SOLOHead(num_classes=4).to(solo_device).eval()

    # whether to resume training
    load_epoch = 35

    path = os.path.join(checkpoints_path, 'solo_epoch' + str(load_epoch))
    print("Loading checkpoint:", path)
    checkpoint = torch.load(path)
    solo_head.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint

    count = 0
    avg_loss = 0

    trues_per_batch = []
    positives_per_batch = []
    match_values = []
    score_values = []

    # loop the image
    progress_bar = tqdm(enumerate(test_loader, 0))
    for iter, data in progress_bar:
        with torch.no_grad():
            img, label_list, mask_list, bbox_list = [
                data[i] for i in range(len(data))
            ]
            img = img.to(resnet_device)

            # fpn is a dict
            backout = resnet50_fpn(img)

            ori_size = img.size()[-2:]

            if solo_device == torch.device(
                    "cpu") and resnet_device == torch.device("cuda:0"):
                fpn_feat_list = [val.cpu() for val in list(backout.values())]
            elif solo_device == torch.device(
                    "cuda:0") and resnet_device == torch.device("cpu"):
                fpn_feat_list = [val.cuda() for val in list(backout.values())]
            else:
                fpn_feat_list = [val for val in list(backout.values())]

            del backout

            cate_pred_list, ins_pred_list = solo_head.forward(
                fpn_feat_list, solo_device, eval=True, ori_size=ori_size)

            del fpn_feat_list

            # compute the target on GPU
            bbox_list = [item.to(solo_device) for item in bbox_list]
            label_list = [item.to(solo_device) for item in label_list]
            mask_list = [item.to(solo_device) for item in mask_list]

            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(
                ins_pred_list,
                bbox_list,
                label_list,
                mask_list,
                device=solo_device)

            del bbox_list

            ins_ind_gts_list = [[fpn.to(solo_device) for fpn in fpn_list]
                                for fpn_list in ins_ind_gts_list]
            cate_gts_list = [[fpn.to(solo_device) for fpn in fpn_list]
                             for fpn_list in cate_gts_list]

            loss, *_ = solo_head.loss(cate_pred_list, ins_pred_list,
                                      ins_gts_list, ins_ind_gts_list,
                                      cate_gts_list, solo_device)

            avg_loss += loss.item()
            count += 1

            del ins_gts_list, ins_ind_gts_list, cate_gts_list

            ins_pred_list_cpu = [item.cpu() for item in ins_pred_list]
            cate_pred_list_cpu = [item.cpu() for item in cate_pred_list]
            label_list_cpu = [item.cpu() for item in label_list]
            mask_list_cpu = [item.cpu() for item in mask_list]

            del cate_pred_list, ins_pred_list, label_list, mask_list

            NMS_sorted_score_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = \
                solo_head.PostProcess(ins_pred_list_cpu, cate_pred_list_cpu,
                                      ori_size, device=torch.device("cpu"))

            match, score, num_true, num_positive = \
                solo_head.solo_evaluation(NMS_sorted_score_list,
                                          NMS_sorted_cate_label_list,
                                          NMS_sorted_ins_list,
                                          label_list_cpu, mask_list_cpu)

            solo_head.PlotInfer(NMS_sorted_score_list,
                                NMS_sorted_cate_label_list,
                                NMS_sorted_ins_list,
                                color_list=["jet", "ocean", "Spectral"],
                                img=img,
                                iter_ind=iter)

            del ins_pred_list_cpu, cate_pred_list_cpu
            del label_list_cpu, mask_list_cpu

            del img

            trues_per_batch.append(num_true)
            positives_per_batch.append(num_positive)
            match_values.append(match)
            score_values.append(score)

            progress_bar.set_description("loss = %f" % loss.item())

    avg_loss /= count
    print("test avg loss = {}".format(avg_loss))

    # organize data
    trues_per_batch = torch.stack(trues_per_batch)
    positives_per_batch = torch.stack(positives_per_batch)
    trues_per_batch = torch.sum(trues_per_batch, dim=0)
    positives_per_batch = torch.sum(positives_per_batch, dim=0)
    match_values = torch.cat(match_values)
    score_values = torch.cat(score_values)

    os.makedirs(mAP_path, exist_ok=True)
    path = os.path.join(mAP_path, 'matches')
    torch.save(
        {
            'trues per batch': trues_per_batch,
            'positives per batch': positives_per_batch,
            'match_values': match_values,
            'score_values': score_values
        }, path)

    # calculate mAP
    list_sorted_recall = []
    list_sorted_precision = []
    list_AP = []

    AP = 0
    cnt = 0
    for class_i in range(3):
        if torch.sum(match_values[:, :, class_i]) > 0:
            area, sorted_recall, sorted_precision = \
                solo_head.average_precision(match_values[:, :, class_i],
                                            score_values[:, :, class_i],
                                            trues_per_batch[class_i],
                                            positives_per_batch[class_i],
                                            threshold=0)
            AP += area
            cnt += 1

            list_sorted_recall.append(sorted_recall)
            list_sorted_precision.append(sorted_precision)
            list_AP.append(area)

    mAP = AP if cnt == 0 else AP / cnt
    # calculate mean loss
    print('testing mAP   {}'.format(mAP))

    path = os.path.join(mAP_path, 'mAP')
    torch.save(
        {
            'sorted_recalls': list_sorted_recall,
            'sorted_precisions': list_sorted_precision,
            'AP': list_AP,
            'mAP': mAP
        }, path)
