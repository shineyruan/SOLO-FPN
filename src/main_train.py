import torch
import cv2
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
        coloredlogs.install(level='DEBUG')
    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "data/hw3_mycocodata_bboxes_comp_zlib.npy"
    checkpoints_path = "checkpoints/"

    if IN_COLAB:
        imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        masks_path = os.path.join(COLAB_ROOT, masks_path)
        labels_path = os.path.join(COLAB_ROOT, labels_path)
        bboxes_path = os.path.join(COLAB_ROOT, bboxes_path)
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)

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
        solo_device = torch.device("cpu")

    torch.cuda.empty_cache()

    resnet50_fpn = Resnet50Backbone(device=resnet_device, eval=True)
    # class number is 4, because consider the background as one category.
    solo_head = SOLOHead(num_classes=4).to(solo_device).train()

    learning_rate = 0.02 / batch_size
    optimizer = torch.optim.SGD(solo_head.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[27, 33],
                                                     gamma=0.1)

    # whether to resume training
    resume = False
    resume_epoch = 17
    start_epoch = 0

    if resume:
        path = os.path.join(checkpoints_path, 'solo_epoch' + str(resume_epoch))
        print("Loading checkpoint:", path)
        checkpoint = torch.load(path)
        solo_head.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    list_avg_loss = []
    list_cate_loss = []
    list_dice_loss = []

    num_epochs = 36
    for epochs in range(start_epoch, num_epochs):
        count = 0
        avg_loss = 0
        avg_cate_loss = 0
        avg_dice_loss = 0

        # loop the image
        progress_bar = tqdm(enumerate(train_loader, 0))
        for iter, data in progress_bar:
            img, label_list, mask_list, bbox_list = [
                data[i] for i in range(len(data))
            ]
            img = img.to(resnet_device)

            # fpn is a dict
            with torch.no_grad():
                backout = resnet50_fpn(img)

            ori_size = img.size()[-2:]
            del img

            if solo_device == torch.device(
                    "cpu") and resnet_device == torch.device("cuda:0"):
                fpn_feat_list = [val.cpu() for val in list(backout.values())]
            elif solo_device == torch.device(
                    "cuda:0") and resnet_device == torch.device("cpu"):
                fpn_feat_list = [val.cuda() for val in list(backout.values())]
            else:
                fpn_feat_list = [val for val in list(backout.values())]

            del backout

            optimizer.zero_grad()
            cate_pred_list, ins_pred_list = solo_head.forward(
                fpn_feat_list, solo_device, eval=False, ori_size=ori_size)

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
            del label_list, mask_list, bbox_list

            ins_ind_gts_list = [[fpn.to(solo_device) for fpn in fpn_list]
                                for fpn_list in ins_ind_gts_list]
            cate_gts_list = [[fpn.to(solo_device) for fpn in fpn_list]
                             for fpn_list in cate_gts_list]

            loss, cate_loss, dice_loss = solo_head.loss(
                cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list,
                cate_gts_list, solo_device)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            avg_cate_loss += cate_loss.item()
            avg_dice_loss += dice_loss.item()
            count += 1

            progress_bar.set_description("loss = %f" % loss.item())

            del cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list

        scheduler.step()

        avg_loss /= count
        avg_cate_loss /= count
        avg_dice_loss /= count
        print(
            "Epoch: {}, loss = {}, cate_loss = {}, avg_dice_loss = {}".format(
                epochs, avg_loss, avg_cate_loss, avg_dice_loss))

        list_avg_loss.append(avg_loss)
        list_cate_loss.append(avg_cate_loss)
        list_dice_loss.append(avg_dice_loss)

        path = os.path.join(checkpoints_path, 'solo_epoch' + str(epochs))
        torch.save(
            {
                'epoch': epochs,
                'model_state_dict': solo_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_running_loss': avg_loss,
                'train_cate_loss': avg_cate_loss,
                'train_dice_loss': avg_dice_loss,
                'list_loss': list_avg_loss,
                'list_cate_loss': list_cate_loss,
                'list_dice_loss': list_dice_loss
            }, path)
