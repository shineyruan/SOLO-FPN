import torch
import cv2
from backbone import Resnet50Backbone
from tqdm import tqdm
from dataset import BuildDataLoader, BuildDataset, visual_bbox_mask
from solo_head import SOLOHead


if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    del train_dataset, test_dataset

    # detect device
    resnet_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solo_device = torch.device("cpu")
    torch.cuda.empty_cache()

    resnet50_fpn = Resnet50Backbone(device=resnet_device, eval=True)
    # class number is 4, because consider the background as one category.
    solo_head = SOLOHead(num_classes=4).to(solo_device)

    learning_rate = 0.02 / batch_size
    optimizer = torch.optim.SGD(solo_head.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=1e-4)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 33], gamma=0.1)

    num_epochs = 36
    for epochs in range(num_epochs):
        # loop the image
        for iter, data in tqdm(enumerate(train_loader, 0)):
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
            img = img.to(resnet_device)

            # fpn is a dict
            backout = resnet50_fpn(img)
            if solo_device == torch.device("cpu"):
                fpn_feat_list = [val.cpu() for val in list(backout.values())]
            else:
                fpn_feat_list = [val for val in list(backout.values())]

            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list,
                                                              solo_device,
                                                              eval=False,
                                                              ori_size=img.size()[-2:])

            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                             bbox_list,
                                                                             label_list,
                                                                             mask_list)
            ins_ind_gts_list = [[fpn.type(torch.bool) for fpn in fpnlist]
                                for fpnlist in ins_ind_gts_list]
            cate_gts_list = [[fpn.type(torch.long) for fpn in fpnlist]
                             for fpnlist in ins_ind_gts_list]

            optimizer.zero_grad()
            loss = solo_head.loss(cate_pred_list,
                                  ins_pred_list,
                                  ins_gts_list,
                                  ins_ind_gts_list,
                                  cate_gts_list)
            print("loss =", loss.item())
            loss.backward()
            optimizer.step()

            del img, label_list, mask_list, bbox_list
            del backout, fpn_feat_list, cate_pred_list, ins_pred_list

        scheduler.step()