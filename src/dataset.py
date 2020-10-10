# Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # load dataset, make mask list
        dataset_imgs = np.array([])
        dataset_masks = np.array([])
        dataset_labels = np.array([])
        dataset_bboxes = np.array([])

        dataset = [dataset_imgs, dataset_masks, dataset_labels, dataset_bboxes]

        for i, p in enumerate(path):
            _, ext = os.path.splitext(p)

            if ext == '.h5':
                f = h5py.File(p, 'r')
                dataset[i] = f['data']

            if ext == '.npy':
                dataset[i] = np.load(p, allow_pickle=True)

        # preprocess when accessing data
        self.imgs_data = dataset[0]  # h5py of numpy arrays
        self.label_data = dataset[2]  # numpy array of arrays
        self.bbox_data = dataset[3]  # numpy array of arrays

        self.masks_data = []  # list of numpy arrays
        mask_id = 0
        for img_id in range(self.label_data.shape[0]):
            mask_list = []
            for mask_id in range(self.label_data[img_id].shape[0]):
                mask_list.append(dataset[1][mask_id].astype(np.uint8))
                mask_id += 1
            mask_list = np.stack(mask_list)
            self.masks_data.append(mask_list)

    def __getitem__(self, index):
        # __getitem__: return torch.tensors
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(
            self.imgs_data[index], self.masks_data[index], self.bbox_data[index])
        label = torch.tensor(self.label_data[index])
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox

    def __len__(self):
        return len(self.imgs_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: n_box*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # image preprocess
        img = img.astype(float)
        img /= 255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = F.interpolate(img, size=1066)
        img = img.permute(0, 2, 1)
        img = F.interpolate(img, size=800)
        img = img.permute(0, 2, 1)
        img = transforms.functional.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = F.pad(img, pad=(11, 11))

        mask = torch.from_numpy(mask.astype(np.uint8))
        mask = F.interpolate(mask, size=1066)
        mask = mask.permute(0, 2, 1)
        mask = F.interpolate(mask, size=800)
        mask = mask.permute(0, 2, 1)
        mask = F.pad(mask, pad=(11, 11))

        bbox[:, 0] = bbox[:, 0] / 400 * 1066 + 11
        bbox[:, 2] = bbox[:, 2] / 400 * 1066 + 11
        bbox[:, 1] = bbox[:, 1] / 300 * 800
        bbox[:, 3] = bbox[:, 3] / 300 * 800
        bbox = torch.tensor(bbox)

        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
    def collect_fn(self, batch):
        # collect_fn
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(transed_img_list, dim=0),\
            label_list,\
            transed_mask_list,\
            transed_bbox_list

    def loader(self):
        # return a dataloader
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)

        # Visualize debugging


if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
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

    batch_size = 2
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        # plot the origin img
        for i in range(batch_size):
            # TODO: plot images with annotations
            plt.savefig("./testfig/visualtrainset" + str(iter) + ".png")
            plt.show()

        if iter == 10:
            break
