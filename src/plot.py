import torch
import os
from matplotlib import pyplot as plt

from solo_head import IN_COLAB
if not IN_COLAB:
    from solo_head import coloredlogs

COLAB_ROOT = "/content/drive/My Drive/CIS680_2019/SOLO-FPN"


if __name__ == "__main__":
    if not IN_COLAB:
        coloredlogs.install(level='INFO')
    # file path and make a list
    imgs_path = 'data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "data/hw3_mycocodata_bboxes_comp_zlib.npy"
    checkpoints_path = "checkpoints/"
    mAP_path = "mAP/"
    figure_path = "outfig/"

    if IN_COLAB:
        imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        masks_path = os.path.join(COLAB_ROOT, masks_path)
        labels_path = os.path.join(COLAB_ROOT, labels_path)
        bboxes_path = os.path.join(COLAB_ROOT, bboxes_path)
        checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)
        mAP_path = os.path.join(COLAB_ROOT, mAP_path)
        figure_path = os.path.join(COLAB_ROOT, figure_path)

    path = os.path.join(mAP_path, 'mAP')
    list_sorted_recall = torch.load(path)['sorted_recalls']
    list_sorted_precision = torch.load(path)['sorted_precisions']
    list_AP = torch.load(path)['AP']

    print(list_AP)

    os.makedirs(figure_path, exist_ok=True)

    for i, sorted_recall in enumerate(list_sorted_recall):
        plt.figure()
        plt.plot(sorted_recall, list_sorted_precision[i], '.-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(figure_path + "class_" + str(i) + ".png")
