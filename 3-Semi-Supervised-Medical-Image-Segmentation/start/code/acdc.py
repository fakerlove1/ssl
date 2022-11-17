import os.path

import torch
import random
import numpy as np
from glob import glob

import torchvision.utils
from torch.utils.data import Dataset, DataLoader
import h5py
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt


class ACDC(Dataset):
    def __init__(self, root=r"E:\note\ssl\data\ACDC", mode="train", transform=None):
        super(ACDC, self).__init__()
        self.mode = mode
        self.root = root
        self.transform = transform
        if self.mode == 'train':
            with open(self.root + "/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.mode == "test":
            with open(self.root + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = np.array(Image.open(os.path.join(self.root, "image", case)))
        mask = np.array(Image.open(os.path.join(self.root, "mask", case)))

        if self.transform is not None:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
        return image, mask


def get_loader(root=r'E:\note\ssl\data\My-ACDC', batch_size=4, label=0.2):
    """

    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_both_aug = A.Compose([
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, p=1, mask_value=255),
        A.RandomCrop(height=256, width=256, p=1),
        A.Cutout(num_holes=8, p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(p=0.6),
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.8)
        ]),
        ToTensorV2()
    ])

    train_img_aug = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.5, 0.5), p=0.9),
            A.RandomGamma(gamma_limit=(50, 200), p=0.8)
        ]),
    ])

    val_both_aug = A.Compose([
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, p=1, mask_value=255),
        A.RandomCrop(height=256, width=256, p=1),
        ToTensorV2()
    ])

    train_dataset = ACDC(root=root, mode="train", transform=train_both_aug)
    l = int(len(train_dataset) * label)
    train_label, train_unlabel = torch.utils.data.random_split(dataset=train_dataset,
                                                               lengths=[l, len(train_dataset) - l])

    test_dataset = ACDC(root=root, mode="test", transform=val_both_aug)

    train_label_dataloader = DataLoader(train_label, batch_size=batch_size, shuffle=True, drop_last=True)
    train_unlabel_dataloader = DataLoader(train_unlabel, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_label_dataloader, train_unlabel_dataloader, test_dataloader


colour_codes = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
])


def show(t):
    plt.figure()
    plt.imshow(t.numpy().squeeze(), cmap="gray")
    plt.show()


def color(mask):
    mask = mask.numpy()
    mask[mask == 255] = 0
    mask = colour_codes[mask]
    plt.figure()
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':

    train_label_dataloader, train_unlabel_dataloader, test_dataloader = get_loader()
    print(len(train_label_dataloader))
    print(len(train_unlabel_dataloader))
    print(len(test_dataloader))

    for sample in train_label_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        show(image[0])
        color(label[0])
        break
