import os.path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import h5py


class ACDC(Dataset):
    def __init__(self, root=r"E:\note\ssl\data\ACDC", split="train", transform=None):
        super(ACDC, self).__init__()
        self.split = split
        self.root = root
        self.transform = transform

        if self.split == "train":
            with open(self.root + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self.root + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self.root + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self.root + "/data/{}.h5".format(case), "r")

        image = np.array(h5f["image"][:])
        mask = np.array(h5f["label"][:])

        if self.transform is not None and self.split == "train":
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        return image, mask


def get_loader(root=r'/kaggle/input/acdc-2/My-ACDC-2', batch_size=4, crop_size=(256, 256)):
    """

    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_both_aug = A.Compose([
        # A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=0, p=1, mask_value=255),
        # A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1),
        A.RandomResizedCrop(height=crop_size[0], width=crop_size[1]),
        # A.ColorJitter(),
        # A.CoarseDropout(max_holes=8, p=0.5),
        # A.OneOf([
        #     A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.5, 0.5), p=0.9),
        #     A.RandomGamma(gamma_limit=(50, 200), p=0.8),
        # ]),
        A.RandomGamma(gamma_limit=(50, 200), p=0.8),
        A.OneOf([
            A.ShiftScaleRotate(p=0.6),
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.8)
        ]),
        ToTensorV2()
    ])
    train_dataset = ACDC(root=root, split="train", transform=train_both_aug)
    test_dataset = ACDC(root=root, split="val")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader


colour_codes = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
])


def show(im):
    im = im.numpy().squeeze()
    plt.figure()
    plt.imshow(im, cmap="gray")
    plt.show()
    Image.fromarray(np.uint8(im)).save("image.png")


def show_label(mask, path="label.jpg"):
    mask[mask == 255] = 0
    mask = colour_codes[mask]
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(np.uint8(mask)).save(path)


def color(mask):
    mask[mask == 255] = 0
    mask = colour_codes[mask]
    return np.uint8(mask)


if __name__ == '__main__':

    train_dataloader, test_dataloader = get_loader(root=r"E:\note\ssl\data\ACDC")
    print(len(train_dataloader))
    print(len(test_dataloader))
    print(len(test_dataloader.dataset))
    for sample in train_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)

        print(np.unique(label.numpy()))
        show(image[0])
        show_label(label[0].numpy())
        break

    for sample in test_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)

        print(np.unique(label.numpy()))
        # show(image[0])
        # show_label(label[0].numpy())
        break
