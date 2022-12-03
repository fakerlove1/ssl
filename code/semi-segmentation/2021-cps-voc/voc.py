import os.path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from torchvision.transforms import transforms
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from skimage import io

class_name = ['background', 'aeroplane', 'bicycle', 'bird',
              'boat',
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable',
              'dog', 'horse', 'motorbike', 'person',
              'pottedplant',
              'sheep', 'sofa', 'train', 'tv/monitor']

class_colors = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128],
                         [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                         [192, 128, 0],
                         [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                         [0, 64, 0],
                         [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])


def label_to_img(label):
    label[label == 255] = 0
    x = class_colors[label]
    return x


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VOC(Dataset):
    def __init__(self, root, mode, transform, label=1464):
        self.root = root
        self.mode = mode
        self.transform = transform
        if self.mode == 'label':
            with open(os.path.join(root, "pascal", str(label), "labeled.txt"), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.mode == "unlabel":
            with open(os.path.join(root, "pascal", str(label), "unlabeled.txt"), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        else:
            with open(os.path.join(root, "pascal", "val.txt"), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        self.img_path = [os.path.join(root, "JPEGImages", item + ".jpg") for item in self.sample_list]
        self.mask_path = [os.path.join(root, "SegmentationClassAug", item + ".png") for item in self.sample_list]

        print("mode-{} load {} images".format(mode, len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert("RGB")
        mask = Image.open(self.mask_path[item]).convert("P")
        img = np.array(img)
        mask = np.array(mask)

        if self.transform is not None:
            result = self.transform(image=img, mask=mask)
            img, mask = result["image"], result["mask"]

        img = image_transform(img).numpy()
        return img, mask


def get_loader(root, label=1464, batch_size=4, crop_size=(512, 512)):
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=0, p=1, mask_value=255),
        A.RandomCrop(height=crop_size[0], width=crop_size[1]),
        # A.RandomResizedCrop(height=crop_size[0], width=crop_size[1]),  # 随机裁剪缩放
        A.HorizontalFlip(p=0.5),  # 随机水平翻转
        A.ColorJitter(p=1),  # 随机 改变亮度，饱和度
        A.RandomBrightnessContrast(p=0.2),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ToTensorV2(),
    ])

    label_dataset = VOC(root=root, mode="label", transform=train_transform, label=label)
    unlabel_dataset = VOC(root=root, mode="unlabel", transform=train_transform, label=label)
    test_dataset = VOC(root=root, mode="test", transform=None, label=label)

    label_dataloader = DataLoader(label_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  drop_last=True, pin_memory=True)
    unlabel_dataloader = DataLoader(unlabel_dataset, batch_size=batch_size * 2, shuffle=True, num_workers=4,
                                    pin_memory=True,
                                    drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    return label_dataloader, unlabel_dataloader, test_dataloader


def show(image):
    im = image.numpy().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(im)
    plt.show()
    io.imsave("image.jpg", im)


def show_label(label, path="label.jpg"):
    im = label_to_img(label)
    plt.figure()
    plt.imshow(im)
    plt.show()
    Image.fromarray(np.uint8(im)).save(path)


if __name__ == '__main__':
    label_dataloader, unlabel_dataloader, test_dataloader = get_loader(
        root=r"E:\note\ssl\data\voc_aug_2\VOCdevkit\VOC2012")

    for i, (image, label) in enumerate(label_dataloader):
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(label[0])
        break

    for image, label in unlabel_dataloader:
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(label[0])
        break
    #
    for image, label in test_dataloader:
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(label[0])
        break
