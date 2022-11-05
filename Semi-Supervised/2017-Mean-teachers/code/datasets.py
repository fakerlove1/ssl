from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from typing import Any, Callable, Optional, Tuple

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url, check_integrity


class CIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    num_class = 10
    split_list = ['label', 'unlabel', 'valid', 'test']

    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, boundary=4000):
        """

        :param root: cifra10路径
        :param split: 区分的
        :param transform: 图片增强
        :param target_transform: label图片增强
        :param download: 是否下载
        :param boundary:
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        # assert (boundary < 10)
        #  boundary=4000,,意味者每个类使用的标签数目为 4000/10=400个。剩下的都会被标记为无标签。
        self.label_num = boundary // self.num_class
        # print("Boundary: ", boundary)
        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        # load data
        # self.data = np.load(os.path.join(self.root,self.base_folder))
        self.data: Any = []
        self.targets = []

        if self.split == 'label' or self.split == 'unlabel':
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.split == 'label' or self.split == 'unlabel':
            train_data = []
            train_labels = []

            train_no_labels_data = []
            train_no_labels = []

            num_labels = [0 for _ in range(self.num_class)]

            for i, temp in enumerate(self.data):
                tmp_label = self.targets[i]
                if num_labels[tmp_label] < self.label_num:
                    train_data.append(temp)
                    train_labels.append(tmp_label)
                    num_labels[tmp_label] += 1
                else:
                    train_no_labels_data.append(temp)
                    train_no_labels.append(-1)

            if self.split == "label":
                self.data = np.array(train_data)
                self.targets = np.array(train_labels)
            elif self.split == "unlabel":
                self.data = np.array(train_no_labels_data)
                self.targets = np.array(train_no_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        #  如果是训练模型，则返回3张图
        # 如果测试，只返回一张图
        if self.split == 'label' or self.split == 'unlabel':
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target
        else:
            img= self.transform(img)
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        import tarfile

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


def get_dataloader(root, split, boundary=0, batch_size=32):
    if split == 'label' or split == 'unlabel':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=1, magnitude=8),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
            transforms.RandomErasing(p=0.25)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    dataset = CIFAR10(root=root, split=split, transform=transform, download=True, boundary=boundary)
    if split == 'label' or split == 'unlabel':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, )
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, )
    return dataloader


if __name__ == '__main__':
    '''
    for i in range(10):
        print("Boundary %d///////////////////////////////////////"%i)
        data_train = CIFAR10('/tmp', split='label', download=True, transform=None, boundary=i)
        data_train_ul = CIFAR10('/tmp', split='unlabel', download=True, transform=None, boundary=i)
        data_valid = CIFAR10('/tmp', split='valid', download=True, transform=None, boundary=i)
        data_test = CIFAR10('/tmp', split='test', download=True, transform=None, boundary=i)

        print("Number of data")
        print(len(data_train))
        print(len(data_train_ul))
        print(len(data_valid))
        print(len(data_test))
    '''

    import torch.utils.data as data
    from math import ceil

    boundary = 4000
    batchsize = 32
    #  有标签的数据集
    label_loader = get_dataloader(r"./data", split="label", boundary=boundary, batch_size=batchsize, )
    #  无标签的数据集
    unlabel_loader = get_dataloader(r"./data", split="unlabel", boundary=boundary, batch_size=batchsize)
    #  无标签的数据集
    test_loader = get_dataloader(r"./data", split="test", batch_size=batchsize )

    print("有标签的数据集长度: ", len(label_loader) * batchsize)
    print("无标签的数据集长度:", len(unlabel_loader) * batchsize)
    print("测试集的长度", len(test_loader))

    for img1,img2,target in label_loader:
        print(img1.shape)
        print(img2.shape)
        print(target.shape)
        break
    for img1, img2, target in unlabel_loader:
        print(img1.shape)
        print(img2.shape)
        print(target.shape)
        break

    for img,target in test_loader:
        print(img.shape)
        print(target.shape)
        break
    # for i in range(90, 256):
    #     batch_size = i
    #     label_size = len(labelset)
    #     unlabel_size = len(unlabelset)
    #     iter_per_epoch = int(ceil(float(label_size + unlabel_size) / batch_size))
    #     batch_size_label = int(ceil(float(label_size) / iter_per_epoch))
    #     batch_size_unlabel = int(ceil(float(unlabel_size) / iter_per_epoch))
    #     iter_label = int(ceil(float(label_size) / batch_size_label))
    #     iter_unlabel = int(ceil(float(unlabel_size) / batch_size_unlabel))
    #     if iter_label == iter_unlabel:
    #         print('Batch size: ', batch_size)
    #         print('Iter/epoch: ', iter_per_epoch)
    #         print('Batch size (label): ', batch_size_label)
    #         print('Batch size (unlabel): ', batch_size_unlabel)
    #         print('Iter/epoch (label): ', iter_label)
    #         print('Iter/epoch (unlabel): ', iter_unlabel)
