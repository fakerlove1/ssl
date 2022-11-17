# 2017-Mean-Teacher NIPS

> 论文题目：Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
>
> 论文链接：[https://arxiv.org/abs/1703.01780](https://arxiv.org/abs/1703.01780)
>
> 论文代码：[https://github.com/CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher)
>
> 发表时间：2017年3月
>
> 引用：Tarvainen A, Valpola H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results[J]. Advances in neural information processing systems, 2017, 30.
>
> 引用数：2644



## 1. 简介



### 1.1 简介

今天我们来学习半监督学习的第2篇文章Mean-Teacher

![image-20221116220735107](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221116220735107.png)



Mean-Teacher是对这篇论文[Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242)做的改进



### 1.2 前提准备



**一致性判定**正是描述了其中一个属性，那就是一个表现很好的模型应该对输入数据以及他的某些变形表现稳定。比如人看到了`一匹马的照片，不管是卡通的，还是旋转的，人都能准确的识别出这是一匹马`。那半监督学习也是一样，我们想要我们的模型表现良好，表现和上限通过大量有标签数据训练的一样（足够鲁棒），那么我们的模型也应该拥有这个属性，即对输入数据的某种变化鲁棒，此类方法代表方法为Teacher-student Model, CCT模型等等，对应的半监督学习假设就是平滑性假设。



## 2. 网络



### 2.1 模型整体架构



![image-20221102163549866](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221102163549866.png)



1. 一个batch里面会同时`有标签图像`和`无标签图像`。然后对一个batch 做两次数据增强。生成2组图片。

2. 分别送入student 模型与Teacher模型。送入student模型里面的`有标签数据`与真实标签做`crossentropy-loss`损失计算。

3. 然后让Teacher模型出来的所有预测与 student模型出来的所有预测，做`mse-loss`。为`一致性损失`

4. 最终的损失`loss`=`crossentropy_loss`+`mse_loss`。然后开始更新参数。student模型是`正常更新`。

5. 但是Teacher模型是 使用EMA的方式。就是下面的公式
   $$
   \theta_{t}^{\prime}=\alpha \theta_{t-1}^{\prime}+(1-\alpha) \theta_{t}
   $$
   啥意思呢？？？

   Teacher模型=$\alpha$%的**Teacher模型参数**+($1-\alpha$)的**student模型参数**



### 2.2 思路

**模型的核心思想：模型即充当学生，又充当老师。作为老师，用来产生学生学习时的目标，作为学生，利用老师模型产生的目标来学习。**

为了克服Temporal Ensembling的局限性，我们建议平均模型权重而不是预测。教师模式是连续学生模式的平均值，因此我们叫它Mean teacher。与直接使用最终的权重相比，将模型权重平均到训练步骤会产生更准确的模型，在训练中可以利用这一点来构建更好的目标。教师模型使用学生模型的EMA权重，而不是与学生模型共享权重。同时，由于权值平均改善了所有层的输出，而不仅仅是顶层输出，目标模型有更好的中间表示。


## 3. 代码

一共3个文件

![image-20221116213035282](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221116213035282.png)

但是这份代码4000label 只能跑到 82% 左右的准确率。和论文的90%还有很大的差距，不知道为啥??`有没有大佬替我解决一下`



### 3.1 定义数据集

~~~python
import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def get_cifar10(root, n_labeled, batch_size=16, K=7,
                transform_train=None, transform_test=None,
                download=True, ):
    """

    :param root: cifra保存的路径
    :param n_labeled: 需要视频label的数量
    :param transform_train: train的数据增强
    :param transform_val: val的数据增强
    :param download: 是否下载，默认是True
    :return:
    """
    if transform_train is None:
        transform_train = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]))
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    # 加载原始数据集
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    # 区分有标签数据与无标签数据。
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, int(n_labeled / 10))
    #  有标签数据集
    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    # 无标签数据集
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True,
                                                transform=transform_train)
    # 验证集
    # val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_test, download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} ")

    train_labeled_dataloader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, batch_size=batch_size * K, shuffle=True,
                                            drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_labeled_dataloader, train_unlabeled_dataloader, test_dataloader


def train_split(labels, n_labeled_per_class):
    """

    :param labels: 全部的标签数据
    :param n_labeled_per_class: 每个标签的数目
    :return: 有标签索引，无标签索引
    """
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


def train_val_split(labels, n_labeled_per_class):
    """

    :param labels: 全部标签数据
    :param n_labeled_per_class: 每个标签的类
    :return:  有标签数据索引，无标签索引，验证集索引
    """
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        # self.data = transpose(normalize(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])


if __name__ == '__main__':
    train_labeled_dataloader, train_unlabeled_dataloader, test_dataloader = get_cifar10("./data", 4000, batch_size=4)
    label_iter = iter(train_labeled_dataloader)
    unlabel_iter = iter(train_unlabeled_dataloader)
    (img1, img2), target_label = next(label_iter)
    (img1_ul, img2_ul), target_no_label = next(unlabel_iter)

    input1 = torch.cat([img1, img1_ul])
    input2 = torch.cat([img2, img2_ul])

    torchvision.utils.save_image(input1, "1.jpg")
    torchvision.utils.save_image(input2, "2.jpg")

~~~







### 3.2 定义网络



~~~python
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes)


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model = build_wideresnet(depth=28, widen_factor=2, dropout=0.1, num_classes=10)
    y = model(x)
    print(y.shape)

~~~



### 3.3 训练

~~~python
from __future__ import print_function
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from cifar10 import get_cifar10
from wideresnet import build_wideresnet


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """
    设置一致性损失的权重
    随着 训练的次数在变得
    :param epoch:
    :param init_ep:
    :param end_ep:
    :param init_w:
    :param end_w:
    :return:
    """
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        # weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
    # print('Consistency weight: %f'%weight_cl)
    return weight_cl


def update_ema_variables(model, model_teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param.data)
    return alpha


def train(model, mean_teacher, device, label_loader, unlabel_loader, lr_scheduler, optimizer, epoch, num_epochs):
    """

    :param model: 学生模型
    :param mean_teacher: 教师模型
    :param device: 数据在那个地方
    :param label_loader: 有标签数据集
    :param unlabel_loader: 无标签数据集
    :param lr_scheduler: 学习率优化器
    :param optimizer: 优化器
    :param epoch: epoch次数
    :return:
    """
    model.train()
    mean_teacher.train()
    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    alpha = 0.999
    for i in range(len_iter):
        # 计算权重 w
        global_step = epoch * len_iter + i
        weight = cal_consistency_weight(global_step, end_ep=(num_epochs // 2) * len_iter, end_w=1.0)

        #  有标签的数据集
        try:
            (img1, img2), target_label = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            (img1, img2), target_label, = next(label_iter)

        batch_size_labeled = img1.shape[0]

        # 加载无标签数据
        (img1_ul, img2_ul), target_no_label = next(unlabel_iter)

        # mini_batch_size = img1.shape[0] + img1_ul.shape[0]
        #  这边是 有标签+无标签数据集进行混合
        # 有标签+无标签数据
        input1 = Variable(torch.cat([img1, img1_ul]).to(device))
        input2 = Variable(torch.cat([img2, img2_ul]).to(device))
        target = Variable(target_label.to(device))

        optimizer.zero_grad()
        output = model(input1)
        ########################### CODE CHANGE HERE ######################################
        # forward pass with mean teacher
        # torch.no_grad() prevents gradients from being passed into mean teacher model
        with torch.no_grad():
            mean_t_output = mean_teacher(input2)

        ########################### CODE CHANGE HERE ######################################
        # consistency loss (example with MSE, you can change)
        # 一致性损失。损失为所有标签(无标签+有标签)
        const_loss = F.mse_loss(output, mean_t_output)

        ########################### CODE CHANGE HERE ######################################
        # set the consistency weight (should schedule)
        #  ignore_index=-1 忽略无标签的数据集。无标签的数据target 为-1
        # 总体损失为
        # 损失为 有标签的交叉熵损失+ 一致性损失(基于平滑性假设，一个模型对于 一个输入及其变形应该保持一致性）
        out_x = output[:batch_size_labeled]
        label_loss = F.cross_entropy(out_x, target)

        loss = label_loss + weight * const_loss
        loss.backward()
        optimizer.step()

        lr_scheduler.step(epoch=epoch)
        lr = optimizer.param_groups[0]["lr"]

        ########################### CODE CHANGE HERE ######################################
        # update mean teacher, (should choose alpha somehow)
        # 更新教师模型
        # Use the true average until the exponential average is more correct
        ema_const = update_ema_variables(model, mean_teacher, alpha, global_step)

        if i % 500 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.6f}\t w:{:.6f} \t alpha:{:.6f} \t lr:{}".format(epoch + 1,
                                                                                                             i,
                                                                                                             len_iter,
                                                                                                             100. * i / len_iter,
                                                                                                             loss.item(),
                                                                                                             weight,
                                                                                                             ema_const,
                                                                                                             lr))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    class arg:
        opt = 'sgd'
        lr = 0.1
        weight_decay = 1e-4
        momentum = 0.9
        epochs = 200
        sched = "multistep"
        decay_milestones = [100, 150]
        decay_rate = 0.1
        min_lr = 1e-6
        warmup_lr = 1e-4
        warmup_epochs = 3
        cooldown_epochs = 10

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_wideresnet(widen_factor=3, depth=28, dropout=0.1, num_classes=10).to(device)
    ########################### CODE CHANGE HERE ######################################
    # initialize mean teacher
    mean_teacher = build_wideresnet(widen_factor=3, depth=28, dropout=0.1, num_classes=10).to(device)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=arg))
    lr_scheduler, num_epochs = create_scheduler(arg, optimizer)

    # 50000个样本，只使用4000label
    # mean-teacher
    # 4000label acc=90%
    # 2000label acc=87%
    # 1000label acc=83%
    # 500 label acc=58%
    # 250 label acc=52%

    batch_size = 16
    K = 7  # 无标签的batchsize是 有标签的K 倍
    label_loader, unlabel_loader, test_loader = get_cifar10(root="./data", n_labeled=4000, batch_size=batch_size, K=K)

    print("有标签的数据集长度: ", len(label_loader) * batch_size)
    print("无标签的数据集长度:", len(unlabel_loader) * batch_size * K)
    print("测试集的长度", len(test_loader))

    for epoch in range(num_epochs):
        train(model, mean_teacher, device, label_loader, unlabel_loader, lr_scheduler, optimizer, epoch, num_epochs)
        test(model, device, test_loader)
        test(mean_teacher, device, test_loader)
        # 保存模型
        torch.save(model.state_dict(), "mean_teacher_cifar10.pt")


if __name__ == '__main__':
    main()

~~~





参考资料

> [Mean Teacher学习笔记（一）_Demon果的博客-CSDN博客_mean teacher](https://blog.csdn.net/demons2/article/details/109825597)