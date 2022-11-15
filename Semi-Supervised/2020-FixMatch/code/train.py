from __future__ import print_function
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from wideresnet import build_wideresnet
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

from cifar10 import get_cifar10


def train(model, eval_step, device, label_loader, unlabel_loader, lr_scheduler, optimizer, epoch, T,
          threshold, lambda_u):
    """

    :param model:模型
    :param eval_step:
    :param device:
    :param label_loader:
    :param unlabel_loader:
    :param lr_scheduler:
    :param optimizer:
    :param epoch:
    :param num_epochs:
    :param T: 温度
    :param threshold: 伪标签的阈值。只有通过阈值的伪标签才能 进行损失计算
    :param lambda_u:
    :return:
    """
    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)

    for i in range(eval_step):
        optimizer.zero_grad()
        #  有标签的数据集
        try:
            img_labeled, target_label = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            img_labeled, target_label, = next(label_iter)

        batch_size = img_labeled.shape[0]

        #  无标签的数据集
        try:
            (img_unlabeled_weak, img_unlabeled_strong), _ = next(unlabel_iter)
        except StopIteration:
            unlabel_iter = iter(unlabel_loader)
            (img_unlabeled_weak, img_unlabeled_strong), _ = next(unlabel_iter)

        #########################################################################################################
        # 拼接输入
        inputs = Variable(torch.cat([img_labeled, img_unlabeled_weak, img_unlabeled_strong]).to(device))
        target_label = Variable(target_label.to(device))  # 让标签在 对应的device 上

        logits = model(inputs)  # 获取输出
        #######################################################################################################
        # 分离输出
        logits_x = logits[:batch_size]  # 有标签输出
        #  有标签的损失
        Lx = F.cross_entropy(logits_x, target_label)

        # 无标签-弱增强-输出，无标签-强增强-输出
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        #  使用弱增强的出来的结果制作 伪标签，因为稳定
        pseudo_label = torch.softmax(logits_u_w.detach() / T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        # 只有通过阈值的伪标签才能进行计算
        mask = max_probs.ge(threshold).float()
        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()

        # 所有损失= 有标签数据弱增广的交叉嫡  +  无标签强增广数据、弱增广伪标签的交叉嫡
        loss = Lx + lambda_u * Lu

        #########################################################################################################
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch=epoch)
        lr = optimizer.param_groups[0]["lr"]

        if i % 400 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.6f} \t \t lr:{}".format(epoch, i,
                                                                                     eval_step,
                                                                                     100. * i / eval_step,
                                                                                     loss.item(),
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
    model.train()


def main():
    class arg:
        opt = 'sgd'
        lr = 0.03
        weight_decay = 1e-4
        momentum = 0.9
        epochs = 200
        sched = "cosine"
        #  缩小比例
        decay_rate = 0.1
        min_lr = 1e-6
        warmup_lr = 1e-4  # 热身
        warmup_epochs = 3
        cooldown_epochs = 10

    num_labeled = 4000

    expand_labels = True
    eval_step = 512
    dataset = "cifar10"
    T = 1
    threshold = 0.95
    lambda_u = 1
    batch_size = 32
    k = 7
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if dataset == "cifar10":
        num_classes = 10
        model = build_wideresnet(depth=28, widen_factor=2, dropout=0.1, num_classes=10).to(device)
    elif dataset == "cifar100":
        num_classes = 100
        model = build_wideresnet(depth=28, widen_factor=8, dropout=0.1, num_classes=10).to(device)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=arg))
    lr_scheduler, num_epochs = create_scheduler(arg, optimizer)

    label_loader, unlabel_loader, test_loader = get_cifar10(num_labeled=num_labeled, num_classes=num_classes,
                                                            expand_labels=expand_labels,
                                                            batch_size=batch_size, eval_step=eval_step,
                                                            root=r"./data", k=7)
    print("有标签的数据集长度: ", len(label_loader))
    print("无标签的数据集长度:", len(unlabel_loader))
    print("测试集的长度", len(test_loader))

    print("有标签的数据集长度: ", len(label_loader) * batch_size)
    print("无标签的数据集长度:", len(unlabel_loader) * batch_size * k)
    print("测试集的长度", len(test_loader))

    for epoch in range(num_epochs):
        train(model=model, eval_step=eval_step, device=device, label_loader=label_loader,
              unlabel_loader=unlabel_loader, lr_scheduler=lr_scheduler, optimizer=optimizer, epoch=epoch, T=T,
              threshold=threshold, lambda_u=lambda_u)
        test(model, device, test_loader)
        # 保存模型
        torch.save(model.state_dict(), "fixmatch_cifar10.pt")


if __name__ == '__main__':
    main()
