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

        student_output = F.softmax(output.detach(), dim=1)
        teacher_output = F.softmax(mean_t_output.detach(), dim=1)
        ########################### CODE CHANGE HERE ######################################
        # consistency loss (example with MSE, you can change)
        # 一致性损失。损失为所有标签(无标签+有标签)
        const_loss = F.mse_loss(student_output, teacher_output)

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
