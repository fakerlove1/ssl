from __future__ import print_function
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from datasets import get_dataloader
from resnet_cifar10 import resnet32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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


def train(args, model, mean_teacher, device, label_loader, unlabel_loader, test_loader, optimizer, epoch):
    """

    :param args: 参数
    :param model: 学生模型
    :param mean_teacher: 教师模型
    :param device: 数据在那个地方
    :param label_loader: 有标签数据集
    :param unlabel_loader: 无标签数据集
    :param test_loader: 测试集
    :param optimizer: 优化器
    :param epoch: epoch次数
    :return:
    """

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    alpha = 0.95
    for i in range(len_iter):
        # 计算权重 w
        global_step = epoch * len_iter + i
        weight = cal_consistency_weight(global_step, end_ep=(args.epochs // 2) * len_iter, end_w=1.0)

        #  有标签的数据集
        try:
            img1, img2, target_label = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            img1, img2, target_label, = next(label_iter)
        # 无标签的数据
        img1_ul, img2_ul, target_no_label = next(unlabel_iter)

        #  这边是 有标签+无标签数据集进行混合

        input1 = Variable(torch.cat([img1, img1_ul]).to(device))
        input2 = Variable(torch.cat([img2, img2_ul]).to(device))
        target = Variable(torch.cat([target_label, target_no_label]).to(device))

        #  整体的batchsize
        sl = img1.shape
        su = img1_ul.shape

        batch_size = sl[0] + su[0]
        optimizer.zero_grad()

        output = model(input1)

        ########################### CODE CHANGE HERE ######################################
        # forward pass with mean teacher
        # torch.no_grad() prevents gradients from being passed into mean teacher model
        with torch.no_grad():
            mean_t_output = mean_teacher(input2)

        ########################### CODE CHANGE HERE ######################################
        # consistency loss (example with MSE, you can change)
        # 一致性损失
        const_loss = F.mse_loss(output, mean_t_output)

        ########################### CODE CHANGE HERE ######################################
        # set the consistency weight (should schedule)
        #  ignore_index=-1 忽略无标签的数据集。无标签的数据target 为-1
        # 总体损失为
        loss = F.cross_entropy(output, target, ignore_index=-1) + weight * const_loss
        loss.backward()
        optimizer.step()

        ########################### CODE CHANGE HERE ######################################
        # update mean teacher, (should choose alpha somehow)
        # 更新教师模型
        # Use the true average until the exponential average is more correct

        alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
        for mean_param, param in zip(mean_teacher.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)
        if i % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.6f}\t w:{:.6f} \t alpha:{:.6f}".format(epoch, i, len_iter,
                                                                                                    100. * i / len_iter,
                                                                                                    loss.item(), weight,
                                                                                                    alpha))


def test(args, model, device, test_loader):
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
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    #     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                         help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args([])
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    # 50000个样本，只使用4000label
    # mean-teacher
    # 4000label acc=90%
    # 2000label acc=87%
    # 1000label acc=83%
    # 500 label  acc=58%
    # 250 label acc=52%

    boundary = 4000
    batchsize = 64

    label_loader = get_dataloader(r"./data", split="label", boundary=boundary, batch_size=batchsize)  # 有标签的数据集
    unlabel_loader = get_dataloader(r"./data", split="unlabel", boundary=boundary, batch_size=batchsize)  # 无标签的数据集
    test_loader = get_dataloader(r"./data", split="test", batch_size=batchsize)  # 无标签的数据集

    print("有标签的数据集长度: ", len(label_loader) * batchsize)
    print("无标签的数据集长度:", len(unlabel_loader) * batchsize)
    print("测试集的长度", len(test_loader))

    model = resnet32().to(device)
    ########################### CODE CHANGE HERE ######################################
    # initialize mean teacher
    mean_teacher = resnet32().to(device)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        train(args, model, mean_teacher, device, label_loader, unlabel_loader, test_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        test(args, mean_teacher, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
