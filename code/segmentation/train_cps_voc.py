import os.path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
import datetime

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from tqdm import tqdm

from model.deeplabv3plus import deeplabv3plus_resnet50, deeplabv3plus_mobilenet
from dataset.voc import get_loader, show_label


def save(lo, ac, mi, title, path):
    """
    保存训练数据
    :param lo:
    :param ac:
    :param mi:
    :param title:
    :param path:
    :return:
    """
    with open(path, "a+") as file:
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        txt = "{}\t {},\t loss-{:.5f},\t acc-{:.5f},\t miou-{:.5f}".format(time, title, lo, ac, mi)
        file.write(txt + "\n")


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def train(epoch, model1, optimizer_1, lr_scheduler_1, model2, optimizer_2, lr_scheduler_2, label_loader, unlabel_loader,
          device, criterion, cps_weight=1.5):
    model1.train()
    model2.train()

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    train_loss = 0

    for i in range(len_iter):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        #  有标签的数据集
        try:
            img_labeled, target_label = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            img_labeled, target_label, = next(label_iter)
        # 加载无标签数据
        img_unlabeled, target_no_label = next(unlabel_iter)

        ##############################################################################
        img_labeled = img_labeled.to(device)
        img_unlabeled = img_unlabeled.to(device)
        target_label = target_label.to(device)

        ##############################################################################
        # 计算结果
        pseudo_labeled_1 = model1(img_labeled)
        pseudo_unlabeled_1 = model1(img_unlabeled)

        pseudo_labeled_2 = model2(img_labeled)
        pseudo_unlabeled_2 = model2(img_unlabeled)

        ##############################################################################
        ### 计算 cps loss ###
        pred_1 = torch.cat([pseudo_labeled_1, pseudo_unlabeled_1], dim=0)
        pred_2 = torch.cat([pseudo_labeled_2, pseudo_unlabeled_2], dim=0)
        _, max_1 = torch.max(pred_1, dim=1)
        _, max_2 = torch.max(pred_2, dim=1)
        max_1 = max_1.long()
        max_2 = max_2.long()
        cps_loss = cps_weight * (criterion(pred_1, max_2) + criterion(pred_2, max_1))

        ### 计算 有监督损失, standard cross entropy loss ###
        loss_1 = criterion(pseudo_labeled_1, target_label)
        loss_2 = criterion(pseudo_labeled_2, target_label)

        loss = cps_loss + loss_1 + loss_2
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()

        lr_scheduler_1.step(epoch)
        lr_scheduler_2.step(epoch)
        lr1 = optimizer_1.param_groups[0]["lr"]
        lr2 = optimizer_2.param_groups[0]["lr"]

        train_loss += loss.item()

        if i % 500 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.6f} \t lr1:{:.6f} \t lr2:{}".format(epoch + 1,
                                                                                                 i,
                                                                                                 len_iter,
                                                                                                 100. * i / len_iter,
                                                                                                 loss.item(),
                                                                                                 lr1,
                                                                                                 lr2))
    return train_loss / len_iter


def test(epoch, model, optimizer, loss_fun, cuda, device, test_loader, num_classes=21):
    model.eval()
    start_t = datetime.datetime.now()
    test_loss = 0.0
    test_acc = []
    test_acc_cls = []
    test_mean_iu = []
    test_fwavacc = []

    print("test-epoch--", epoch)
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        optimizer.zero_grad()

        # 如果使用了gpu
        if cuda:
            images = images.to(device).float()
            labels = labels.to(device).long()
        else:
            images = images.float()
            labels = labels.long()

        out = model(images)
        loss = loss_fun(out, labels)
        test_loss += loss.item()

        if cuda:
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
        else:
            label_pred = out.max(dim=1)[1].data.numpy()
            label_true = labels.data.numpy()

        #  计算miou
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            test_acc.append(acc)
            test_acc_cls.append(acc_cls)
            test_mean_iu.append(mean_iu)
            test_fwavacc.append(fwavacc)
        # # 展示一下
        if idx == 0 and epoch % 5 == 0:
            print("pred", np.unique(label_pred))
            print("true", np.unique(label_true))
            show_label(label_pred[0], "label_pred_epoch{}.jpg".format(epoch))
            show_label(label_true[0], "label_true_epoch{}.jpg".format(epoch))

    avg_loss = test_loss / len(test_loader.dataset)
    avg_acc = np.mean(test_acc)
    avg_miou = np.mean(test_mean_iu)
    end_t = datetime.datetime.now()
    epoch_str = (
        'Epoch: {}, test Loss:{:.5f}, test Acc:{:.5f},test Mean IU:{:.5f},time :{:.2f} min '.format(epoch, avg_loss,
                                                                                                    avg_acc, avg_miou, (
                                                                                                            end_t - start_t).seconds / 60))
    print(epoch_str)
    return avg_loss, avg_acc, avg_miou


def main():
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model1 = deeplabv3plus_resnet50(num_classes=21)
    model2 = deeplabv3plus_resnet50(num_classes=21)

    # 3. 准备学习策略
    class arg:
        opt = 'sgd'
        lr = 0.01
        weight_decay = 1e-4
        momentum = 0.9
        #  训练的epoch
        epochs = 200
        sched = 'cosine'
        # sched = "step"
        decay_epochs = 2.4
        decay_rate = .969
        # decay_milestones = [10, 20]
        # decay_epochs = [30, 50]
        #  缩小比例
        # decay_rate = 0.1
        min_lr = 1e-6
        # 热身
        warmup_lr = 1e-4
        warmup_epochs = 3
        cooldown_epochs = 10

    #   是否使用gpu
    cuda = True
    label = 1464  # 使用有标签的数量
    batch_size = 4  #
    crop_size = (512, 512)  # 裁剪大小
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    label_loader, unlabel_loader, test_loader = get_loader(r"E:\note\ssl\data\voc_aug_2\VOCdevkit\VOC2012",
                                                           label=label, batch_size=batch_size, crop_size=crop_size)
    #  定义两个优化器
    optimizer_1 = create_optimizer_v2(model1, **optimizer_kwargs(cfg=arg))
    lr_scheduler_1, num_epochs = create_scheduler(arg, optimizer_1)

    optimizer_2 = create_optimizer_v2(model2, **optimizer_kwargs(cfg=arg))
    lr_scheduler_2, num_epochs = create_scheduler(arg, optimizer_2)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion_csst = nn.MSELoss()

    for epoch in range(num_epochs):
        train(epoch=epoch,
              model1=model1, optimizer_1=optimizer_1, lr_scheduler_1=lr_scheduler_1,
              model2=model2, optimizer_2=optimizer_2, lr_scheduler_2=lr_scheduler_2,
              label_loader=label_loader, unlabel_loader=unlabel_loader,
              device=device, criterion=criterion)
        test(epoch, model1, optimizer_1, criterion, cuda, device, test_loader)
        test(epoch, model2, optimizer_2, criterion, cuda, device, test_loader)


if __name__ == '__main__':
    main()
