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
from utils.stream_metrics import StreamSegMetrics


def save(epoch, score, path):
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
        txt = "{}\t epoch:{} \t {}".format(time, epoch, score)
        file.write(txt + "\n")


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
        img_labeled = img_labeled.to(device).float()
        img_unlabeled = img_unlabeled.to(device).float()
        target_label = target_label.to(device).long()

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
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.5f} \t lr1:{:.5f} \t lr2:{:.5f}".format(epoch + 1,
                                                                                                     i,
                                                                                                     len_iter,
                                                                                                     100. * i / len_iter,
                                                                                                     loss.item(),
                                                                                                     lr1,
                                                                                                     lr2))
    return train_loss / len_iter


def test(epoch, model, optimizer, device, test_loader, metrics):
    model.eval()
    metrics.reset()
    print("test-epoch--", epoch)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):
            optimizer.zero_grad()
            # 如果使用了gpu
            images = images.to(device).float()
            labels = labels.to(device).long()

            out = model(images)
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            metrics.update(label_true, label_pred)

            # # 展示一下
            if idx == 0 and epoch % 5 == 0:
                print("pred", np.unique(label_pred))
                print("true", np.unique(label_true))
                show_label(label_pred[0], "label_pred_epoch{}.jpg".format(epoch))
                show_label(label_true[0], "label_true_epoch{}.jpg".format(epoch))

    score = metrics.get_results()
    return score


def main():
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
    num_classes = 21
    label = 1464  # 使用有标签的数量
    batch_size = 2  #
    crop_size = (512, 512)  # 裁剪大小
    cuda = True
    ckpt1 = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行
    ckpt2 = None  # 模型2 恢复地址。如果模型突然终断。可以重新 添加地址运行
    start_epoch = 0
    if not cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path = r"E:\note\ssl\data\voc_aug_2\VOCdevkit\VOC2012"
    label_loader, unlabel_loader, test_loader = get_loader(root=path, label=label, batch_size=batch_size,
                                                           crop_size=crop_size)
    metrics = StreamSegMetrics(num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model1 = deeplabv3plus_resnet50(num_classes=21).to(device)
    model2 = deeplabv3plus_resnet50(num_classes=21).to(device)

    #  定义两个优化器
    optimizer_1 = create_optimizer_v2(model1, **optimizer_kwargs(cfg=arg))
    lr_scheduler_1, num_epochs = create_scheduler(arg, optimizer_1)

    optimizer_2 = create_optimizer_v2(model2, **optimizer_kwargs(cfg=arg))
    lr_scheduler_2, num_epochs = create_scheduler(arg, optimizer_2)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion_csst = nn.MSELoss()
    best_miou = 0.0
    #  加载原模型
    if ckpt1 is not None and os.path.isfile(ckpt1):
        state_dict = torch.load(ckpt1)
        start_epoch = state_dict["epoch"]
        model1 = state_dict["model"]
        optimizer_1 = state_dict["optimizer"]
        lr_scheduler_1 = state_dict["scheduler"]
        best_miou = state_dict["best_miou"]

    if ckpt2 is not None and os.path.isfile(ckpt2):
        state_dict = torch.load(ckpt2)
        start_epoch = state_dict["epoch"]
        model2 = state_dict["model"]
        optimizer_2 = state_dict["optimizer"]
        lr_scheduler_2 = state_dict["scheduler"]
        best_miou = state_dict["best_miou"]

    for epoch in range(start_epoch, num_epochs):
        train(epoch=epoch,
              model1=model1, optimizer_1=optimizer_1, lr_scheduler_1=lr_scheduler_1,
              model2=model2, optimizer_2=optimizer_2, lr_scheduler_2=lr_scheduler_2,
              label_loader=label_loader, unlabel_loader=unlabel_loader,
              device=device, criterion=criterion)

        val_score = test(epoch=epoch, model=model1, optimizer=optimizer_1, device=device, test_loader=test_loader,
                         metrics=metrics)
        print(metrics.to_str(val_score))
        save(epoch, val_score, "model1.txt")
        best_miou = update(val_score, epoch, model1, optimizer_1, lr_scheduler_1, best_miou, "best_model1.pth")

        val_score = test(epoch=epoch, model=model2, optimizer=optimizer_2,
                         device=device, test_loader=test_loader, metrics=metrics)
        print(metrics.to_str(val_score))
        save(epoch, val_score, "model2.txt")
        best_miou = update(val_score, epoch, model1, optimizer_1, lr_scheduler_1, best_miou, "best_model2.pth")


def update(val_score, epoch, model, optimizer, scheduler, best_miou, path):
    if val_score['Mean IoU'] > best_miou:
        best_miou = val_score['Mean IoU']
        save_ckpt(path, epoch, model, optimizer, scheduler, best_miou)
    return best_miou


def save_ckpt(path, epoch, model, optimizer, scheduler, best_miou):
    """ save current model
    """
    torch.save({
        "epoch": epoch,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "best_miou": best_miou,
    }, path)

    print("Model saved as %s" % path)


if __name__ == '__main__':
    main()
