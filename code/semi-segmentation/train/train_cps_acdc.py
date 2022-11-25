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

from model.unet import UNet
from dataset.acdc import get_loader, show_label
from utils.medical_metrics import Medical_Metric
from utils.losses import DiceLoss


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


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


consistency = 0.1
consistency_rampup = 200


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def train(epoch,
          model1, optimizer_1, lr_scheduler_1,
          model2, optimizer_2, lr_scheduler_2,
          label_loader, unlabel_loader,
          device, criterion, dice_loss, cps_weight=1.5, ):
    model1.train()
    model2.train()

    label_iter = iter(label_loader)
    len_iter = len(unlabel_loader)
    train_loss = 0

    for i, (img_unlabeled, target_no_label) in enumerate(tqdm(unlabel_loader)):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        consistency_weight = get_current_consistency_weight(epoch=epoch)

        #  有标签的数据集
        try:
            img_labeled, target_label = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            img_labeled, target_label, = next(label_iter)

        ##############################################################################
        img_labeled = img_labeled.to(device).float()
        img_unlabeled = img_unlabeled.to(device).float()
        target_label = target_label.to(device).long()

        ##############################################################################
        # 计算结果

        pseudo_labeled_1 = torch.softmax(model1(img_labeled), dim=1)
        pseudo_unlabeled_1 = torch.softmax(model1(img_unlabeled), dim=1)
        pseudo_labeled_2 = torch.softmax(model2(img_labeled), dim=1)
        pseudo_unlabeled_2 = torch.softmax(model2(img_unlabeled), dim=1)

        ##############################################################################
        ### 计算 cps loss ###
        pred_1 = torch.cat([pseudo_labeled_1, pseudo_unlabeled_1], dim=0)
        pred_2 = torch.cat([pseudo_labeled_2, pseudo_unlabeled_2], dim=0)
        max_1 = torch.argmax(pred_1.detach(), dim=1, keepdim=False).long()
        max_2 = torch.argmax(pred_2.detach(), dim=1, keepdim=False).long()
        cps_loss = cps_weight * (criterion(pred_1, max_2) + criterion(pred_2, max_1))

        ### 计算 有监督损失, standard cross entropy loss ###
        loss_1 = 0.5 * (
                criterion(pseudo_labeled_1, target_label) + dice_loss(pseudo_labeled_1, target_label.unsqueeze(1)))

        loss_2 = 0.5 * (
                criterion(pseudo_labeled_2, target_label) + dice_loss(pseudo_labeled_2, target_label.unsqueeze(1)))

        loss = consistency_weight * cps_loss + loss_1 + loss_2
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
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.5f} \t lr1:{:.5f} \t lr2:{:.5f} \t w:{:.5f}".format(
                    epoch + 1,
                    i,
                    len_iter,
                    100. * i / len_iter,
                    loss.item(),
                    lr1,
                    lr2, consistency_weight))
    return train_loss / len_iter


def test(epoch, model, optimizer, lr_scheduler, device, test_loader, metrics, save_path, best_score, name="model1.txt"):
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
            metrics.update(label_pred, label_true)

            # # 展示一下
            if idx == 0 and epoch % 5 == 0:
                print("pred", np.unique(label_pred))
                print("true", np.unique(label_true))
                show_label(label_pred[0], os.path.join(save_path, "label_pred_epoch{}.jpg".format(epoch)))
                show_label(label_true[0], os.path.join(save_path, "label_true_epoch{}.jpg".format(epoch)))

    score = metrics.get_results()
    metrics.to_str(score)
    # 保存训练数据
    save(epoch, score, os.path.join(save_path, name))
    #  对模型进行更新
    best_score = update(score, epoch, model, optimizer, lr_scheduler, best_score,
                        os.path.join(save_path, "best_model1.pth"))

    return best_score, score


def mk_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def draw(title, data, path):
    fig = plt.figure()
    plt.plot(range(len(data)), data, label='train', linestyle="-", color="red")
    plt.title(title)
    plt.show()
    fig.savefig('{}'.format(path))


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
    num_classes = 4
    label = 0.2  # 使用多少有标签
    batch_size = 4  #
    crop_size = (256, 256)  # 裁剪大小
    cuda = True
    ckpt1 = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行
    ckpt2 = None  # 模型2 恢复地址。如果模型突然终断。可以重新 添加地址运行
    start_epoch = 0
    if not cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path = r"/root/autodl-tmp/my-acdc/My-ACDC-2"

    save_path = "checkpoint/11-23"
    mk_path(save_path)
    label_loader, unlabel_loader, test_loader = get_loader(root=path, batch_size=batch_size, crop_size=crop_size,
                                                           label=label)
    metrics = Medical_Metric(num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model1 = UNet(n_classes=num_classes, n_channels=1).to(device)
    model2 = UNet(n_classes=num_classes, n_channels=1).to(device)

    #  定义两个优化器
    optimizer_1 = create_optimizer_v2(model1, **optimizer_kwargs(cfg=arg))
    lr_scheduler_1, num_epochs = create_scheduler(arg, optimizer_1)

    optimizer_2 = create_optimizer_v2(model2, **optimizer_kwargs(cfg=arg))
    lr_scheduler_2, num_epochs = create_scheduler(arg, optimizer_2)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion_csst = nn.MSELoss()
    dice_loss = DiceLoss(num_classes)

    best_dice = 0.0
    best_hd95 = 0.0
    #  加载原模型
    if ckpt1 is not None and os.path.isfile(ckpt1):
        state_dict = torch.load(ckpt1)
        start_epoch = state_dict["epoch"]
        model1 = state_dict["model"]
        optimizer_1 = state_dict["optimizer"]
        lr_scheduler_1 = state_dict["scheduler"]
        best_dice = state_dict["best_score"]

    if ckpt2 is not None and os.path.isfile(ckpt2):
        state_dict = torch.load(ckpt2)
        start_epoch = state_dict["epoch"]
        model2 = state_dict["model"]
        optimizer_2 = state_dict["optimizer"]
        lr_scheduler_2 = state_dict["scheduler"]
        best_dice = state_dict["best_score"]

    all_dice = []
    all_hd95 = []

    for epoch in range(start_epoch, num_epochs):
        train(epoch=epoch,
              model1=model1, optimizer_1=optimizer_1, lr_scheduler_1=lr_scheduler_1,
              model2=model2, optimizer_2=optimizer_2, lr_scheduler_2=lr_scheduler_2,
              label_loader=label_loader, unlabel_loader=unlabel_loader,
              device=device, criterion=criterion, dice_loss=dice_loss)

        best_dice, score = test(epoch=epoch, model=model1, lr_scheduler=lr_scheduler_1, optimizer=optimizer_1,
                                device=device,
                                test_loader=test_loader,
                                metrics=metrics, save_path=save_path,
                                best_score=best_dice, name="model1.txt")

        all_dice.append(score["mean_dice"])
        all_hd95.append(score["mean_hd952"])
        draw(title="epoch:{} Dice".format(epoch), data=all_dice, path=os.path.join(save_path, "dice.jpg"))
        draw(title="epoch:{} Hd95".format(epoch), data=all_hd95, path=os.path.join(save_path, "hd95.jpg"))

        best_dice, score = test(epoch=epoch, model=model2, optimizer=optimizer_2, lr_scheduler=lr_scheduler_2,
                                device=device, test_loader=test_loader,
                                metrics=metrics, save_path=save_path,
                                best_score=best_dice, name="model2.txt")


def update(val_score, epoch, model, optimizer, scheduler, best_score, path):
    if val_score['mean_dice'] > best_score:
        best_score = val_score['mean_dice']
        save_ckpt(path, epoch, model, optimizer, scheduler, best_score)
    return best_score


def save_ckpt(path, epoch, model, optimizer, scheduler, best_score):
    """ save current model
    """
    torch.save({
        "epoch": epoch,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "best_score": best_score,
    }, path)

    print("Model saved as %s" % path)


if __name__ == '__main__':
    main()
