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

from swin_unet import get_swin_unet
from acdc import get_loader, show_label
from medical_metrics import Medical_Metric
from losses import DiceLoss
from easydict import EasyDict
from utils import PolyLR


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
    model.train()
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
    args = EasyDict()
    args.lr = 0.0001
    args.weight_decay = 0.01
    args.total_itrs = 10000
    args.step_size = 500

    #   是否使用gpu
    args.num_classes = 4
    args.label = 0.2  # 使用多少有标签
    args.batch_size = 8  #
    args.crop_size = (256, 256)  # 裁剪大小
    args.cuda = True
    args.ckpt1 = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行
    args.ckpt2 = None  # 模型2 恢复地址。如果模型突然终断。可以重新 添加地址运行
    start_epoch = 0
    if not args.cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # path = r"/root/autodl-tmp/my-acdc/My-ACDC-2"
    path=r"/home/ubuntu/data/My-ACDC-2"

    save_path = "checkpoint/11-27"
    mk_path(save_path)
    label_loader, unlabel_loader, test_loader = get_loader(root=path, batch_size=args.batch_size,
                                                           crop_size=args.crop_size,
                                                           label=args.label)

    metrics = Medical_Metric(args.num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model1 = get_swin_unet(num_classes=4, image_size=256, in_channel=3).to(device)
    model2 = get_swin_unet(num_classes=4, image_size=256, in_channel=3).to(device)

    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    lr_scheduler1 = PolyLR(optimizer1, max_iters=args.total_itrs, power=0.9)

    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    lr_scheduler2 = PolyLR(optimizer2, max_iters=args.total_itrs, power=0.9)
    #  定义两个优化器

    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    print("max_epoch length: {}".format(max_epoch))
    consistency = 1.5
    consistency_rampup = max_epoch

    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return consistency * sigmoid_rampup(epoch, consistency_rampup)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    best_dice1 = 0.0
    best_dice2 = 0.0

    #  加载原模型
    if args.ckpt1 is not None and os.path.isfile(args.ckpt1):
        state_dict = torch.load(args.ckpt1)
        start_epoch = state_dict["epoch"]
        model1 = state_dict["model"]
        optimizer1 = state_dict["optimizer"]
        lr_scheduler1 = state_dict["scheduler"]
        best_dice1 = state_dict["best_score"]

    if args.ckpt2 is not None and os.path.isfile(args.ckpt2):
        state_dict = torch.load(args.ckpt2)
        start_epoch = state_dict["epoch"]
        model2 = state_dict["model"]
        optimizer2 = state_dict["optimizer"]
        lr_scheduler2 = state_dict["scheduler"]
        best_dice1 = state_dict["best_score"]

    all_dice = []
    all_hd95 = []

    model1.train()
    model2.train()
    cur_itrs = 0
    unlabel_len = len(unlabel_loader)
    label_iter = iter(label_loader)
    train_loss = 0
    while True:
        for i, (img_unlabeled, target_no_label) in enumerate(tqdm(unlabel_loader)):
            cur_itrs += 1

            cps_weight = get_current_consistency_weight(epoch=cur_itrs // unlabel_len)

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
            cps_loss = criterion(pred_1, max_2) + criterion(pred_2, max_1)

            ### 计算 有监督损失, standard cross entropy loss ###
            loss_1 = 0.5 * (
                    criterion(pseudo_labeled_1, target_label) + dice_loss(pseudo_labeled_1, target_label.unsqueeze(1)))

            loss_2 = 0.5 * (
                    criterion(pseudo_labeled_2, target_label) + dice_loss(pseudo_labeled_2, target_label.unsqueeze(1)))

            loss = cps_weight * cps_loss + loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            lr_scheduler1.step()
            lr_scheduler2.step()

            lr1 = optimizer1.param_groups[0]["lr"]
            lr2 = optimizer2.param_groups[0]["lr"]

            train_loss += loss.item()

            if cur_itrs % 500 == 0:
                print(
                    "Train [{}/{} ({:.0f}%)]\t loss: {:.5f} \t lr1: {:.5f} \t lr2: {:.5f} \t w: {:.5f}".format(
                        cur_itrs,
                        args.total_itrs,
                        100. * cur_itrs / args.total_itrs,
                        loss.item(),
                        lr1,
                        lr2, cps_weight))

            if cur_itrs % args.step_size == 0:
                epoch = cur_itrs // len(unlabel_loader)
                save(epoch, train_loss, os.path.join(save_path, "loss.txt"))
                best_dice1, score = test(epoch=epoch, model=model1, lr_scheduler=lr_scheduler1, optimizer=optimizer1,
                                         device=device,
                                         test_loader=test_loader,
                                         metrics=metrics, save_path=save_path,
                                         best_score=best_dice1, name="model1.txt")
                all_dice.append(score["mean_dice"])
                all_hd95.append(score["mean_hd952"])
                draw(title="epoch:{} Dice".format(epoch), data=all_dice, path=os.path.join(save_path, "dice.jpg"))
                draw(title="epoch:{} Hd95".format(epoch), data=all_hd95, path=os.path.join(save_path, "hd95.jpg"))

                best_dice2, score = test(epoch=epoch, model=model2, optimizer=optimizer2, lr_scheduler=lr_scheduler2,
                                         device=device, test_loader=test_loader,
                                         metrics=metrics, save_path=save_path,
                                         best_score=best_dice2, name="model2.txt")
                train_loss = 0
            if cur_itrs > args.total_itrs:
                return


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
        # "scheduler": scheduler,
        "best_score": best_score,
    }, path)

    print("Model saved as %s" % path)


if __name__ == '__main__':
    main()
