import os.path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
import datetime
from tqdm import tqdm

from acdc import get_loader, show_label, color
from losses import DiceLoss
from easydict import EasyDict
from utils import PolyLR,_get_logger
from val_2D import test_single_volume
from swin_unet import get_swin_unet
from tensorboardX import SummaryWriter
import logging


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


def test(epoch, cur_itrs, model, optimizer, lr_scheduler, device, test_loader, save_path, best_score, writer,
         name="model1.txt"):
    model.eval()
    metric_list = 0.0
    #  选取第一张图片

    for i_batch, sampled_batch in enumerate(test_loader):
        image = sampled_batch[0].to(device)
        label = sampled_batch[1].to(device)
        metric_i = test_single_volume(image, label, model, classes=4, patch_size=(224, 224))
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(test_loader.dataset)

    performance2 = np.mean(metric_list, axis=0)[0]
    mean_hd952 = np.mean(metric_list, axis=0)[1]

    # 保存训练数据
    save(epoch, "dice: {}".format(performance2), os.path.join(save_path, name))
    #  对模型进行更新
    best_score = update(performance2, epoch, model, optimizer, lr_scheduler, best_score,
                        os.path.join(save_path, "best_model1.pth"))
    model.train()
    return best_score, performance2


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
    args.lr = 0.01
    args.weight_decay = 0.0001
    args.total_itrs = 40000
    args.step_size = 500

    #   是否使用gpu
    args.num_classes = 4
    args.batch_size = 32  #
    args.crop_size = (224, 224)  # 裁剪大小
    args.cuda = True
    args.ckpt1 = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行

    if not args.cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # path = r"/root/autodl-tmp/my-acdc/My-ACDC-2"
    path = r"/home/ubuntu/data/ACDC"
    # path = r"/root/autodl-tmp/data/ACDC"
    save_path = "./checkpoint/12-3-swinunet"
    mk_path(save_path)
    train_loader, test_loader = get_loader(root=path, batch_size=args.batch_size,
                                           crop_size=args.crop_size,
                                           )
    writer = SummaryWriter(save_path + '/log')
    logger = _get_logger(save_path + '/info.log', 'info')

    # metrics = Medical_Metric(args.num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model = get_swin_unet(image_size=args.crop_size[0], num_classes=args.num_classes, in_channel=3).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = PolyLR(optimizer, max_iters=args.total_itrs, power=0.9)

    #  定义两个优化器

    max_epoch = args.total_itrs // len(train_loader) + 1
    logger.info("max_epoch length: {}".format(max_epoch))

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    best_dice1 = 0.0

    #  加载原模型
    if args.ckpt1 is not None and os.path.isfile(args.ckpt1):
        state_dict = torch.load(args.ckpt1)
        start_epoch = state_dict["epoch"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        lr_scheduler = state_dict["scheduler"]
        best_dice1 = state_dict["best_score"]

    all_dice = []
    all_hd95 = []

    model.train()
    cur_itrs = 0
    train_loss = 0
    lr = 0
    epoch = 0
    while True:
        for i, (img_labeled, target_label) in enumerate(tqdm(train_loader)):
            cur_itrs += 1
            ##############################################################################
            img_labeled = img_labeled.to(device).float()
            target_label = target_label.to(device).long()
            pseudo_labeled = model(img_labeled)
            ### 计算 有监督损失, standard cross entropy loss ###
            # loss = dice_loss(pseudo_labeled, target_label.unsqueeze(1))
            # loss = 0.5 * (criterion(pseudo_labeled, target_label) + dice_loss(pseudo_labeled,
            #                                                                   target_label.unsqueeze(1)))
            loss_ce = criterion(pseudo_labeled, target_label)
            loss_dice = dice_loss(pseudo_labeled, target_label.unsqueeze(1), softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - cur_itrs / args.total_itrs) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # lr_scheduler.step()
            # lr1 = optimizer.param_groups[0]["lr"]
            train_loss += loss.item()
            writer.add_scalar('lr', lr, cur_itrs)
            writer.add_scalar('loss_item', loss.item(), cur_itrs)

            # if cur_itrs % 500 == 0:
            #     infor = "Train [{}/{} ({:.0f}%)]\t loss: {:.5f} \t lr1: {:.5f}  ".format(
            #         cur_itrs,
            #         args.total_itrs,
            #         100. * cur_itrs / args.total_itrs,
            #         loss.item(),
            #         lr,
            #     )
            #     logging.info(infor)

            if cur_itrs % args.step_size == 0:
                writer.add_image('train/Image', img_labeled[0], cur_itrs)
                outputs = torch.argmax(torch.softmax(
                    pseudo_labeled, dim=1), dim=1, keepdim=False)
                outputs = color(outputs[0].cpu().numpy())
                writer.add_image('train/label_pred', outputs, cur_itrs, dataformats='HWC')  # 添加预测
                label_true = color(target_label[0].cpu().numpy())  # 添加真实值
                writer.add_image('train/label_true', label_true, cur_itrs, dataformats='HWC')

                best_dice1, score = test(epoch=epoch,
                                         model=model,
                                         cur_itrs=cur_itrs,
                                         lr_scheduler=lr_scheduler,
                                         optimizer=optimizer,
                                         device=device,
                                         test_loader=test_loader,
                                         save_path=save_path,
                                         best_score=best_dice1,
                                         name="model1.txt",
                                         writer=writer)
                all_dice.append(score)
                writer.add_scalar("dice", score, cur_itrs)

            if cur_itrs > args.total_itrs:
                return

        infor = "Train [{}/{} ({:.0f}%)]\t loss: {:.5f} \t lr1: {:.5f}\t ".format(
            cur_itrs,
            args.total_itrs,
            100. * cur_itrs / args.total_itrs,
            train_loss / len(train_loader),
            lr,
        )
        train_loss = 0
        epoch += 1
        save(epoch, infor, os.path.join(save_path, "loss.txt"))
        logger.info(infor)


def update(score, epoch, model, optimizer, scheduler, best_score, path):
    if score > best_score:
        best_score = score
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
