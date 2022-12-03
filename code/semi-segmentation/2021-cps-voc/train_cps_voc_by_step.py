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

from deeplabv3plus_resnet_atrous import deeplabv3plus
from voc import get_loader, show_label
from stream_metrics import StreamSegMetrics
import matplotlib.pyplot as plt
from utils import PolyLR
from easydict import EasyDict


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


def test(epoch, model, optimizer, device, test_loader, metrics, save_path):
    model.eval()
    metrics.reset()
    print("test-epoch--", epoch)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            optimizer.zero_grad()
            # 如果使用了gpu
            images = images.to(device).float()
            labels = labels.to(device).long()

            out = model(images)
            label_pred = out.detach().max(dim=1)[1].data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            metrics.update(label_true, label_pred)

            # # 展示一下
            if idx == 0 and epoch % 5 == 0:
                print("pred", np.unique(label_pred))
                print("true", np.unique(label_true))
                show_label(label_pred[0], os.path.join(save_path, "label_pred_epoch{}.jpg".format(epoch)))
                show_label(label_true[0], os.path.join(save_path, "label_true_epoch{}.jpg".format(epoch)))

    score = metrics.get_results()
    model.train()
    return score


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
    args.total_itrs = 40000
    args.step_size = 2000
    #   是否使用gpu
    args.num_classes = 21
    args.label = 1464  # 使用有标签的数量
    args.batch_size = 4  #
    args.crop_size = (512, 512)  # 裁剪大小
    args.cuda = True
    args.ckpt1 = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行
    args.ckpt2 = None  # 模型2 恢复地址。如果模型突然终断。可以重新 添加地址运行
    start_epoch = 0

    if not args.cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # path = r"/root/autodl-tmp/my_voc/voc_aug_2/VOCdevkit/VOC2012"
    path = r"/root/my_voc_2/voc_aug_2/VOCdevkit/VOC2012"
    save_path = "checkpoint/11-28"
    mk_path(save_path)

    label_loader, unlabel_loader, test_loader = get_loader(root=path, label=args.label, batch_size=args.batch_size,
                                                           crop_size=args.crop_size)
    print("label_loader length: ",len(label_loader))
    print("unlabel_loader length: ",len(unlabel_loader))
    print("test_loader length: ",len(test_loader))
    metrics = StreamSegMetrics(args.num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model1 = deeplabv3plus(num_classes=args.num_classes, backbone="resnet101", pretrained=False).to(device)
    model2 = deeplabv3plus(num_classes=args.num_classes, backbone="resnet101", pretrained=False).to(device)

    #  定义两个优化器
    # optimizer1 = torch.optim.SGD(model1.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler1 = PolyLR(optimizer1, max_iters=args.total_itrs, power=0.9)

    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler2 = PolyLR(optimizer2, max_iters=args.total_itrs, power=0.9)

    max_epoch = args.total_itrs // len(unlabel_loader) + 1

    consistency = 1.5
    consistency_rampup = max_epoch

    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return consistency * sigmoid_rampup(epoch, consistency_rampup)

    print("=================> total {}epoch ==================".format(max_epoch))

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    best_miou1 = 0.0
    best_miou2 = 0.0
    #  加载原模型
    if args.ckpt1 is not None and os.path.isfile(args.ckpt1):
        state_dict = torch.load(args.ckpt1)
        start_epoch = state_dict["epoch"]
        model1 = state_dict["model"]
        optimizer1 = state_dict["optimizer"]
        lr_scheduler1 = state_dict["scheduler"]
        best_miou = state_dict["best_miou"]

    if args.ckpt2 is not None and os.path.isfile(args.ckpt2):
        state_dict = torch.load(args.ckpt2)
        start_epoch = state_dict["epoch"]
        model2 = state_dict["model"]
        optimizer2 = state_dict["optimizer"]
        lr_scheduler2 = state_dict["scheduler"]
        best_miou = state_dict["best_miou"]

    all_miou = []
    all_acc = []
    cur_itrs = 0
    label_iter = iter(label_loader)
    unlabel_len = len(unlabel_loader)
    print("===============> train start ! ==========================================")
    while True:
        model1.train()
        model2.train()

        train_loss = 0
        for i, (img_unlabeled, target_no_label) in enumerate(tqdm(unlabel_loader)):
            cur_itrs += 1

            #  有标签的数据集
            try:
                img_labeled, target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(label_loader)
                img_labeled, target_label, = next(label_iter)

            cps_weight = get_current_consistency_weight(epoch=int(cur_itrs // unlabel_len))

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
            ### 计算 有监督损失, standard cross entropy loss ###
            loss_1 = criterion(pseudo_labeled_1, target_label)
            loss_2 = criterion(pseudo_labeled_2, target_label)
            ##############################################################################
            ### 计算 cps loss ###
            pred_1 = torch.cat([pseudo_labeled_1, pseudo_unlabeled_1], dim=0)
            pred_2 = torch.cat([pseudo_labeled_2, pseudo_unlabeled_2], dim=0)
            _, max_1 = torch.max(pred_1.detach(), dim=1)
            _, max_2 = torch.max(pred_2.detach(), dim=1)
            max_1 = max_1.long()
            max_2 = max_2.long()
            cps_loss = cps_weight * criterion(pred_1, max_2) + cps_weight * criterion(pred_2, max_1)

            loss = cps_loss + loss_1 + loss_2

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

            if i % 500 == 0:
                print(
                    "Train  [{}/{} ({:.0f}%)]\t loss: {:.5f} \t lr1: {:.5f} \t lr2: {:.5f} \t cps_weight: {:.5f}".format(
                        cur_itrs,
                        args.total_itrs,
                        100. * cur_itrs /  args.total_itrs,
                        loss.item(),
                        lr1,
                        lr2, cps_weight))

            if cur_itrs %  args.step_size == 0:
                epoch = int(cur_itrs // len(unlabel_loader))
                save(epoch, train_loss, os.path.join(save_path, "loss.txt"))
                val_score = test(epoch=epoch, model=model1, optimizer=optimizer1, device=device,
                                 test_loader=test_loader,
                                 metrics=metrics, save_path=save_path)
                print(metrics.to_str(val_score))
                save(epoch, val_score, os.path.join(save_path, "model1.txt"))
                best_miou1 = update(val_score, epoch, model1, optimizer1, lr_scheduler1, best_miou1,
                                    os.path.join(save_path, "best_model1.pth"))
                all_miou.append(val_score["Mean IoU"])
                all_acc.append(val_score["Overall Acc"])
                draw(title="epoch:{} Miou".format(epoch), data=all_miou, path=os.path.join(save_path, "miou.jpg"))
                draw(title="epoch:{} Acc".format(epoch), data=all_acc, path=os.path.join(save_path, "acc.jpg"))

                val_score = test(epoch=epoch, model=model2, optimizer=optimizer2,
                                 device=device, test_loader=test_loader, metrics=metrics, save_path=save_path)
                print(metrics.to_str(val_score))
                save(epoch, val_score, os.path.join(save_path, "model2.txt"))
                best_miou2 = update(val_score, epoch, model1, optimizer1, lr_scheduler1, best_miou2,
                                    os.path.join(save_path, "best_model2.pth"))
                train_loss = 0

            if cur_itrs > args.total_itrs:
                return


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
        # "scheduler": scheduler,
        "best_miou": best_miou,
    }, path)

    print("Model saved as %s" % path)


if __name__ == '__main__':
    main()
