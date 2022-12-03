import os.path
import numpy as np
import torch
import torch.nn as nn
import datetime

from torch.nn import DataParallel
from tqdm import tqdm

from voc import get_loader, show_label
from deeplabv3plus_resnet_atrous import deeplabv3plus
from stream_metrics import StreamSegMetrics
from utils import PolyLR, create_lr_scheduler
import matplotlib.pyplot as plt
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
            label_pred = out.detach().max(dim=1)[1].cpu().numpy()
            # label_pred=torch.argmax(out,dim=1).cpu().numpy()
            label_true = labels.cpu().numpy()
            metrics.update(label_true, label_pred)

            # # 展示一下
            if idx == 0:
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
    args.lr = 0.001
    args.weight_decay = 0.0005
    args.num_classes = 21
    args.batch_size = 16  #
    args.crop_size = (512, 512)  # 裁剪大小
    args.total_itrs = 160000
    args.step_size = 4000
    args.cuda = True  # 是否使用gpu
    args.ckpt = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行
    start_epoch = 0

    if not args.cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path = r"/root/autodl-tmp/my_voc/voc_aug_2/VOCdevkit/VOC2012"
    # path = r"/root/my_voc_2/voc_aug_2/VOCdevkit/VOC2012"
    save_path = "checkpoint/11-27"
    mk_path(save_path)
    label_loader, test_loader = get_loader(root=path, batch_size=args.batch_size, crop_size=args.crop_size)

    print("训练集 length: ",len(label_loader))
    print("测试集 length: ",len(test_loader))
    metrics = StreamSegMetrics(args.num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model = deeplabv3plus(num_classes=21, backbone="resnet101", pretrained=True).to(device)
    #  定义两个优化器

    optimizer = torch.optim.SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                                 {'params': [param for name, param in model.named_parameters()
                                             if 'backbone' not in name],
                                  'lr': args.lr * 10}],
                                lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # max_epoch=int
    # lr_scheduler = create_lr_scheduler(optimizer=optimizer, epochs=int(total_itrs // step_size + 1),
    #                                    num_step=step_size,
    #                                    warmup_factor=1e-4, warmup_epochs=1)
    # lr_scheduler = PolyLR(optimizer, max_iters=args.total_itrs, power=0.9)
    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou = 0.0
    #  加载原模型
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt)
        start_epoch = state_dict["epoch"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        lr_scheduler = state_dict["scheduler"]
        best_miou = state_dict["best_miou"]

    all_miou = []
    all_acc = []
    cur_itrs = 0

    print("===============> train start ! ==========================================")
    model.train()
    while True:
        train_loss = 0
        for i, (img_labeled, target_label) in enumerate(tqdm(label_loader)):
            cur_itrs += 1
            img_labeled = img_labeled.to(device).float()
            target_label = target_label.to(device).long()

            pred = model(img_labeled)
            loss = criterion(pred, target_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_scheduler.step()
            lr = args.lr * (1 - cur_itrs / args.total_itrs) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            train_loss += loss.item()
            if cur_itrs % 500 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    "Train  [{}/{} ({:.0f}%)]\t loss:{:.5f} \t lr:{:.5f} \t".format(
                        cur_itrs,
                        args.total_itrs,
                        100. * cur_itrs / args.total_itrs,
                        loss.item(),
                        lr))

            #  开始验证
            if cur_itrs % 1500 == 0:
                epoch = int(cur_itrs // len(label_loader.dataset))
                #  验证
                val_score = test(epoch=epoch, model=model, optimizer=optimizer,
                                 device=device, test_loader=test_loader,
                                 metrics=metrics, save_path=save_path)
                print(metrics.to_str(val_score))
                save(epoch, train_loss, os.path.join(save_path, "loss.txt"))
                save(epoch, val_score, os.path.join(save_path, "result.txt"))
                best_miou = update(val_score, epoch, model, optimizer, best_miou,
                                   os.path.join(save_path, "best_model.pth"))
                all_miou.append(val_score["Mean IoU"])
                all_acc.append(val_score["Overall Acc"])
                draw(title="epoch:{} Miou".format(epoch), data=all_miou, path=os.path.join(save_path, "miou.jpg"))
                draw(title="epoch:{} Acc".format(epoch), data=all_acc, path=os.path.join(save_path, "acc.jpg"))
                train_loss = 0

            if cur_itrs > args.total_itrs:
                return


def update(val_score, epoch, model, optimizer, best_miou, path):
    if val_score['Mean IoU'] > best_miou:
        best_miou = val_score['Mean IoU']
        save_ckpt(path, epoch, model, optimizer, best_miou)
    return best_miou


def save_ckpt(path, epoch, model, optimizer, best_miou):
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
