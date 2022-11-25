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

from voc import get_loader, show_label
from deeplabv3plus_resnet import deeplabv3plus
from stream_metrics import StreamSegMetrics
#
import matplotlib.pyplot as plt


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


def train(epoch, model, optimizer, lr_scheduler, label_loader,
          device, criterion, ):
    model.train()

    train_loss = 0
    len_iter = len(label_loader)
    for i, (img_labeled, target_label) in enumerate(tqdm(label_loader)):
        optimizer.zero_grad()
        img_labeled = img_labeled.to(device).float()
        target_label = target_label.to(device).long()

        pred = model(img_labeled)
        loss = criterion(pred, target_label)

        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)

        train_loss += loss.item()
        if i % 500 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss:{:.5f} \t lr:{:.5f} \t".format(
                    epoch + 1,
                    i,
                    len_iter,
                    100. * i / len_iter,
                    loss.item(),
                    lr))
    return train_loss / len_iter


def test(epoch, model, optimizer, device, test_loader, metrics, save_path):
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
                show_label(label_pred[0], os.path.join(save_path, "label_pred_epoch{}.jpg".format(epoch)))
                show_label(label_true[0], os.path.join(save_path, "label_true_epoch{}.jpg".format(epoch)))

    score = metrics.get_results()

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
    # 3. 准备学习策略
    class arg:
        opt = 'sgd'
        lr = 0.01
        weight_decay = 0.0005
        momentum = 0.9
        epochs = 100
        sched = 'poly'
        decay_rate = .9
        min_lr = 1e-4
        warmup_lr = 1e-4
        warmup_epochs = 3
        cooldown_epochs = 10

    #   是否使用gpu
    num_classes = 21
    batch_size = 8  #
    crop_size = (512, 512)  # 裁剪大小
    cuda = True
    ckpt = None  # 模型1 恢复地址。如果模型突然终断。可以重新 添加地址运行
    start_epoch = 0

    if not cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # path = r"/root/autodl-tmp/my_voc/voc_aug_2/VOCdevkit/VOC2012"
    path = r"/root/my_voc_2/voc_aug_2/VOCdevkit/VOC2012"
    save_path = "checkpoint/11-24"
    mk_path(save_path)
    label_loader,  test_loader = get_loader(root=path, batch_size=batch_size, crop_size=crop_size)
    metrics = StreamSegMetrics(num_classes)
    #  对于这两个网络，我们使用相同的结构，但是不同的初始化。
    #  对网络用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。
    #  网络已经进行过初始化了
    model = deeplabv3plus(num_classes=21, pretrained=True).to(device)

    #  定义两个优化器
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=arg))
    lr_scheduler, num_epochs = create_scheduler(arg, optimizer)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou = 0.0
    #  加载原模型
    if ckpt is not None and os.path.isfile(ckpt):
        state_dict = torch.load(ckpt)
        start_epoch = state_dict["epoch"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        lr_scheduler = state_dict["scheduler"]
        best_miou = state_dict["best_miou"]

    all_miou = []
    all_acc = []

    for epoch in range(start_epoch, num_epochs):
        train(epoch=epoch,
              model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
              label_loader=label_loader,
              device=device, criterion=criterion)

        val_score = test(epoch=epoch, model=model, optimizer=optimizer, device=device, test_loader=test_loader,
                         metrics=metrics, save_path=save_path)
        print(metrics.to_str(val_score))
        save(epoch, val_score, os.path.join(save_path, "model1.txt"))
        best_miou = update(val_score, epoch, model, optimizer, lr_scheduler, best_miou,
                           os.path.join(save_path, "best_model1.pth"))
        all_miou.append(val_score["Mean IoU"])
        all_acc.append(val_score["Overall Acc"])
        draw(title="epoch:{} Miou".format(epoch), data=all_miou, path=os.path.join(save_path, "miou.jpg"))
        draw(title="epoch:{} Acc".format(epoch), data=all_acc, path=os.path.join(save_path, "acc.jpg"))


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
