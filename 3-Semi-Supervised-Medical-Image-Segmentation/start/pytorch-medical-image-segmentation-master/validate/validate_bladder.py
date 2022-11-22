import os
import cv2
import torch
import shutil
import utils.image_transforms as joint_transforms
from torch.utils.data import DataLoader
import utils.transforms as extended_transforms
from datasets import bladder
from utils.loss import *
from networks.u_net import Baseline
from tqdm import tqdm

crop_size = 256
val_path = r'..\media/Datasets/Bladder/raw_data'
center_crop = joint_transforms.CenterCrop(crop_size)
val_input_transform = extended_transforms.NpyToTensor()
target_transform = extended_transforms.MaskToTensor()

val_set = bladder.Dataset(val_path, 'val', 1,
                              joint_transform=None, transform=val_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

palette = bladder.palette
num_classes = bladder.num_classes

net = Baseline(img_ch=1, num_classes=num_classes, depth=2).cuda()
net.load_state_dict(torch.load("../checkpoint/unet_depth=2_fold_1_dice_348055.pth"))
net.eval()


def auto_val(net):
    # 效果展示图片数
    dices = 0
    class_dices = np.array([0] * (num_classes - 1), dtype=np.float)

    save_path = './results'
    if os.path.exists(save_path):
        # 若该目录已存在，则先删除，用来清空数据
        shutil.rmtree(os.path.join(save_path))
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, 'pred')
    gt_path = os.path.join(save_path, 'gt')
    os.makedirs(img_path)
    os.makedirs(pred_path)
    os.makedirs(gt_path)

    val_dice_arr = []
    for (input, mask), file_name in tqdm(val_loader):
        file_name = file_name[0].split('.')[0]

        X = input.cuda()
        pred = net(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()

        # pred[pred < 0.5] = 0
        # pred[np.logical_and(pred > 0.5, pred == 0.5)] = 1

        # 原图
        m1 = np.array(input.squeeze())
        m1 = helpers.array_to_img(np.expand_dims(m1, 2))

        # gt
        gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette)
        gt = helpers.array_to_img(gt)

        # pred
        save_pred = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette)
        save_pred_png = helpers.array_to_img(save_pred)

        # png格式
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))
        save_pred_png.save(os.path.join(pred_path, file_name + '.png'))

        class_dice = []
        for i in range(1, num_classes):
            class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

        mean_dice = sum(class_dice) / len(class_dice)
        val_dice_arr.append(class_dice)
        dices += mean_dice
        class_dices += np.array(class_dice)
        print('mean_dice: {:.4} - dice_bladder: {:.4} - dice_tumor: {:.4}'
                  .format(mean_dice, class_dice[0], class_dice[1]))

    val_mean_dice = dices / (len(val_loader) / 1)
    val_class_dice = class_dices / (len(val_loader) / 1)
    print('Val mean_dice: {:.4} - dice_bladder: {:.4} - dice_tumor: {:.4}'.format(val_mean_dice, val_class_dice[0], val_class_dice[1]))


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    auto_val(net)