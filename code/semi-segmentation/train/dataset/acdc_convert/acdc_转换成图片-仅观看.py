import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import torchvision.utils
from PIL import Image
from skimage import io
from tqdm import tqdm
import nibabel as nli


to_path = r"E:\note\ssl\data\My-ACDC"

root = r"E:\note\ssl\data\ACDC-T\training"


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


img_path = os.path.join(to_path, "image")
mask_path = os.path.join(to_path, "mask")
make_path(img_path)
make_path(mask_path)


def get_image(path):
    image = sitk.ReadImage(path)
    # print(image.GetSize())
    image = sitk.RescaleIntensity(image, 0, 255)
    image = sitk.GetArrayFromImage(image)
    # image.GetSize() 返回的值为 [w,h,c]
    # SimpleITK 返回的文件格式为[c,h,w]

    # 如果使用 nibabel 读取文件,返回的是 [w,h,c]
    # 宽度，后高度，再深度
    # (216, 256, 10)
    # 所以需要最后的转置.
    return np.array(image,dtype=np.uint8)


def get_label(label_path):
    image = sitk.ReadImage(label_path)
    image = sitk.GetArrayFromImage(image)
    return np.array(image,dtype=np.uint8)


def save_image(ndarr, path):
    im = Image.fromarray(ndarr)
    im.save(path)
    # io.imsave(path, ndarr)


def save_txt(txt, path):
    with open(path, "a+") as file:
        file.write(txt + "\n")


for dir_path in tqdm(os.listdir(root)):
    #  获取用户信息

    patient_info = os.path.join(root, dir_path, "Info.cfg")

    file = open(patient_info, 'r').readlines()
    # 心脏拍片一共 有 4个部分
    # ED: 1
    # ES: 12
    # patient001_4d.nii.gz   3d图
    # patient001_frame01.nii.gz  心脏舒张末期图片
    # patient001_frame01_gt.nii.gz 心脏舒张掩膜图
    # patient001_frame12.nii.gz  收缩末期图片
    # patient001_frame12_gt.nii.gz  收缩末期掩膜图
    #  舒张末期 ED(end diastolic) ,收缩末期 ES(end-systolic)

    ED_frame = int(file[0].split(":")[1])
    ES_frame = int(file[1].split(":")[1])

    # 加载4个文件。
    img_path_ED = os.path.join(root, dir_path, "{}_frame{:02d}.nii.gz".format(dir_path, ED_frame))
    img_path_ES = os.path.join(root, dir_path, "{}_frame{:02d}.nii.gz".format(dir_path, ES_frame))
    label_path_ED = os.path.join(root, dir_path, "{}_frame{:02d}_gt.nii.gz".format(dir_path, ED_frame))
    label_path_ES = os.path.join(root, dir_path, "{}_frame{:02d}_gt.nii.gz".format(dir_path, ES_frame))

    img_ED = get_image(img_path_ED)
    img_ES = get_image(img_path_ES)
    label_ED = get_label(label_path_ED)
    label_ES = get_label(label_path_ED)

    # patient001_frame01.nii.gz 一个文件可能含有9-13张图片不等。
    # patient001 有10张 。所以维度为[10,256,216]
    # SimpleITK 返回的文件格式为[c,h,w]
    # 如果使用 nibabel 读取文件,返回的是 [w,h,c]

    batch_size = img_ED.shape[0]
    for i in range(batch_size):
        name = "{}_frame_ED_01_{:02d}.png".format(dir_path, i + 1)
        imgpath = os.path.join(img_path, name)
        maskpath = os.path.join(mask_path, name)
        print(np.unique(np.array(label_ED[i])))
        # save_image(img_ED[i], imgpath)
        # save_image(label_ED[i], maskpath)
        # save_txt(name, os.path.join(to_path, "all.txt"))

    batch_size = img_ES.shape[0]
    for i in range(batch_size):
        name = "{}_frame_ES_02_{:02d}.png".format(dir_path, i + 1)
        imgpath = os.path.join(img_path, name)
        maskpath = os.path.join(mask_path, name)
        print(np.unique(np.array(label_ES[i])))
        # save_image(img_ES[i], imgpath)
        # save_image(label_ES[i], maskpath)
        # save_txt(name, os.path.join(to_path, "all.txt"))


# with open(os.path.join(to_path, "all.txt"), "r") as file:
#     all = np.array(file.readlines())
#     np.random.shuffle(all)
#     val = all[:int(len(all) * 0.2)]
#     print(len(val))
#     train = all[int(len(all) * 0.2):]
#
# with open(os.path.join(to_path, "train.txt"), "a+") as file:
#     file.writelines(train)
#
# with open(os.path.join(to_path, "val.txt"), "a+") as file:
#     file.writelines(val)
