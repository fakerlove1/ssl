import glob
import os

import h5py
import numpy
import numpy as np
import SimpleITK as sitk
import torchvision.utils
from PIL import Image
from skimage import io
from tqdm import tqdm
import nibabel as nli

to_path = r"E:\note\ssl\data\My-ACDC-2"

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
    image = sitk.GetArrayFromImage(image)
    image = (image - image.min()) / (image.max() - image.min())
    return np.array(image)


def get_label(label_path):
    image = sitk.ReadImage(label_path)
    image = sitk.GetArrayFromImage(image)
    return np.array(image,dtype=np.uint8)


def save_npy(ndarr, path):
    np.save(path, ndarr)


def save_txt(txt, path):
    with open(path, "a+") as file:
        file.write(txt + "\n")


def read(path):
    with open(path, "r") as f:
        sample_list = f.readlines()
        sample_list = [item.split("_")[0] for item in sample_list]
    return sample_list


train_list = read("train.list")
test_list = read("test.list")
val_list = read("val.list")

for dir_path in tqdm(os.listdir(root)):
    #  获取用户信息

    patient_info = os.path.join(root, dir_path, "Info.cfg")
    file = open(patient_info, 'r').readlines()
    ED_frame = int(file[0].split(":")[1])
    ES_frame = int(file[1].split(":")[1])

    if dir_path in train_list:
        txt_path = "train.txt"
    elif dir_path in test_list:
        txt_path = "test.txt"
    elif dir_path in val_list:
        txt_path = "val.txt"


    # 加载4个文件。
    img_path_ED = os.path.join(root, dir_path, "{}_frame{:02d}.nii.gz".format(dir_path, ED_frame))
    img_path_ES = os.path.join(root, dir_path, "{}_frame{:02d}.nii.gz".format(dir_path, ES_frame))
    label_path_ED = os.path.join(root, dir_path, "{}_frame{:02d}_gt.nii.gz".format(dir_path, ED_frame))
    label_path_ES = os.path.join(root, dir_path, "{}_frame{:02d}_gt.nii.gz".format(dir_path, ES_frame))

    img_ED = get_image(img_path_ED)
    img_ES = get_image(img_path_ES)
    label_ED = get_label(label_path_ED)
    label_ES = get_label(label_path_ED)

    batch_size = img_ED.shape[0]
    for i in range(batch_size):
        name = "{}_frame_ED_01_{:02d}.npy".format(dir_path, i + 1)
        imgpath = os.path.join(img_path, name)
        maskpath = os.path.join(mask_path, name)
        print(np.unique(np.array(label_ED[i])))
        save_npy(img_ED[i], imgpath)
        save_npy(label_ED[i], maskpath)
        save_txt(name, os.path.join(to_path, txt_path))
        save_txt(name, os.path.join(to_path, "train_val_test.txt"))

    batch_size = img_ES.shape[0]
    for i in range(batch_size):
        name = "{}_frame_ES_02_{:02d}.npy".format(dir_path, i + 1)
        imgpath = os.path.join(img_path, name)
        maskpath = os.path.join(mask_path, name)
        print(np.unique(np.array(label_ES[i])))
        save_npy(img_ES[i], imgpath)
        save_npy(label_ES[i], maskpath)
        save_txt(name, os.path.join(to_path, txt_path))
        save_txt(name, os.path.join(to_path, "train_val_test.txt"))

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
