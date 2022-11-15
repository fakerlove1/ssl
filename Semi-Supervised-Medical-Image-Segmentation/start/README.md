# ACDC数据集介绍



## 1. 下载



ACDC数据集：[ACDC Challenge (insa-lyon.fr)](https://acdc.creatis.insa-lyon.fr/description/databases.html)

界面长下面这个样子

![image-20221115111337554](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221115111337554.png)

点击Traing dataset.点击 下面的Online evalutation

![image-20221115111413164](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221115111413164.png)

有两个任务，看你需求。然后点击下载

![image-20221115111508729](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221115111508729.png)



>下面是百度网盘链接，我是做分割的，所以是分割的数据集
>
>训练集大概在1.5G左右，测试集也是一样
>
>链接：https://pan.baidu.com/s/1Rs-7eFTRhZdh9AzQUnYtGQ 
>提取码：tht9 
>--来自百度网盘超级会员V5的分享

## 2. 结构

医学图像基本格式`.dcm`、`.nii`和`nii.gz`

![image-20221115111851684](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221115111851684.png)



train 共100名用户，每名用户有6个文件 。test 共有50名用户，但没有掩膜mask信息。需要自己提交

info.cfg记录用户。用户年龄，体重，身高，舒张末期的id,收缩末期的id

~~~bash
ED: 1
ES: 12
Group: DCM
Height: 184.0
NbFrame: 30
Weight: 95.0
~~~

下面那个4d的文件是3d图，我做的是 2d分割。所以没用

所有文件介绍

~~~bash
# info.cfg 用户基本信息
# patient001_4d.nii.gz   3d图
# patient001_frame01.nii.gz  心脏舒张末期图片
# patient001_frame01_gt.nii.gz 心脏舒张掩膜图
# patient001_frame12.nii.gz  收缩末期图片
# patient001_frame12_gt.nii.gz  收缩末期掩膜图
~~~

舒张末期 ED(end diastolic) ,收缩末期 ES(end-systolic)



## 3. 转换

### 3.1 格式转换工具

`SimpelITK`是一个用于医学图像分析的Python库。

安装 最新版的可能有问题。反正我用的时候会报错。所以安装2.1.0以下版本就行

~~~
pip install SimpleITK=2.0.2
~~~

使用教程[SimpleITK读取医学图像 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/360556496)

----

`nibabel` 这个工具也可以加载 `nii`格式的医学数据

NiBabel包是可以对常见的医学和神经影像文件格式进行读写。

~~~python
import os
import numpy as np
import nibabel as nib

example_filename = os.path.join(data_path, 'example.nii.gz')
img = nib.load(example_filename)
img.get_data()
~~~





### 3.2 转换

> 可以使用别人处理好的数据。下面是链接
>
> [hritam-98/ICT-MedSeg: Code implementation for our paper ACCEPTED in ISBI, 2022 (github.com)](https://github.com/hritam-98/ICT-MedSeg)

----

下面是我自己处理的过程，不知道对不对。你可以直接使用别人的

~~~python
import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import torchvision.utils
from PIL import Image
from skimage import io
from tqdm import tqdm

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
    return np.array(image, dtype=np.uint8)


def get_label(path):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    return np.array(image, dtype=np.uint8)


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
        save_image(img_ED[i], imgpath)
        save_image(label_ED[i], maskpath)
        save_txt(name, os.path.join(to_path, "all.txt"))

    batch_size = img_ES.shape[0]
    for i in range(batch_size):
        name = "{}_frame_ES_02_{:02d}.png".format(dir_path, i + 1)
        imgpath = os.path.join(img_path, name)
        maskpath = os.path.join(mask_path, name)
        save_image(img_ES[i], imgpath)
        save_image(label_ES[i], maskpath)
        save_txt(name, os.path.join(to_path, "all.txt"))


with open(os.path.join(to_path, "all.txt"), "r") as file:
    all = np.array(file.readlines())
    np.random.shuffle(all)
    val = all[:int(len(all) * 0.2)]
    print(len(val))
    train = all[int(len(all) * 0.2):]

with open(os.path.join(to_path, "train.txt"), "a+") as file:
    file.writelines(train)

with open(os.path.join(to_path, "val.txt"), "a+") as file:
    file.writelines(val)

~~~



