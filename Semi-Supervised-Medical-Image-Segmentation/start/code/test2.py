import h5py
import numpy as np
import SimpleITK as sitk
import torchvision.utils
from PIL import Image
from skimage import io
from PIL import Image
path=r"E:\note\ssl\data\ACDC-T\training\patient001\patient001_frame01.nii.gz"
# image = sitk.ReadImage(path)
# # image = sitk.RescaleIntensity(image, 0, 255)
# image = sitk.GetArrayFromImage(image)
# #
# io.imsave("1.png",image[0])
# # import cv2
# # cv2.imwrite("2.png",image[0])
# # image=Image.open(r"img.png")
# # print(np.array(image))

image=Image.open(r"E:\note\ssl\data\My-ACDC\mask\patient001_frame_ED_01_07.png")
print(np.unique(np.array(image)))