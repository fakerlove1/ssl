# from torchvision.transforms import transforms
import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
transform=A.Compose([
    A.RandomResizedCrop(height=512,width=512, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
    A.HorizontalFlip(p=0.5),
    # A.RandomGamma(),
    A.ColorJitter(0.1, 0.1, 0.1),
])

image=Image.open(r"E:\note\cv\data\LIDC\image\LIDC_3894.png")

image=transform(image=np.array(image))

plt.imshow(image['image'],cmap="gray")
plt.show()
plt.savefig("1.png")
