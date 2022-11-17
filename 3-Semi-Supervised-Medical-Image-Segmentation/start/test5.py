import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image=Image.open(r"E:\note\cv\data\mini_city\leftImg8bit\bochum_000000_003674_leftImg8bit.png")

image=np.array(image)
print(image.shape)

plt.figure()
plt.imshow(image[:,:,0],cmap="gray")
plt.show()