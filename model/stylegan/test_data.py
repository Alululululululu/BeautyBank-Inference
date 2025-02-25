import os
from PIL import Image
import numpy as np

# 定义图像的路径
path = "/home/s25223/newdisk/AEI-Net6/data/img_bareskin/"

# 读取路径下所有的.png文件
images = []
for filename in os.listdir(path):
    if filename.endswith(".png"):
        file_path = os.path.join(path, filename)
        # 打开并转换图像为需要的格式，例如转换为RGB
        with Image.open(file_path).convert("RGB") as img:
            # 可以在此进行图像的处理，例如改变大小等
            img_resized = img.resize((256, 256))
            # 将处理后的图像添加到列表中
            images.append(np.array(img_resized))
            print(len(images))

print(f"Loaded {len(images)} images.")