import torch
from tqdm import tqdm

from torchvision.io import read_image
from torchvision.transforms import functional as TF
from pathlib import Path

p = Path("./data/VOC2010")
paths = [str(file) for file in p.rglob("*.jpg")]

image_tensors = []

mean = torch.zeros(3)
var = torch.zeros(3)
std = torch.zeros(3)
n_pixels = torch.tensor(0)
sum = 0
sum_sq = 0

for i in tqdm(range(len(paths))):
    #    image_tensors.append(read_image(paths[i]) / 255.0)
    image_tensor = read_image(paths[i]) / 255.0
    resized_tensor = TF.resize(image_tensor, [224, 224])
    c, h, w = resized_tensor.shape
    sum += torch.sum(resized_tensor, dim=[1, 2])
    sum_sq += torch.sum(resized_tensor**2, dim=[1, 2])
    n_pixels += h * w

mean = sum / n_pixels
var = sum_sq / n_pixels - mean**2
std = torch.sqrt(var)

print(mean.tolist())  # [0.4563639461994171, 0.4365175664424896, 0.4032033085823059]
print(std.tolist())  # [0.27053767442703247, 0.26697292923927307, 0.28019198775291443]
