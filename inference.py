import os
import torch
import torch.nn as nn

from torchvision import transforms

from pathlib import Path
from skimage import io
from PIL import Image

from model import Compressor
from model import DoubleConv, DownConv, UpConv
from utils import denormalize


MEAN = [0.4563639461994171, 0.4365175664424896, 0.4032033085823059]
STD = [0.27053767442703247, 0.26697292923927307, 0.28019198775291443]

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


class Encoder(nn.Module):
    def __init__(self, modules):
        super().__init__()

        self.doubleconv1 = modules[0]
        self.downconv1 = modules[1]
        self.downconv2 = modules[2]
        self.downconv3 = modules[3]
        self.downconv4 = modules[4]

    def forward(self, x):
        # x is the image to compress
        x1 = self.doubleconv1(x)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x = self.downconv4(x4)

        print(
            "Saving the compressed image to disk"
        )  # basically the latent space tensor
        torch.save(
            x1, "compressed/compressed_1.pt"
        )  # first concatenation (top to bottom)
        torch.save(x2, "compressed/compressed_2.pt")  # second concat etc.
        torch.save(x3, "compressed/compressed_3.pt")
        torch.save(x4, "compressed/compressed_4.pt")
        torch.save(x, "compressed/compressed_5.pt")

        return x


class Decoder(nn.Module):
    def __init__(self, modules, concat):
        super().__init__()
        self.upconv1 = modules[0]
        self.doubleconv2 = modules[1]

        self.upconv2 = modules[2]
        self.doubleconv3 = modules[3]

        self.upconv3 = modules[4]
        self.doubleconv4 = modules[5]

        self.upconv4 = modules[6]
        self.doubleconv5 = modules[7]

        self.out = modules[8]
        self.relu = modules[9]

        self.concat = concat

    def forward(self, x):
        # x is the latent space of the data
        x = self.upconv1(x)
        x = torch.cat([self.concat[1], x], dim=1)
        x = self.doubleconv2(x)

        x = self.upconv2(x)
        x = torch.cat([self.concat[3], x], dim=1)
        x = self.doubleconv3(x)

        x = self.upconv3(x)
        x = torch.cat([self.concat[2], x], dim=1)
        x = self.doubleconv4(x)

        x = self.upconv4(x)
        x = torch.cat([self.concat[0], x], dim=1)
        x = self.doubleconv5(x)

        x = self.relu(self.out(x))

        return x


# Image prep
DATA_DIR = str(Path(__file__).resolve().parent.parent) + "/data/VOC2010"
image_path = DATA_DIR + "/2007_000333.jpg"
size_in_bytes = os.path.getsize(image_path)
print(f"Initial image size (bytes): {size_in_bytes}")

img = io.imread(image_path)
init_h, init_w = img.shape[0], img.shape[1]
print(f"Initial image size (dimensions): {init_h}x{init_w}")
img = Image.fromarray(img)
img = transform(img)

# compressor prep
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("models/COMPRESSOR.pth").to(device)
img = img.to(device)
img = torch.unsqueeze(img, 0)  # add batch (necessary for Upsample() module)


encoder_module_names = [
    "doubleconv1",
    "downconv1",
    "downconv2",
    "downconv3",
    "downconv4",
]
decoder_module_names = [
    "upconv1",
    "doubleconv2",
    "upconv2",
    "doubleconv3",
    "upconv3",
    "doubleconv4",
    "upconv4",
    "doubleconv5",
    "out",
    "relu",
]

encoder_modules = [
    mod
    for m_name in encoder_module_names
    for name, mod in model.named_children()
    if name == m_name
]
decoder_modules = [
    mod
    for m_name in decoder_module_names
    for name, mod in model.named_children()
    if name == m_name
]

concat = [torch.load(f) for f in Path("compressed/").iterdir()]

encoder = Encoder(encoder_modules).to(device)
decoder = Decoder(decoder_modules, concat).to(device)
encoder.eval()  # no batchnorm / dropout used but i will put it anyway
decoder.eval()  # idem

with torch.no_grad():
    # init
    print("Compressing Image")
    compress = encoder(img)
    size_in_bytes = 0
    for f in Path("compressed").rglob("*.pt"):
        size_in_bytes += os.path.getsize(f)
    print(f"Image size after compression (bytes): {size_in_bytes}")

    print("Decompressing Image")
    decompress = decoder(compress)
    decompress = torch.squeeze(decompress, 0)  # remove batch

    # Need to return image to original stats
    decompress = transforms.Resize([init_h, init_w], antialias=True)(decompress)
    final_image = denormalize(decompress.cpu(), MEAN, STD)
    pil_image = transforms.ToPILImage()(final_image)
    pil_image.save("testing/decompressed_image.jpg")


decompressed_image_path = (
    str(Path(__file__).resolve().parent) + "/testing/decompressed_image.jpg"
)
size_in_bytes = os.path.getsize(decompressed_image_path)
print(f"Image size after decompressing {size_in_bytes}")
decompressed_img = io.imread(decompressed_image_path)
post_h, post_w = decompressed_img.shape[0], decompressed_img.shape[1]
print(f"Initial image size (dimensions): {post_h}x{post_w}")
