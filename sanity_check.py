import os
import torch
import torch.nn as nn

from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from model import Compressor


def set_seed(seed):
    torch.manual_seed(seed)


set_seed(42)
writer = SummaryWriter("../runs/sanity_check")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4563639461994171, 0.4365175664424896, 0.4032033085823059],
            std=[0.27053767442703247, 0.26697292923927307, 0.28019198775291443],
        ),
    ]
)


class minibatch(Dataset):
    def __init__(self, img_dir, img_count, transforms):
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = []
        self.img_count = img_count
        self.transform = transforms

        img_dir = Path(img_dir)
        for i, img_name in enumerate(img_dir.iterdir()):
            if i == self.img_count:
                break

            self.img_paths.append(img_name.name)

    def __len__(self):
        return self.img_count

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        img = self.transform(Image.open(img_path))
        # img = self.transform(np.asarray(img))

        return img, img


minibatch_set = minibatch(
    "/home/cosmin/Projects/nn-compression/data/VOC2010", 12, transform
)
minibatch_loader = DataLoader(dataset=minibatch_set, batch_size=12, shuffle=True)


def train(model, device, epochs):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(epochs):
        optimizer.zero_grad()
        total_loss = 0

        for img, l_img in tqdm(minibatch_loader):
            img, l_img = img.to(device), l_img.to(device)

            output = model(img)
            loss = criterion(output, l_img)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        with torch.no_grad():
            avg_loss = total_loss / len(minibatch_loader)
            print(f"Loss at epoch {_}: {avg_loss}")


model = Compressor()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working on {device}")
model.to(device)
train(model, device, 50)
