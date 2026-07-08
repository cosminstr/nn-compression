import torch
import torch.nn as nn

from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from skimage import io
from torchvision import transforms
from PIL import Image

from pytorch_msssim import MS_SSIM
from tqdm import tqdm

from model import Compressor
from utils import get_comparison_figure, denormalize


def set_seed(seed):
    torch.manual_seed(seed)


set_seed(42)
writer = SummaryWriter("../runs/nn_compressor")
data_dir = (
    str(Path(__file__).resolve().parent.parent) + "/data/VOC2010"
)  # <project_path>/data/VOC2010
BATCH_SIZE = 10
EPOCHS = 50
IMAGES_TO_TRAIN_COMPRESSOR_ON = 11000
MEAN = [0.4563639461994171, 0.4365175664424896, 0.4032033085823059]
STD = [0.27053767442703247, 0.26697292923927307, 0.28019198775291443]

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ]
)


class MixedLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.ss = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.alpha = alpha

    def forward(self, X, Y):
        # must denormalize images for MS-SSIM to work
        # as per the implementation at
        # https://github.com/VainF/pytorch-msssim?tab=readme-ov-file#2-normalized-input
        X_ = denormalize(tensor=X.cpu(), mean=MEAN, std=STD).to(device)
        Y_ = denormalize(tensor=Y.cpu(), mean=MEAN, std=STD).to(device)
        ssim = self.ss(X_, Y_)  # Has values between [0, 1]
        mse = nn.MSELoss()(X, Y)

        return self.alpha * (1 - ssim) + (1 - self.alpha) * mse


class voc_dataset(Dataset):
    def __init__(self, DATA_DIR, transform):
        super().__init__()
        self.img_paths = [str(f) for f in Path(DATA_DIR).rglob("*.jpg")]
        self.transform = transform

    def __len__(self):
        # return len(self.img_paths)
        return IMAGES_TO_TRAIN_COMPRESSOR_ON

    def __getitem__(self, idx):
        img = io.imread(self.img_paths[idx])
        img = Image.fromarray(
            img
        )  # transforms.Resize requires either PIL or torch tensor

        img = self.transform(img)

        return img


dataset = voc_dataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(model, loader, device, epochs):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = MixedLoss(alpha=0.90)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[35, 45], gamma=0.5
    )
    # criterion = nn.MSELoss()

    global_loss = torch.inf
    for _ in range(epochs):
        optimizer.zero_grad()
        total_loss = 0

        for img_batch in tqdm(loader):
            imgs = img_batch.to(device)

            output = model(imgs)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item()

        total_loss = total_loss / len(loader)
        print(f"Loss at epoch {_ + 1}/{epochs} -> {total_loss:.4f}")

        if total_loss < global_loss:
            print("Saving compressor state")
            torch.save(model, "models/compressor.pth")

            global_loss = total_loss

        # Visual tracking of model perfomance inside tensorboard
        comparison_fig = get_comparison_figure(
            model,
            loader,
            device,
            mean=MEAN,
            std=STD,
            num_samples=10,
        )
        writer.add_figure(
            "'decompressed' images vs actual images",
            comparison_fig,
        )

        scheduler.step()


nn_compressor = Compressor()
device = "cuda" if torch.cuda.is_available() else "cpu"
nn_compressor.to(device)

print("Training image compressor")
train(nn_compressor, dataloader, device, EPOCHS)
