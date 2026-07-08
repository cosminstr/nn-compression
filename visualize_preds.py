import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import Compressor
from utils import denormalize


class voc_dataset(Dataset):
    def __init__(self, DATA_DIR, transform):
        super().__init__()
        self.img_paths = [str(f) for f in Path(DATA_DIR).rglob("*.jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = io.imread(self.img_paths[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        return img


# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = torch.load("models/COMPRESSOR.pth")
model.to(device)
model.eval()

# Data setup
data_dir = str(Path(__file__).resolve().parent.parent) + "/data/VOC2010"
mean = [0.4563639461994171, 0.4365175664424896, 0.4032033085823059]
std = [0.27053767442703247, 0.26697292923927307, 0.28019198775291443]

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

dataset = voc_dataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Visualize predictions
num_samples = 5
fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

count = 0
with torch.no_grad():
    for input_img in dataloader:
        input_img = input_img.to(device)
        output = model(input_img)

        # Denormalize for visualization
        original = denormalize(input_img[0].cpu(), mean, std).permute(1, 2, 0).numpy()
        prediction = denormalize(output[0].cpu(), mean, std).permute(1, 2, 0).numpy()

        # Plot
        axes[count, 0].imshow(original)
        axes[count, 0].set_title(f"Ground Truth {count + 1}")
        axes[count, 0].axis("off")

        axes[count, 1].imshow(prediction)
        axes[count, 1].set_title(f"Model Output {count + 1}")
        axes[count, 1].axis("off")

        count += 1
        if count >= num_samples:
            break

plt.tight_layout()
plt.show()
