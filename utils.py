import torch
import matplotlib.pyplot as plt


def denormalize(tensor, mean, std):
    """Denormalize images using provided mean and std."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def get_comparison_figure(model, loader, device, mean, std, num_samples=3):
    """
    Generate a matplotlib figure comparing model predictions with ground truth.
    """
    model.eval()
    count = 0
    clean_imgs = []
    predictions = []

    with torch.no_grad():
        for input_img in loader:
            input_img = input_img.to(device)

            output = model(input_img)

            for i in range(min(num_samples - count, input_img.shape[0])):
                clean_imgs.append(input_img[i].cpu())
                predictions.append(output[i].cpu())
                count += 1
                if count >= num_samples:
                    break

            if count >= num_samples:
                break

    # Create figure with 3 rows and 3 columns
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        # Decompressed
        axes[idx, 0].imshow(
            denormalize(predictions[idx], mean, std).permute(1, 2, 0).numpy()
        )
        axes[idx, 0].set_title(f"Model Output {idx + 1}")
        axes[idx, 0].axis("off")

        # Original
        axes[idx, 1].imshow(
            denormalize(clean_imgs[idx], mean, std).permute(1, 2, 0).numpy()
        )
        axes[idx, 1].set_title(f"Clean Image {idx + 1}")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    return fig
