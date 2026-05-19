import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_road import UNet, RoadsDataset, path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load("unet_road.pth", map_location=device))
        print("веса модели загружены")
    except FileNotFoundError:
        print("unet_road.pth не найдеа")
        return

    model.eval()

    TARGET_SIZE = (256, 256)
    dataset = RoadsDataset(path, target_size=TARGET_SIZE)

    image_path = dataset.images[0]
    mask_path = dataset.masks[0]

    with Image.open(image_path).convert("RGB") as img:
        img_resized = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with Image.open(mask_path).convert("L") as msk:
        msk_resized = msk.resize(TARGET_SIZE, Image.Resampling.NEAREST)
        true_mask = (np.array(msk_resized) == 82).astype(np.float32)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask_thresh = (pred_mask > 0.5).astype(np.float32)

    difference = np.abs(true_mask - pred_mask_thresh)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(true_mask, cmap="gray")
    axes[0].set_title("Исходная маска", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(pred_mask_thresh, cmap="gray")
    axes[1].set_title("Предсказанная", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(difference, cmap="hot")
    axes[2].set_title("Разница", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()