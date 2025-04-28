#!/usr/bin/env python3
"""
Fine-tune DeepLabV3+ (ResNet-50) for binary building segmentation.
"""

import os
import platform
import argparse
from pathlib import Path

import cv2
import torch
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip,
    RandomBrightnessContrast, GridDistortion,
    ElasticTransform, Normalize
)
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import DeepLabV3Plus, losses as smp_losses
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --------------------- 1) Dataset --------------------- #
class SatDataset(Dataset):
    """Loads image + mask, applies Albumentations, ensures mask is 1×H×W float32."""
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.fnames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        name = self.fnames[idx]
        img = cv2.imread(str(self.img_dir / name))[..., ::-1]       # BGR→RGB numpy
        mask_np = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_GRAYSCALE)  # numpy 0–255

        if self.transform:
            aug = self.transform(image=img, mask=mask_np)
            img_tensor = aug["image"]                               # torch.FloatTensor C×H×W
            mask = aug["mask"]                                      # either torch.Tensor H×W or numpy
        else:
            img_tensor = ToTensorV2()(image=img)["image"]
            mask = mask_np

        # Now normalize mask: ensure torch.FloatTensor, shape 1×H×W, values 0.0 or 1.0
        if isinstance(mask, torch.Tensor):
            m = mask
        else:
            m = torch.from_numpy(mask)

        if m.ndim == 2:
            m = m.unsqueeze(0)
        m = m.float()
        if m.max() > 1.0:
            m = m / 255.0

        return img_tensor, m

# --------------------- 2) Transforms --------------------- #
def get_transforms(tile_size):
    return Compose([
        RandomResizedCrop(tile_size, tile_size, scale=(0.8, 1.0)),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        GridDistortion(p=0.2),
        ElasticTransform(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --------------------- main --------------------- #
def main():
    p = argparse.ArgumentParser(description="Fine-tune DeepLabV3+ on satellite tiles")
    p.add_argument("--images", "-i", required=True,
                   help="Path to folder with input .png tiles")
    p.add_argument("--masks", "-m", required=True,
                   help="Path to folder with ground-truth .png masks")
    p.add_argument("--batch", "-b", type=int, default=8,
                   help="Batch size")
    p.add_argument("--epochs", "-e", type=int, default=20,
                   help="Number of training epochs")
    p.add_argument("--tile-size", "-t", type=int, default=512,
                   help="Tile size (px)")
    p.add_argument("--lr-head", type=float, default=1e-3,
                   help="LR for decoder/head")
    p.add_argument("--lr-backbone", type=float, default=1e-5,
                   help="LR for encoder/backbone")
    args = p.parse_args()

    # Data & Loader
    transform = get_transforms(args.tile_size)
    train_ds = SatDataset(args.images, args.masks, transform)
    num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    # Model
    model = DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None  # raw logits
    )
    # freeze encoder initially
    for p_enc in model.encoder.parameters():
        p_enc.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss & Optimizer
    dice_loss = smp_losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()
    def criterion(pred, target):
        return dice_loss(pred, target) + bce_loss(pred, target)

    optimizer = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": args.lr_head},
        {"params": model.encoder.parameters(), "lr": args.lr_backbone},
    ])

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        total_loss = 0.0

        for imgs, masks in prog:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                preds = model(imgs)
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            prog.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # Unfreeze backbone after 5 epochs
        if epoch == 5:
            for p_enc in model.encoder.parameters():
                p_enc.requires_grad = True
            print(">>> Encoder unfrozen for full fine-tuning")

    # Save
    out_path = Path("models") / "deeplabv3plus_building.pth"
    out_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f"Model weights saved to {out_path}")

if __name__ == "__main__":
    main()
