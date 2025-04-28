#!/usr/bin/env python3
import torch
import cv2
import numpy as np
from segmentation_models_pytorch import DeepLabV3Plus
from torchvision import transforms

def load_model(weights_path, device):
    """Instantiate model, load weights, and return eval-mode model."""
    model = DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,      # we're loading our own weights
        in_channels=3,
        classes=1,
        activation=None            # raw logits
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def preprocess_image(img_path, device):
    """Read image, convert to RGB tensor, normalize, and batchify."""
    # Read BGR→RGB
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform to Tensor & Normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # scales to [0,1]
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ])
    tensor = preprocess(img).unsqueeze(0).to(device)  # 1×3×H×W
    return tensor

def infer(model, input_tensor):
    """Run model, apply sigmoid, and return a binary mask (H×W uint8)."""
    with torch.no_grad():
        logits = model(input_tensor)           # 1×1×H×W
        probs  = torch.sigmoid(logits)[0,0]    # H×W
        mask   = (probs > 0.5).cpu().numpy().astype(np.uint8) * 255
    return mask

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inference for DeepLabV3+ building segmentation"
    )
    parser.add_argument(
        "--weights", "-w", default="deeplabv3plus_building.pth",
        help="Path to model weights .pth"
    )
    parser.add_argument(
        "--input", "-i", default="test_tile.png",
        help="Path to input RGB image"
    )
    parser.add_argument(
        "--output", "-o", default="pred_mask.png",
        help="Path to save predicted mask"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.weights, device)
    inp    = preprocess_image(args.input, device)
    mask   = infer(model, inp)

    # Save the mask
    cv2.imwrite(args.output, mask)
    print(f"Mask saved to {args.output}")

if __name__ == "__main__":
    main()
