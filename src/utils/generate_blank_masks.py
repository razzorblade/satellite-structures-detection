#!/usr/bin/env python3
import os
import argparse
from PIL import Image

def create_black_mask(img_path, out_path):
    """
    Create a black (zeroed) mask for the image at img_path,
    saving it to out_path with identical dimensions.
    """
    with Image.open(img_path) as img:
        w, h = img.size
    # 'L' mode = single‐channel grayscale; 0 is black
    mask = Image.new('L', (w, h), 0)
    mask.save(out_path)

def main():
    parser = argparse.ArgumentParser(
        description="Generate black masks for every .png in a folder."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Folder containing source .png images."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Folder where blank masks will be saved."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith(".png"):
            continue
        src = os.path.join(args.input_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        create_black_mask(src, dst)
        print(f"Created mask for {fname}")
        count += 1

    print(f"Done — generated {count} black masks in '{args.output_dir}'")

if __name__ == "__main__":
    main()
