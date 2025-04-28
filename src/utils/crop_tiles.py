#!/usr/bin/env python3
import os
import argparse
from PIL import Image

def crop_image_to_tiles(img_path, out_dir, tile_size=512):
    """
    Crop the image at img_path into non-overlapping tile_size x tile_size tiles,
    saving them into out_dir.
    """
    img = Image.open(img_path)
    w, h = img.size

    nx = w // tile_size
    ny = h // tile_size

    basename = os.path.splitext(os.path.basename(img_path))[0]
    count = 0

    for iy in range(ny):
        for ix in range(nx):
            left = ix * tile_size
            upper = iy * tile_size
            right = left + tile_size
            lower = upper + tile_size

            tile = img.crop((left, upper, right, lower))
            out_name = f"{basename}_x{ix:02d}_y{iy:02d}.png"
            tile.save(os.path.join(out_dir, out_name))
            count += 1

    return count

def main():
    parser = argparse.ArgumentParser(
        description="Crop all .png images in a folder into 512x512 tiles."
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="Path to folder containing .png images."
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Path where cropped tiles will be written."
    )
    parser.add_argument(
        "--tile_size", "-t", type=int, default=512,
        help="Size of the square tile (default: 512)."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_tiles = 0
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(".png"):
            continue
        img_path = os.path.join(args.input_dir, fname)
        tiles = crop_image_to_tiles(img_path, args.output_dir, args.tile_size)
        print(f"{fname}: extracted {tiles} tiles")
        total_tiles += tiles

    print(f"Done — total tiles written: {total_tiles}")

if __name__ == "__main__":
    main()
