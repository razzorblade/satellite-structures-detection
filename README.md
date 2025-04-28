# Satellite Structures Detection

Satellite Structures Detector is a DeepLabV3+-based neural network pipeline for detecting buildings and man-made structures in satellite imagery. Designed for small-to-medium custom datasets generated from Google Static Maps or similar raster sources. Fast prototyping with manual masks, easy fine-tuning, and simple inference ready for real-world applications.

---

## Installation

```bash
# 1. Clone repo
$ git clone https://github.com/razzorblade/satellite-structures-detection.git
$ cd satellite-structures-detection

# 2. Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # or venv/Scripts/activate on Windows # or call venv/Scripts/activate.bat in Anaconda

# 3. Install dependencies
$ pip install -r requirements.txt
```

---

## Project Structure

```
├── data/
│   └── training_example/
│       ├── images/
│       └── masks/
│
├── src/
│   ├── models/
│   │   ├── deeplabv3plus_building.pth # first example trained on very few images
│   │   └── download.txt # use for download of above model
│   ├── utils/           # only helper functions for preparing data
│   │   ├── crop_tiles.py				
│   │   └── generate_blank_masks.py
│   ├── inference.py
│   ├── requirements.txt
│   ├── pred_mask.png    # (ignored, quick test example)
│   ├── test_tile.png    # (ignored, quick test example)
│   ├── train.py
│   └── usage.txt
│
├── venv/                # virtual environment
└── README.md
```

---

## Dataset Setup

Expected folder structure:

```
dataset/
  images/
    tile_000.png
    tile_001.png
  masks/
    tile_000.png  # manual mask (white buildings / black background)
    tile_001.png
```

Tiles should be 512x512 px. Masks must match images exactly.


---

## Utilities

### Crop Large Images into Tiles
This will convert large satellite imagery into 512x512 tiles.

```bash
python crop_tiles.py --input_dir path/to/large_images --output_dir path/to/tiles
```

### Generate Blank Masks
Generate a completely black mask for each tile for ease of use. Then manually draw structures in white (255) on these masks.
```bash
python generate_blank_masks.py --input_dir path/to/tiles --output_dir path/to/masks
```

---

## Training

```bash
python train.py \
  --images data/images \
  --masks  data/masks \
  --batch   8 \
  --epochs  20 \
  --tile-size 512 \
  --lr-head    1e-3 \
  --lr-backbone 1e-5
```

- Trains DeepLabV3+ (ResNet-50) with frozen encoder for 5 epochs, then full fine-tuning.
- Saves weights to `deeplabv3plus_building.pth`.
- Move manually to models/ folder for better organization

Example:

```bash
python train.py --images dataset/images --masks dataset/masks
```
---

## Inference

```bash
python inference.py \
  --weights models/deeplabv3plus_building.pth \
  --input test_tile.png \
  --output pred_mask.png
```

- Outputs predicted mask as `pred_mask.png` (white=structure, black=background).

---

## Future Improvements

- **Training on Larger Datasets:**
  - Use hundreds of manually labeled tiles.
  - Increase model capacity (e.g., DeepLabV3+ with EfficientNet encoders).

- **Data Augmentation:**
  - Cloud simulation, random occlusions, seasonal variations.

- **Multi-class Segmentation:**
  - Detect roads, rivers, farmlands, etc. along with structures.

- **Cloud Dataset / Automation:**
  - Integrate with Google Static Maps API fully automated scraping and tiling.

- **Real-time or Edge Deployment:**
  - Optimize model for ONNX export and inference on mobile or embedded devices.

---

## License

MIT License. Free to use, modify, and deploy.

