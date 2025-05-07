# masks_creator
An interactive annotation tool built with Python and Tkinter that leverages Meta's Segment Anything Model (SAM) for efficient water body segmentation. Create accurate masks with just a few clicks, save annotations, and generate metadata for datasets.

# SAM-Annotator: Interactive Segmentation Tool

A lightweight, efficient annotation tool that utilizes Meta's Segment Anything Model (SAM) to create high-quality segmentation masks for any type of dataset with minimal user input.

![SAM-Annotator Demo](https://path-to-your-demo.gif)

## 🌟 Features

- **One-Click Segmentation**: Create precise masks with just a few clicks
- **Interactive Interface**: User-friendly Tkinter GUI for rapid annotation
- **Batch Processing**: Efficiently work through entire datasets
- **Metadata Generation**: Automatically tracks statistics and annotations
- **Optimized Performance**: Configured for both GPU and CPU environments

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/sam-annotator.git
cd sam-annotator
```

2. Clone the Segment Anything repository:
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

3. Install dependencies:
```bash
pip install opencv-python torch numpy pandas pillow
```

4. Download the SAM checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P segment-anything/
```

## 📊 Dataset Structure

Prepare your dataset with the following structure:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── masks/
└── metadata.csv (will be generated)
```

## 🚀 Usage

1. Place your images in the `dataset/images` directory
2. Run the application:
```bash
python sam_annotator.py
```
3. Use the interface to:
   - Click on objects to create masks
   - Add multiple clicks to refine segmentation
   - Add masks for multiple objects in the same image
   - Save masks and move to the next image
   - Skip images when needed
   
4. When finished, the tool will save:
   - All masks to `dataset/masks/`
   - A comprehensive metadata CSV with area, bounding box, and confidence data

## 🔧 Configuration

Modify these variables in the script to customize:

```python
IMAGE_DIR = "dataset/images"    # Location of input images
MASK_DIR = "dataset/masks"      # Where masks will be saved
METADATA_PATH = "dataset/metadata.csv"  # Path for metadata
CHECKPOINT_PATH = "segment-anything/sam_vit_h_4b8939.pth"  # SAM model
MODEL_TYPE = "vit_h"  # Model type (vit_h, vit_l, vit_b)
```

You can also modify the window size, compression settings, and other parameters in the code to suit your specific needs.

## 📝 Metadata

The generated metadata.csv includes:
- Image filename
- Mask filename
- Click coordinates
- Segmentation area
- Bounding box dimensions
- Confidence score

This metadata can be used for further analysis, training machine learning models, or filtering your dataset.

## 💡 Tips for Better Segmentation

- Click near the center of objects for best results
- For complex objects, add multiple clicks
- For objects with clear boundaries, fewer clicks work better
- Adjust the model type based on your computational resources:
  - `vit_h`: Highest quality, but slower and more resource-intensive
  - `vit_l`: Good balance of quality and speed
  - `vit_b`: Fastest, but may have lower quality for complex objects

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta Research
- Contributors to the open-source libraries used in this project

---

Made with ❤️ for efficient dataset annotation
