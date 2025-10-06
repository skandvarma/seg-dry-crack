# Prompted Segmentation for Drywall QA

Text-conditioned segmentation model for detecting drywall taping areas and wall cracks using CLIPSeg.

## Project Structure

```
.
├── train_clipseg.py        # Model training script
├── evaluate.py             # Model evaluation and metrics
├── generate_sam_masks.py   # Ground truth mask generation
├── data/                   # Dataset directory
│   ├── drywall/           # Drywall taping dataset
│   ├── cracks/            # Wall cracks dataset
│   └── masks/             # Generated segmentation masks
└── outputs/               # Model outputs
    ├── checkpoints/       # Saved model weights
    ├── predictions/       # Test predictions
    └── visualizations/    # Visual comparisons
```

## File Descriptions

### generate_sam_masks.py
Generates ground truth segmentation masks from COCO bounding box annotations using Segment Anything Model (SAM).

**Key Functions:**
- `generate_masks()`: Converts bounding boxes to pixel-level masks
- Downloads SAM checkpoint automatically if not present
- Processes train/valid/test splits for both datasets
- Outputs binary masks (0/255) as PNG files

**Usage:**
```bash
python generate_sam_masks.py
```

### train_clipseg.py
Fine-tunes CLIPSeg model for prompted segmentation on the combined dataset.

**Key Components:**
- `SegmentationDataset`: Custom PyTorch dataset loader
  - Loads images and masks from both datasets
  - Associates correct text prompts with each dataset
  - Handles COCO annotation parsing
  
- `dice_loss()`: Implements Dice coefficient loss for segmentation
  
- Training Configuration:
  - Model: CIDAS/clipseg-rd64-refined
  - Batch size: 8
  - Learning rate: 1e-4
  - Optimizer: AdamW
  - Loss: BCE + Dice combination
  - Epochs: 20
  - Automatic best model checkpointing

**Usage:**
```bash
python train_clipseg.py
```

### evaluate.py
Evaluates trained model on test sets and generates predictions.

**Key Features:**
- Loads fine-tuned checkpoint
- Processes both test datasets with appropriate prompts
- Calculates mIoU and Dice metrics
- Generates prediction masks following naming convention
- Creates visualization comparisons (original | ground truth | prediction)
- Outputs comprehensive results table

**Usage:**
```bash
python evaluate.py
```

## Installation

### Requirements
```bash
pip install torch torchvision
pip install transformers
pip install segment-anything
pip install opencv-python
pip install pillow
pip install numpy
```

### Dataset Setup
1. Download datasets from Roboflow:
   - [Drywall Dataset](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
   - [Cracks Dataset](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)

2. Extract to `data/` directory maintaining structure:
   ```
   data/
   ├── drywall/
   │   ├── train/
   │   ├── valid/
   │   └── test/
   └── cracks/
       ├── train/
       ├── valid/
       └── test/
   ```

## Running the Pipeline

1. **Generate ground truth masks:**
   ```bash
   python generate_sam_masks.py
   ```
   This creates pixel-level masks from bounding boxes.

2. **Train the model:**
   ```bash
   python train_clipseg.py
   ```
   Training automatically saves best checkpoint based on validation loss.

3. **Evaluate and generate predictions:**
   ```bash
   python evaluate.py
   ```
   Produces metrics and prediction masks for submission.

## Model Architecture

**Base Model:** CLIPSeg (CIDAS/clipseg-rd64-refined)
- Vision-language model for zero-shot and prompted segmentation
- Combines CLIP embeddings with dense prediction decoder
- Input: RGB image + text prompt
- Output: 352x352 segmentation logits

**Fine-tuning Strategy:**
- Full model fine-tuning (not frozen backbone)
- Combined loss: Binary Cross-Entropy + Dice
- Resizes predictions to match ground truth resolution
- Prompt mapping:
  - Drywall: "segment taping area"
  - Cracks: "segment crack"

## Results

| Dataset | Prompt | mIoU | Dice | Test Images |
|---------|--------|------|------|-------------|
| Drywall | segment taping area | 0.3763 | 0.5083 | 150 |
| Cracks | segment crack | 0.4273 | 0.5613 | 508 |
| **Overall** | - | **0.4018** | **0.5348** | 658 |

## Output Format

### Predictions
- Location: `outputs/predictions/`
- Format: Single-channel PNG, values {0, 255}
- Naming: `{image_id}__{prompt}.png`
- Example: `123__segment_crack.png`

### Visualizations
- Location: `outputs/visualizations/`
- Format: Side-by-side comparison (Original | GT | Prediction)
- Generated for first 4 test images per dataset

## Hardware Requirements
- GPU recommended (CUDA-capable)
- Minimum 8GB VRAM for batch size 8
- ~4GB for model weights and data

## Training Details
- Random seed: Not explicitly set (add `torch.manual_seed(42)` for reproducibility)
- Training time: ~30 minutes on NVIDIA RTX 3080
- Inference time: ~0.15 seconds per image
- Model size: 461MB (CLIPSeg-rd64)

## Known Limitations
- Model struggles with very thin cracks
- Performance drops on heavily textured surfaces
- Requires good lighting in input images
- Fixed input resolution may lose fine details

## Future Improvements
- Implement data augmentation
- Experiment with different prompt formulations
- Try prompt ensembling
- Add post-processing refinement
- Test larger CLIPSeg variants
