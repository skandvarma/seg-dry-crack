import os
import torch
import numpy as np
from PIL import Image
import json
import cv2
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


def calculate_metrics(pred_binary, gt_binary):
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 0.0, 0.0

    iou = intersection / union
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-6)

    return iou, dice


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    model.eval()

    datasets = {
        'drywall': 'segment taping area',
        'cracks': 'segment crack'
    }

    results = {}
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)

    for dataset_name, prompt in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        test_dir = f"data/{dataset_name}/test"
        mask_dir = f"data/masks/{dataset_name}/test"
        anno_file = f"{test_dir}/_annotations.coco.json"

        if not os.path.exists(anno_file):
            print(f"Skipping {dataset_name} - test annotations not found")
            continue

        with open(anno_file, 'r') as f:
            coco = json.load(f)

        ious = []
        dices = []
        processed = 0

        for img_info in coco['images']:
            img_file = img_info['file_name']
            img_id = img_info['id']

            img_path = os.path.join(test_dir, img_file)
            if not os.path.exists(img_path):
                img_path = os.path.join(test_dir, 'images', img_file)

            if not os.path.exists(img_path):
                continue

            mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png').replace('.jpeg', '.png'))

            if not os.path.exists(mask_path):
                continue

            image = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L"))

            # Fix 1: Process inputs correctly
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")

            # Fix 2: Use inputs correctly with .to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'].to(device),
                    pixel_values=inputs['pixel_values'].to(device),
                    attention_mask=inputs['attention_mask'].to(device) if 'attention_mask' in inputs else None
                )

            logits = outputs.logits

            # Fix 3: Handle logits reshaping correctly
            if logits.dim() == 2:
                # Reshape from (batch_size * H * W,) to (batch_size, H, W)
                # CLIPSeg outputs 352x352 by default
                logits = logits.view(1, 352, 352)

            # Move to CPU and numpy
            logits = logits.squeeze(0).cpu().numpy()

            if processed == 0:
                print(f"  First image debug:")
                print(f"    Logits shape: {logits.shape}")
                print(f"    Logits min/max/mean: {logits.min():.4f} / {logits.max():.4f} / {logits.mean():.4f}")
                print(f"    GT mask shape: {gt_mask.shape}")
                print(f"    GT mask has {(gt_mask > 127).sum()} positive pixels")

            # Resize logits to match ground truth mask size
            logits_resized = cv2.resize(logits, (gt_mask.shape[1], gt_mask.shape[0]))
            pred_probs = 1 / (1 + np.exp(-logits_resized))

            if processed == 0:
                print(
                    f"    Pred probs min/max/mean: {pred_probs.min():.4f} / {pred_probs.max():.4f} / {pred_probs.mean():.4f}")
                print(f"    Pixels > 0.5: {(pred_probs > 0.5).sum()}")

            pred_binary = pred_probs > 0.5
            gt_binary = gt_mask > 127

            iou, dice = calculate_metrics(pred_binary, gt_binary)
            ious.append(iou)
            dices.append(dice)

            pred_save = (pred_binary * 255).astype(np.uint8)
            prompt_clean = prompt.replace(' ', '_')
            pred_filename = f"{img_id}__{prompt_clean}.png"
            cv2.imwrite(f"outputs/predictions/{pred_filename}", pred_save)

            if processed < 4:
                orig = cv2.imread(img_path)
                if orig.shape[:2] != gt_binary.shape:
                    orig = cv2.resize(orig, (gt_binary.shape[1], gt_binary.shape[0]))
                gt_viz = cv2.cvtColor((gt_binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                pred_viz = cv2.cvtColor(pred_save, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([orig, gt_viz, pred_viz])
                cv2.imwrite(f"outputs/visualizations/{dataset_name}_{img_id}.png", combined)

            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} images...")

        results[dataset_name] = {
            'prompt': prompt,
            'mIoU': np.mean(ious),
            'Dice': np.mean(dices),
            'count': len(ious)
        }
        print(f"  Completed {len(ious)} images")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Prompt':<25} | {'mIoU':<8} | {'Dice':<8} | {'Images':<8}")
    print("-" * 60)
    for dataset, metrics in results.items():
        print(f"{metrics['prompt']:<25} | {metrics['mIoU']:<8.4f} | {metrics['Dice']:<8.4f} | {metrics['count']:<8}")

    overall_miou = np.mean([r['mIoU'] for r in results.values()])
    overall_dice = np.mean([r['Dice'] for r in results.values()])
    print("-" * 60)
    print(f"{'Overall Average':<25} | {overall_miou:<8.4f} | {overall_dice:<8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()