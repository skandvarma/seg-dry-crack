import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image
import json
import torch.nn.functional as F
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, dataset_names, split, processor):
        self.data = []
        self.processor = processor

        for dataset in dataset_names:
            base_dir = f"data/{dataset}/{split}"
            mask_dir = f"data/masks/{dataset}/{split}"
            anno_file = f"{base_dir}/_annotations.coco.json"

            if not os.path.exists(anno_file):
                continue

            with open(anno_file, 'r') as f:
                coco = json.load(f)

            prompt = "segment taping area" if dataset == "drywall" else "segment crack"

            for img_info in coco['images']:
                img_file = img_info['file_name']
                img_path = os.path.join(base_dir, img_file)
                if not os.path.exists(img_path):
                    img_path = os.path.join(base_dir, 'images', img_file)
                if not os.path.exists(img_path):
                    continue

                mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png').replace('.jpeg', '.png'))
                if not os.path.exists(mask_path):
                    continue

                self.data.append((img_path, mask_path, prompt))

        print(f"Loaded {len(self.data)} samples for {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, prompt = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        inputs = self.processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")

        mask_array = np.array(mask) / 255.0
        mask_tensor = torch.from_numpy(mask_array).float()

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'mask': mask_tensor
        }


def dice_loss(pred, target):
    smooth = 1.
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)

    train_dataset = SegmentationDataset(['drywall', 'cracks'], 'train', processor)
    val_dataset = SegmentationDataset(['drywall', 'cracks'], 'valid', processor)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    epochs = 20
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'].to(device),
                pixel_values=batch['pixel_values'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )

            logits = outputs.logits
            mask = batch['mask'].to(device)

            if logits.dim() == 2:
                batch_size = mask.shape[0]
                logits = logits.view(batch_size, 352, 352)

            logits = logits.unsqueeze(1)
            mask = mask.unsqueeze(1)

            logits_resized = F.interpolate(logits, size=(mask.shape[2], mask.shape[3]), mode='bilinear',
                                           align_corners=False)

            loss = bce_loss(logits_resized, mask) + dice_loss(torch.sigmoid(logits_resized), mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    pixel_values=batch['pixel_values'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )

                logits = outputs.logits

                if logits.dim() == 2:
                    batch_size = logits.shape[0]
                    logits = logits.view(batch_size, 352, 352)

                logits = logits.unsqueeze(1)
                mask = batch['mask'].to(device).unsqueeze(1)

                logits_resized = F.interpolate(logits, size=(mask.shape[2], mask.shape[3]), mode='bilinear',
                                               align_corners=False)

                loss = bce_loss(logits_resized, mask) + dice_loss(torch.sigmoid(logits_resized), mask)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("outputs/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pth")
            print(f"  Saved checkpoint")

    print("Training complete")


if __name__ == "__main__":
    train()