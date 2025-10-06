import os
import json
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def generate_masks(dataset_name, split):
    base_path = f"data/{dataset_name}/{split}"
    anno_file = f"{base_path}/_annotations.coco.json"
    output_path = f"data/masks/{dataset_name}/{split}"

    if not os.path.exists(anno_file):
        print(f"Skipping {dataset_name}/{split} - annotation file not found")
        return

    os.makedirs(output_path, exist_ok=True)

    with open(anno_file, 'r') as f:
        coco_data = json.load(f)

    existing_masks = set(os.listdir(output_path)) if os.path.exists(output_path) else set()
    images_to_process = [img for img in coco_data['images']
                         if img['file_name'].replace('.jpg', '.png').replace('.jpeg', '.png') not in existing_masks]

    if not images_to_process:
        print(f"{dataset_name}/{split}: already done, skipping")
        return

    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to("cuda")
    predictor = SamPredictor(sam)

    for img_data in images_to_process:
        img_id = img_data['id']
        img_file = img_data['file_name']

        img_path = os.path.join(base_path, img_file)
        if not os.path.exists(img_path):
            img_path = os.path.join(base_path, 'images', img_file)

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        annos = [a for a in coco_data['annotations'] if a['image_id'] == img_id]

        if not annos:
            continue

        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for anno in annos:
            bbox = anno['bbox']
            x, y, w, h = bbox
            input_box = np.array([x, y, x + w, y + h])

            masks, _, _ = predictor.predict(
                box=input_box,
                multimask_output=False
            )

            combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8)

        final_mask = combined_mask * 255
        mask_filename = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(os.path.join(output_path, mask_filename), final_mask)

    print(f"{dataset_name}/{split}: done")


if __name__ == "__main__":
    if not os.path.exists("sam_vit_b_01ec64.pth"):
        print("Downloading SAM checkpoint...")
        os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    else:
        print("SAM checkpoint exists, skipping download")

    datasets = ['drywall', 'cracks']
    splits = ['train', 'valid', 'test']

    for dataset in datasets:
        for split in splits:
            generate_masks(dataset, split)

    print("All masks generated")