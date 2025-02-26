import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TMJDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        # Загружаем COCO-аннотации
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)

        self.image_ids = [img['id'] for img in self.coco['images']]

        # Создаем маппинг image_id -> filename
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}

        # Создаем маппинг image_id -> аннотации
        self.image_id_to_annotations = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco['annotations']:
            self.image_id_to_annotations[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, self.image_id_to_filename[image_id])
        image = Image.open(image_path).convert("RGB")

        annotations = self.image_id_to_annotations[image_id]
        boxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])  # Преобразуем в [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]

        return image, target
