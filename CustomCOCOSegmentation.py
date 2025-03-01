import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch
import torchvision.transforms.functional as F

class CustomCOCOSegmentation(Dataset):
    def __init__(self, root_dir: str, ann_file: str, transform=None):
        """
        :param root_dir: Папка с изображениями
        :param ann_file: Путь к файлу аннотации COCO (json)
        :param transform: Преобразования для данных
        """
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())  # Исправлено: imgToAnns → imgs
        self.transform = transform

    def __getitem__(self, idx):
        # Загружаем ID изображения
        img_id = self.ids[idx]

        # Получаем информацию об изображении
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # Загружаем изображение
        img = Image.open(img_path).convert("RGB")

        # Загружаем аннотации для данного изображения
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Создаём маску
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            class_id = ann['category_id']
            mask = np.maximum(mask, self.coco.annToMask(ann) * class_id)

        # Преобразования
        if self.transform:
            augmented = self.transform(image=np.array(img), mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = F.to_tensor(img)  # В `FloatTensor` [0,1]
            mask = torch.tensor(mask, dtype=torch.long)  # В `LongTensor`

        return img, mask

    def __len__(self):
        return len(self.ids)
