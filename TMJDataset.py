import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class TMJDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Получаем пути к изображению и маске
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Читаем изображение
        image = cv2.imread(img_path)

        # Загружаем маску из .npy
        mask = np.load(mask_path)  # Загружаем маску в формате .npy
        mask = mask.astype('long')  # Убедитесь, что маска целочисленная для сегментации

        # Применяем трансформации (если они есть)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
