import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image


class TMJDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        """
        Конструктор для загрузки изображений и масок.

        :param image_dir: Путь к папке с изображениями (например, .jpg или .png).
        :param mask_dir: Путь к папке с масками в формате .npy.
        :param transform: Преобразования, которые нужно применить к данным.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        # Список всех изображений
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        """
        Возвращает количество элементов в датасете.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Загружает изображение и соответствующую маску по индексу.

        :param idx: Индекс изображения в датасете.
        :return: Кортеж (изображение, маска).
        """
        # Загружаем изображение
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")  # Преобразуем изображение в RGB формат
        img = np.array(image, dtype=np.uint8)

        filename = os.path.splitext(self.image_files[idx])[0]
        mask_name = f"mask_{filename}.npy"  # Формат имени маски
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.load(mask_path)  # Маска размером (584, 584)

        # Преобразуем маску в многоканальную
        msk = self.create_multichannel_mask(mask)
        msk = np.transpose(msk, (1, 2, 0))  # Получаем маску с двумя каналами (2, 584, 584)

        augmented = self.transforms(image=img, mask=msk)

        return augmented["image"].float(), np.transpose(augmented["mask"].float(), (2, 0, 1))

    @staticmethod
    def create_multichannel_mask(mask):
        # Создаём два канала, где:
        # - Канал 0 будет для головки (1 для головки, 0 для остального)
        # - Канал 1 будет для ямки (1 для ямки, 0 для остального)

        head_mask = (mask == 1).astype(np.uint8)  # 1 - головка, 0 - не головка
        pit_mask = (mask == 2).astype(np.uint8)  # 1 - ямка, 0 - не ямка

        multichannel_mask = np.stack([head_mask, pit_mask], axis=0)
        return multichannel_mask