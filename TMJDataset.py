import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

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

        # Преобразуем маску в многоканальную (one-hot) форму
        msk = self.create_multichannel_mask(mask)  # Маска теперь будет размерности (584, 584, 2)
        msk = np.transpose(msk, (1, 2, 0))

        # Применяем преобразования, если они есть
        augmented = self.transforms(image=img, mask=msk)

        # Возвращаем изображение и маску в формате, который PyTorch понимает
        return augmented["image"].float(), np.transpose(augmented["mask"].float(), (2, 0, 1))

    @staticmethod
    def create_multichannel_mask(mask):
        """
        Создает маску в формате one-hot для трех классов:
        0 — фон, 1 — головка, 2 — ямка.
        """
        # Канал 0 - фон (1 для фона, 0 для остального)
        background_mask = (mask == 0).astype(np.uint8)

        # Канал 1 - головка (1 для головки, 0 для остального)
        head_mask = (mask == 1).astype(np.uint8)

        # Канал 2 - ямка (1 для ямки, 0 для остального)
        pit_mask = (mask == 2).astype(np.uint8)

        # Стек из трех каналов (фон, головка и ямка)
        multichannel_mask = np.stack([background_mask, head_mask, pit_mask], axis=0)

        return multichannel_mask

    def visualize(self, idx: int):
        image, mask = self[idx]
        image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Маска: разделим на три канала (фон, головка, ямка)
        background_mask = mask[0].cpu().numpy()
        head_mask = mask[1].cpu().numpy()
        pit_mask = mask[2].cpu().numpy()

        # Создаем фигуру для визуализации
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Визуализируем изображение
        axes[0].imshow(image)
        axes[0].set_title("Image (RGB)")
        axes[0].axis('off')

        # Визуализируем маску для фона, головки и ямки
        combined_mask = np.zeros((background_mask.shape[0], background_mask.shape[1], 3), dtype=np.uint8)

        combined_mask[background_mask == 1] = [0, 0, 0]

        # Красный для головки
        combined_mask[head_mask == 1] = [255, 0, 0]  # Красный для головки

        # Синий для ямки
        combined_mask[pit_mask == 1] = [0, 0, 255]  # Синий для ямки

        axes[1].imshow(combined_mask)
        axes[1].set_title("Combined Background, Head and Pit Masks")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
