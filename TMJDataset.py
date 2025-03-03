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
        mask = np.load(mask_path)

        assert self.transforms is not None

        augmented = self.transforms(image=img, mask=mask)

        return augmented["image"].float(), augmented["mask"].float()

    def visualize(self, idx):
        """
        Визуализирует изображение и соответствующую маску по индексу.

        :param idx: Индекс элемента в датасете.
        """
        # Получаем изображение и маску
        img, mask = self[idx]

        # Если маска имеет более одного канала (например, цвета), то для визуализации будем показывать только один канал
        if mask.ndimension() == 3:
            mask = mask[0]  # Предполагаем, что маска представлена в одном канале

        # Преобразуем в numpy для отображения
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask = mask.cpu().numpy().astype(np.uint8)

        # Визуализируем
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        plt.show()
