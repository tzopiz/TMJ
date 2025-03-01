import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def rename_files_and_update_annotations(folder_path):
    annotation_file = os.path.join(folder_path, "annotations.json")
    folder_name = os.path.basename(os.path.normpath(folder_path))

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    for img in coco_data["images"]:
        img_id = img["id"]
        old_name = img["file_name"]
        new_name = f"{folder_name}_{img_id:05d}.jpg"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            img["file_name"] = new_name
            print(f"{old_name} -> {new_name}")
        else:
            print(f"Файл {old_name} не найден")

    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)


def rename_files_in_folder(folder_path):
    folder_name = os.path.basename(os.path.abspath(folder_path))
    files = sorted(os.listdir(folder_path))

    for i, file in enumerate(files):
        old_path = os.path.join(folder_path, file)
        if not os.path.isfile(old_path):
            continue

        index = i
        while True:
            new_name = f"{folder_name}_{index:05d}{os.path.splitext(file)[1]}"
            new_path = os.path.join(folder_path, new_name)
            if not os.path.exists(new_path):
                break
            index += 1

        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")


def visualize_samples(dataset, idx_range):
    """
    Функция для визуализации изображений и масок по заданному диапазону индексов.

    :param dataset: Датасет, содержащий изображения и маски
    :param idx_range: Диапазон индексов, для которых нужно отобразить изображения
    """
    # Размеры графиков
    num_samples = len(idx_range)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))

    # Если только один элемент в диапазоне, axes будет не 2D, нужно обработать
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(idx_range):
        img, mask = dataset[idx]
        img = np.array(img)

        # Отображаем изображение
        axes[i, 0].imshow(img.transpose(1, 2, 0))  # Изменяем оси для корректного отображения
        axes[i, 0].set_title(f"Image {idx}")
        axes[i, 0].axis('off')

        # Отображаем маску
        axes[i, 1].imshow(mask.numpy(), alpha=1.0)
        axes[i, 1].set_title(f"Mask {idx}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

