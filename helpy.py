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

import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

def _create_mask(coco, image_id, class_name_to_index):
    """
    Создаёт многоканальную маску для изображения.
    :param coco: объект COCO
    :param image_id: ID изображения
    :param class_name_to_index: словарь {имя_класса: индекс_в_маске}
    :return: np.array маски с несколькими каналами
    """
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    # Создаем пустую многоканальную маску
    mask_shape = (coco.imgs[image_id]['height'], coco.imgs[image_id]['width'], len(class_name_to_index))
    mask = np.zeros(mask_shape, dtype=np.uint8)

    for ann in anns:
        category_id = ann["category_id"]
        class_index = class_name_to_index.get(coco.cats[category_id]["name"], None)

        if class_index is not None:
            mask[..., class_index] += coco.annToMask(ann)

    return mask

def create_masks_for_all_images(coco, output_folder, class_name_to_index):
    os.makedirs(output_folder, exist_ok=True)
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        mask = _create_mask(coco, image_id, class_name_to_index)

        # Сохраняем каждый канал как отдельный PNG
        for class_name, class_index in class_name_to_index.items():
            mask_pil = Image.fromarray(mask[..., class_index] * 255)
            mask_pil.save(os.path.join(output_folder, f"mask_{class_name}_{str(image_id).zfill(4)}.png"))

