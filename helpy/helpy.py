import os
import json
import numpy as np
import cv2
import torch

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


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


def save_coco_masks(coco_annotation_path, output_dir):
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Создаём выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Создаём словарь соответствия image_id -> file_name и размеров изображений
    image_info = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco_data["images"]}

    # Группируем аннотации по image_id
    annotations_by_image = {}
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Обрабатываем изображения
    for image_id, annotations in annotations_by_image.items():
        file_name, width, height = image_info.get(image_id, (f"image_{image_id}.png", None, None))
        if width is None or height is None:
            continue

        # Создаём пустую цветную маску
        mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Генерируем цвета для классов
        np.random.seed(42)  # Фиксируем цвета для воспроизводимости
        category_colors = {}

        # Добавляем маски в изображение
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id not in category_colors:
                category_colors[category_id] = np.random.randint(0, 255, 3, dtype=np.uint8)

            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list):  # Полигональный формат
                for polygon in segmentation:
                    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], color=category_colors[category_id].tolist())
            elif isinstance(segmentation, dict):  # RLE формат
                binary_mask = maskUtils.decode(segmentation)
                mask[binary_mask == 1] = category_colors[category_id]
            else:
                raise ValueError("Формат сегментации не поддерживается.")

        # Сохраняем маску
        mask_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_mask.png")
        cv2.imwrite(mask_path, mask)

    print(f"Маски сохранены в {output_dir}")


def save_coco_masks_npy(coco_annotation_path, output_dir):
    """
    Сохраняет маски из аннотаций COCO в формате .npy без потерь качества.

    :param coco_annotation_path: Путь к JSON файлу с аннотациями COCO
    :param output_dir: Директория для сохранения масок
    """
    # Загружаем аннотации
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Создаём выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Создаём словарь соответствия image_id -> размеры изображений
    image_info = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco_data["images"]}

    # Группируем аннотации по image_id
    annotations_by_image = {}
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Обрабатываем изображения
    for image_id, annotations in annotations_by_image.items():
        file_name, width, height = image_info.get(image_id, (f"image_{image_id}.png", None, None))
        if width is None or height is None:
            continue

        # Создаём пустую маску (одноканальное изображение)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Заполняем маску аннотациями
        for annotation in annotations:
            category_id = annotation["category_id"]  # Значение класса
            segmentation = annotation["segmentation"]

            if isinstance(segmentation, list):  # Полигональный формат
                for polygon in segmentation:
                    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], color=category_id)  # Заполняем область значением category_id
            elif isinstance(segmentation, dict):  # RLE формат
                binary_mask = maskUtils.decode(segmentation)
                mask[binary_mask == 1] = category_id
            else:
                raise ValueError("Формат сегментации не поддерживается.")

        # Сохраняем маску в формате .npy с именем mask_{filename}.npy
        mask_path = os.path.join(output_dir, f"mask_{os.path.splitext(file_name)[0]}.npy")
        np.save(mask_path, mask)

    print(f"Маски сохранены в {output_dir} в формате .npy")


def merge_coco_json(json_files, output_file):
    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_id_offset = 0
    existing_category_ids = set()

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset['images']:
            image['id'] += image_id_offset
            merged_annotations['images'].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset['annotations']:
            annotation['id'] += annotation_id_offset
            annotation['image_id'] += image_id_offset
            merged_annotations['annotations'].append(annotation)

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset['categories']:
            if category['id'] not in existing_category_ids:
                category['id'] += category_id_offset
                merged_annotations['categories'].append(category)
                existing_category_ids.add(category['id'])

        image_id_offset = len(merged_annotations['images'])
        annotation_id_offset = len(merged_annotations['annotations'])
        category_id_offset = len(merged_annotations['categories'])

    # Save merged annotations to output file
    with open(output_file, 'w') as f:
        json.dump(merged_annotations, f)


def scale_head_mask(mask, scale_factor=4, padding=10):
    # Извлекаем маску головки
    head_mask = mask[1].cpu().numpy() if isinstance(mask, torch.Tensor) else mask[1]

    # Находим координаты непустых пикселей
    y_indices, x_indices = np.where(head_mask > 0)
    if len(y_indices) == 0:
        raise ValueError("Головка не найдена в маске")

    # Определяем границы
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Вырезаем область головки
    cropped = head_mask[y_min:y_max + 1, x_min:x_max + 1]

    # Масштабируем изображение с интерполяцией
    h, w = cropped.shape
    resized = cv2.resize(
        cropped, (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_NEAREST
    )

    # Добавляем отступы
    padded = np.pad(resized, padding, mode='constant', constant_values=0)

    return padded