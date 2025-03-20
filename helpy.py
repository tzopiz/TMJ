import os
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cv2
import torch

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms



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


def visualize_npy(file_path):
    # Загружаем данные из .npy файла
    data = np.load(file_path)

    # Если данные многомерные (например, изображение)
    if len(data.shape) == 2:
        plt.imshow(data, cmap='gray')
        plt.colorbar()
        plt.show()
    # Если это массив изображений (например, батч из изображений)
    elif len(data.shape) == 3:
        # Показать первое изображение в батче
        plt.imshow(data[0], cmap='gray')
        plt.colorbar()
        plt.show()
    else:
        print("Данные имеют неподдерживаемую форму:", data.shape)

def compute_class_weights(dataset, num_classes):
    class_pixel_count = torch.zeros(num_classes)

    for image, mask in dataset:
        # Маска будет иметь форму (num_classes, height, width)
        mask_tensor = mask

        for c in range(num_classes):
            class_pixel_count[c] += torch.sum(mask_tensor[c] == 1)

    # Нормализуем веса
    total_pixels = torch.sum(class_pixel_count)
    class_weights = total_pixels / (num_classes * class_pixel_count)

    return class_weights


def visualize_prediction(model, dataset, index, device, threshold=0.5):
    # Загружаем изображение и маску
    image, mask = dataset[index]

    # Получаем предсказания модели
    logits = model(image.unsqueeze(0).to(device)).squeeze(0)

    # Преобразуем изображение в numpy
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Разделяем маску на два канала
    head_mask = mask[1].cpu().numpy()
    pit_mask = mask[2].cpu().numpy()

    # Предсказанные маски
    head_mask_logits = logits[1].cpu().detach().numpy()
    pit_mask_logits = logits[2].cpu().detach().numpy()

    # Создаем фигуру для визуализации
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Визуализируем изображение
    axes[0].imshow(image)
    axes[0].set_title("Image (RGB)")

    # Визуализируем истинную маску (красный — головка, синий — ямка)
    combined_mask = np.zeros(
        (head_mask.shape[0], head_mask.shape[1], 3), dtype=np.uint8
    )
    combined_mask[head_mask == 1] = [255, 0, 0]  # Красный для головки
    combined_mask[pit_mask == 1] = [0, 0, 255]  # Синий для ямки
    axes[1].imshow(combined_mask)
    axes[1].set_title("Ground Truth Mask")

    # Визуализируем предсказанную моделью маску
    combined_mask_pred = np.zeros(
        (head_mask_logits.shape[0], head_mask_logits.shape[1], 3), dtype=np.uint8
    )
    combined_mask_pred[head_mask_logits > threshold] = [255, 0, 0]
    combined_mask_pred[pit_mask_logits > threshold] = [0, 0, 255]
    axes[2].imshow(combined_mask_pred)
    axes[2].set_title("Prediction")

    plt.tight_layout()
    plt.show()


def calculate_head_height(mask, intersection_point):
    x_intersect, y_intersect = intersection_point

    # Берем все точки маски, которые лежат на этой вертикальной оси
    y_indices = np.where(mask[:, x_intersect] > 0)[0]

    if len(y_indices) == 0:
        raise ValueError("Нет точек маски на оси X = x_intersect")

    # Самая верхняя точка среди них
    y_top = y_indices.min()

    # Высота — разница по оси Y
    height = abs(y_intersect - y_top)

    return height



def visualize_with_axes(scaled_mask, x_top, y_extreme, height=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(scaled_mask, cmap='binary')

    # Рисуем оси
    plt.axvline(x=x_top, color='red', linestyle='--', linewidth=1)
    plt.axhline(y=y_extreme, color='red', linestyle='--', linewidth=1)
    plt.scatter(x_top, y_extreme, color='red', s=50)

    # Если передана высота, добавляем визуальное обозначение
    if height is not None:
        plt.gca().add_patch(patches.FancyArrow(
            x_top + 10,
            y_extreme - 5,
            0,
            0,
            width=1,
            head_width=5,
            head_length=5,
            color='blue')
        )
        plt.gca().add_patch(patches.FancyArrow(
            x_top + 10,
            y_extreme - 5,
            0,
            -height + 20,
            width=1,
            head_width=5,
            head_length=5,
            color='blue')
        )
        # Подпись высоты
        plt.text(
            x_top + 15,
            y_extreme - height / 2,
            f"Высота: {height}",
            color='blue',
            fontsize=12,
            verticalalignment='center',
            weight='bold'
        )

    plt.axis("off")
    plt.show()


def visualize_mask(mask_tensor, title="Маска"):
    plt.figure(figsize=(5, 5))
    plt.imshow(mask_tensor.numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()


def find_axes(mask):
    ys, xs = np.where(mask > 0)

    y_top = ys.min()
    x_top = xs[np.argmin(ys)]
    is_right = x_top < mask.shape[1] // 2
    x_min, x_max = xs.min(), xs.max()

    if is_right:
        x_extreme = x_max
    else:
        x_extreme = x_min
    y_extreme = ys[xs == x_extreme].mean().astype(int)

    x_intersect, y_intersect = x_top, y_extreme

    # Коррекция точки пересечения, если она не принадлежит маске
    while x_intersect >= 0 and x_intersect < mask.shape[1] and mask[y_intersect, x_intersect] == 0:
        if is_right:
            x_intersect += 1  # Двигаем влево
        else:
            x_intersect -= 1  # Двигаем вправо

    return x_intersect, y_intersect


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

def compute_areas(mask):
    background_area = torch.sum(mask[0]).item()  # Количество пикселей фона
    head_area = torch.sum(mask[1]).item()        # Количество пикселей головки
    pit_area = torch.sum(mask[2]).item()         # Количество пикселей ямки

    return {
        "background": background_area,
        "head": head_area,
        "pit": pit_area
    }