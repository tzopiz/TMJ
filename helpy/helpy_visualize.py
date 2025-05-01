import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch


def get_bounding_box(mask, padding=20, image_shape=None):
    """Определяет ограничивающий прямоугольник вокруг маски и добавляет отступ."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None  # Маска пустая

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Добавляем паддинг, но не выходим за границы
    if image_shape:
        height, width = image_shape[:2]
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(height, y_max + padding)
        x_max = min(width, x_max + padding)

    return x_min, y_min, x_max, y_max

def crop_and_resize(image, box, target_size=(256, 256)):
    """Обрезает изображение по bounding box и изменяет его размер."""
    x_min, y_min, x_max, y_max = box
    cropped = image[y_min:y_max, x_min:x_max]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

def visualize_prediction(
        model,
        dataset,
        index,
        device,
        threshold=0.5,
        zoom=True,
        padding=40,
        target_size=(512, 512),
        name=None
):
    # Загружаем изображение и маску
    image, mask = dataset[index]

    # Получаем предсказания модели
    logits = model(image.unsqueeze(0).to(device)).squeeze(0)
    probas = torch.sigmoid(logits).detach().cpu().numpy()

    # Преобразуем изображение в numpy
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Истинные маски
    head_mask = mask[1].cpu().numpy()
    pit_mask = mask[2].cpu().numpy()

    # Предсказанные маски
    head_mask_logits = probas[1]
    pit_mask_logits = probas[2]

    # Объединённая маска для определения границ
    full_mask = (head_mask > 0) | (pit_mask > 0) | (head_mask_logits > threshold) | (pit_mask_logits > threshold)

    # Определяем bounding box + паддинг
    rect_coords = get_bounding_box(full_mask, padding, image.shape)

    if zoom and rect_coords:
        # Обрезаем по единому bounding box
        image_zoomed = crop_and_resize(image, rect_coords, target_size)
        combined_mask = np.zeros_like(image)
        combined_mask[head_mask > 0] = [255, 0, 0]  # Красный - головка
        combined_mask[pit_mask > 0] = [0, 0, 255]  # Синий - ямка
        combined_mask_zoomed = crop_and_resize(combined_mask, rect_coords, target_size)

        combined_mask_pred = np.zeros_like(image)
        combined_mask_pred[head_mask_logits > threshold] = [255, 0, 0]
        combined_mask_pred[pit_mask_logits > threshold] = [0, 0, 255]
        combined_mask_pred_zoomed = crop_and_resize(combined_mask_pred, rect_coords, target_size)

        # Добавляем прямоугольник на исходное изображение
        x_min, y_min, x_max, y_max = rect_coords
        rect_padding = padding // 2
        x_min += rect_padding
        y_min += rect_padding
        x_max -= rect_padding
        y_max -= rect_padding
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Зеленый прямоугольник
    else:
        image_zoomed = image
        combined_mask_zoomed = np.zeros_like(image)
        combined_mask_pred_zoomed = np.zeros_like(image)

    # Создаем фигуру для визуализации
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")

    axes[1].imshow(combined_mask_zoomed)
    axes[1].set_title("Ground Truth Mask (Zoomed)" if zoom else "Ground Truth Mask")

    axes[2].imshow(combined_mask_pred_zoomed)
    axes[2].set_title("Prediction (Zoomed)" if zoom else "Prediction")

    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    else:
        plt.show()

def visualize_with_axes(scaled_mask, x_top, y_extreme, height=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(scaled_mask, cmap='binary')

    # Рисуем оси
    plt.axvline(x=x_top, color='white', linestyle='--', linewidth=1)
    plt.axhline(y=y_extreme, color='white', linestyle='--', linewidth=1)
    plt.scatter(x_top, y_extreme, color='white', s=25)

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
            x_top + 70,
            y_extreme - height / 2,
            f"{(height / 20):.1f} mm",
            color='black',
            fontsize=12,
            verticalalignment='center',
            weight='regular'
        )

    plt.axis("off")
    plt.show()


def visualize_with_angle(
        mask,
        upper_point,
        intersection_point,
        lower_point,
        angle,
        is_back_angle: bool = False,
        top_axis_point=None
):
    plt.figure(figsize=(7, 7))
    plt.imshow(mask, cmap='binary')

    plt.axvline(x=upper_point[0] if top_axis_point is None else top_axis_point[0], color='white', linestyle='--', linewidth=1)
    plt.axhline(y=intersection_point[1], color='white', linestyle='--', linewidth=1)
    plt.scatter(upper_point[0] if top_axis_point is None else top_axis_point[0], intersection_point[1], color='white', s=25)

    # Рисуем линии
    plt.plot(
        [upper_point[0], intersection_point[0]], [upper_point[1], intersection_point[1]],
        'b-',
        linewidth=2
    )
    plt.plot(
        [intersection_point[0], lower_point[0]], [intersection_point[1], lower_point[1]],
        'b-',
        linewidth=2
    )

    # Отмечаем точки
    plt.scatter(upper_point[0], upper_point[1], color='blue', s=50)
    plt.scatter(intersection_point[0], intersection_point[1], color='blue', s=50)
    plt.scatter(lower_point[0], lower_point[1], color='blue', s=50)

    text_pos = [
        intersection_point[0] + 15,
        intersection_point[1] - 10
    ] if is_back_angle else [
        intersection_point[0] - 75,
        intersection_point[1] - 10
    ]
    plt.text(
        text_pos[0],
        text_pos[1],
        f"{angle:.1f}°",
        color='black',
        fontsize=12,
        weight='regular'
    )

    plt.axis("off")
    plt.show()


def visualize_mask(mask_tensor, title="Маска"):
    plt.figure(figsize=(5, 5))
    plt.imshow(mask_tensor.numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()


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