import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_with_axes(scaled_mask, x_top, y_extreme, height=None):
    plt.figure(figsize=(7, 7))
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
            color='green')
        )
        plt.gca().add_patch(patches.FancyArrow(
            x_top + 10,
            y_extreme - 5,
            0,
            -height + 20,
            width=1,
            head_width=5,
            head_length=5,
            color='green')
        )
        # Подпись высоты
        plt.text(
            x_top + 15,
            y_extreme - height / 2,
            f"Высота: {height}",
            color='green',
            fontsize=12,
            verticalalignment='center',
            weight='bold'
        )

    plt.axis("off")
    plt.show()


def visualize_with_back_angle(mask, upper_point, intersection_point, lower_point, angle):
    plt.figure(figsize=(7, 7))
    plt.imshow(mask, cmap='binary')

    plt.axvline(x=upper_point[0], color='white', linestyle='--', linewidth=1)
    plt.axhline(y=intersection_point[1], color='white', linestyle='--', linewidth=1)
    plt.scatter(upper_point[0], intersection_point[1], color='white', s=25)

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

    is_right = upper_point[0] < mask.shape[1] // 2

    # Отмечаем точки
    plt.scatter(upper_point[0], upper_point[1], color='blue', s=50)
    plt.scatter(intersection_point[0], intersection_point[1], color='blue', s=50)
    plt.scatter(lower_point[0], lower_point[1], color='blue', s=50)

    # Подписываем угол
    plt.text(
        intersection_point[0] + 20,
        intersection_point[1] - 10,
        f"{angle:.2f}°",
        color='green',
        fontsize=12,
        weight='bold'
    )

    # Рисуем дугу угла
    arc = patches.Arc(
        intersection_point,
        width=15,
        height=15,
        angle=(-angle // 2 + 185) if is_right else -angle // 2,
        theta1=0,
        theta2=angle,
        color='green',
        linewidth=3
    )
    plt.gca().add_patch(arc)

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

