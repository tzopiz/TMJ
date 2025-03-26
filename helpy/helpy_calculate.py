import numpy as np
import torch


def calculate_back_axes(mask):
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
    while 0 < x_intersect < mask.shape[1] and mask[y_intersect, x_intersect] == 0:
        if is_right:
            x_intersect += 1  # Двигаем влево
        else:
            x_intersect -= 1  # Двигаем вправо

    return (x_top, y_top), (x_extreme, y_extreme), (x_intersect, y_intersect)


def find_intersection(mask, x_coord):
    height = mask.shape[0]
    for y in range(height - 1, -1, -1):  # Идем снизу вверх
        if mask[y, x_coord] > 0:
            return (x_coord, y)
    return None


def calculate_angle(p1, p2, p3):
    """Вычисляет угол между двумя линиями: p1->p2 и p2->p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return None  # Нельзя вычислить угол

    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

    return angle


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


def calculate_front_axes(mask):
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

    while 0 < x_intersect < mask.shape[1] and mask[y_intersect, x_intersect + 1] != 0:
        if is_right:
            x_intersect -= 1  # Двигаем влево
        else:
            x_intersect += 1  # Двигаем вправо

    for i in range(0, 500):
        if is_right:
            y_intersect += 1
        else:
            if y_intersect + 1 < mask.shape[1] and mask[x_intersect + 1, y_intersect + 1] == 0:
                y_intersect += 1
            else:
                break

            if xs[ys == y_intersect].max() > x_intersect and x_intersect != x_extreme:
                break
            else:
                x_intersect = xs[ys == y_intersect].max()

    return (x_top, y_top), (x_extreme, y_extreme), (x_intersect, y_intersect)


def calculate_areas(mask):
    background_area = torch.sum(mask[0]).item()  # Количество пикселей фона
    head_area = torch.sum(mask[1]).item()        # Количество пикселей головки
    pit_area = torch.sum(mask[2]).item()         # Количество пикселей ямки

    return {
        "background": background_area,
        "head": head_area,
        "pit": pit_area
    }


def calculate_class_weights(dataset, num_classes):
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
