import numpy as np
import torch
import cv2

from scipy.spatial.distance import directed_hausdorff


def compute_curvature(contour):
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
    return curvature

def find_key_points(mask: np.ndarray, use_avarage: bool = False):
    mask = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]

    min_y = np.min(contour[:, 1])
    top_points = contour[contour[:, 1] == min_y]

    top_point = tuple(top_points[0])
    if len(top_points) > 1 and use_avarage:
        top_point = tuple(top_points[0] + (top_points[-1] - top_points[0]) / 2)

    # Найдем точку с максимальным X и минимальным Y среди точек с максимальным X
    max_x = np.max(contour[:, 0])
    right_points = contour[contour[:, 0] == max_x]
    right_point = tuple(min(right_points, key=lambda p: p[1]))
    if len(right_points) > 1 and use_avarage:
        right_point = tuple(right_points[0] + (right_points[-1] - right_points[0]) / 2)

    x_center = top_point[0]

    left_half = contour[contour[:, 0] < x_center]
    max_x_points = left_half

    filtered_points = []
    for point in max_x_points:
        x, y = point
        left_side = left_half[left_half[:, 1] < y]
        right_side = left_half[left_half[:, 1] > y]
        if (left_side[:, 0] < x).any() and (right_side[:, 0] < x).any():
            filtered_points.append(point)

    # Вычисляем кривизну
    curvature = compute_curvature(contour)

    min_curvature_idx = np.argmax(curvature)
    concave_point = tuple(contour[min_curvature_idx])

    return top_point, right_point, concave_point


def calculate_front_axes(mask):
    mask = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]

    curvature = compute_curvature(contour)
    min_curvature_idx = np.argmin(curvature)
    concave_point = tuple(contour[min_curvature_idx])

    return concave_point


def find_intersection(mask, x_coord):
    height = mask.shape[0]
    intersection = None

    # Округляем до ближайшего целого
    x_coord = int(round(x_coord))

    # Идем снизу вверх
    for y in range(height - 1, -1, -1):
        if mask[y, x_coord] > 0:
            max_x = -1
            for x in range(mask.shape[1] - 1, -1, -1):
                if mask[y, x] > 0:
                    max_x = x
                    break

            if max_x != -1:
                intersection = (max_x, y)
                break

    return intersection


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

    # Округляем x до ближайшего индекса
    x_intersect = int(round(x_intersect))

    # Берем все точки маски, которые лежат на этой вертикальной оси
    y_indices = np.where(mask[:, x_intersect] > 0)[0]

    if len(y_indices) == 0:
        raise ValueError("Нет точек маски на оси X = x_intersect")

    # Самая верхняя точка среди них
    y_top = y_indices.min()

    # Высота — разница по оси Y
    height = abs(y_intersect - y_top)

    return height


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


def compute_metrics(pred, target, eps=1e-6):
    # Преобразуем numpy массивы в тензоры PyTorch
    pred = torch.tensor(pred).bool()
    target = torch.tensor(target).bool()

    TP = (pred & target).sum().float()
    FP = (pred & ~target).sum().float()
    FN = (~pred & target).sum().float()
    TN = (~pred & ~target).sum().float()

    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    specificity = (TN + eps) / (TN + FP + eps)
    f1_score = (2 * precision * recall) / (precision + recall + eps)

    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Accuracy": accuracy.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "Specificity": specificity.item(),
        "F1 Score": f1_score.item()
    }


def hausdorff_distance(pred, target):
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.inf

    return max(
        directed_hausdorff(pred_points, target_points)[0],
        directed_hausdorff(target_points, pred_points)[0]
    )


def evaluate_model_on_test_set(model, test_dataloader, device):
    model.eval()

    # Для накопления метрик
    metrics_sum = {
        "Dice": [0, 0],
        "IoU": [0, 0],
        "Accuracy": [0, 0],
        "Precision": [0, 0],
        "Recall": [0, 0],
        "Specificity": [0, 0],
        "F1 Score": [0, 0],
        "Hausdorff": [0, 0]
    }
    count = 0

    # Оценка на всем тестовом датасете
    with torch.no_grad():
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.cpu().numpy()  # Истинные маски (batch, 2, 512, 512)

            logits = model(images)  # (batch, 2, 512, 512)
            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas > 0.5)  # Бинаризация

            for i in range(images.shape[0]):  # Перебираем батч
                for cls in range(2):  # Два класса: головка (0) и впадина (1)
                    met = compute_metrics(preds[i, cls], masks[i, cls])
                    for key in met:
                        metrics_sum[key][cls] += met[key]

                    # Hausdorff отдельно
                    hd = hausdorff_distance(preds[i, cls], masks[i, cls])
                    metrics_sum["Hausdorff"][cls] += hd

                count += 1

    # Усреднение
    for key in metrics_sum:
        metrics_sum[key] = [x / count for x in metrics_sum[key]]

    # Возвращаем результаты
    return metrics_sum


