import torch
import torch.nn as nn


class CustomMeanIoU(nn.Module):
    def __init__(self, num_classes: int, class_weights: torch.Tensor = None, eps: float = 1e-6):
        super(CustomMeanIoU, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            assert class_weights.shape == (num_classes,)
            self.class_weights = class_weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Вычисление Mean IoU.

        :param preds: Предсказания модели (логиты или вероятности) (N, C, H, W)
        :param targets: Истинные маски (one-hot encoded) (N, C, H, W)
        :return: Средний IoU с учетом весов классов
        """
        preds = torch.argmax(preds, dim=1)  # (N, H, W)
        targets = torch.argmax(targets, dim=1)  # (N, H, W)

        iou_per_class = []
        for c in range(self.num_classes):
            pred_c = (preds == c).float()
            target_c = (targets == c).float()

            intersection = (pred_c * target_c).sum(dim=(1, 2))  # (N,)
            union = (pred_c + target_c).clamp(0, 1).sum(dim=(1, 2))  # (N,)

            iou = (intersection + self.eps) / (union + self.eps)
            iou_per_class.append(iou)

        iou_per_class = torch.stack(iou_per_class, dim=1)  # (N, C)
        weighted_iou = iou_per_class * self.class_weights.to(iou_per_class.device)
        mean_iou = weighted_iou.mean(dim=1)  # усреднение по классам

        return mean_iou.mean()  # усреднение по батчу
