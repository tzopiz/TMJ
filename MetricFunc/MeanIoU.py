import torch
from torch import Tensor, nn

EPSILON = 1e-6


class MeanIoU(nn.Module):
    def __init__(
            self,
            threshold: float = 0.5,
            epsilon: float = EPSILON,
            weights: Tensor | None = None
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        classes_num = inputs.shape[1]  # Количество классов

        # Проверяем, что размеры совпадают
        assert inputs.size() == targets.size(), "Размеры inputs и targets должны совпадать!"

        preds = binarize_probs(torch.softmax(inputs, dim=1))

        # Вычисляем и возвращаем среднее значение IoU по всем каналам
        return torch.mean(
            compute_miou_per_channel(
                probs=preds,
                targets=targets,
                epsilon=self.epsilon,
                weights=self.weights
            )
        )

def compute_miou_per_channel(
        probs: Tensor,
        targets: Tensor,
        epsilon: float = EPSILON,
        weights: Tensor | None = None
) -> Tensor:
    assert probs.size() == targets.size(), "Размеры inputs и targets должны совпадать!"

    # Транспонируем и выравниваем тензоры
    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    # Вычисляем пересечение (Intersection)
    intersection = (probs * targets).sum(-1)

    # Вычисляем объединение (Union)
    union = (probs + targets - (probs * targets)).sum(-1)

    # Применяем веса, если они есть
    if weights is not None:
        intersection = weights * intersection

    # Вычисляем IoU
    iou = intersection / union.clamp(min=epsilon)

    # Усредняем по каналам
    return torch.mean(iou, dim=1)

def binarize_probs(inputs: Tensor) -> Tensor:
    one_hot = torch.zeros(inputs.shape, dtype=torch.float, device=inputs.device)
    return one_hot.scatter_(1, torch.argmax(inputs, dim=1, keepdim=True), 1)
