import torch
from torch import Tensor, nn

EPSILON = 1e-6


class MeanIoU(nn.Module):
    def __init__(
            self,
            threshold: float = 0.5,
            epsilon: float = EPSILON,
            onehot_conversion: bool = False,
            binarize: bool = True
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon
        self.onehot_conversion = onehot_conversion
        self.binarize = binarize

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        classes_num = inputs.shape[1]  # Количество классов

        # Конвертируем целевые значения в one-hot, если указано
        if self.onehot_conversion:
            targets = convert_to_one_hot(targets, classes_num=classes_num)

        # Проверяем, что размеры совпадают
        assert inputs.size() == targets.size(), "Размеры inputs и targets должны совпадать!"

        # Бинаризуем предсказания, если это указано
        if self.binarize:
            preds = binarize_probs(
                inputs=torch.sigmoid(inputs),  # Применяем сигмоиду к входным данным
                classes_num=classes_num,
                threshold=self.threshold
            )
        else:
            preds = torch.sigmoid(inputs)

        # Вычисляем и возвращаем среднее значение IoU по всем каналам
        return torch.mean(
            compute_miou_per_channel(probs=preds, targets=targets, epsilon=self.epsilon)
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

    # Вычисляем числитель IoU
    numerator = (probs * targets).sum(-1)

    # Применяем веса, если они есть
    if weights is not None:
        numerator = weights * numerator

    # Вычисляем знаменатель IoU
    denominator = (probs + targets).clamp(max=1).sum(-1)

    # Вычисляем IoU и усредняем по каналам
    return torch.mean(numerator / denominator.clamp(min=epsilon), dim=1)


def convert_to_one_hot(targets: Tensor, classes_num: int) -> Tensor:
    assert targets.dim() == 4, "Ожидается 4D тензор (N, С, H, W)"

    # Добавляем канал классов и создаём one-hot encoding
    targets = targets.unsqueeze(1)
    shape = list(targets.shape)
    shape[1] = classes_num  # Устанавливаем количество классов
    return torch.zeros(shape, dtype=torch.float, device=targets.device).scatter_(1, targets.long(), 1)


def binarize_probs(inputs: Tensor, classes_num: int, threshold: float = 0.5) -> Tensor:
    if classes_num == 1:
        # Для бинарного случая применяем порог
        return (inputs > threshold).float()

    # Для многоклассовой классификации выбираем максимальные значения
    return torch.zeros_like(inputs, dtype=torch.float).scatter_(
        1, torch.argmax(inputs, dim=1, keepdim=True), 1
    )
