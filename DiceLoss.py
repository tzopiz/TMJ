import torch
from torch import Tensor, nn


EPSILON = 1e-6


class DiceLoss(nn.Module):
    def __init__(self, weights: Tensor | None = None, epsilon: float = EPSILON) -> None:
        super().__init__()
        self.register_buffer("weights", weights)
        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Применяем сигмоиду к входным данным, чтобы получить вероятности
        probs = nn.Sigmoid()(inputs)

        # Вычисляем Dice коэффициент для каждого канала
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )

        # Возвращаем среднее значение Dice Loss по всем каналам
        return 1.0 - torch.mean(per_channel_dice)


def compute_dice_per_channel(
        probs: Tensor,
        targets: Tensor,
        epsilon: float = EPSILON,
        weights: Tensor | None = None
) -> Tensor:
    assert probs.size() == targets.size()

    # Транспонируем и выравниваем тензоры
    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    # Вычисляем числитель формулы Dice
    numerator = (probs * targets).sum(-1)
    if weights is not None:
        numerator = weights * numerator

    # Вычисляем знаменатель формулы Dice
    denominator = (probs + targets).sum(-1)

    # Вычисляем и возвращаем среднее значение Dice по всем каналам
    return torch.mean(2 * (numerator / denominator.clamp(min=epsilon)), dim=1)
