import torch
from torch import Tensor, nn

EPSILON = 1e-6


class DiceLoss(nn.Module):
    def __init__(self, weights: Tensor | None = None, epsilon: float = EPSILON) -> None:
        super().__init__()
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None
        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Применяем сигмоиду к входным данным
        probs = torch.sigmoid(inputs)

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
    assert probs.size() == targets.size(), "Размеры inputs и targets должны совпадать!"

    # Транспонируем и выравниваем тензоры (C, N, H*W)
    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    # Вычисляем числитель формулы Dice
    numerator = (probs * targets).sum(-1)

    # Если есть веса, применяем их к числителю
    if weights is not None:
        numerator = weights * numerator

    # Вычисляем знаменатель формулы Dice
    denominator = (probs + targets).sum(-1)

    # Предотвращаем деление на ноль (можно сделать через torch.where, но clamp проще)
    dice_score = 2 * (numerator / denominator.clamp(min=epsilon))

    # Вычисляем среднее значение Dice по всем каналам
    return torch.mean(dice_score, dim=1)
