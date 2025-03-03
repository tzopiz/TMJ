import torch
from torch import Tensor, nn

EPSILON = 1e-6

class DiceLoss(nn.Module):
    def __init__(self, weights: Tensor | None = None, epsilon: float = EPSILON) -> None:
        super().__init__()
        self.register_buffer("weights", weights if weights is not None else torch.ones(1))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(inputs)  # Сигмоид на входе
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )
        return 1.0 - per_channel_dice.mean()

def compute_dice_per_channel(
        probs: Tensor,
        targets: Tensor,
        epsilon: float = EPSILON,
        weights: Tensor | None = None
) -> Tensor:
    assert probs.size() == targets.size(), "Размеры inputs и targets должны совпадать!"

    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    numerator = (probs * targets).sum(dim=-1)
    denominator = (probs + targets).sum(dim=-1)

    if weights is not None and weights.shape[0] == numerator.shape[0]:  # Проверка размеров
        numerator = numerator * weights.unsqueeze(1)  # Расширяем размерность весов

    dice_score = (2 * numerator) / denominator.clamp(min=epsilon)
    return dice_score.mean(dim=1)  # Среднее по batch
