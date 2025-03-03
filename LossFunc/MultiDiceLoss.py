import torch
import torch.nn.functional as F
from torch import nn

EPSILON = 1e-6


class MulticlassDiceLoss(nn.Module):
    def __init__(self, eps: float = EPSILON) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1)

        intersection = (targets * probas).sum((0, 2, 3)).clamp_min(self.eps)
        cardinality = (targets + probas).sum((0, 2, 3)).clamp_min(self.eps)

        dice_coefficient = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        dice_loss = 1.0 - dice_coefficient

        mask = targets.sum((0, 2, 3)) > 0
        dice_loss *= mask

        return dice_loss.mean()


class MulticlassCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=input, target=torch.argmax(target, dim=1))