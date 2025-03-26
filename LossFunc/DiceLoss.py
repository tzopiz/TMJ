import torch
from torch import Tensor, nn

EPSILON = 1e-6

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probas = torch.sigmoid(logits)  # Преобразуем логиты в вероятности

        intersection = (targets * probas).sum((0, 2, 3)).clamp_min(self.eps)
        cardinality = (targets + probas).sum((0, 2, 3)).clamp_min(self.eps)

        dice_coefficient = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_coefficient

        return dice_loss.mean()
