import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        y_pred = torch.sigmoid(y_pred)

        # Рассчитываем binary cross-entropy
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        # Рассчитываем фокусировку на сложных примерах
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()
