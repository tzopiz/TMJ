from LossFunc.DiceLoss import DiceLoss


class WeightedDiceLoss(DiceLoss):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        dice_loss = super().forward(y_pred, y_true)
        weighted_loss = self.weights * dice_loss
        return weighted_loss
