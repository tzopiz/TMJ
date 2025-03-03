from MetricFunc import MeanIoU
from torch import Tensor

class WeightedMeanIoU(MeanIoU):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        miou = super().forward(inputs, targets)

        if self.weights is not None:
            weighted_miou = miou * self.weights
            return weighted_miou

        return miou
