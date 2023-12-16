import torch
from torch.nn import CrossEntropyLoss


class CrossEntropyWrapper(CrossEntropyLoss):
    def __init__(self, weight, device):
        super().__init__(weight=torch.tensor(weight).to(device))

    def forward(self, logits, label, **kwargs):
        return super().forward(
            logits, label
        )

