import torch
from torch import nn

class ComboLoss(nn.Module):
    """
    Helper class that allows for weighted summation of different losses.
    """
    def __init__(self, losses, coeffs):
        super().__init__()
        self.losses = losses
        self.coeffs = coeffs
    
    def forward(self, inputs, targets):
        loss = 0
        for i, f in enumerate(self.losses):
            if isinstance(f, nn.HuberLoss):
                x = torch.argmax(inputs, dim=1).float() # Fix for Huber Loss
            else:
                x = inputs
            loss += self.coeffs[i] * f(x, targets)
        return loss.mean()