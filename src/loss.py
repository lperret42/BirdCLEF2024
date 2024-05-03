import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function for addressing class imbalance by reducing the loss
    contribution from easy examples and increasing the importance of correcting misclassified
    examples.

    Attributes:
        alpha (float): Balancing factor, default is 0.25.
        gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
        bce_with_logits (nn.Module): Binary Cross Entropy loss with logits.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean') -> None:
        """
        Initializes the Focal Loss function.

        Args:
            alpha (float): Weighting factor for the rate of positive samples.
            gamma (float): Modulating factor to adjust the focal loss.
            reduction (str): Method for reducing the batch loss to a single value; options are 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the focal loss between the inputs and targets.

        Args:
            inputs (torch.Tensor): Predicted probabilities.
            targets (torch.Tensor): Ground truth labels.
            weight (torch.Tensor, optional): Sample-wise weights. If provided, it must match the batch size.

        Returns:
            torch.Tensor: Computed focal loss. Depending on 'reduction', it may be scalar or tensor.
        """
        # Compute BCE loss
        bce_loss = self.bce_with_logits(inputs, targets)
        # Calculate the probability tensor
        pt = torch.exp(-bce_loss)
        # Calculate the final focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply weights if provided
        if weight is not None:
            weight = weight.unsqueeze(1).expand_as(focal_loss)
            focal_loss *= weight

        # Apply the reduction method
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
