import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


class ClassificationLossFactory:
    @staticmethod
    def create(loss_name, weight=None, focal_gamma=2.0):
        normalized_name = str(loss_name).strip().lower()

        if normalized_name == "ce":
            return nn.CrossEntropyLoss()
        if normalized_name == "weighted_ce":
            return nn.CrossEntropyLoss(weight=weight)
        if normalized_name == "focal":
            return FocalLoss(gamma=focal_gamma)
        if normalized_name == "weighted_focal":
            return FocalLoss(gamma=focal_gamma, weight=weight)

        raise ValueError(
            f"Unsupported loss type: {loss_name}. "
            f"Use one of: ce, weighted_ce, focal, weighted_focal."
        )
