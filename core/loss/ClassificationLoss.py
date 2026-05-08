import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        # self.weight = weight
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        # focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = (torch.clamp(1-pt, min=1e-6)**self.gamma) * ce_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


def compute_class_balanced_weights(samples_per_class, beta=0.999):
    """Compute class-balanced weights using effective number of samples.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number
    of Samples", CVPR 2019.

    Args:
        samples_per_class: tensor of per-class sample counts.
        beta: hyperparameter in (0, 1). Higher beta = stronger re-weighting.

    Returns:
        Normalized weight tensor (sums to num_classes).
    """
    effective_num = 1.0 - torch.pow(beta, samples_per_class.float())
    weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
    weights = weights / weights.sum() * len(samples_per_class)
    return weights
    

class ClassificationLossFactory:
    @staticmethod
    def create(loss_name, weight=None, focal_gamma=2.0,
              samples_per_class=None, cb_beta=0.999):
        normalized_name = str(loss_name).strip().lower()

        if normalized_name == "ce":
            return nn.CrossEntropyLoss()
        if normalized_name == "weighted_ce":
            return nn.CrossEntropyLoss(weight=weight)
        if normalized_name == "focal":
            return FocalLoss(gamma=focal_gamma)
        if normalized_name == "weighted_focal":
            return FocalLoss(gamma=focal_gamma, weight=weight)
        if normalized_name == "cb_focal":
            if samples_per_class is None:
                raise ValueError("cb_focal requires samples_per_class")
            cb_weight = compute_class_balanced_weights(samples_per_class, cb_beta)
            return FocalLoss(gamma=focal_gamma, weight=cb_weight)
        if normalized_name == "cb_ce":
            if samples_per_class is None:
                raise ValueError("cb_ce requires samples_per_class")
            cb_weight = compute_class_balanced_weights(samples_per_class, cb_beta)
            return nn.CrossEntropyLoss(weight=cb_weight)

        raise ValueError(
            f"Unsupported loss type: {loss_name}. "
            f"Use one of: ce, weighted_ce, focal, weighted_focal, cb_focal, cb_ce."
        )
