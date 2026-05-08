import torch
import torch.nn as nn

# Liebel L, Körner M. Auxiliary tasks in multi-task learning[J]. arXiv preprint arXiv:1805.06334, 2018.
# https://github.com/Mikoto10032/AutomaticWeightedLoss

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            
            # Clamp sigma^2 from below to prevent numerical explosion
            # when a loss component approaches zero (e.g. focal loss on majority class).
            # Without this, 0.5 / sigma^2 -> inf, causing NaN gradients.
            sigma_sq = torch.clamp(self.params[i] ** 2, min=1e-2)
            loss_sum += 0.5 / sigma_sq * loss + torch.log(1 + sigma_sq)
        return loss_sum