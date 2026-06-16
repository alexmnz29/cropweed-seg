"""Loss functions for segmentation experiments.

The baseline uses class-weighted cross-entropy. Focal and Dice are alternative
ways to handle the 84/12/3.7 imbalance, compared against that baseline. Focal
and Dice handle imbalance through their own mechanism (focal down-weights easy
pixels via gamma; dice optimizes overlap directly), so they do not take the
per-class weights. The experiment compares imbalance-handling strategies, not
just a drop-in loss swap.

All losses take logits (N, C, H, W) and integer targets (N, H, W), matching the
model output and dataset masks. Losses come from segmentation-models-pytorch
rather than hand-rolled, same reasoning as the model choice (ADR 0002).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_NAMES = ("weighted_ce", "focal", "dice", "focal_dice")


class FocalLoss(nn.Module):
    """Multiclass focal loss, computed via cross_entropy to stay MPS-compatible.

    SMP's focal loss casts targets with tensor.type(str), which is invalid on
    MPS. This implementation uses F.cross_entropy (which works on MPS) and
    applies the focal modulation (1 - p_t)^gamma on top. It also accepts
    per-class weights, so gamma and class weights can combine if needed.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ComboLoss(nn.Module):
    """Sum of focal and Dice loss, equally weighted.

    Focal works at the pixel level (penalizes hard pixels); Dice works at the
    region level (optimizes mask overlap). Summing gives both signals at once.
    Lambda is fixed at 1 (simple sum); tuning the focal/dice ratio is out of
    scope and would not justify a hyperparameter search framework on its own.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, weight=None)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, target) + self.dice(logits, target)


def build_criterion(
    name: str,
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Build a loss criterion by name.

    weighted_ce uses the per-class weights; focal and dice ignore them and
    handle imbalance through their own mechanism.
    """
    if name == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif name == "focal":
        criterion = FocalLoss(gamma=2.0, weight=None)
    elif name == "dice":
        criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    elif name == "focal_dice":
        criterion = ComboLoss(gamma=2.0)
    else:
        raise ValueError(f"unknown loss: {name!r}. Options: {LOSS_NAMES}")

    if device is not None:
        criterion = criterion.to(device)
    return criterion
