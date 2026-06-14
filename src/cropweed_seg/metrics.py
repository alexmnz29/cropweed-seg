"""Segmentation metrics built on an accumulated confusion matrix.

Per-class IoU is computed from a confusion matrix accumulated over a whole
split, not by averaging per-image IoU. With 29.5% of images lacking crop,
per-image IoU would be undefined for absent classes; accumulating counts
sidesteps that. Per-class IoU is always reported separately, never as
aggregate mIoU alone (see notebook 01 for why the imbalance demands this).
"""

import torch

from cropweed_seg.masks import CLASS_NAMES, N_CLASSES


class ConfusionMatrixMetric:
    """Accumulate a confusion matrix across batches, then derive IoU.

    Rows are ground truth, columns are predictions. update() is called per
    batch; compute() returns per-class IoU and mean IoU at the end.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        self.n_classes = n_classes
        self.confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    def reset(self) -> None:
        self.confusion.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch. preds and targets are (N, H, W) integer class maps."""
        preds = preds.flatten().cpu()
        targets = targets.flatten().cpu()
        # bincount of (truth * n + pred) fills the matrix in one pass
        idx = targets * self.n_classes + preds
        counts = torch.bincount(idx, minlength=self.n_classes**2)
        self.confusion += counts.reshape(self.n_classes, self.n_classes)

    def compute(self) -> dict[str, float]:
        """Return per-class IoU and mean IoU from the accumulated matrix."""
        conf = self.confusion.float()
        tp = conf.diag()
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        denom = tp + fp + fn
        # IoU is NaN for a class entirely absent from both preds and targets
        iou = tp / denom

        result = {f"iou_{name}": iou[i].item() for i, name in enumerate(CLASS_NAMES)}
        # mean IoU ignores NaN classes (absent everywhere in this split)
        valid = ~torch.isnan(iou)
        result["miou"] = iou[valid].mean().item()
        return result
