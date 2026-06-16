"""Model construction for crop-weed segmentation.

Baseline: U-Net with a ResNet34 encoder pretrained on ImageNet.
DeepLabV3+ is available as an alternative for comparison; its decoder and atrous
convolutions are meant to preserve fine detail, motivated by the error analysis
in notebook 05 (U-Net misses thin filamentous weed). Architecture and the use of
segmentation-models-pytorch are recorded in docs/decisions/. Output is raw logits
with one channel per class; softmax/argmax happens at loss and inference time.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

from cropweed_seg.masks import N_CLASSES

DEFAULT_ARCH = "unet"
DEFAULT_ENCODER = "resnet34"

ARCHITECTURES = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(
    architecture: str = DEFAULT_ARCH,
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a segmentation model by architecture name.

    architecture: one of ARCHITECTURES. encoder_weights="imagenet" for the
    pretrained baseline; None initializes the encoder from scratch (see ADR 0001).
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture: {architecture!r}. Options: {tuple(ARCHITECTURES)}"
        )
    model_cls = ARCHITECTURES[architecture]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=N_CLASSES,
    )
