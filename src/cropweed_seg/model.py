"""Model construction for crop-weed segmentation.

Baseline: U-Net with a ResNet34 encoder pretrained on ImageNet.
Architecture choice and the use of segmentation-models-pytorch are recorded
in docs/decisions/ (see the baseline ADR). Output is raw logits with one
channel per class; softmax/argmax happens at loss and inference time.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

from cropweed_seg.masks import N_CLASSES

DEFAULT_ENCODER = "resnet34"


def build_model(
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build the segmentation model.

    encoder_weights="imagenet" for the pretrained baseline; pass None to
    initialize the encoder from scratch (used only in from-scratch experiments,
    where dataset-specific normalization would also apply; see ADR 0001).
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=N_CLASSES,
    )
