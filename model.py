"""
model.py  —  MobileNet-v2 for CIFAR-10 (Q1b).

Configuration choices:
  width_mult = 1.0       — matches the pretrained ImageNet checkpoint exactly;
                           changing it would invalidate all pretrained layer shapes.
  dropout    = 0.2       — torchvision default; light regularisation in the head.
  BatchNorm  — default PyTorch settings (eps=1e-5, momentum=0.1); pretrained
               running statistics were computed at these values.

Stride surgery (_adapt_strides_for_small_images):
  The original MobileNet-v2 has a total spatial downsampling factor of 32×
  (stride-2 stem + four stride-2 InvertedResidual blocks). Applying this to
  a 32×32 image collapses the feature map to 1×1, which kills spatial
  information entirely.

  We flip the stem (features[0][0]) and the first stride-2 bottleneck
  (features[2].conv[1][0]) from stride 2 to stride 1, reducing the total
  downsampling to 8×. The final feature map is then 4×4, which is a
  reasonable spatial resolution for a 32×32 input.

  Alternative: upscale CIFAR images to 224×224 in the transforms — works
  but is ~7× slower per epoch and gains nothing architecturally.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def _adapt_strides_for_small_images(model: nn.Module) -> nn.Module:
    # Stem: features[0] is Conv2dNormActivation, features[0][0] is its Conv2d.
    model.features[0][0].stride = (1, 1)
    # First stride-2 InvertedResidual block is features[2].
    # Its depthwise conv sits at .conv[1][0].
    model.features[2].conv[1][0].stride = (1, 1)
    return model


def build_mobilenetv2(num_classes: int = 10,
                      pretrained: bool = True,
                      adapt_for_small_images: bool = True) -> nn.Module:
    """
    Builds MobileNet-v2 configured for CIFAR-10.

    Args:
        num_classes:            10 for CIFAR-10.
        pretrained:             Load ImageNet1k weights before adapting.
        adapt_for_small_images: Apply stride surgery for 32×32 inputs.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.mobilenet_v2(weights=weights, width_mult=1.0)

    if adapt_for_small_images:
        model = _adapt_strides_for_small_images(model)

    # Replace classifier head: ImageNet has 1000 classes, CIFAR-10 has 10.
    # Keep Dropout(0.2); only swap the Linear output size.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_mobilenetv2(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Output shape : {y.shape}")          # [2, 10]
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {total:,}")          # ~3.4 M
