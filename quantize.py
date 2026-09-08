"""
quantize.py  —  Custom weight quantization (no torch.quantization used).

Two core functions:
  compute_qparams  — derives (scale, zero_point) from a tensor's range.
  fake_quantize    — rounds to the nearest N-bit representable value,
                     then converts back to float32 so normal PyTorch ops
                     can continue. The compression ratio comes from the
                     reduced *bit-width*, not from actually changing dtypes.

Design choices (Q2a):
  Symmetric quantization for weights (zero_point = 0):
    Weights are roughly zero-centred, so fixing zero_point = 0 wastes
    ≤1 representable level while halving the metadata overhead (no need
    to store zero_point at all, though we count it conservatively in
    utils.py for a fair comparison).

  Per-channel quantization (one scale per output channel):
    Weights in different output channels of the same conv can have very
    different magnitude ranges. Assigning each channel its own scale
    reduces quantisation error substantially at low bit-widths with
    only a small metadata overhead (n_channels × 4 bytes of extra storage).

  BatchNorm parameters are NOT quantised:
    BN parameters (weight, bias, running_mean, running_var) directly
    control activation scale. Their combined storage is < 1 % of model
    size, so quantising them delivers negligible savings while causing
    disproportionate accuracy damage.

  Stem + classifier are optionally skipped (see SKIP_WEIGHT_LAYERS in test.py):
    The stem has only 3 input channels — its parameter count is tiny and
    it sets the tone for all subsequent features. The classifier is the
    last layer before the loss and is most sensitive to numerical error.
    Keeping both at fp32 costs almost nothing while providing an accuracy
    safety margin.
"""

import torch
import torch.nn as nn


def compute_qparams(
    tensor: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0,
):
    """
    Compute (scale, zero_point) for quantising `tensor` to `bits` bits.

    symmetric=True:   zero_point = 0; scale = max(|x|) / qmax.
                       Use for weights (zero-centred distributions).
    symmetric=False:  scale and zero_point from actual [min, max].
                       Use for activations (e.g. post-ReLU, always ≥ 0).
    per_channel=True: one (scale, zero_point) pair per output channel.
    """
    qmin = -(2 ** (bits - 1))       # e.g. bits=8 → -128
    qmax =   2 ** (bits - 1) - 1    # e.g. bits=8 →  127

    if per_channel:
        # Reduce over every dimension except channel_dim.
        reduce_dims = [d for d in range(tensor.dim()) if d != channel_dim]
        if symmetric:
            max_abs   = tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
            scale     = max_abs / qmax
            zero_point = torch.zeros_like(scale)
        else:
            t_min      = tensor.amin(dim=reduce_dims, keepdim=True)
            t_max      = tensor.amax(dim=reduce_dims, keepdim=True)
            scale      = ((t_max - t_min) / (qmax - qmin)).clamp(min=1e-8)
            zero_point = torch.round(qmin - t_min / scale)
    else:
        if symmetric:
            max_abs    = tensor.abs().max().clamp(min=1e-8)
            scale      = max_abs / qmax
            zero_point = torch.zeros_like(scale)
        else:
            t_min      = tensor.min()
            t_max      = tensor.max()
            scale      = ((t_max - t_min) / (qmax - qmin)).clamp(min=1e-8)
            zero_point = torch.round(qmin - t_min / scale)

    return scale, zero_point


def fake_quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Simulate N-bit quantisation in float32:
        q     = clamp(round(x / scale) + zero_point, qmin, qmax)
        x_hat = scale * (q − zero_point)
    """
    qmin = -(2 ** (bits - 1))
    qmax =   2 ** (bits - 1) - 1
    q     = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    x_hat = scale * (q - zero_point)
    return x_hat


def apply_weight_quantization(
    model: nn.Module,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = True,
    skip_layers: list = None,
) -> list:
    """
    Walks all Conv2d and Linear layers in `model` and replaces their
    weight tensors with fake-quantised versions (in-place).

    Returns the list of layer names that were quantised.
    """
    if skip_layers is None:
        skip_layers = []

    quantized_layer_names = []
    for name, module in model.named_modules():
        if name in skip_layers:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                scale, zp = compute_qparams(
                    module.weight.data, bits=bits,
                    symmetric=symmetric, per_channel=per_channel, channel_dim=0,
                )
                module.weight.data = fake_quantize(module.weight.data, scale, zp, bits)
            quantized_layer_names.append(name)

    return quantized_layer_names


if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(64, 32, 3, 3) * 0.15   # typical conv-weight magnitude

    print("Per-tensor symmetric quantisation:")
    for bits in [8, 4, 2]:
        scale, zp = compute_qparams(w, bits=bits, symmetric=True, per_channel=False)
        w_hat = fake_quantize(w, scale, zp, bits)
        mse   = (w - w_hat).pow(2).mean().item()
        print(f"  bits={bits:2d}  scale={scale.item():.5f}  MSE={mse:.2e}")

    print("\nPer-channel symmetric quantisation:")
    for bits in [8, 4, 2]:
        scale, zp = compute_qparams(w, bits=bits, symmetric=True, per_channel=True)
        w_hat = fake_quantize(w, scale, zp, bits)
        mse   = (w - w_hat).pow(2).mean().item()
        print(f"  bits={bits:2d}  MSE={mse:.2e}   (per-channel is always ≤ per-tensor MSE)")
