"""
utils.py  —  Storage and compression-ratio accounting (Q2c, Q4a, Q4b, Q4d).

Two separate accounting scopes:

  1. compute_weight_storage()
       What gets written to disk in a compressed checkpoint.
       Covers: quantised weight payloads + biases + BN parameters +
               per-channel scale/zero_point metadata.
       Answers: Q4a (weight compression ratio), Q4d (final model size in MB).

  2. compute_activation_storage()
       Per-image runtime activation memory for the tensors we quantise
       (ReLU6 outputs), measured as the SUM across all quantised layers.
       This is an upper-bound estimate — in practice not all intermediate
       tensors are live simultaneously, but it gives a reproducible,
       comparable number.
       Answers: Q4b (activation compression ratio).

Storage overhead (Q2c):
  Symmetric weight quantisation:   1 float32 scale  per output channel
                                   1 float32 zero_point per channel (= 0,
                                   but stored for format compatibility)
  Asymmetric activation quant:     1 float32 scale + 1 float32 zero_point
                                   per quantised layer (per-tensor scheme)
  Biases:                          always kept at fp32 (negligible size,
                                   high sensitivity)
  BatchNorm params + running stats: always kept at fp32
"""

import torch.nn as nn

FP32_BITS = 32
INT64_BITS = 64


def compute_weight_storage(
    model: nn.Module,
    quantized_layers: dict,    # {layer_name: {"bits": int, "per_channel": bool}}
    skip_layers: list = None,
) -> dict:
    """
    Computes compressed vs original model storage in bits/MB.

    Parameters
    ----------
    model            : the MobileNet-v2 model (post fake-quantisation).
    quantized_layers : maps layer name → quantisation config, as returned
                       by apply_weight_quantization().
    skip_layers      : names left at fp32 (not in quantized_layers anyway,
                       but kept as an explicit guard).

    Returns
    -------
    dict with keys: original_MB, compressed_MB, compression_ratio,
                    metadata_MB (overhead due to scale/zero_point tensors).
    """
    if skip_layers is None:
        skip_layers = []

    original_bits  = 0
    compressed_bits = 0
    metadata_bits  = 0

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            n_w    = module.weight.numel()
            n_bias = module.bias.numel() if module.bias is not None else 0

            original_bits += (n_w + n_bias) * FP32_BITS

            if name in quantized_layers and name not in skip_layers:
                cfg  = quantized_layers[name]
                bits = cfg["bits"]

                # Quantised weight payload
                compressed_bits += n_w * bits
                # Bias stays at fp32
                compressed_bits += n_bias * FP32_BITS

                # Metadata: one (scale, zero_point) pair per quantisation group.
                # Per-channel → one group per output channel.
                n_groups        = module.weight.shape[0] if cfg.get("per_channel") else 1
                meta            = n_groups * 2 * FP32_BITS
                compressed_bits += meta
                metadata_bits   += meta

            else:
                # Not quantised — full fp32
                compressed_bits += (n_w + n_bias) * FP32_BITS

        elif isinstance(module, nn.BatchNorm2d):
            # BN: weight, bias (via .parameters()), running_mean, running_var,
            # and num_batches_tracked (int64 scalar).
            n_params  = sum(p.numel() for p in module.parameters())
            n_rm      = module.running_mean.numel()
            n_rv      = module.running_var.numel()
            bits_here = (n_params + n_rm + n_rv) * FP32_BITS + INT64_BITS

            original_bits   += bits_here
            compressed_bits += bits_here   # BN never quantised

    return {
        "original_MB":       original_bits   / 8 / 1_000_000,
        "compressed_MB":     compressed_bits / 8 / 1_000_000,
        "metadata_MB":       metadata_bits   / 8 / 1_000_000,
        "compression_ratio": original_bits   / compressed_bits,
    }


def compute_activation_storage(activation_stats: dict, bits: int) -> dict:
    """
    Estimates per-image activation memory for all quantised ReLU6 outputs.

    How activations are measured (Q4b):
      We record the element count (C×H×W) at each ReLU6 output during
      calibration. The "original" cost assumes every activation is stored
      as float32; the "compressed" cost assumes N-bit integer storage plus
      one (scale, zero_point) pair per layer (per-tensor scheme).

    Parameters
    ----------
    activation_stats : output of ActivationCalibrator.stats
                       {layer_name: {"min", "max", "numel"}}
    bits             : activation bit-width.

    Returns
    -------
    dict with keys: original_KB, compressed_KB, metadata_KB, compression_ratio.
    """
    original_bits   = 0
    compressed_bits = 0
    metadata_bits   = 0

    for s in activation_stats.values():
        n = s["numel"]
        original_bits   += n * FP32_BITS
        compressed_bits += n * bits
        meta             = 2 * FP32_BITS   # scale + zero_point, per layer
        compressed_bits += meta
        metadata_bits   += meta

    return {
        "original_KB":       original_bits   / 8 / 1_000,
        "compressed_KB":     compressed_bits / 8 / 1_000,
        "metadata_KB":       metadata_bits   / 8 / 1_000,
        "compression_ratio": original_bits   / compressed_bits,
    }


def print_compression_summary(weight_result: dict, activation_result: dict,
                               weight_bits: int, act_bits: int):
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  Compression summary  (weights={weight_bits}-bit, acts={act_bits}-bit)")
    print(sep)
    print(f"\n  WEIGHTS")
    print(f"    Original size      : {weight_result['original_MB']:.3f} MB")
    print(f"    Compressed size    : {weight_result['compressed_MB']:.3f} MB")
    print(f"    Metadata overhead  : {weight_result['metadata_MB']:.4f} MB  (scale/zero_pt)")
    print(f"    Compression ratio  : {weight_result['compression_ratio']:.3f}×")

    print(f"\n  ACTIVATIONS  (per image, sum over all ReLU6 outputs)")
    print(f"    Original           : {activation_result['original_KB']:.2f} KB")
    print(f"    Compressed         : {activation_result['compressed_KB']:.2f} KB")
    print(f"    Metadata overhead  : {activation_result['metadata_KB']:.2f} KB")
    print(f"    Compression ratio  : {activation_result['compression_ratio']:.3f}×")
    print(f"\n  Final approx model size : {weight_result['compressed_MB']:.3f} MB")
    print(sep)
