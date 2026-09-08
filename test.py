"""
test.py  —  Single quantisation experiment (Q3, Q4).

Matches the assignment's expected CLI:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8

run_experiment() is the authoritative implementation of one evaluation run;
sweep.py imports it directly to avoid duplicating this logic.

What one run does:
  1. Load the fine-tuned checkpoint (falls back to ImageNet pretrained if absent).
  2. Evaluate baseline (fp32) accuracy before any quantisation.
  3. Fake-quantise weights in-place.
  4. Recalibrate BN running stats to match quantised conv outputs.
  5. Calibrate activation ranges on training data (not test data).
  6. Attach activation quantisation hooks.
  7. Evaluate quantised accuracy on the test set.
  8. Compute and report weight/activation compression ratios and model size.

SKIP_WEIGHT_LAYERS:
  "features.0.0" — the stem Conv2d (3-channel input, tiny parameter count,
                   but first layer in the network sets the feature space for
                   all subsequent processing; high accuracy sensitivity).
  "classifier.1" — the final Linear layer, closest to the loss function and
                   most sensitive to numerical noise.
  Both are kept at fp32.  Their combined parameter share is < 0.5% of total.
"""

import argparse
import copy
import os
import torch

from data            import get_dataloaders
from model           import build_mobilenetv2
from quantize        import apply_weight_quantization
from activation_quant import find_relu6_layers, ActivationCalibrator, attach_activation_quant_hooks
from utils           import compute_weight_storage, compute_activation_storage, print_compression_summary

SKIP_WEIGHT_LAYERS = ["features.0.0", "classifier.1"]


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _recalibrate_bn(model: torch.nn.Module, loader, num_batches: int, device) -> None:
    """
    Update BatchNorm running statistics after weight fake-quantisation.

    Fake-quantising weights shifts the distribution of conv outputs, making
    the pre-trained BN running mean/var incorrect. Running a few training
    batches in model.train() mode forces BN to collect fresh running stats
    that match the quantised conv outputs. This can recover 5–15% accuracy
    at 4-bit and even more at 2-bit compared to skipping this step.
    """
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images.to(device))
    model.eval()


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def _load_model(checkpoint_path: str, device):
    exists = os.path.isfile(checkpoint_path)
    model  = build_mobilenetv2(num_classes=10, pretrained=not exists)
    if exists:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint : {checkpoint_path}")
    else:
        print(
            f"WARNING: '{checkpoint_path}' not found. "
            f"Using ImageNet-pretrained backbone + untrained head. "
            f"Run train.py first for meaningful accuracy numbers."
        )
    return model.to(device)


# --------------------------------------------------------------------------- #
#  Core experiment function (imported by sweep.py)                            #
# --------------------------------------------------------------------------- #

def run_experiment(
    weight_quant_bits: int,
    activation_quant_bits: int,
    checkpoint_path: str  = "best_model.pth",
    batch_size: int       = 128,
    num_calib_batches: int = 50,
    device                = None,
    dataloader_fn         = get_dataloaders,
    verbose: bool         = True,
) -> dict:
    """
    Run one (weight_bits, activation_bits) configuration and return a flat
    results dict whose keys match the Wandb parallel coordinates columns:
        activation_quant_bits, weight_quant_bits,
        compression_ratio, activation_compression_ratio,
        model_size_mb, baseline_acc, quantized_acc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = dataloader_fn(batch_size=batch_size)

    # ── 1. Load model ────────────────────────────────────────────────────────
    model = _load_model(checkpoint_path, device)

    # ── 2. Baseline accuracy (fp32, no quantisation) ─────────────────────────
    baseline_acc = _evaluate(model, test_loader, device)
    if verbose:
        print(f"\nBaseline fp32 accuracy : {baseline_acc:.2f}%")

    # Work on a deep copy so the original checkpoint weights are preserved.
    model_q = copy.deepcopy(model)

    # ── 3. Weight quantisation ────────────────────────────────────────────────
    quantized_names = apply_weight_quantization(
        model_q, bits=weight_quant_bits,
        symmetric=True, per_channel=True,
        skip_layers=SKIP_WEIGHT_LAYERS,
    )
    quantized_layers = {n: {"bits": weight_quant_bits, "per_channel": True}
                        for n in quantized_names}
    if verbose:
        print(f"Weight-quantised layers : {len(quantized_names)}")

    # ── 4. BN recalibration ───────────────────────────────────────────────────
    # After fake-quantising weights the BatchNorm running mean/var are stale —
    # they were estimated on the original fp32 weights, but the quantised conv
    # outputs have different statistics. Running a few training batches in
    # model.train() mode forces BN to update its running stats to match the
    # quantised model's actual activations. Without this, 4-bit accuracy can
    # drop 10–20% compared to the theoretical optimum.
    _recalibrate_bn(model_q, train_loader, num_batches=num_calib_batches, device=device)
    if verbose:
        print(f"BN recalibrated over {num_calib_batches} batches")

    # ── 5. Activation calibration (on training data only) ────────────────────
    act_layers    = find_relu6_layers(model_q)
    calibrator    = ActivationCalibrator(act_layers)
    act_stats     = calibrator.calibrate(
        model_q, train_loader,
        num_batches=num_calib_batches, device=device,
    )
    act_qparams   = calibrator.compute_qparams(bits=activation_quant_bits)
    hook_handles  = attach_activation_quant_hooks(act_layers, act_qparams, bits=activation_quant_bits)
    if verbose:
        print(f"Activation-quantised layers : {len(act_layers)}")

    # ── 7. Quantised accuracy ─────────────────────────────────────────────────
    quantized_acc = _evaluate(model_q, test_loader, device)
    for h in hook_handles:
        h.remove()

    # ── 8. Storage accounting ─────────────────────────────────────────────────
    weight_result = compute_weight_storage(
        model_q, quantized_layers, skip_layers=SKIP_WEIGHT_LAYERS
    )
    act_result = compute_activation_storage(act_stats, bits=activation_quant_bits)

    if verbose:
        print_compression_summary(weight_result, act_result, weight_quant_bits, activation_quant_bits)
        print(f"\nQuantised accuracy     : {quantized_acc:.2f}%")
        print(f"Accuracy drop          : {baseline_acc - quantized_acc:.2f}%")

    return {
        # --- columns for wandb parallel coordinates ---
        "weight_quant_bits":           weight_quant_bits,
        "activation_quant_bits":       activation_quant_bits,
        "compression_ratio":           weight_result["compression_ratio"],
        "activation_compression_ratio": act_result["compression_ratio"],
        "model_size_mb":               weight_result["compressed_MB"],
        "metadata_mb":                 weight_result["metadata_MB"],
        "baseline_acc":                baseline_acc,
        "quantized_acc":               quantized_acc,
        "accuracy_drop":               baseline_acc - quantized_acc,
    }


# --------------------------------------------------------------------------- #
#  CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run a single quantisation experiment")
    parser.add_argument("--weight_quant_bits",     type=int, default=8)
    parser.add_argument("--activation_quant_bits", type=int, default=8)
    parser.add_argument("--checkpoint_path",       type=str, default="best_model.pth")
    parser.add_argument("--batch_size",            type=int, default=128)
    parser.add_argument("--num_calib_batches",     type=int, default=50)
    args = parser.parse_args()

    results = run_experiment(
        weight_quant_bits     = args.weight_quant_bits,
        activation_quant_bits = args.activation_quant_bits,
        checkpoint_path       = args.checkpoint_path,
        batch_size            = args.batch_size,
        num_calib_batches     = args.num_calib_batches,
        verbose               = True,
    )

    print("\n=== Raw results dict ===")
    for k, v in results.items():
        print(f"  {k:35s}: {v:.4f}" if isinstance(v, float) else f"  {k:35s}: {v}")


if __name__ == "__main__":
    main()
