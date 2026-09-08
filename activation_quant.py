"""
activation_quant.py  —  Post-Training Quantisation (PTQ) for activations.

We quantise at the output of every ReLU6 in the network. ReLU6 outputs are
passed directly to the next Conv2d as its input, so quantising them faithfully
models the scenario where "the next layer receives an N-bit input."

Two-phase workflow:
  Phase 1 (calibration):  run a number of real training batches through the
      *unmodified* float model and record per-layer activation statistics.
      We use the training set for calibration and the test set only for
      evaluation — using test data for calibration would leak test distribution
      information into the model's effective quantisation behaviour.
  Phase 2 (inference):    attach forward hooks that fake-quantise each ReLU6
      output using (scale, zero_point) derived from Phase 1.

Asymmetric quantisation for activations (Q2a design choice):
  ReLU6 output is always in [0, 6].  Symmetric quantisation (zero_point=0)
  would waste half the integer range on values that provably never occur.
  Asymmetric quantisation maps [t_min, t_max] to [qmin, qmax] exactly.

Percentile-based range estimation:
  Rather than using the global min/max (which can be dominated by rare
  outliers that stretch the scale and waste quantisation precision), we clip
  to the 99.9th percentile of observed activation values. This is a standard
  PTQ calibration technique that meaningfully improves accuracy at 4-bit.
"""

import torch
import torch.nn as nn


def find_relu6_layers(model: nn.Module) -> dict:
    """Returns {name: module} for every ReLU6 in the model."""
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.ReLU6)}


class ActivationCalibrator:
    """
    Phase 1: collect per-layer activation statistics over calibration batches.

    After calling .calibrate(), self.stats contains:
        {layer_name: {"min": float, "max": float, "numel": int}}
    where numel is the element count for a SINGLE image (batch dimension excluded).
    This per-image numel is used by utils.py to estimate per-image activation
    storage cost for the compression ratio reported in Q4b.

    We use percentile clipping (99.9%) on the collected histogram to avoid
    outlier activations inflating the scale and wasting precision levels.
    """

    def __init__(self, layers: dict, percentile: float = 99.9):
        self.layers     = layers
        self.percentile = percentile
        # Store all observed values per layer for percentile computation.
        self._samples: dict = {n: [] for n in layers}
        self.stats:    dict = {n: {"min": 0.0, "max": 6.0, "numel": 0} for n in layers}
        self._handles:  list = []

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                # Flatten and downsample to keep memory use manageable.
                flat = output.detach().float().flatten()
                # Keep at most 4096 values per batch to avoid OOM on large tensors.
                if flat.numel() > 4096:
                    idx  = torch.randperm(flat.numel(), device=flat.device)[:4096]
                    flat = flat[idx]
                self._samples[name].append(flat.cpu())
                self.stats[name]["numel"] = output[0].numel()   # per single image
        return hook

    def attach(self):
        for name, module in self.layers.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def calibrate(self, model: nn.Module, loader, num_batches: int = 50,
                  device: str = "cpu") -> dict:
        """Run `num_batches` training batches through the model to collect stats."""
        model.eval()
        self.attach()
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images.to(device))
        self.remove()

        # Compute percentile-clipped range from collected samples.
        for name in self.layers:
            if not self._samples[name]:
                continue
            all_vals = torch.cat(self._samples[name])
            lo = torch.kthvalue(all_vals, max(1, int((1.0 - self.percentile / 100.0) * all_vals.numel()))).values.item()
            hi = torch.kthvalue(all_vals, min(all_vals.numel(), int(self.percentile / 100.0 * all_vals.numel()))).values.item()
            self.stats[name]["min"] = max(0.0, lo)   # ReLU6 can't be negative
            self.stats[name]["max"] = min(6.0, hi)   # ReLU6 can't exceed 6

        return self.stats

    def compute_qparams(self, bits: int) -> dict:
        """Convert per-layer min/max into (scale, zero_point) using asymmetric formulae."""
        qmin = -(2 ** (bits - 1))
        qmax =   2 ** (bits - 1) - 1
        qparams = {}
        for name, s in self.stats.items():
            t_min, t_max = s["min"], s["max"]
            scale      = max((t_max - t_min) / (qmax - qmin), 1e-8)
            zero_point = round(qmin - t_min / scale)
            qparams[name] = {"scale": scale, "zero_point": zero_point}
        return qparams


def attach_activation_quant_hooks(layers: dict, qparams: dict, bits: int) -> list:
    """
    Phase 2: attach hooks that fake-quantise each ReLU6 output.
    Returns hook handles so they can be removed later.
    """
    qmin = -(2 ** (bits - 1))
    qmax =   2 ** (bits - 1) - 1
    handles = []

    def _make_quant_hook(scale: float, zero_point: float):
        def hook(module, inputs, output):
            q = torch.clamp(torch.round(output / scale) + zero_point, qmin, qmax)
            return scale * (q - zero_point)
        return hook

    for name, module in layers.items():
        p = qparams[name]
        handles.append(module.register_forward_hook(
            _make_quant_hook(p["scale"], p["zero_point"])
        ))
    return handles


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    from model import build_mobilenetv2

    torch.manual_seed(0)
    model = build_mobilenetv2(num_classes=10, pretrained=False)

    x = torch.randn(32, 3, 32, 32)
    loader = DataLoader(TensorDataset(x, torch.zeros(32).long()), batch_size=8)

    layers = find_relu6_layers(model)
    print(f"ReLU6 layers found: {len(layers)}")

    cal = ActivationCalibrator(layers)
    stats = cal.calibrate(model, loader, num_batches=4)
    first = next(iter(stats))
    print(f"Stats for '{first}': {stats[first]}")

    qparams  = cal.compute_qparams(bits=8)
    handles  = attach_activation_quant_hooks(layers, qparams, bits=8)
    with torch.no_grad():
        out = model(x[:2])
    print(f"Output with quantised activations: {out.shape}")
    for h in handles:
        h.remove()
