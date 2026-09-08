# CS6886 — Assignment 2: MobileNet-v2 Compression on CIFAR-10

## Repository structure

```
├── data.py              # CIFAR-10 loading + augmentation
├── model.py             # MobileNet-v2 adapted for 32×32 inputs
├── train.py             # Fine-tuning script (Q1)
├── quantize.py          # Custom weight quantisation from scratch (Q2)
├── activation_quant.py  # Calibration-based activation quantisation (Q2)
├── utils.py             # Storage / compression-ratio accounting (Q2c, Q4)
├── test.py              # Single experiment runner (Q3, Q4)
├── sweep.py             # Wandb grid sweep for parallel coordinates chart (Q3b)
├── plot_curves.py       # Generate loss/accuracy curves from saved history (Q1c)
└── requirements.txt
```

---

## Environment

```bash
# Python 3.10+  recommended
pip install -r requirements.txt

# Log in to Wandb (needed only for sweep.py)
wandb login
```

---

## Reproducing results

### Step 1 — Train the baseline model (Q1)

```bash
# Fine-tune from ImageNet pretrained weights (recommended; ~93% test accuracy)
python train.py --epochs 30 --batch_size 128 --lr 0.01 --seed 42

# Optional: log training curves to Wandb
python train.py --epochs 30 --batch_size 128 --lr 0.01 --seed 42 --use_wandb

# Train from scratch (lower accuracy, ~75–80%, but no pretrained weights)
python train.py --epochs 100 --no_pretrained --lr 0.05 --seed 42
```

Outputs:
- `best_model.pth` — checkpoint with best test accuracy
- `training_history.json` — per-epoch metrics

### Step 2 — Plot training curves (Q1c)

```bash
python plot_curves.py --history training_history.json --out training_curves.png
```

### Step 3 — Single quantisation experiment (Q3, Q4)

```bash
# 8-bit weights + 8-bit activations (high accuracy, ~4× compression)
python test.py --weight_quant_bits 8 --activation_quant_bits 8

# 4-bit weights + 4-bit activations (good trade-off)
python test.py --weight_quant_bits 4 --activation_quant_bits 4

# 2-bit weights + 2-bit activations (maximum compression, accuracy drops)
python test.py --weight_quant_bits 2 --activation_quant_bits 2
```

### Step 4 — Full Wandb sweep for parallel coordinates chart (Q3b)

```bash
python sweep.py --wandb_project cs6886-mobilenetv2-compression
```

Runs a 3×3 grid: `weight_bits ∈ {2,4,8}` × `activation_bits ∈ {2,4,8}` (9 runs total).

After the sweep finishes, go to your Wandb project → **Parallel Coordinates** chart, and select columns:
`weight_quant_bits`, `activation_quant_bits`, `compression_ratio`, `model_size_mb`, `quantized_acc`.

---

## Model configuration (Q1b)

| Setting | Value | Reason |
|---|---|---|
| width_mult | 1.0 | Matches ImageNet pretrained checkpoint |
| dropout | 0.2 | Torchvision default; regularises head |
| BatchNorm | default (eps=1e-5, momentum=0.1) | Pretrained stats computed at these values |
| Stem stride | 1 (modified) | 32×32 → 1×1 collapse without this fix |
| features[2] stride | 1 (modified) | Reduces total downsample to 8× |
| Optimizer | SGD + Nesterov momentum=0.9 | Standard for CNNs; better generalisation than Adam |
| weight_decay | 4e-5 | Small because depthwise layers are already compact |
| Initial LR | 0.01 | Fine-tuning; lower than scratch to protect pretrained weights |
| LR schedule | Warmup (5 epochs) + Cosine annealing | Smooth decay, no manual milestones |
| Epochs | 30 | Sufficient for fine-tuning; ~4× faster than scratch |
| Batch size | 128 | Standard CIFAR-10 batch |
| Label smoothing | 0.1 | Regularises loss without shrinking parameter count |
| Gradient clipping | 1.0 | Prevents instability in early warmup epochs |

---

## Compression method (Q2)

### Weights — symmetric per-channel fake quantisation

For each `Conv2d` / `Linear` layer (excluding stem and classifier):

```
scale_c     = max(|W_c|) / (2^(bits-1) − 1)   # one per output channel c
q_c         = clamp(round(W_c / scale_c), qmin, qmax)
W_hat_c     = scale_c × q_c                    # zero_point = 0 (symmetric)
```

**Why per-channel?** Weights in different output channels have very different magnitudes. A single global scale overfits to the largest channel and wastes range for smaller ones. Per-channel scale costs only `n_channels × 4 bytes` extra metadata.

**Why symmetric?** Weights are roughly zero-centred. Fixing `zero_point = 0` wastes ≤1 representable level while halving metadata and simplifying inference arithmetic.

### Activations — asymmetric per-tensor calibration

Measured at the output of every `ReLU6` (which feeds directly into the next Conv2d):

```
Calibration: run 10 training batches → record min, max per layer
scale       = (max − min) / (2^bits − 1)
zero_point  = round(qmin − min / scale)
fake_quant  → applied via forward hook during inference
```

**Why asymmetric?** `ReLU6` output is always in `[0, 6]`. Symmetric quantisation would waste half the integer range on negative values that never occur.

### Layers skipped (Q2b)

| Layer | Reason |
|---|---|
| `features.0.0` (stem Conv2d) | Tiny parameter count; first layer; high accuracy sensitivity |
| `classifier.1` (final Linear) | Closest to loss; most sensitive to numerical error |
| All `BatchNorm2d` | Combined <1% of model size; direct effect on activation scale |
| All biases | Always kept at fp32; negligible size |

### Storage overheads (Q2c)

| Item | Storage |
|---|---|
| Quantised weight payload | `n_weights × bits` |
| Bias (fp32, not quantised) | `n_bias × 32 bits` |
| Weight metadata (per output channel) | `n_channels × 2 × 32 bits` (scale + zero_pt) |
| Activation metadata (per layer) | `2 × 32 bits` (scale + zero_pt) |
| BatchNorm params + running stats | `(weight + bias + mean + var) × 32 bits + 64 bits` |

---

## Expected results (Q4)

| Config | Weight CR | Act CR | Model size | Test accuracy |
|---|---|---|---|---|
| fp32 baseline | 1.00× | 1.00× | ~13.4 MB | ~93.5% |
| w8 a8 | ~3.8× | ~3.8× | ~3.6 MB | ~93.2% |
| w4 a4 | ~7.4× | ~7.4× | ~1.9 MB | ~91.5% |
| w2 a2 | ~13.5× | ~13.5× | ~1.1 MB | ~75–85% |

*(Exact values depend on your hardware, seed, and whether pretrained weights are used.)*

---

## Seed configuration (Q5b)

All scripts accept `--seed` (default `42`). The seed is set at the start of `train.py`:

```python
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
```

For DataLoader worker reproducibility, set `--num_workers 0` or configure `worker_init_fn` if strict reproducibility across runs is needed.
