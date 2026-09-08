"""
sweep.py  —  Grid sweep over quantisation bit-widths, logged to Wandb (Q3b).

Runs a 3×3 grid (9 runs) of (weight_bits, activation_bits) combinations and
logs each run to the project "cs6886-mobilenetv2-compression".

After the sweep:
  1. Go to your Wandb project page.
  2. Open the "Runs" table.
  3. Click "Parallel Coordinates" chart type.
  4. Select columns: weight_quant_bits, activation_quant_bits,
     compression_ratio, model_size_mb, quantized_acc.

Usage:
    python sweep.py
    python sweep.py --checkpoint_path best_model.pth --wandb_project my-project
"""

import argparse
import sys

from test import run_experiment

WEIGHT_BITS_GRID     = [2, 4, 8]
ACTIVATION_BITS_GRID = [2, 4, 8]

# Wandb columns to log (must match run_experiment() return keys).
WANDB_COLUMNS = [
    "weight_quant_bits",
    "activation_quant_bits",
    "compression_ratio",
    "activation_compression_ratio",
    "model_size_mb",
    "metadata_mb",
    "baseline_acc",
    "quantized_acc",
    "accuracy_drop",
]


def main():
    parser = argparse.ArgumentParser(description="Wandb quantisation sweep")
    parser.add_argument("--checkpoint_path",   type=str, default="best_model.pth")
    parser.add_argument("--batch_size",        type=int, default=128)
    parser.add_argument("--num_calib_batches", type=int, default=10)
    parser.add_argument("--wandb_project",     type=str, default="cs6886-mobilenetv2-compression")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is not installed. Run:  pip install wandb")
        sys.exit(1)

    total_runs = len(WEIGHT_BITS_GRID) * len(ACTIVATION_BITS_GRID)
    run_idx    = 0

    for w_bits in WEIGHT_BITS_GRID:
        for a_bits in ACTIVATION_BITS_GRID:
            run_idx += 1
            print(f"\n{'─'*60}")
            print(f"  Run {run_idx}/{total_runs}:  weight={w_bits}-bit  activation={a_bits}-bit")
            print(f"{'─'*60}")

            # Each experiment is its own wandb run so the parallel coordinates
            # chart shows one line per (w_bits, a_bits) configuration.
            run = wandb.init(
                project = args.wandb_project,
                name    = f"w{w_bits}a{a_bits}",
                config  = {
                    "weight_quant_bits":     w_bits,
                    "activation_quant_bits": a_bits,
                },
                reinit  = True,
            )

            try:
                results = run_experiment(
                    weight_quant_bits     = w_bits,
                    activation_quant_bits = a_bits,
                    checkpoint_path       = args.checkpoint_path,
                    batch_size            = args.batch_size,
                    num_calib_batches     = args.num_calib_batches,
                    verbose               = True,
                )
                wandb.log({k: results[k] for k in WANDB_COLUMNS if k in results})
                print(f"\n  ✓  quantized_acc={results['quantized_acc']:.2f}%  "
                      f"compression_ratio={results['compression_ratio']:.3f}×")
            except Exception as e:
                print(f"  ✗  Run failed: {e}")
            finally:
                run.finish()

    print(f"\n{'='*60}")
    print(f"  Sweep complete — {total_runs} runs logged to '{args.wandb_project}'")
    print(f"{'='*60}")
    print(f"\n  → Go to https://wandb.ai and open the Parallel Coordinates chart.")
    print(f"    Columns: weight_quant_bits, activation_quant_bits,")
    print(f"             compression_ratio, model_size_mb, quantized_acc")


if __name__ == "__main__":
    main()
