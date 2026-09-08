"""
train.py  —  MobileNet-v2 fine-tuning on CIFAR-10 (Q1b, Q1c).

Training strategy:
  Optimizer   : SGD + Nesterov momentum (0.9), weight_decay=4e-5.
                SGD generalises better than Adam for CNNs on standard
                image benchmarks and is what the original MobileNet paper uses.
                Small weight_decay because depthwise-separable layers are
                already parameter-efficient; heavy L2 would underfit.
  LR schedule : Linear warmup (5 epochs) → CosineAnnealingLR to ~0.
                Warmup prevents large early gradients from damaging pretrained
                features; cosine decay removes the need to hand-tune milestones.
  Initial LR  : 0.01  (lower than train-from-scratch because pretrained weights
                are already in a good loss basin).
  Epochs      : 30  (fine-tuning converges much faster than scratch training).
  Batch size  : 128  (standard CIFAR-10 batch; fits on a single 4 GB GPU).
  Label smooth: 0.1  (regularises the loss without shrinking the small parameter
                count further).

Usage:
  python train.py                            # pretrained, 30 epochs
  python train.py --epochs 50 --use_wandb   # with wandb logging
  python train.py --no_pretrained            # train from scratch (slower)
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn

from data  import get_dataloaders
from model import build_mobilenetv2


# --------------------------------------------------------------------------- #
#  Training / evaluation helpers                                               #
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# --------------------------------------------------------------------------- #
#  Main training loop                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train MobileNet-v2 on CIFAR-10")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=0.01)
    parser.add_argument("--momentum",        type=float, default=0.9)
    parser.add_argument("--weight_decay",    type=float, default=4e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--checkpoint_path", type=str,   default="best_model.pth")
    parser.add_argument("--history_path",    type=str,   default="training_history.json")
    parser.add_argument("--pretrained",      action="store_true", default=True)
    parser.add_argument("--no_pretrained",   dest="pretrained", action="store_false")
    parser.add_argument("--use_wandb",       action="store_true", default=False)
    parser.add_argument("--wandb_project",   type=str,   default="cs6886-mobilenetv2")
    args = parser.parse_args()

    # ---- Reproducibility -----------------------------------------------
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Pretrained  : {args.pretrained}")
    print(f"Epochs      : {args.epochs}  (warmup={args.warmup_epochs})")
    print(f"Batch size  : {args.batch_size}")
    print(f"Initial LR  : {args.lr}")

    # ---- Data ----------------------------------------------------------
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ---- Model ---------------------------------------------------------
    model = build_mobilenetv2(num_classes=10, pretrained=args.pretrained).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total_params:,}")

    # ---- Optimiser + scheduler -----------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True,
    )

    # Warmup: linearly ramp LR from lr/warmup_epochs to lr over the first
    # `warmup_epochs` epochs, then hand off to cosine decay.
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-5
    )

    # ---- Wandb (optional) ----------------------------------------------
    use_wandb = args.use_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"train_pretrained={args.pretrained}_epochs={args.epochs}",
            )
        except ImportError:
            print("WARNING: wandb not installed; disabling wandb logging.")
            use_wandb = False

    # ---- Training loop -------------------------------------------------
    best_acc = 0.0
    history  = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        current_lr = get_lr(optimizer)

        # Step the appropriate scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.checkpoint_path)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.2f}% | "
            f"lr={current_lr:.5f} | {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "test_loss":  test_loss,
                "test_acc":   test_acc,
                "lr":         current_lr,
            })

        # Checkpoint history after every epoch so a crash doesn't lose data
        with open(args.history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest test accuracy : {best_acc:.2f}%")
    print(f"Checkpoint saved   : {args.checkpoint_path}")
    print(f"History saved      : {args.history_path}")

    if use_wandb:
        import wandb
        wandb.summary["best_test_acc"] = best_acc
        wandb.finish()

    return history, best_acc


if __name__ == "__main__":
    main()
