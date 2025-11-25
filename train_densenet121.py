#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

CLASSES = ["beer_cider_bottle", "beer_cider_cup", "wine", "non_alcoholic_beverage"]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(train_dataset, use_weights=True):
    if not use_weights:
        return None
    counts = np.zeros(len(train_dataset.classes), dtype=np.int64)
    for _, label in train_dataset.samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (counts.astype(np.float32) * len(counts))
    return torch.tensor(weights, dtype=torch.float32)

def accuracy(output, target):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
    return correct / max(total, 1)

def confusion_matrix(num_classes: int, preds: torch.Tensor, targets: torch.Tensor):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds.view(-1), targets.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def precision_recall_f1_from_cm(cm: torch.Tensor):
    eps = 1e-12
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=eps)
    return precision, recall, f1

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_checkpoint_if_exists(model, optimizer, scaler, ckpt_path: Path):
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[resume] Loaded checkpoint from {ckpt_path} (epoch {start_epoch-1}, best_acc={best_acc:.4f})")
        return start_epoch, best_acc
    return 0, 0.0

def build_transforms(img_size=224, center_crop=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(center_crop * 256 / 224)),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def build_dataloaders(data_root: Path, img_size=224, batch_size=64, workers=8, pin_memory=True):
    train_tf, eval_tf = build_transforms(img_size, img_size)

    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds  = datasets.ImageFolder(test_dir, transform=eval_tf)

    if train_ds.classes != CLASSES:
        name_to_idx = {name: i for i, name in enumerate(train_ds.classes)}
        try:
            target_indices = [name_to_idx[c] for c in CLASSES]
        except KeyError as e:
            raise RuntimeError(f"Dataset class folder missing: {e}. Found={train_ds.classes}, Expected={CLASSES}")
        new_class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        def remap_samples(ds):
            remapped = []
            for path, y in ds.samples:
                cls_name = ds.classes[y]
                new_y = new_class_to_idx[cls_name]
                remapped.append((path, new_y))
            ds.samples = remapped
            ds.targets = [y for _, y in remapped]
            ds.classes = CLASSES
            ds.class_to_idx = new_class_to_idx
        remap_samples(train_ds)
        remap_samples(val_ds)
        remap_samples(test_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_ds

def build_model(num_classes=4, dropout=0.2, pretrained=True):
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
    except Exception:
        model = models.densenet121(pretrained=pretrained)
    in_features = model.classifier.in_features
    classifier = []
    if dropout and dropout > 0.0:
        classifier.append(nn.Dropout(p=dropout))
    classifier.append(nn.Linear(in_features, num_classes))
    model.classifier = nn.Sequential(*classifier)
    return model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_acc += (outputs.argmax(dim=1) == targets).float().sum().item()
        n += targets.size(0)

    return total_loss / max(n, 1), total_acc / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=4):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        total_acc += (outputs.argmax(dim=1) == targets).float().sum().item()
        n += targets.size(0)

        preds = outputs.argmax(dim=1)
        for p, t in zip(preds.cpu(), targets.cpu()):
            cm[t.long(), p.long()] += 1

    eps = 1e-12
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=eps)

    metrics = {
        "loss": total_loss / max(n, 1),
        "acc": total_acc / max(n, 1),
        "cm": cm.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }
    return metrics

def save_checkpoint(path: Path, model, optimizer, scaler, epoch, best_acc):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        "best_acc": best_acc
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Dataset root containing train/ val/ test/")
    parser.add_argument("--out", type=str, required=True, help="Output dir for logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  |  cuda available: {torch.cuda.is_available()}")

    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        data_root=data_root, img_size=224, batch_size=args.batch, workers=args.workers, pin_memory=True
    )

    mapping = {"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx}
    save_json(mapping, out_dir / "label_to_index.json")
    print("[classes]", mapping)

    model = build_model(num_classes=len(train_ds.classes), dropout=args.dropout, pretrained=not args.no_pretrained)
    model.to(device)

    class_weights = compute_class_weights(train_ds, use_weights=not args.no_class_weights)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("[loss] Using class weights:", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch, best_acc = 0, 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint_if_exists(model, optimizer, scaler, Path(args.resume))

    history = []
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(max(args.warmup_epochs, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes=len(train_ds.classes))

        print(f"  train: loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"  val  : loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}")
        print(f"  val per-class F1:")
        for idx, c in enumerate(train_ds.classes):
            print(f"    - {c:<24}  P={val_metrics['precision'][idx]:.3f}  R={val_metrics['recall'][idx]:.3f}  F1={val_metrics['f1'][idx]:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"]
        })
        save_json(history, out_dir / "history.json")

        save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, epoch, best_acc)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, epoch, best_acc)
            epochs_no_improve = 0
            print(f"  ✅ New best acc: {best_acc:.4f} (checkpoint saved)")
        else:
            epochs_no_improve += 1
            print(f"  no improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= args.patience:
            print("  ⏹ Early stopping triggered")
            break

    best_ckpt = out_dir / "best.pt"
    if best_ckpt.exists():
        print(f"\n[load best for test] {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, criterion, device, num_classes=len(train_ds.classes))
    print(f"\n[Test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f}")
    print("Test per-class metrics:")
    for idx, c in enumerate(train_ds.classes):
        print(f"  - {c:<24}  P={test_metrics['precision'][idx]:.3f}  R={test_metrics['recall'][idx]:.3f}  F1={test_metrics['f1'][idx]:.3f}")

    save_json(test_metrics, out_dir / "test_metrics.json")
    print(f"\nArtifacts saved in: {out_dir}")
    print(f"Total time: {(time.time()-t0)/60.0:.1f} min")

if __name__ == "__main__":
    main()
