#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, precision_score)

CLASSES = ["beer_cider_bottle", "beer_cider_cup", "wine", "non_alcoholic_beverage"]
AUTOTUNE = tf.data.AUTOTUNE

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

def has_splits(root: Path) -> bool:
    return all([(root / split).exists() for split in ["train", "val", "test"]])

def prepare_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_class_files(source: Path) -> Dict[str, List[Path]]:
    data = {}
    for cls in CLASSES:
        cls_dir = source / cls
        if not cls_dir.exists():
            raise RuntimeError(f"Missing class folder: {cls_dir}")
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend(cls_dir.glob(ext))
        data[cls] = sorted(files)
    return data

def split_indices(n: int, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, seed=42):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(round(n * train_ratio))
    n_test = int(round(n * test_ratio))
    n_val = n - n_train - n_test
    return idxs[:n_train], idxs[n_train:n_train+n_test], idxs[n_train+n_test:]

def create_splits_if_needed(data_root: Path, out_root: Path, do_split: bool):
    if not do_split and has_splits(data_root):
        return data_root / "train", data_root / "val", data_root / "test"

    source = data_root
    if not has_splits(data_root):
        print("[split] No train/val/test found. Will create splits from a single root:", data_root)
    else:
        print("[split] --split was requested; will recreate splits under:", out_root)

    train_dir = out_root / "train"
    val_dir   = out_root / "val"
    test_dir  = out_root / "test"
    for d in [train_dir, val_dir, test_dir]:
        prepare_dir(d)

    per_class_files = collect_class_files(source)

    for cls in CLASSES:
        files = per_class_files[cls]
        if len(files) == 0:
            raise RuntimeError(f"No images for class {cls} in {source/cls}")
        train_idx, test_idx, val_idx = split_indices(len(files), 0.7, 0.2, 0.1)
        for split_name, idxs in [("train", train_idx), ("test", test_idx), ("val", val_idx)]:
            dest = (out_root / split_name / cls)
            prepare_dir(dest)
            for i in idxs:
                src = files[i]
                dst = dest / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        print(f"[split:{cls}] total={len(files)}  train={len(train_idx)}  test={len(test_idx)}  val={len(val_idx)}")

    return train_dir, val_dir, test_dir

def save_label_map(out_dir: Path, class_names: List[str]):
    mapping = {"classes": class_names, "class_to_index": {c:i for i,c in enumerate(class_names)}}
    out = out_dir / "label_to_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print("[labels] Saved:", out)

def build_datasets(train_dir: Path, val_dir: Path, test_dir: Path, img_size=(224,224), batch=64):
    def make_ds(directory: Path, shuffle: bool):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=CLASSES,
            image_size=img_size,
            batch_size=batch,
            shuffle=shuffle
        )
        return ds

    train_ds = make_ds(train_dir, shuffle=True)
    val_ds   = make_ds(val_dir, shuffle=False)
    test_ds  = make_ds(test_dir, shuffle=False)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ])

    preprocess = tf.keras.applications.densenet.preprocess_input

    def preprocess_map(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess(image)
        return image, label

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(preprocess_map, num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(preprocess_map, num_parallel_calls=AUTOTUNE)
    test_ds  = test_ds.map(preprocess_map, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

def build_model(num_classes=4, dropout=0.4, train_base=True):
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base.trainable = train_base
    inputs = tf.keras.Input(shape=(224,224,3))
    x = base(inputs, training=train_base)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def plot_history(history, out_dir: Path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    out_path = out_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("[plot] Saved:", out_path)

def evaluate_and_plots(model, val_ds, test_ds, out_dir: Path):
    y_true_val, y_pred_val = [], []
    for batch_x, batch_y in val_ds:
        probs = model.predict(batch_x, verbose=0)
        y_true_val.extend(batch_y.numpy().tolist())
        y_pred_val.extend(np.argmax(probs, axis=1).tolist())

    acc_val = accuracy_score(y_true_val, y_pred_val)
    prec_val = precision_score(y_true_val, y_pred_val, average="macro", zero_division=0)
    print(f"[Val] acc={acc_val:.4f}  precision_macro={prec_val:.4f}")

    labels = CLASSES
    cm = confusion_matrix(y_true_val, y_pred_val, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, values_format="d")
    plt.title("Validation Confusion Matrix")
    cm_path = out_dir / "val_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("[plot] Saved:", cm_path)

    plt.figure(figsize=(5,4))
    plt.bar(["Accuracy", "Precision"], [acc_val, prec_val])
    plt.ylim(0,1)
    plt.title("Val: Accuracy vs Precision")
    bar_path = out_dir / "val_acc_vs_precision.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print("[plot] Saved:", bar_path)

    report = classification_report(y_true_val, y_pred_val, target_names=labels, zero_division=0)
    with open(out_dir / "val_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("[report] Saved:", out_dir / "val_classification_report.txt")

    y_true_test, y_pred_test = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)
        y_true_test.extend(batch_y.numpy().tolist())
        y_pred_test.extend(np.argmax(probs, axis=1).tolist())
    acc_test = accuracy_score(y_true_test, y_pred_test)
    prec_test = precision_score(y_true_test, y_pred_test, average="macro", zero_division=0)
    print(f"[Test]  acc={acc_test:.4f}  precision_macro={prec_test:.4f}")
    with open(out_dir / "test_scores.txt", "w", encoding="utf-8") as f:
        f.write(f"acc={acc_test:.4f}  precision_macro={prec_test:.4f}\n")

def demo_predictions(model, test_dir: Path, out_dir: Path, num_images=8):
    rng = random.Random(123)
    candidates = []
    for cls in CLASSES:
        cls_dir = test_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            candidates.extend(list(cls_dir.glob(ext)))
    rng.shuffle(candidates)
    samples = candidates[:num_images]

    preprocess = tf.keras.applications.densenet.preprocess_input

    plt.figure(figsize=(12, 8))
    cols = 4
    for i, path in enumerate(samples):
        img = tf.keras.utils.load_img(path, target_size=(224,224))
        arr = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(arr, axis=0).astype("float32")
        x = preprocess(x)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = CLASSES[pred_idx]
        conf = float(np.max(probs))

        plt.subplot(int(np.ceil(num_images/cols)), cols, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{pred_name}\n{conf:.2%}")
    demo_path = out_dir / "demo_predictions.png"
    plt.tight_layout()
    plt.savefig(demo_path, dpi=150)
    plt.close()
    print("[demo] Saved:", demo_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Either a single-folder dataset (per-class subfolders) or a root with train/val/test")
    parser.add_argument("--out", type=str, required=True, help="Output dir for runs (plots, weights, reports)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split", action="store_true", help="Force-create 70/20/10 splits from single root")
    parser.add_argument("--freeze", action="store_true", help="Freeze DenseNet121 base (feature extractor)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    gpu_memory_growth()

    data_root = Path(args.data).expanduser().resolve()
    out_dir   = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_splits(data_root) and not args.split:
        train_dir, val_dir, test_dir = data_root / "train", data_root / "val", data_root / "test"
        split_root = data_root
    else:
        split_root = out_dir / "splits_70_20_10"
        train_dir, val_dir, test_dir = create_splits_if_needed(data_root, split_root, do_split=True)

    save_label_map(out_dir, CLASSES)

    train_ds, val_ds, test_ds = build_datasets(train_dir, val_dir, test_dir, img_size=(224,224), batch=args.batch)

    model = build_model(num_classes=len(CLASSES), dropout=0.4, train_base=not args.freeze)

    opt = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ckpt_path = out_dir / "best_densenet121.keras"
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    plot_history(history, out_dir)

    evaluate_and_plots(model, val_ds, test_ds, out_dir)

    final_path = out_dir / "final_densenet121.keras"
    model.save(final_path)
    model.save('best_model_wine_detection_densenet121.h5')
    print("[save] Final model:", final_path)

    demo_predictions(model, test_dir, out_dir, num_images=8)

    print("\nAll artifacts saved under:", out_dir)
    print("Splits at:", split_root)

if __name__ == "__main__":
    main()
