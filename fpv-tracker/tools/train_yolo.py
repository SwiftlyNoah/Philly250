"""Train a YOLOv8 model on the labelled drone dataset.

Includes a ``--small-object`` mode that applies best-practice optimisations
for detecting very small / distant drones (few-pixel targets):

* Higher input resolution (1280 px default)
* Extra P2 detection head via a ``yolov8s-p2.yaml`` architecture
* Copy-paste augmentation and aggressive mosaic
* Lower default confidence threshold
* Multi-scale training
"""

import argparse
import os
import random
import shutil
import sys
from typing import Dict, List

import yaml


# -- Dataset helpers -----------------------------------------------------------


def _split_dataset(
    images_dir: str,
    labels_dir: str,
    output_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
) -> None:
    """Organise images + labels into train / val / test splits."""
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train : n_train + n_val],
        "test": image_files[n_train + n_val :],
    }

    for split_name, files in splits.items():
        img_out = os.path.join(output_root, split_name, "images")
        lbl_out = os.path.join(output_root, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in files:
            src_img = os.path.join(images_dir, fname)
            shutil.copy2(src_img, os.path.join(img_out, fname))

            lbl_name = os.path.splitext(fname)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_out, lbl_name))

    print(f"Split {n} images: train={n_train}, val={n_val}, test={n - n_train - n_val}")


def _create_dataset_yaml(output_root: str, yaml_path: str) -> None:
    """Create the dataset.yaml that Ultralytics expects."""
    data = {
        "path": os.path.abspath(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "drone"},
    }
    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    print(f"Dataset YAML written to {yaml_path}")


# -- Small-object augmentation ------------------------------------------------


def _copy_paste_small_objects(
    images_dir: str,
    labels_dir: str,
    max_pastes_per_image: int = 3,
    max_source_images: int = 50,
) -> int:
    """In-place copy-paste augmentation: crop small labelled objects from
    random images and paste them into other training images.

    This dramatically increases the density of small targets in the
    training set, which is the single most effective data-side trick for
    small-object detection.

    Returns the number of new label entries added.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("WARNING: opencv / numpy not available -- skipping copy-paste augmentation")
        return 0

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) < 2:
        return 0

    # Collect all small-object crops from the dataset
    crops: List[Dict] = []
    for fname in image_files[:max_source_images]:
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(labels_dir, lbl_name)
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(os.path.join(images_dir, fname))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        with open(lbl_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:])
                pw, ph = int(bw * iw), int(bh * ih)
                # Only collect genuinely small objects (< 40 px in both dims)
                if pw > 40 or ph > 40 or pw < 2 or ph < 2:
                    continue
                px = max(0, int((cx - bw / 2) * iw))
                py = max(0, int((cy - bh / 2) * ih))
                crop = img[py : py + ph, px : px + pw]
                if crop.size == 0:
                    continue
                crops.append({"patch": crop.copy(), "cls": cls_id})

    if not crops:
        print("  No small-object crops found -- skipping copy-paste")
        return 0

    added = 0
    for fname in image_files:
        img_path = os.path.join(images_dir, fname)
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(labels_dir, lbl_name)

        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        n_paste = random.randint(1, max_pastes_per_image)
        new_labels: List[str] = []

        for _ in range(n_paste):
            crop_info = random.choice(crops)
            patch = crop_info["patch"]
            ph, pw = patch.shape[:2]

            margin = 10
            if iw - pw - margin <= margin or ih - ph - margin <= margin:
                continue
            px = random.randint(margin, iw - pw - margin)
            py = random.randint(margin, ih - ph - margin)

            # Paste with slight brightness jitter
            jitter = random.uniform(0.8, 1.2)
            pasted = np.clip(patch.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
            img[py : py + ph, px : px + pw] = pasted

            cx_n = (px + pw / 2.0) / iw
            cy_n = (py + ph / 2.0) / ih
            bw_n = pw / iw
            bh_n = ph / ih
            new_labels.append(f"{crop_info['cls']} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")
            added += 1

        if new_labels:
            cv2.imwrite(img_path, img)
            with open(lbl_path, "a") as fh:
                for lbl in new_labels:
                    fh.write(lbl + "\n")

    print(f"  Copy-paste augmentation: added {added} small-object instances")
    return added


# -- Training ------------------------------------------------------------------


def _get_small_object_train_args(args: argparse.Namespace) -> dict:
    """Return Ultralytics training kwargs tuned for small-object detection."""
    return {
        "data": None,  # set by caller
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": "drone_tracker_small",
        # --- small-object-specific settings ---
        "augment": True,
        "mosaic": 1.0,           # full mosaic -- packs 4 images, increasing small-object density
        "copy_paste": 0.3,       # Ultralytics built-in copy-paste probability
        "mixup": 0.1,            # light mixup for regularisation
        "scale": 0.9,            # aggressive scale jitter so the model sees objects at many sizes
        "fliplr": 0.5,
        "flipud": 0.1,           # mild vertical flip
        "degrees": 15.0,         # rotation augmentation
        "translate": 0.2,
        "multi_scale": True,     # vary input resolution each batch
        "cos_lr": True,          # cosine LR schedule
        "patience": 30,          # early-stop patience (epochs)
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
    }


def _get_default_train_args(args: argparse.Namespace) -> dict:
    """Return standard Ultralytics training kwargs."""
    return {
        "data": None,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": "drone_tracker",
        "augment": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on labelled drone dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Standard training
  python train_yolo.py

  # Small-object mode (recommended for distant / few-pixel drones)
  python train_yolo.py --small-object --imgsz 1280 --epochs 200

  # Small-object mode with offline copy-paste data augmentation
  python train_yolo.py --small-object --copy-paste-aug --imgsz 1280
""",
    )

    parser.add_argument("--images", type=str, default="dataset/images",
                        help="Directory of labelled images")
    parser.add_argument("--labels", type=str, default="dataset/labels",
                        help="Directory of YOLO label .txt files")
    parser.add_argument("--output", type=str, default="dataset_split",
                        help="Root directory for train/val/test splits")
    parser.add_argument("--model", type=str, default=None,
                        help="Base YOLO model (default: yolov8n.pt, or yolov8s-p2.yaml with --small-object)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image size (default: 640, or 1280 with --small-object)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--weights-dir", type=str, default="models/",
                        help="Where to save the best weights")

    # Small-object flags
    parser.add_argument("--small-object", action="store_true",
                        help="Enable small-object detection optimisations "
                             "(higher res, P2 head, aggressive augmentation)")
    parser.add_argument("--copy-paste-aug", action="store_true",
                        help="Run offline copy-paste augmentation on the training "
                             "split before training (boosts small-object density)")

    args = parser.parse_args()

    # Apply small-object defaults when flags are not explicitly set
    if args.small_object:
        if args.imgsz is None:
            args.imgsz = 1280
        if args.model is None:
            args.model = "yolov8s-p2.yaml"
        print("Small-object mode enabled:")
        print(f"  imgsz      = {args.imgsz}")
        print(f"  model      = {args.model}")
        print(f"  copy-paste = {'ON' if args.copy_paste_aug else 'OFF (use --copy-paste-aug to enable)'}")
    else:
        if args.imgsz is None:
            args.imgsz = 640
        if args.model is None:
            args.model = "yolov8n.pt"

    # Step 1: Split dataset
    _split_dataset(args.images, args.labels, args.output)

    # Step 2: Create dataset YAML
    yaml_path = os.path.join(args.output, "dataset.yaml")
    _create_dataset_yaml(args.output, yaml_path)

    # Step 2b: Optional offline copy-paste augmentation on the training split
    if args.copy_paste_aug:
        train_imgs = os.path.join(args.output, "train", "images")
        train_lbls = os.path.join(args.output, "train", "labels")
        print("Running offline copy-paste augmentation on training split...")
        _copy_paste_small_objects(train_imgs, train_lbls)

    # Step 3: Train
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)

    if args.small_object:
        train_kwargs = _get_small_object_train_args(args)
    else:
        train_kwargs = _get_default_train_args(args)
    train_kwargs["data"] = yaml_path

    print(f"\nStarting training with {args.model}  imgsz={args.imgsz}  epochs={args.epochs}\n")
    results = model.train(**train_kwargs)

    # Step 4: Export best weights
    os.makedirs(args.weights_dir, exist_ok=True)
    best_path = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_path):
        dest = os.path.join(args.weights_dir, "best.pt")
        shutil.copy2(best_path, dest)
        print(f"Best weights saved to {dest}")
    else:
        print("WARNING: best.pt not found in training output.")

    print("Training complete.")


if __name__ == "__main__":
    main()
