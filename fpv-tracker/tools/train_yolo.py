"""Train a YOLOv8 model on the labelled drone dataset."""

import argparse
import os
import random
import shutil
import sys

import yaml


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on labelled drone dataset")
    parser.add_argument("--images", type=str, default="dataset/images",
                        help="Directory of labelled images")
    parser.add_argument("--labels", type=str, default="dataset/labels",
                        help="Directory of YOLO label .txt files")
    parser.add_argument("--output", type=str, default="dataset_split",
                        help="Root directory for train/val/test splits")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base YOLO model (nano, small, etc.)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--weights-dir", type=str, default="models/",
                        help="Where to save the best weights")
    args = parser.parse_args()

    # Step 1: Split dataset
    _split_dataset(args.images, args.labels, args.output)

    # Step 2: Create dataset YAML
    yaml_path = os.path.join(args.output, "dataset.yaml")
    _create_dataset_yaml(args.output, yaml_path)

    # Step 3: Train
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="drone_tracker",
        augment=True,
    )

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
