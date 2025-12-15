#!/usr/bin/env python3
"""Dataset preparation scripts for emotion detection.

This script helps download and preprocess CK+ and RAF-DB datasets
to match the FER-2013 directory structure.

Usage:
    python prepare_datasets.py --dataset ckplus --input /path/to/ckplus --output data/ckplus
    python prepare_datasets.py --dataset rafdb --input /path/to/rafdb --output data/rafdb
    python prepare_datasets.py --dataset verify --input data/fer2013
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Standard class names (matching FER-2013)
CLASS_NAMES = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

# CK+ emotion labels (from FACS coding)
# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
CKPLUS_LABEL_MAP = {
    0: "neutral",
    1: "angry",
    2: None,  # contempt - exclude
    3: "disgusted",
    4: "fearful",
    5: "happy",
    6: "sad",
    7: "surprised",
}

# RAF-DB labels (1-indexed)
RAFDB_LABEL_MAP = {
    1: "surprised",
    2: "fearful",
    3: "disgusted",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}


def resize_image(img, target_size = (48, 48)):
    """Resize image to target size."""
    return img.resize(target_size, Image.BILINEAR)


def convert_to_grayscale(img):
    """Convert image to grayscale."""
    return img.convert("L")


def process_ckplus(
    input_dir,
    output_dir,
    target_size = (48, 48),
    val_ratio = 0.2,
):
    """Process CK+ dataset.
    
    Expected input structure (original):
        input_dir/
            cohn-kanade-images/
                S005/
                    001/
                        S005_001_00000001.png
                        ...
            Emotion/
                S005/
                    001/
                        S005_001_00000011_emotion.txt
    
    Or Kaggle format:
        input_dir/
            angry/
            disgusted/
            ...
    """
    print(f"Processing CK+ from {input_dir}")
    
    images_dir = input_dir / "cohn-kanade-images"
    emotions_dir = input_dir / "Emotion"
    
    if not images_dir.exists():
        # Try Kaggle/preprocessed structure
        if (input_dir / "angry").exists() or (input_dir / "Angry").exists():
            _process_ckplus_kaggle(input_dir, output_dir, target_size, val_ratio)
            return
        # Try CK+48 subdirectory
        for subdir in ["CK+48", "ck+", "CK+"]:
            if (input_dir / subdir).exists():
                _process_ckplus_kaggle(input_dir / subdir, output_dir, target_size, val_ratio)
                return
        raise FileNotFoundError(f"Could not find CK+ images in {input_dir}")
    
    # Process original CK+ format
    samples_by_class = defaultdict(list)
    
    for subject_dir in sorted(images_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        for sequence_dir in sorted(subject_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue
            
            # Find emotion label
            emotion_file = emotions_dir / subject_dir.name / sequence_dir.name
            emotion_files = list(emotion_file.glob("*_emotion.txt"))
            
            if not emotion_files:
                continue
            
            # Read emotion label
            label_text = emotion_files[0].read_text().strip()
            try:
                label_idx = int(float(label_text))
            except ValueError:
                continue
            
            class_name = CKPLUS_LABEL_MAP.get(label_idx)
            if class_name is None:
                continue  # Skip contempt
            
            # Get peak expression frame (last frame)
            image_files = sorted(sequence_dir.glob("*.png"))
            if image_files:
                samples_by_class[class_name].append(image_files[-1])
            
            # Also get neutral from first few frames
            if len(image_files) >= 3:
                samples_by_class["neutral"].append(image_files[0])
    
    # Split and save
    _save_dataset(samples_by_class, output_dir, target_size, val_ratio)
    
    print(f"CK+ processing complete!")
    for cls, samples in samples_by_class.items():
        print(f"  {cls}: {len(samples)} images")


def _process_ckplus_kaggle(
    images_dir,
    output_dir,
    target_size,
    val_ratio,
):
    """Process pre-organized CK+ from Kaggle."""
    print("Processing CK+ (Kaggle/preprocessed format)")
    
    samples_by_class = defaultdict(list)
    
    for class_dir in images_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name.lower()
        # Handle different naming conventions
        class_name = class_name.replace("disgust", "disgusted")
        class_name = class_name.replace("fear", "fearful")
        class_name = class_name.replace("surprise", "surprised")
        class_name = class_name.replace("sadness", "sad")
        class_name = class_name.replace("anger", "angry")
        class_name = class_name.replace("happiness", "happy")
        
        if class_name not in CLASS_NAMES:
            print(f"  Skipping unknown class: {class_dir.name}")
            continue
        
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                samples_by_class[class_name].append(img_path)
    
    _save_dataset(samples_by_class, output_dir, target_size, val_ratio)


def process_rafdb(
    input_dir,
    output_dir,
    target_size = (48, 48),
):
    """Process RAF-DB dataset.
    
    Expected input structure (original):
        input_dir/
            basic/
                Image/
                    aligned/
                        train_00001_aligned.jpg
                        test_00001_aligned.jpg
                        ...
                EmoLabel/
                    list_patition_label.txt
    
    Or Kaggle format:
        input_dir/
            train/
                angry/
                ...
            test/
                angry/
                ...
    """
    print(f"Processing RAF-DB from {input_dir}")
    
    # Try Kaggle format first (easier)
    if (input_dir / "train").exists():
        _process_rafdb_kaggle(input_dir, output_dir, target_size)
        return
    
    # Try different possible structures for original format
    aligned_dir = None
    for possible_path in [
        input_dir / "basic" / "Image" / "aligned",
        input_dir / "aligned",
        input_dir / "Image" / "aligned",
    ]:
        if possible_path.exists():
            aligned_dir = possible_path
            break
    
    if aligned_dir is None:
        raise FileNotFoundError(f"Could not find RAF-DB aligned images in {input_dir}")
    
    # Find label file
    label_file = None
    for possible_path in [
        input_dir / "basic" / "EmoLabel" / "list_patition_label.txt",
        input_dir / "list_patition_label.txt",
        input_dir / "EmoLabel" / "list_patition_label.txt",
    ]:
        if possible_path.exists():
            label_file = possible_path
            break
    
    if label_file is None:
        raise FileNotFoundError(f"Could not find RAF-DB label file")
    
    # Read labels
    labels = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                label_idx = int(parts[1])
                labels[img_name] = label_idx
                # Also store aligned version
                labels[img_name.replace(".jpg", "_aligned.jpg")] = label_idx
    
    # Collect samples
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)
    
    for img_path in aligned_dir.glob("*.jpg"):
        img_name = img_path.name
        
        # Try to find label
        label_idx = labels.get(img_name)
        if label_idx is None:
            base_name = img_name.replace("_aligned.jpg", ".jpg")
            label_idx = labels.get(base_name)
        
        if label_idx is None:
            continue
        
        class_name = RAFDB_LABEL_MAP.get(label_idx)
        if class_name is None:
            continue
        
        if img_name.startswith("train_"):
            train_samples[class_name].append(img_path)
        elif img_name.startswith("test_"):
            test_samples[class_name].append(img_path)
    
    # Save datasets
    for split, samples_dict in [("train", train_samples), ("test", test_samples)]:
        for class_name, samples in samples_dict.items():
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i, src_path in enumerate(samples):
                dst_path = class_dir / f"rafdb_{i:05d}.png"
                _process_and_save(src_path, dst_path, target_size)
    
    print(f"RAF-DB processing complete!")
    print(f"Train samples:")
    for cls, samples in sorted(train_samples.items()):
        print(f"  {cls}: {len(samples)}")
    print(f"Test samples:")
    for cls, samples in sorted(test_samples.items()):
        print(f"  {cls}: {len(samples)}")


def _process_rafdb_kaggle(
    input_dir,
    output_dir,
    target_size,
):
    """Process RAF-DB from Kaggle format."""
    print("Processing RAF-DB (Kaggle format)")
    
    for split in ["train", "test"]:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name.lower()
            # Handle different naming conventions
            class_name = class_name.replace("disgust", "disgusted")
            class_name = class_name.replace("fear", "fearful")
            class_name = class_name.replace("surprise", "surprised")
            class_name = class_name.replace("sadness", "sad")
            class_name = class_name.replace("anger", "angry")
            class_name = class_name.replace("happiness", "happy")
            
            if class_name not in CLASS_NAMES:
                print(f"  Skipping unknown class: {class_dir.name}")
                continue
            
            out_class_dir = output_dir / split / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    dst_path = out_class_dir / f"rafdb_{count:05d}.png"
                    _process_and_save(img_path, dst_path, target_size)
                    count += 1
            
            print(f"  {split}/{class_name}: {count} images")


def _save_dataset(
    samples_by_class,
    output_dir,
    target_size,
    val_ratio = 0.2,
):
    """Split and save dataset."""
    rng = np.random.default_rng(42)
    
    for class_name, samples in samples_by_class.items():
        samples = samples.copy()
        rng.shuffle(samples)
        
        n_test = max(1, int(len(samples) * val_ratio))
        if len(samples) < 5:
            # Too few samples - all to train
            train_samples = samples
            test_samples = []
        else:
            test_samples = samples[:n_test]
            train_samples = samples[n_test:]
        
        # Save train
        train_dir = output_dir / "train" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        for i, src_path in enumerate(train_samples):
            dst_path = train_dir / f"{class_name}_{i:05d}.png"
            _process_and_save(src_path, dst_path, target_size)
        
        # Save test
        if test_samples:
            test_dir = output_dir / "test" / class_name
            test_dir.mkdir(parents=True, exist_ok=True)
            for i, src_path in enumerate(test_samples):
                dst_path = test_dir / f"{class_name}_{i:05d}.png"
                _process_and_save(src_path, dst_path, target_size)
        
        print(f"  {class_name}: {len(train_samples)} train, {len(test_samples)} test")


def _process_and_save(
    src_path,
    dst_path,
    target_size,
):
    """Load, process, and save a single image."""
    try:
        with Image.open(src_path) as img:
            img = convert_to_grayscale(img)
            img = resize_image(img, target_size)
            img.save(dst_path)
    except Exception as e:
        print(f"Warning: Could not process {src_path}: {e}")


def verify_dataset(data_dir):
    """Verify dataset structure and print statistics."""
    print(f"\nVerifying dataset at {data_dir}")
    
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  {split}: NOT FOUND")
            continue
        
        print(f"\n  {split}:")
        total = 0
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
                print(f"    {class_name}: {count}")
                total += count
            else:
                print(f"    {class_name}: 0 (missing)")
        print(f"    TOTAL: {total}")


def main():
    parser = argparse.ArgumentParser(description="Prepare emotion detection datasets")
    parser.add_argument(
        "--dataset",
        choices=["ckplus", "rafdb", "verify"],
        required=True,
        help="Dataset to process",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory with raw dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=48,
        help="Target image size (default: 48)",
    )
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    if args.dataset == "ckplus":
        if not args.input or not args.output:
            parser.error("--input and --output required for ckplus")
        process_ckplus(args.input, args.output, target_size)
        verify_dataset(args.output)
    
    elif args.dataset == "rafdb":
        if not args.input or not args.output:
            parser.error("--input and --output required for rafdb")
        process_rafdb(args.input, args.output, target_size)
        verify_dataset(args.output)
    
    elif args.dataset == "verify":
        if not args.input:
            parser.error("--input required for verify")
        verify_dataset(args.input)


if __name__ == "__main__":
    main()
