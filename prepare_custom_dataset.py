import argparse
import hashlib
import json
import os
import shutil
import tempfile
from typing import Optional

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from generate_keypoints import process_video, tqdm_joblib


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
TARGET_SPLITS = ("train", "val", "test")


def normalize_label(raw_label: str) -> str:
    label = "".join([c for c in raw_label if c.isalpha()]).lower()
    if not label:
        raise ValueError(f"Label '{raw_label}' becomes empty after normalization.")
    return label


def resolve_split_dir(data_dir: str, split: str) -> Optional[str]:
    direct = os.path.join(data_dir, split)
    if os.path.isdir(direct):
        return direct

    # Common alternative used in some datasets.
    if split == "val":
        alt = os.path.join(data_dir, "eval")
        if os.path.isdir(alt):
            return alt
    return None


def collect_split_samples(split_dir: str):
    """
    Recursively collects videos from:
      <split_dir>/<class_name>/**/<video_file>

    Class label is the first folder under split_dir.
    """
    samples = []
    for root, _, files in os.walk(split_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, split_dir)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                # Must contain at least: class_name/file
                continue
            class_dir = parts[0]
            label = normalize_label(class_dir)
            samples.append((src, label))
    return sorted(samples)


def link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def build_flat_class_layout(samples, split_tmp_dir: str):
    """
    Build a temporary flat layout:
      <tmp>/<normalized_class>/<unique_video_name.ext>
    so process_video() reads the correct class name from parent folder.
    """
    flat_paths = []
    for src, label in samples:
        stem, ext = os.path.splitext(os.path.basename(src))
        digest = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
        dst = os.path.join(split_tmp_dir, label, f"{stem}_{digest}{ext}")
        link_or_copy(src, dst)
        flat_paths.append(dst)
    return flat_paths


def save_label_map(dataset_name: str, labels) -> str:
    label_map_dir = "label_maps"
    os.makedirs(label_map_dir, exist_ok=True)
    sorted_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}

    # This naming is what load_label_map(dataset) expects.
    label_map_path = os.path.join(label_map_dir, f"label_map_{dataset_name}.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    return label_map_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert nested custom dataset into keypoint JSON split folders compatible "
            "with runner.py"
        )
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path containing train/val/test (or eval instead of val).",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Output root for <dataset_name>_{train,val,test}_keypoints.",
    )
    parser.add_argument(
        "--dataset_name",
        default="custom",
        help="Dataset prefix used by runner.py (e.g., custom).",
    )
    parser.add_argument("--jobs", default=4, type=int, help="Parallel workers.")
    parser.add_argument(
        "--use_holistic",
        action="store_true",
        help="Use MediaPipe Holistic (pose+hands+face).",
    )
    parser.add_argument(
        "--face_mode",
        default="full",
        choices=["none", "eyebrows", "full"],
        help="Face landmarks mode when --use_holistic is set.",
    )
    parser.add_argument(
        "--write_placeholders",
        action="store_true",
        help="Write placeholder JSONs when a video fails to decode.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    all_labels = []
    temp_roots = []
    split_counts = {}

    try:
        for split in TARGET_SPLITS:
            split_dir = resolve_split_dir(args.data_dir, split)
            if split_dir is None:
                print(f"Warning: split '{split}' not found under {args.data_dir}. Skipping.")
                continue

            print(f"\n--- Scanning split: {split} ({split_dir}) ---")
            samples = collect_split_samples(split_dir)
            if not samples:
                print(f"Warning: no videos found for split '{split}'.")
                continue

            labels = [label for _, label in samples]
            all_labels.extend(labels)
            split_counts[split] = len(samples)

            split_save_dir = os.path.join(
                args.save_dir, f"{args.dataset_name}_{split}_keypoints"
            )
            os.makedirs(split_save_dir, exist_ok=True)

            # Flatten to ensure class label is the direct parent folder.
            split_tmp_dir = tempfile.mkdtemp(prefix=f"flat_{split}_")
            temp_roots.append(split_tmp_dir)
            flat_paths = build_flat_class_layout(samples, split_tmp_dir)

            face_mode = args.face_mode if args.use_holistic else "none"
            print(f"Found {len(flat_paths)} videos for {split}. Extracting keypoints...")

            with tqdm_joblib(tqdm(total=len(flat_paths), desc=f"Extracting {split}")):
                Parallel(n_jobs=args.jobs, backend="multiprocessing")(
                    delayed(process_video)(
                        path=path,
                        save_dir=split_save_dir,
                        use_holistic=args.use_holistic,
                        face_mode=face_mode,
                        write_placeholders=args.write_placeholders,
                    )
                    for path in flat_paths
                )

        if not all_labels:
            raise SystemExit("No videos found in any split. Nothing to process.")

        label_map_path = save_label_map(args.dataset_name, all_labels)

        print("\nDone.")
        for split in TARGET_SPLITS:
            if split in split_counts:
                print(f"  {split}: {split_counts[split]} videos")
        print(f"  label map: {label_map_path}")
        print(
            f"\nNext: run runner.py with --dataset {args.dataset_name} "
            f"and --data_dir {args.save_dir}"
        )

    finally:
        for tmp in temp_roots:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
