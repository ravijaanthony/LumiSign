import argparse
import glob
import os
from typing import Dict, List, Set


SPLITS = ("train", "val", "test")


def load_split_uids(data_dir: str, dataset: str, split: str) -> List[str]:
    pattern = os.path.join(data_dir, f"{dataset}_{split}_keypoints", "*.json")
    file_paths = sorted(glob.glob(pattern))
    return [os.path.splitext(os.path.basename(path))[0] for path in file_paths]


def canonical_uid(uid: str, dark_suffix: str) -> str:
    if dark_suffix and uid.endswith(dark_suffix):
        return uid[: -len(dark_suffix)]
    return uid


def summarize_overlaps(split_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    return {
        "train_val": split_sets["train"] & split_sets["val"],
        "train_test": split_sets["train"] & split_sets["test"],
        "val_test": split_sets["val"] & split_sets["test"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check train/val/test leakage across canonical UIDs."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="directory containing <dataset>_{train,val,test}_keypoints folders",
    )
    parser.add_argument(
        "--dataset",
        default="isl_split_dataset",
        help="dataset prefix used in split folders (e.g. isl_split_dataset)",
    )
    parser.add_argument(
        "--dark_suffix",
        default="__dark",
        help="suffix used for train dark variants",
    )
    args = parser.parse_args()

    split_uids = {split: load_split_uids(args.data_dir, args.dataset, split) for split in SPLITS}
    for split in SPLITS:
        print(f"{split} JSON files: {len(split_uids[split])}")

    train_uids = split_uids["train"]
    train_dark_count = sum(1 for uid in train_uids if uid.endswith(args.dark_suffix))
    train_raw_count = len(train_uids) - train_dark_count

    print(f"raw train count: {train_raw_count}")
    print(f"dark train count: {train_dark_count}")

    val_dark = [uid for uid in split_uids["val"] if uid.endswith(args.dark_suffix)]
    test_dark = [uid for uid in split_uids["test"] if uid.endswith(args.dark_suffix)]

    canonical_sets = {
        split: {canonical_uid(uid, args.dark_suffix) for uid in uids}
        for split, uids in split_uids.items()
    }

    print("canonical split sizes:")
    for split in SPLITS:
        print(f"  {split}: {len(canonical_sets[split])}")

    overlaps = summarize_overlaps(canonical_sets)
    print("canonical overlap counts:")
    for name in ("train_val", "train_test", "val_test"):
        print(f"  {name}: {len(overlaps[name])}")

    failed = False
    if val_dark:
        failed = True
        print(
            f"ERROR: validation split contains {len(val_dark)} dark-suffixed UIDs. "
            f"Example: {val_dark[0]}"
        )
    if test_dark:
        failed = True
        print(
            f"ERROR: test split contains {len(test_dark)} dark-suffixed UIDs. "
            f"Example: {test_dark[0]}"
        )

    for name in ("train_val", "train_test", "val_test"):
        if overlaps[name]:
            failed = True
            sample = sorted(overlaps[name])[:5]
            print(
                f"ERROR: overlap detected for {name} ({len(overlaps[name])}). "
                f"Example canonical UIDs: {sample}"
            )

    if failed:
        print("Leakage check FAILED.")
        return 1

    print("Leakage check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
