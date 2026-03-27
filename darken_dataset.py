import argparse
import glob
import os
import hashlib

import cv2
import numpy as np
from tqdm.auto import tqdm

from video_preprocess import darken_frame


def scan_videos(include_dir: str) -> list[str]:
    exts = ("*.MOV", "*.mov", "*.MP4", "*.mp4", "*.AVI", "*.avi", "*.MKV", "*.mkv")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(include_dir, "*", ext)))
    return sorted(set(videos))


def _seed_from_path(path: str) -> int:
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def darken_video(
    src_path: str,
    dst_path: str,
    darken_min: float,
    darken_max: float,
    rng: np.random.Generator,
):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    writer = cv2.VideoWriter(
        dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    factor = float(rng.uniform(darken_min, darken_max))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = darken_frame(frame, factor)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Create a darkened copy of the dataset")
    parser.add_argument("--include_dir", required=True, help="path to original dataset")
    parser.add_argument("--output_dir", required=True, help="output folder for darkened dataset")
    parser.add_argument("--darken_min", default=0.3, type=float)
    parser.add_argument("--darken_max", default=0.8, type=float)
    parser.add_argument("--ratio", default=1.0, type=float, help="ratio of videos to darken (0.0-1.0)")
    args = parser.parse_args()

    videos = scan_videos(args.include_dir)
    if not videos:
        raise SystemExit("No videos found in include_dir")
    
    # Select a subset of videos to darken based on the specified ratio
    nums_to_darken = int(len(videos) * args.ratio)
    print(f"Selected {nums_to_darken} out of {len(videos)} videos to darken ({args.ratio*100}%)")
    
    #sort and use a fixed seed so the random selection is reprodusable if run again
    videos.sort()
    rng = np.random.default_rng(42)
    videos_to_darken = rng.choice(videos, size=nums_to_darken, replace=False)

    for src in tqdm(videos, desc="Darkening videos"):
        label = os.path.basename(os.path.dirname(src))
        dst = os.path.join(args.output_dir, label, os.path.basename(src))
        rng = np.random.default_rng(_seed_from_path(src))
        darken_video(src, dst, args.darken_min, args.darken_max, rng)


if __name__ == "__main__":
    main()
