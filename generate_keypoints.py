import os
import json
import glob
import multiprocessing
import argparse
import os.path
import shutil
import cv2
import mediapipe as mp
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import numpy as np
import gc
import warnings
import hashlib
from contextlib import contextmanager
import joblib

from video_preprocess import PreprocessConfig, apply_darken_then_brighten


# MediaPipe FaceMesh landmark indices (subset) for eyebrows.
# These are commonly used indices for left/right eyebrows.
_EYEBROW_IDXS = [
    # left eyebrow
    70,
    63,
    105,
    66,
    107,
    55,
    65,
    52,
    53,
    46,
    # right eyebrow
    336,
    296,
    334,
    293,
    300,
    276,
    283,
    282,
    295,
    285,
]


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib so tqdm reports completed tasks (not just dispatched)."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _label_from_path(path: str) -> str:
    # Assumes dataset layout: <include_dir>/<label>/<video_file>
    label = path.split("/")[-2]
    return "".join([i for i in label if i.isalpha()]).lower()


def _uid_from_path(path: str, label: str) -> str:
    uid = os.path.splitext(os.path.basename(path))[0]
    return "_".join([label, uid])


def _try_open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap

    # Try forcing FFmpeg backend if available.
    if hasattr(cv2, "CAP_FFMPEG"):
        cap2 = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if cap2.isOpened():
            return cap2

    return cap


def _pose25_from_mp_pose(pose_landmarks):
    if pose_landmarks is None:
        return [np.nan] * 25, [np.nan] * 25

    lm = pose_landmarks.landmark

    def get_xy(i: int):
        return lm[i].x, lm[i].y

    def avg_xy(a: int, b: int):
        ax, ay = get_xy(a)
        bx, by = get_xy(b)
        return (ax + bx) / 2.0, (ay + by) / 2.0

    # Map MediaPipe Pose (33) -> OpenPose BODY_25-like layout used by this repo (25).
    # 0 Nose
    nose = get_xy(0)
    # 1 Neck (approx: midpoint of shoulders)
    neck = avg_xy(11, 12)

    rshoulder = get_xy(12)
    relbow = get_xy(14)
    rwrist = get_xy(16)

    lshoulder = get_xy(11)
    lelbow = get_xy(13)
    lwrist = get_xy(15)

    midhip = avg_xy(23, 24)

    rhip = get_xy(24)
    rknee = get_xy(26)
    rankle = get_xy(28)

    lhip = get_xy(23)
    lknee = get_xy(25)
    lankle = get_xy(27)

    # Eyes / ears
    reye = get_xy(5)
    leye = get_xy(2)
    rear = get_xy(8)
    lear = get_xy(7)

    # Feet (MediaPipe doesn't provide small-toe separately; we duplicate foot_index)
    lbigtoe = get_xy(31)
    lsmalltoe = get_xy(31)
    lheel = get_xy(29)

    rbigtoe = get_xy(32)
    rsmalltoe = get_xy(32)
    rheel = get_xy(30)

    points = [
        nose,
        neck,
        rshoulder,
        relbow,
        rwrist,
        lshoulder,
        lelbow,
        lwrist,
        midhip,
        rhip,
        rknee,
        rankle,
        lhip,
        lknee,
        lankle,
        reye,
        leye,
        rear,
        lear,
        lbigtoe,
        lsmalltoe,
        lheel,
        rbigtoe,
        rsmalltoe,
        rheel,
    ]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return xs, ys


def _hand21_from_landmarks(hand_landmarks):
    if hand_landmarks is None:
        return [np.nan] * 21, [np.nan] * 21
    xs, ys = [], []
    for landmark in hand_landmarks.landmark:
        xs.append(landmark.x)
        ys.append(landmark.y)
    return xs, ys


def _face_from_landmarks(face_landmarks, face_mode: str):
    if face_mode == "none":
        return None, None

    if face_mode == "full":
        idxs = None
        n = 468
    elif face_mode == "eyebrows":
        idxs = _EYEBROW_IDXS
        n = len(_EYEBROW_IDXS)
    else:
        raise ValueError(f"Unsupported face_mode: {face_mode}")

    if face_landmarks is None:
        return [np.nan] * n, [np.nan] * n

    lm = face_landmarks.landmark
    if idxs is None:
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        return xs, ys

    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x)
        ys.append(lm[i].y)
    return xs, ys


def process_video(
    path: str,
    save_dir: str,
    *,
    use_holistic: bool = False,
    face_mode: str = "none",
    write_placeholders: bool = False,
    preprocess_config: PreprocessConfig | None = None,
    export_darkened_dir: str | None = None,
    uid_suffix: str = "",
):
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "mediapipe does not expose mp.solutions. "
            "This usually means an incompatible mediapipe build. "
            "Use Python 3.9/3.10 and reinstall mediapipe==0.10.14."
        )
    if not os.path.isfile(path):
        warnings.warn(path + " file not found")
        return

    label = _label_from_path(path)
    uid = _uid_from_path(path, label)
    if uid_suffix:
        uid = f"{uid}{uid_suffix}"

    cap = _try_open_video(path)
    if not cap.isOpened():
        warnings.warn(f"OpenCV could not open video: {path}")
        return

    pose_points_x, pose_points_y = [], []
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []
    face_points_x, face_points_y = [], []

    n_frames = 0
    rng = None
    darken_factor = None
    writer = None

    if preprocess_config is not None:
        seed = int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        if preprocess_config.apply_darken:
            darken_factor = float(
                rng.uniform(preprocess_config.darken_min, preprocess_config.darken_max)
            )

    if export_darkened_dir:
        os.makedirs(export_darkened_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(export_darkened_dir, f"{uid}.mp4")
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    if use_holistic:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                if preprocess_config is not None:
                    image = apply_darken_then_brighten(
                        image,
                        config=preprocess_config,
                        rng=rng,
                        darken_factor=darken_factor,
                    )
                if writer is not None:
                    writer.write(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                pose_x, pose_y = _pose25_from_mp_pose(results.pose_landmarks)
                # Use holistic's left/right hands (subject perspective)
                h1x, h1y = _hand21_from_landmarks(results.left_hand_landmarks)
                h2x, h2y = _hand21_from_landmarks(results.right_hand_landmarks)

                fx, fy = _face_from_landmarks(results.face_landmarks, face_mode)

                pose_points_x.append(pose_x)
                pose_points_y.append(pose_y)
                hand1_points_x.append(h1x)
                hand1_points_y.append(h1y)
                hand2_points_x.append(h2x)
                hand2_points_y.append(h2y)

                if face_mode != "none":
                    face_points_x.append(fx)
                    face_points_y.append(fy)

                n_frames += 1
        finally:
            holistic.close()
    else:
        # Legacy path: Pose + Hands separately (kept for backward compatibility).
        hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                if preprocess_config is not None:
                    image = apply_darken_then_brighten(
                        image,
                        config=preprocess_config,
                        rng=rng,
                        darken_factor=darken_factor,
                    )
                if writer is not None:
                    writer.write(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                hand_results = hands.process(image)
                pose_results = pose.process(image)

                # Hands
                hand1 = (
                    hand_results.multi_hand_landmarks[0]
                    if hand_results.multi_hand_landmarks
                    else None
                )
                hand2 = (
                    hand_results.multi_hand_landmarks[1]
                    if hand_results.multi_hand_landmarks
                    and len(hand_results.multi_hand_landmarks) > 1
                    else None
                )
                h1x, h1y = _hand21_from_landmarks(hand1)
                h2x, h2y = _hand21_from_landmarks(hand2)

                # Pose (mapped to 25)
                pose_x, pose_y = _pose25_from_mp_pose(pose_results.pose_landmarks)

                pose_points_x.append(pose_x)
                pose_points_y.append(pose_y)
                hand1_points_x.append(h1x)
                hand1_points_y.append(h1y)
                hand2_points_x.append(h2x)
                hand2_points_y.append(h2y)

                n_frames += 1
        finally:
            hands.close()
            pose.close()

    cap.release()
    if writer is not None:
        writer.release()

    if n_frames == 0 and not write_placeholders:
        warnings.warn(f"No frames decoded (n_frames=0). Skipping: {path}")
        return

    # If no frames decoded but placeholders requested, mimic previous behavior.
    if n_frames == 0 and write_placeholders:
        pose_points_x = [[np.nan] * 25]
        pose_points_y = [[np.nan] * 25]
        hand1_points_x = [[np.nan] * 21]
        hand1_points_y = [[np.nan] * 21]
        hand2_points_x = [[np.nan] * 21]
        hand2_points_y = [[np.nan] * 21]
        if face_mode != "none":
            n_face = 468 if face_mode == "full" else len(_EYEBROW_IDXS)
            face_points_x = [[np.nan] * n_face]
            face_points_y = [[np.nan] * n_face]

    save_data = {
        "uid": uid,
        "label": label,
        "pose_x": pose_points_x,
        "pose_y": pose_points_y,
        "hand1_x": hand1_points_x,
        "hand1_y": hand1_points_y,
        "hand2_x": hand2_points_x,
        "hand2_y": hand2_points_y,
        "n_frames": n_frames,
    }
    if face_mode != "none":
        save_data["face_x"] = face_points_x
        save_data["face_y"] = face_points_y

    with open(os.path.join(save_dir, f"{uid}.json"), "w") as f:
        json.dump(save_data, f)

    del save_data
    gc.collect()


def _build_label_dir_map(include_dir: str) -> dict[str, str]:
    """Maps lowercased directory name -> actual directory name."""
    mapping: dict[str, str] = {}
    try:
        for name in os.listdir(include_dir):
            full = os.path.join(include_dir, name)
            if os.path.isdir(full):
                mapping[name.lower()] = name
                cleaned = ''.join([c for c in name if c.isalpha()]).lower()
                if cleaned and cleaned not in mapping:
                    mapping[cleaned] = name
    except FileNotFoundError:
        return mapping
    return mapping


def _resolve_video_path(include_dir: str, rel_path: str, label_dir_map: dict[str, str]) -> str:
    """Resolves train_test_paths entries against different dataset folder layouts.

    Supports:
    1) Official INCLUDE layout: <include_dir>/<Category>/<N. Label>/<file>
    2) Flat label layout:      <include_dir>/<Label>/<file>
    """
    direct = os.path.join(include_dir, rel_path)
    if os.path.isfile(direct):
        return direct

    # Try <include_dir>/<label>/<filename>
    parts = [p for p in rel_path.split('/') if p]
    filename = os.path.basename(rel_path)
    label = None
    if len(parts) >= 2:
        label_part = parts[-2]
        label = ''.join([c for c in label_part if c.isalpha()]).lower()

    if label:
        dir_name = label_dir_map.get(label)
        if dir_name:
            cand = os.path.join(include_dir, dir_name, filename)
            if os.path.isfile(cand):
                return cand

        # Sometimes folders are title-cased (e.g., Court)
        for variant in (label.capitalize(), label.title()):
            dir_name = label_dir_map.get(variant.lower())
            if dir_name:
                cand = os.path.join(include_dir, dir_name, filename)
                if os.path.isfile(cand):
                    return cand

    return direct


def load_file(path, include_dir):
    with open(path, "r") as fp:
        rel_paths = [ln.strip() for ln in fp.read().splitlines() if ln.strip()]

    label_dir_map = _build_label_dir_map(include_dir)
    return [_resolve_video_path(include_dir, rel, label_dir_map) for rel in rel_paths]


def scan_include_dir_videos(include_dir: str) -> list[str]:
    """Scans include_dir for videos in a flat label-folder layout.

    Expected layout:
      <include_dir>/<Label>/*.MOV (or mp4/avi/mkv)
    """
    exts = ("*.MOV", "*.mov", "*.MP4", "*.mp4", "*.AVI", "*.avi", "*.MKV", "*.mkv")
    videos: list[str] = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(include_dir, "*", ext)))
    # Deduplicate + stable order
    return sorted(set(videos))


def split_paths(
    paths: list[str], seed: int = 0, train_ratio: float = 0.8, val_ratio: float = 0.1
):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            f"train_ratio must be in (0, 1), got {train_ratio}. "
            "Example: --train_ratio 0.7"
        )
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(
            f"val_ratio must be in [0, 1), got {val_ratio}. "
            "Example: --val_ratio 0.15"
        )
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio:.4f}."
        )
    if not paths:
        return [], [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    paths = [paths[i] for i in idx]

    n = len(paths)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]
    return train, val, test


def _hash_fraction(seed: int, path: str) -> float:
    digest = hashlib.md5(f"{seed}|{path}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return value / float(2**32)


def select_paths_by_probability(
    paths: list[str], *, probability: float, seed: int
) -> list[str]:
    if probability <= 0.0:
        return []
    if probability >= 1.0:
        return list(paths)
    return [p for p in paths if _hash_fraction(seed, p) < probability]


def load_train_test_val_paths(include_dir: str, dataset: str):
    train_paths = load_file(f"train_test_paths/{dataset}_train.txt", include_dir)
    val_paths = load_file(f"train_test_paths/{dataset}_val.txt", include_dir)
    test_paths = load_file(f"train_test_paths/{dataset}_test.txt", include_dir)
    return train_paths, val_paths, test_paths


def save_keypoints(
    *,
    dataset: str,
    file_paths: list[str],
    mode: str,
    save_root: str,
    n_jobs: int,
    use_holistic: bool,
    face_mode: str,
    limit: int,
    no_parallel: bool,
    write_placeholders: bool,
    preprocess_config: PreprocessConfig | None,
    export_darkened_dir: str | None,
    uid_suffix: str = "",
):
    save_dir = os.path.join(save_root, f"{dataset}_{mode}_keypoints")
    os.makedirs(save_dir, exist_ok=True)

    if limit and limit > 0:
        file_paths = file_paths[:limit]

    existing = [p for p in file_paths if os.path.isfile(p)]
    missing = len(file_paths) - len(existing)
    if missing:
        warnings.warn(f"{missing} {mode} videos missing on disk (skipping them)")

    if no_parallel or n_jobs == 1:
        for path in tqdm(existing, desc=f"processing {mode} videos"):
            process_video(
                path,
                save_dir,
                use_holistic=use_holistic,
                face_mode=face_mode,
                write_placeholders=write_placeholders,
                preprocess_config=preprocess_config,
                export_darkened_dir=export_darkened_dir,
                uid_suffix=uid_suffix,
            )
        return

    if not existing:
        return

    with tqdm_joblib(tqdm(total=len(existing), desc=f"processing {mode} videos")):
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(process_video)(
                path,
                save_dir,
                use_holistic=use_holistic,
                face_mode=face_mode,
                write_placeholders=write_placeholders,
                preprocess_config=preprocess_config,
                export_darkened_dir=export_darkened_dir,
                uid_suffix=uid_suffix,
            )
            for path in existing
        )


def preflight_videos(video_paths: list[str], n: int = 10):
    """Quickly checks existence + OpenCV decode for a small sample."""
    sample = video_paths[: max(0, n)]
    exists = 0
    opened = 0
    first_read_ok = 0

    for path in sample:
        if not os.path.isfile(path):
            continue
        exists += 1
        cap = _try_open_video(path)
        if not cap.isOpened():
            cap.release()
            continue
        opened += 1
        ok, _frame = cap.read()
        if ok:
            first_read_ok += 1
        cap.release()

    print('Preflight results:')
    print(f'  sample size: {len(sample)}')
    print(f'  exists on disk: {exists}')
    print(f'  OpenCV opened: {opened}')
    print(f'  first frame decoded: {first_read_ok}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keypoints from MediaPipe")
    parser.add_argument(
        "--include_dir",
        default="",
        type=str,
        required=True,
        help="path to the location of INCLUDE/INCLUDE50 videos",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        required=True,
        help="location to output json files",
    )
    parser.add_argument(
        "--dataset", default="include", type=str, help="options: include or include50"
    )
    parser.add_argument(
        "--video",
        default=None,
        type=str,
        help="process a single video path (bypasses train_test_paths)",
    )
    parser.add_argument(
        "--use_holistic",
        action="store_true",
        help="use MediaPipe Holistic (pose+hands+face) instead of separate pose/hands",
    )
    parser.add_argument(
        "--face_mode",
        default="none",
        choices=["none", "eyebrows", "full"],
        help="face landmarks to save when using holistic",
    )
    parser.add_argument(
        "--splits",
        default="all",
        choices=["train", "val", "test", "all"],
        help="which splits to process",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="process only the first N videos per split (useful for testing)",
    )
    parser.add_argument(
        "--jobs",
        default=multiprocessing.cpu_count(),
        type=int,
        help="number of parallel workers",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="disable multiprocessing (easier debugging)",
    )
    parser.add_argument(
        "--write_placeholders",
        action="store_true",
        help="write placeholder NaN JSONs even when videos can't be decoded",
    )
    parser.add_argument(
        "--apply_darken",
        action="store_true",
        help="apply darken then brighten preprocessing before MediaPipe",
    )
    parser.add_argument(
        "--apply_brighten",
        action="store_true",
        help="apply brighten preprocessing without darkening",
    )
    parser.add_argument(
        "--darken_min",
        default=0.3,
        type=float,
        help="minimum brightness multiplier for darkening",
    )
    parser.add_argument(
        "--darken_max",
        default=0.8,
        type=float,
        help="maximum brightness multiplier for darkening",
    )
    parser.add_argument(
        "--brighten_method",
        default="clahe",
        choices=["clahe", "gamma"],
        help="brighten method used after darkening",
    )
    parser.add_argument(
        "--brighten_gamma_min",
        default=1.2,
        type=float,
        help="minimum gamma for gamma brighten",
    )
    parser.add_argument(
        "--brighten_gamma_max",
        default=1.8,
        type=float,
        help="maximum gamma for gamma brighten",
    )
    parser.add_argument(
        "--export_darkened_dir",
        default=None,
        type=str,
        help="optional directory to save darkened+brightened videos",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="only check that videos exist and can be decoded, then exit",
    )
    parser.add_argument(
        "--preflight_n",
        default=10,
        type=int,
        help="number of videos to check in preflight",
    )

    parser.add_argument(
        "--scan",
        action="store_true",
        help="ignore train_test_paths and scan include_dir/<label>/* for videos, then make a random train/val/test split",
    )
    parser.add_argument(
        "--split_seed",
        default=0,
        type=int,
        help="seed used when --scan creates train/val/test splits",
    )
    parser.add_argument(
        "--train_ratio",
        default=0.8,
        type=float,
        help="train split ratio used with --scan",
    )
    parser.add_argument(
        "--val_ratio",
        default=0.1,
        type=float,
        help="validation split ratio used with --scan",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="delete selected split output folders before generating keypoints",
    )
    parser.add_argument(
        "--train_dark_prob",
        default=0.0,
        type=float,
        help="probability of creating an extra darkened sample for each train video",
    )
    parser.add_argument(
        "--train_dark_suffix",
        default="__dark",
        type=str,
        help="suffix added to uid for train darkened variants",
    )
    parser.add_argument(
        "--train_dark_seed",
        default=None,
        type=int,
        help="seed for selecting train videos for dark variants (defaults to --split_seed)",
    )

    args = parser.parse_args()

    if not (0.0 <= args.train_dark_prob <= 1.0):
        raise ValueError(
            f"--train_dark_prob must be in [0, 1], got {args.train_dark_prob}."
        )
    if args.train_dark_prob > 0 and not args.train_dark_suffix:
        raise ValueError("--train_dark_suffix must be non-empty when --train_dark_prob > 0.")
    if "/" in args.train_dark_suffix or "\\" in args.train_dark_suffix:
        raise ValueError("--train_dark_suffix must not contain path separators.")

    preprocess_config = None
    if args.apply_darken or args.apply_brighten:
        preprocess_config = PreprocessConfig(
            apply_darken=args.apply_darken,
            apply_brighten=True,
            darken_min=args.darken_min,
            darken_max=args.darken_max,
            brighten_method=args.brighten_method,
            brighten_gamma_min=args.brighten_gamma_min,
            brighten_gamma_max=args.brighten_gamma_max,
        )

    if args.video is not None:
        single_dir = os.path.join(args.save_dir, f"{args.dataset}_single_keypoints")
        os.makedirs(single_dir, exist_ok=True)
        process_video(
            args.video,
            single_dir,
            use_holistic=args.use_holistic,
            face_mode=args.face_mode,
            write_placeholders=args.write_placeholders,
            preprocess_config=preprocess_config,
            export_darkened_dir=args.export_darkened_dir,
        )
        print(f"Saved single-video keypoints to: {single_dir}")
        raise SystemExit(0)

    if args.scan:
        all_paths = scan_include_dir_videos(args.include_dir)
        train_paths, val_paths, test_paths = split_paths(
            all_paths,
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        train_paths, val_paths, test_paths = load_train_test_val_paths(
            args.include_dir, args.dataset
        )

    if args.preflight:
        paths = []
        if args.splits in ("train", "all"):
            paths.extend(train_paths)
        if args.splits in ("val", "all"):
            paths.extend(val_paths)
        if args.splits in ("test", "all"):
            paths.extend(test_paths)
        preflight_videos(paths, n=args.preflight_n)
        raise SystemExit(0)

    selected_splits = (
        ["train", "val", "test"] if args.splits == "all" else [args.splits]
    )

    if args.clean_output:
        for split in selected_splits:
            split_dir = os.path.join(args.save_dir, f"{args.dataset}_{split}_keypoints")
            if os.path.isdir(split_dir):
                shutil.rmtree(split_dir)
                print(f"Removed existing output directory: {split_dir}")

    if args.splits in ("val", "all"):
        save_keypoints(
            dataset=args.dataset,
            file_paths=val_paths,
            mode="val",
            save_root=args.save_dir,
            n_jobs=args.jobs,
            use_holistic=args.use_holistic,
            face_mode=args.face_mode,
            limit=args.limit,
            no_parallel=args.no_parallel,
            write_placeholders=args.write_placeholders,
            preprocess_config=preprocess_config,
            export_darkened_dir=args.export_darkened_dir,
        )

    if args.splits in ("test", "all"):
        save_keypoints(
            dataset=args.dataset,
            file_paths=test_paths,
            mode="test",
            save_root=args.save_dir,
            n_jobs=args.jobs,
            use_holistic=args.use_holistic,
            face_mode=args.face_mode,
            limit=args.limit,
            no_parallel=args.no_parallel,
            write_placeholders=args.write_placeholders,
            preprocess_config=preprocess_config,
            export_darkened_dir=args.export_darkened_dir,
        )

    if args.splits in ("train", "all"):
        save_keypoints(
            dataset=args.dataset,
            file_paths=train_paths,
            mode="train",
            save_root=args.save_dir,
            n_jobs=args.jobs,
            use_holistic=args.use_holistic,
            face_mode=args.face_mode,
            limit=args.limit,
            no_parallel=args.no_parallel,
            write_placeholders=args.write_placeholders,
            preprocess_config=preprocess_config,
            export_darkened_dir=args.export_darkened_dir,
        )

    if args.splits in ("train", "all") and args.train_dark_prob > 0:
        train_dark_seed = (
            args.train_dark_seed if args.train_dark_seed is not None else args.split_seed
        )
        train_candidates = train_paths[: args.limit] if args.limit and args.limit > 0 else train_paths
        dark_train_paths = select_paths_by_probability(
            train_candidates, probability=args.train_dark_prob, seed=train_dark_seed
        )
        print(
            f"Selected {len(dark_train_paths)}/{len(train_candidates)} train videos "
            f"for dark variants (prob={args.train_dark_prob}, seed={train_dark_seed})."
        )
        if dark_train_paths:
            dark_preprocess_config = PreprocessConfig(
                apply_darken=True,
                apply_brighten=False,
                darken_min=args.darken_min,
                darken_max=args.darken_max,
                brighten_method=args.brighten_method,
                brighten_gamma_min=args.brighten_gamma_min,
                brighten_gamma_max=args.brighten_gamma_max,
            )
            save_keypoints(
                dataset=args.dataset,
                file_paths=dark_train_paths,
                mode="train",
                save_root=args.save_dir,
                n_jobs=args.jobs,
                use_holistic=args.use_holistic,
                face_mode=args.face_mode,
                limit=0,
                no_parallel=args.no_parallel,
                write_placeholders=args.write_placeholders,
                preprocess_config=dark_preprocess_config,
                export_darkened_dir=args.export_darkened_dir,
                uid_suffix=args.train_dark_suffix,
            )
