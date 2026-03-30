import argparse
import os
import shutil
import tempfile
import cv2
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data

from configs import LstmConfig, TransformerConfig
from generate_keypoints import process_video
from models import LSTM, Transformer
from utils import load_json, load_label_map
from video_preprocess import PreprocessConfig


class KeypointsDataset(data.Dataset):
    def __init__(self, keypoints_dir, max_frame_len=169, frame_length=1080, frame_width=1920):
        self.files = sorted(
            [f for f in os.listdir(keypoints_dir) if f.endswith(".json")]
        )
        self.keypoints_dir = keypoints_dir
        self.max_frame_len = max_frame_len
        self.frame_length = frame_length
        self.frame_width = frame_width

    def interpolate(self, arr):
        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        if np.count_nonzero(~np.isnan(arr_x)) == 0:
            arr_x = np.zeros(arr_x.shape)
        if np.count_nonzero(~np.isnan(arr_y)) == 0:
            arr_y = np.zeros(arr_y.shape)

        arr_x = arr_x * self.frame_width
        arr_y = arr_y * self.frame_length

        return np.stack([arr_x, arr_y], axis=-1)

    def combine_xy(self, x, y):
        x, y = np.array(x), np.array(y)
        _, length = x.shape
        x = x.reshape((-1, length, 1))
        y = y.reshape((-1, length, 1))
        return np.concatenate((x, y), -1).astype(np.float32)

    def __getitem__(self, idx):
        file_path = os.path.join(self.keypoints_dir, self.files[idx])
        row = pd.read_json(file_path, typ="series")

        pose = self.combine_xy(row.pose_x, row.pose_y)
        h1 = self.combine_xy(row.hand1_x, row.hand1_y)
        h2 = self.combine_xy(row.hand2_x, row.hand2_y)

        pose = self.interpolate(pose)
        h1 = self.interpolate(h1)
        h2 = self.interpolate(h2)

        pose = pose.reshape(-1, 50).astype(np.float32)
        h1 = h1.reshape(-1, 42).astype(np.float32)
        h2 = h2.reshape(-1, 42).astype(np.float32)
        final_data = np.concatenate((pose, h1, h2), -1)
        final_data = np.pad(
            final_data,
            ((0, self.max_frame_len - final_data.shape[0]), (0, 0)),
            "constant",
        )
        return {"uid": row.uid, "data": torch.FloatTensor(final_data)}

    def __len__(self):
        return len(self.files)


def _pretrained_name(dataset: str, model_type: str, transformer_size: str) -> str:
    name = dataset
    name += "_no_cnn"
    if model_type == "lstm":
        name += "_lstm.pth"
    elif model_type == "transformer":
        name += "_transformer"
        name += "_large.pth" if transformer_size == "large" else "_small.pth"
    return name


def load_model(
    dataset: str,
    model_type: str,
    transformer_size: str,
    checkpoint_path: Optional[str],
    label_map_path: Optional[str] = None,
):
    label_map = load_json(label_map_path) if label_map_path else load_label_map(dataset)
    n_classes = len(label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "transformer":
        config = TransformerConfig(size=transformer_size, max_position_embeddings=256)
        model = Transformer(config=config, n_classes=n_classes)
    elif model_type == "lstm":
        config = LstmConfig()
        model = LSTM(config=config, n_classes=n_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = model.to(device)

    if checkpoint_path is None:
        pretrained_links = load_json("pretrained_links.json")
        model_name = _pretrained_name(dataset, model_type, transformer_size)
        
        # 1. First, check if you manually uploaded the file to the app folder
        if os.path.isfile(model_name):
            checkpoint_path = model_name
        else:
            # 2. If not, set the path to the writable /tmp directory
            checkpoint_path = os.path.join("/tmp", model_name)
            
            # 3. Download it to /tmp if it isn't already there
            if not os.path.isfile(checkpoint_path):
                link = pretrained_links.get(model_name)
                if not link or link == "link":
                    raise FileNotFoundError(f"No pretrained link for {model_name}")
                torch.hub.download_url_to_file(link, checkpoint_path, progress=True)
            checkpoint_path = model_name

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, label_map


@torch.no_grad()
def predict_video(
    video_path: str,
    dataset: str,
    model,
    label_map: dict,
    preprocess_config: PreprocessConfig,
    *,
    max_frame_len: int = 169,
    use_holistic: bool = True,
    face_mode: str = "full",
):
    temp_dir = tempfile.mkdtemp(prefix="keypoints_")
    try:
        process_video(
            video_path,
            temp_dir,
            use_holistic=use_holistic,
            face_mode=face_mode,
            preprocess_config=preprocess_config,
        )
        dataset_obj = KeypointsDataset(keypoints_dir=temp_dir, max_frame_len=max_frame_len)
        dataloader = data.DataLoader(dataset_obj, batch_size=1, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        id_to_label = dict(zip(label_map.values(), label_map.keys()))
        for batch in dataloader:
            input_data = batch["data"].to(device)
            logits = model(input_data)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            idx = int(np.argmax(probs))
            return {
                "uid": batch["uid"][0],
                "label": id_to_label[idx],
                "score": float(probs[idx]),
            }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    raise RuntimeError("No frames processed from the video.")

# smart brighten function that checks if the frame is too dark before applying CLAHE
def smart_brighten(frame, darkness_threshold=80):
    """
    Checks if a frame is too dark. If it is, applies CLAHE to enhance 
    the hand/pose visibility without blowing out the highlights.
    """
    # 1. Convert to grayscale just to calculate the average brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    
    # 2. If it's bright enough, leave it alone to save processing time
    if average_brightness > darkness_threshold:
        return frame
        
    # 3. If it's dark, apply CLAHE to the L channel (Lightness)
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to the lightness channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    merged = cv2.merge((cl, a, b))
    brightened_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return brightened_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single video")
    parser.add_argument("--video", required=True, help="path to the video file")
    parser.add_argument("--dataset", default="include50", help="include or include50")
    parser.add_argument("--model", default="transformer", choices=["transformer", "lstm"])
    parser.add_argument("--transformer_size", default="small", choices=["small", "large"])
    parser.add_argument("--checkpoint", default=None, help="path to model checkpoint")
    parser.add_argument("--apply_darken", action="store_true", help="apply darken before brighten")
    parser.add_argument("--darken_min", default=0.3, type=float)
    parser.add_argument("--darken_max", default=0.8, type=float)
    parser.add_argument("--brighten_method", default="clahe", choices=["clahe", "gamma"])
    parser.add_argument("--brighten_gamma_min", default=1.2, type=float)
    parser.add_argument("--brighten_gamma_max", default=1.8, type=float)
    parser.add_argument(
        "--max_frame_len",
        default=169,
        type=int,
        help="sequence length used during training (must match inference)",
    )
    args = parser.parse_args()

    preprocess_config = PreprocessConfig(
        apply_darken=args.apply_darken,
        apply_brighten=True,
        darken_min=args.darken_min,
        darken_max=args.darken_max,
        brighten_method=args.brighten_method,
        brighten_gamma_min=args.brighten_gamma_min,
        brighten_gamma_max=args.brighten_gamma_max,
    )

    model, label_map = load_model(
        dataset=args.dataset,
        model_type=args.model,
        transformer_size=args.transformer_size,
        checkpoint_path=args.checkpoint,
    )
    result = predict_video(
        args.video,
        args.dataset,
        model,
        label_map,
        preprocess_config=preprocess_config,
        max_frame_len=args.max_frame_len,
    )
    print(result)
