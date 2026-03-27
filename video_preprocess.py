import hashlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    apply_darken: bool = False
    apply_brighten: bool = True
    darken_min: float = 0.3
    darken_max: float = 0.8
    brighten_method: str = "clahe"  # "clahe" or "gamma"
    brighten_gamma_min: float = 1.2
    brighten_gamma_max: float = 1.8


def _hash_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _clamp_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def darken_frame(frame: np.ndarray, factor: float) -> np.ndarray:
    return _clamp_uint8(frame.astype(np.float32) * factor)


def brighten_frame(frame: np.ndarray, method: str = "clahe", gamma: float = 1.5, darkness_threshold: int = 80) -> np.ndarray:
    # the smarter way to brighten is to first check if the frame is actually dark, and only apply brighten if it is. This way we avoid over-brightening already bright frames and introducing noise.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    
    # If the frame is already bright enough, return it as is to save processing time and avoid over-brightening
    if average_brightness > darkness_threshold:
        return frame
    
    method = (method or "clahe").lower()
    if method == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    if method == "gamma":
        inv_gamma = 1.0 / max(gamma, 1e-6)
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)], dtype=np.uint8
        )
        return cv2.LUT(frame, table)
    raise ValueError(f"Unsupported brighten method: {method}")


def apply_darken_then_brighten(
    frame: np.ndarray,
    *,
    config: PreprocessConfig,
    rng: np.random.Generator,
    darken_factor: Optional[float] = None,
) -> np.ndarray:
    processed = frame
    if config.apply_darken:
        if darken_factor is None:
            darken_factor = float(rng.uniform(config.darken_min, config.darken_max))
        processed = darken_frame(processed, darken_factor)
    if config.apply_brighten:
        gamma = float(rng.uniform(config.brighten_gamma_min, config.brighten_gamma_max))
        processed = brighten_frame(processed, method=config.brighten_method, gamma=gamma)
    return processed
