---
title: LumiSign
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
sdk_version: 4.19.2
python_version: 3.9
app_file: app.py
pinned: false
short_description: Indian Sign Language Recognition Web UI
---

# LumiSign

Indian Sign Language recognition with FastAPI + React UI, using a Transformer model trained on `isl-split-dataset`.

This README is written for developers who are new to the project.

## What You Need

1. Python `3.9`
2. Node.js `18+` and npm
3. Git
4. A Linux/macOS shell (or WSL on Windows)

## Project Layout

```text
LumiSign/
  app.py
  inference.py
  runner.py
  prepare_custom_dataset.py
  check_split_leakage.py
  transformer_large.pth
  label_maps/
    label_map_isl_split_dataset.json
  isl-split-dataset/
    train/
    eval/
    test/
  ui/
```

## Dataset Layout Expected

`prepare_custom_dataset.py` expects this format:

```text
isl-split-dataset/
  train/
    bank/
    court/
    store or shop/
  eval/
    bank/
    court/
    store or shop/
  test/
    bank/
    court/
    store or shop/
```

Labels are normalized internally:
- `store or shop` becomes `storeorshop`

## Quick Start (Use Existing `transformer_large.pth`)

Run these from the project root.

### 1) Create and install environment

```bash
python3.9 -m venv venv
source venv/bin/activate
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

### 2) Build frontend once

```bash
cd ui
npm install
npm run build
cd ..
```

### 3) Start backend with ISL model

```bash
MODEL_CHECKPOINT=./transformer_large.pth \
MODEL_LABEL_MAP_PATH=./label_maps/label_map_isl_split_dataset.json \
MODEL_DATASET=isl_split_dataset \
MODEL_TYPE=transformer \
MODEL_TRANSFORMER_SIZE=large \
MODEL_MAX_FRAME_LEN=169 \
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8070 --reload
```

Open:
- UI: `http://localhost:8070`
- API docs: `http://localhost:8070/docs`

## Full Pipeline (Prepare Data -> Train -> Evaluate)

Use this if you want to regenerate keypoints and retrain.

### 1) Set paths

```bash
PROJECT_ROOT="$(pwd)"
ISL_SPLIT_ROOT="$PROJECT_ROOT/isl-split-dataset"
ISL_DATASET_NAME="isl_split_dataset"
ISL_PROCESSED_DIR="$PROJECT_ROOT/processed_data_islsplit"
ISL_CKPT_DIR="$PROJECT_ROOT/checkpoints_islsplit"

mkdir -p "$ISL_PROCESSED_DIR" "$ISL_CKPT_DIR"
```

### 2) Convert split folders to keypoint JSON files

```bash
venv/bin/python prepare_custom_dataset.py \
  --data_dir "$ISL_SPLIT_ROOT" \
  --save_dir "$ISL_PROCESSED_DIR" \
  --dataset_name "$ISL_DATASET_NAME" \
  --jobs 4 \
  --use_holistic \
  --face_mode full \
  --write_placeholders
```

This creates:
- `processed_data_islsplit/isl_split_dataset_train_keypoints`
- `processed_data_islsplit/isl_split_dataset_val_keypoints`
- `processed_data_islsplit/isl_split_dataset_test_keypoints`
- `label_maps/label_map_isl_split_dataset.json`

### 3) Check split leakage

```bash
venv/bin/python check_split_leakage.py \
  --data_dir "$ISL_PROCESSED_DIR" \
  --dataset "$ISL_DATASET_NAME" \
  --dark_suffix __dark
```

### 4) Train Transformer (large)

```bash
venv/bin/python runner.py \
  --dataset "$ISL_DATASET_NAME" \
  --model transformer \
  --transformer_size large \
  --max_frame_len 169 \
  --data_dir "$ISL_PROCESSED_DIR" \
  --save_path "$ISL_CKPT_DIR" \
  --batch_size 8 \
  --early_stop_metric val_loss \
  --early_stop_patience 5
```

### 5) Evaluate on test split

```bash
venv/bin/python runner.py \
  --dataset "$ISL_DATASET_NAME" \
  --model transformer \
  --transformer_size large \
  --max_frame_len 169 \
  --data_dir "$ISL_PROCESSED_DIR" \
  --save_path "$ISL_CKPT_DIR" \
  --batch_size 1 \
  --epochs 0 \
  --eval_split test
```

## Single Video CLI Inference

```bash
venv/bin/python inference.py \
  --video /absolute/path/to/video.mp4 \
  --dataset isl_split_dataset \
  --model transformer \
  --transformer_size large \
  --checkpoint ./transformer_large.pth \
  --label_map_path ./label_maps/label_map_isl_split_dataset.json \
  --max_frame_len 169
```

## Run UI + API in Development

Terminal 1:

```bash
MODEL_CHECKPOINT=./transformer_large.pth \
MODEL_LABEL_MAP_PATH=./label_maps/label_map_isl_split_dataset.json \
MODEL_DATASET=isl_split_dataset \
MODEL_TYPE=transformer \
MODEL_TRANSFORMER_SIZE=large \
MODEL_MAX_FRAME_LEN=169 \
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8070 --reload
```

Terminal 2:

```bash
cd ui
npm install
npm run dev
```

If Vite starts on `5173`, open `http://localhost:5173`.

## Common Errors and Fixes

### 1) `size mismatch for l2.weight` when loading checkpoint

Cause:
- Label map class count does not match checkpoint output classes.

Fix:

```bash
MODEL_CHECKPOINT=./transformer_large.pth \
MODEL_LABEL_MAP_PATH=./label_maps/label_map_isl_split_dataset.json \
MODEL_DATASET=isl_split_dataset
```

Your ISL checkpoint expects 3 classes: `bank`, `court`, `storeorshop`.

### 2) CUDA warning like `Error 804: forward compatibility was attempted`

Cause:
- GPU driver/CUDA compatibility mismatch.

Fix:
- Run on CPU, or align NVIDIA driver + CUDA + PyTorch versions.
- This warning is not the same as the class mismatch error.

### 3) `No label map found for dataset ...`

Fix:
- Ensure `label_maps/label_map_isl_split_dataset.json` exists.
- Pass `MODEL_LABEL_MAP_PATH` explicitly.

## Useful Entrypoints

- `app.py`: FastAPI server startup and model loading
- `inference.py`: model load and single-video prediction
- `prepare_custom_dataset.py`: convert split video folders to keypoint JSON
- `check_split_leakage.py`: verify train/val/test split leakage
- `runner.py`: train and evaluate models

## Reference Commands

- Full command history used in this project is in `commands.txt`.

## Citation

If you use this work, cite INCLUDE :

```bibtex
@inproceedings{10.1145/3394171.3413528,
author = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
title = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
doi = {10.1145/3394171.3413528},
series = {MM '20}
}
```
