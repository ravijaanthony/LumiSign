# LumiSign

Indian Sign Language recognition with FastAPI + React UI, using a Transformer model trained on `isl-split-dataset`.

This README is written for developers who are new to the project.

## What You Need

1. Python `3.9`
2. Node.js `18+` and npm
3. Git
4. A Linux/macOS shell (or WSL on Windows)
5. Pre-trained model checkpoint: `transformer_large.pth`
6. Label map: `label_maps/label_map_isl_split_dataset.json`
7. Dataset: `isl-split-dataset/` (see Dataset Layout section)

---

## ⚡ Quick Start - Local Development (Backend + Frontend)

**Start from the project root directory.**

### Step 1: Set up Python environment

```bash
python3.9 -m venv venv
source venv/bin/activate
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

### Step 2: Set up frontend (build once)

```bash
cd ui
npm install
npm run build
cd ..
```

### Step 3: Run backend (Terminal 1)

```bash
MODEL_CHECKPOINT=./transformer_large.pth \
MODEL_LABEL_MAP_PATH=./label_maps/label_map_isl_split_dataset.json \
MODEL_DATASET=isl_split_dataset \
MODEL_TYPE=transformer \
MODEL_TRANSFORMER_SIZE=large \
MODEL_MAX_FRAME_LEN=169 \
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8070 --reload
```

### Step 4: Run frontend dev server (Terminal 2)

```bash
cd ui
npm run dev
```

**Access the app:**
- **UI**: `http://localhost:8070` (or Vite dev server at `http://localhost:5173` if you're developing frontend)
- **API docs**: `http://localhost:8070/docs`

> **Note**: Development mode uses the FastAPI `/docs` endpoint. For production, the React build in `ui/dist` is served by FastAPI.

---

## How It Works: UI Workflow & Features

### Landing Page

When you access LumiSign at `http://localhost:8070`, you'll see the main application interface:

![LumiSign Landing Page](poc/LandingPage)

**Landing Page shows:**
- Header: "Indian Sign Language Translator" with subtitle "Real-time ISL to English Translation"
- Language selector dropdown (top right)
- Export Results button
- Video input section with two options:
  - **Use Camera**: Record live sign language in real-time
  - **Upload Video**: Upload pre-recorded ISL gesture video
- Sidebar with navigation: New Translation, History, Saved, Analytics, Documentation, Help Center, Settings

### Step 1: Video Input

Choose one of two ways to provide video input:

![Video Input Section](poc/Input)

**Option A: Use Camera**
- Click "Use Camera" button to capture live video from your webcam
- Record one of the 3 recognized ISL gestures:
  - **bank** - Banking/money-related gesture
  - **court** - Legal/court-related gesture
  - **store or shop** - Shopping/retail-related gesture
- Camera opens and captures your sign language gesture
- Perfect for live testing and demonstration

**Option B: Upload Video** (Shown in image above)
- Click "Upload Video" button to select a pre-recorded video file
- Use test videos from the dataset: `isl-split-dataset/eval/` or `isl-split-dataset/test/`
- Each folder contains videos named like: `bank_001.mp4`, `court_005.mp4`, `storeorshop_010.mp4`
- **Recommended for testing** since you don't need to know sign language yourself
- You can watch the example video in the upload preview

**Video Preview after Upload:**
- **Left side**: Original Video Feed - Shows your exact uploaded/recorded video
- **Right side**: Enhanced Video Preview - Shows video with brightness corrections applied
- This lets you see what the model will actually process
- Helps verify video quality before starting translation

### Step 2: Start Translation

After uploading or recording, click the **"Start Translation"** button:

![Start Translation Button](poc/TranslateButton)

- Red button located at bottom right
- Initiates the complete processing pipeline
- Takes you to the processing status screen

### Step 3: Processing Pipeline

The app processes your video through a **3-step pipeline** with real-time progress:

![Processing Pipeline in Progress](poc/Process)

**Pipeline Steps:**

#### Step 1: Video Enhancement ✓ Done
- Status badge shows green checkmark
- Brightens darkened video frames
- Normalizes lighting conditions
- Completes almost instantly

#### Step 2: Keypoint Extraction ✓ Done
- Status badge shows green checkmark
- **Hand Landmarks**: Detects 21 keypoints per hand (e.g., "21 keypoints detected")
- **Pose Landmarks**: Detects 33 keypoints for body/torso (e.g., "33 keypoints detected")
- **Face Landmarks**: Detects 468 keypoints with full face mode (e.g., "468 keypoints detected")
- Uses MediaPipe Holistic for robust detection
- Processes entire video frame-by-frame

#### Step 3: Transformer Translation ⧗ In Progress
- Status badge shows blue spinner (in progress)
- Passes keypoint sequences through trained Transformer model
- Performs sequence classification → ISL gesture labels
- Generates confidence scores (0-100%)
- Typically 1-2 seconds (most time-intensive step)

**Overall Progress:**
- Shows total pipeline completion percentage (e.g., "79%")
- Progress bar indicates real-time advancement
- Updates dynamically as each step completes

### Step 4: Translation Output

Once processing completes, you see the **Translation Output section:**

![Final Output with Results](poc/Output)

**Output displays:**

- **Predicted Gesture**: The recognized ISL sign (in quotes)
  - Example: `"court"` - Model recognized this as the court gesture
  - One of: court, bank, storeorshop

- **Accuracy Score**: Confidence percentage (0-100%)
  - Example: `94.5%` - High confidence prediction
  - Higher = more reliable

- **Confidence Level**: Qualitative assessment
  - Example: `High` (>85%), Medium (70-85%), Low (<70%)
  - Helps quickly assess reliability

- **Processing Time**: Total time for complete pipeline
  - Example: `1.92s` - Completed in under 2 seconds
  - Typical range: 1-3 seconds (depends on video length)

### Step 5: Save Results

Click **"Save Results"** button to store the translation:
- Permanently saves gesture, accuracy, timestamp, and metadata
- Result appears in **Saved Tab** on sidebar
- Perfect for batch testing and performance analysis

**Review Your Results:**
- **Saved Tab**: All saved translations with gesture, accuracy, timestamp
- **History Tab**: Complete history of all attempts (saved and unsaved)
- **Analytics**: Aggregate statistics across all predictions
- **Export Results**: Generate comprehensive reports

---

## Supported Testing Workflows

### Testing with Dataset Videos (Recommended)

**Best for:** Validating model accuracy with known labels

1. Click **"Upload Video"** button
2. Navigate to `isl-split-dataset/eval/` or `isl-split-dataset/test/`
3. Select a test video (e.g., `bank_001.mp4`, `court_005.mp4`)
4. Review original and enhanced video preview
5. Click **"Start Translation"** button
6. Watch the 3-step processing pipeline in real-time
7. Verify predicted gesture matches the video's class
8. Check accuracy score and processing time
9. Click **"Save Results"** to keep the result
10. Repeat with different videos to test multiple samples
11. Use **Saved Tab** to compare results across multiple tests

### Live Testing with Camera

**Best for:** Live demonstration and manual testing

1. Click **"Use Camera"** button
2. Ensure webcam is accessible and has permission
3. Perform one of the 3 recognized ISL signs:
   - **bank**: Banking/money gesture
   - **court**: Legal/court gesture
   - **shop/store**: Shopping gesture
4. Stop recording when done
5. Review captured video on both sides
6. Click **"Start Translation"** button
7. See the predicted gesture and confidence level
8. Check if prediction matches what you signed
9. Save the result if correct

### Batch Analysis

**Best for:** Comprehensive model evaluation

1. Upload multiple test videos one by one
2. Let each process completely
3. Save all results you want to keep
4. Go to **Saved Tab** to see all saved translations
5. Go to **History Tab** to see all attempts
6. Click **Export Results** for comprehensive report
7. Analyze accuracy patterns and processing performance

---

## Project Layout

```text
LumiSign/
  ├── app.py                      # FastAPI server
  ├── inference.py                # Single-video prediction
  ├── runner.py                   # Training entrypoint
  ├── prepare_custom_dataset.py   # Keypoint generation from videos
  ├── requirements.txt
  ├── transformer_large.pth       # ⚠️ Pre-trained model (git-ignored)
  │
  ├── label_maps/
  │   └── label_map_isl_split_dataset.json  # Class labels (generated on first train)
  │
  ├── isl-split-dataset/          # ⚠️ Dataset folder (git-ignored)
  │   ├── train/
  │   │   ├── bank/
  │   │   ├── court/
  │   │   └── store or shop/
  │   ├── eval/
  │   │   ├── bank/
  │   │   ├── court/
  │   │   └── store or shop/
  │   └── test/
  │       ├── bank/
  │       ├── court/
  │       └── store or shop/
  │
  ├── poc/                        # Proof of Concept screenshots
  │   ├── LandingPage             # Landing page UI
  │   ├── Input                   # Video input section
  │   ├── TranslateButton         # Translate button state
  │   ├── Process                 # Processing pipeline
  │   └── Output                  # Final output display
  │
  ├── ui/                         # React + Vite frontend
  │   ├── src/
  │   ├── package.json
  │   └── vite.config.ts
  │
  └── models/                     # PyTorch model definitions
      ├── transformer.py
      ├── lstm.py
      └── cnn.py
```

⚠️ **Files not tracked by git** (see `.gitignore`):
- `transformer_large.pth` (model weights)
- `label_maps/` (generated during first training)
- `isl-split-dataset/` (data — you must provide)
- `processed_data/`, `checkpoints/`, `outputs/` (generated)
- `venv/` (local virtual environment)

---

## What You Must Provide

### 1. Pre-trained model: `transformer_large.pth`

Obtain by **either**:

- **Option A**: Download from [Kaggle Models](https://www.kaggle.com/models/ravijavitharana/transformer-large)
  ```bash
  # After downloading from Kaggle, place in project root
  ```
- **Option B**: Train from scratch (see "Training from Scratch" section below)
- **Option C**: Use an existing checkpoint file you have

### 2. Dataset: `isl-split-dataset/`

Download the dataset from [Kaggle - isl-split-dataset](https://www.kaggle.com/datasets/ravijavitharana/isl-split-dataset).

**Dataset Structure**: The folder MUST be organized as:

```text
isl-split-dataset/
  train/
    bank/           # Video files: bank_000.mp4, bank_001.mp4, ...
    court/          # Video files: court_000.mp4, court_001.mp4, ...
    store or shop/  # Video files: storeorshop_000.mp4, ...
  eval/
    bank/
    court/
    store or shop/
  test/
    bank/
    court/
    store or shop/
```

- Each video file should be a valid MP4/MOV (or other video format OpenCV can read).
- Label names are normalized internally: `store or shop` → `storeorshop`.

**Note**: This dataset contains **3 classes** (bank, court, store/shop) due to resource limitations in Indian Sign Language recognition research. This scope is suitable for proof-of-concept and initial development. For production systems with more gesture vocabulary, consider extending the dataset.

### 3. Label map: `label_maps/label_map_isl_split_dataset.json`

Obtain by **either**:

- **Option A**: Run training (it auto-generates this)
- **Option B**: Use an existing label map file you have

This maps class names to indices.

---

## Usage Scenarios

### I have a pre-trained model and just want to use it

1. Place `transformer_large.pth` in the project root
2. Place `label_maps/label_map_isl_split_dataset.json` in `label_maps/`
3. Place dataset in `isl-split-dataset/` (optional for just inference)
4. Follow **Quick Start** above
5. Test using the web UI at `http://localhost:8070`
6. Use the workflows above to test with dataset videos or camera

### I want to train the model from scratch

See **Training from Scratch** section below.

### I want to run single-video inference from CLI

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

---

## Training from Scratch

If you want to regenerate keypoints and retrain the model.

### Prerequisites

- Ensure dataset is in `isl-split-dataset/` (see Project Layout above)

### Full Pipeline

Run from project root.

#### 1) Set paths

```bash
PROJECT_ROOT="$(pwd)"
ISL_SPLIT_ROOT="$PROJECT_ROOT/isl-split-dataset"
ISL_DATASET_NAME="isl_split_dataset"
ISL_PROCESSED_DIR="$PROJECT_ROOT/processed_data_islsplit"
ISL_CKPT_DIR="$PROJECT_ROOT/checkpoints_islsplit"

mkdir -p "$ISL_PROCESSED_DIR" "$ISL_CKPT_DIR"
```

#### 2) Convert video folders to keypoint JSON files

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

This generates:
- `processed_data_islsplit/isl_split_dataset_train_keypoints`
- `processed_data_islsplit/isl_split_dataset_val_keypoints`
- `processed_data_islsplit/isl_split_dataset_test_keypoints`
- `label_maps/label_map_isl_split_dataset.json`

#### 3) Check for train/val/test leakage

```bash
venv/bin/python check_split_leakage.py \
  --data_dir "$ISL_PROCESSED_DIR" \
  --dataset "$ISL_DATASET_NAME" \
  --dark_suffix __dark
```

#### 4) Train Transformer (large)

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

This saves checkpoints to `checkpoints_islsplit/`. The final model is also saved as `transformer_large.pth` in the project root.

#### 5) Evaluate on test split

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

---

## Useful Entrypoints

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server startup and model loading |
| `inference.py` | Load model and run single-video prediction |
| `prepare_custom_dataset.py` | Convert split video folders to keypoint JSON |
| `check_split_leakage.py` | Verify train/val/test split leakage |
| `runner.py` | Train and evaluate models |

---

## Common Errors and Fixes

### 1) `size mismatch for l2.weight` when loading checkpoint

**Cause**: Label map class count does not match checkpoint output classes.

**Fix**: Ensure your label map and checkpoint match:
```bash
MODEL_LABEL_MAP_PATH=./label_maps/label_map_isl_split_dataset.json
```

Your ISL checkpoint expects 3 classes: `bank`, `court`, `storeorshop`.

### 2) CUDA warning: `Error 804: forward compatibility was attempted`

**Cause**: GPU driver/CUDA compatibility mismatch.

**Fix**: 
- Run on CPU, or
- Align NVIDIA driver + CUDA + PyTorch versions

This warning is NOT the same as the class mismatch error above.

### 3) `No label map found for dataset ...`

**Fix**: Ensure `label_maps/label_map_isl_split_dataset.json` exists.
- Either train the model (which generates it), or
- Place an existing label map in `label_maps/`

### 4) `FileNotFoundError: isl-split-dataset not found`

**Fix**: Place your dataset in `isl-split-dataset/` following the folder structure in Project Layout above.

---

## Development Commands

For local development workflows, see `commands.txt`. This file contains all useful command snippets for running, training, and debugging locally.

---

## Local Development & Hardware

### Tested Local Environment

This project has been successfully developed and tested locally on:

- **Device**: ASUS TUF Gaming A15 (AN515-57)
- **CPU**: Intel Core i7-11800H
- **RAM**: 32 GB
- **Storage**: 512 GB SSD
- **GPU**: NVIDIA RTX 3050
- **OS**: Linux

**Local Setup Validation**: All development, setup steps, and documentation workflows run without issues on this configuration. This ensures the README and local setup instructions are accurate and tested.

### Why Use Cloud for Training?

While the model **can** be trained locally, we recommend using cloud platforms (Kaggle, Google Colab) for the following reasons:

1. **Training Speed**: Cloud GPUs (V100, T4) provide significantly faster training compared to RTX 3050
2. **Resource Efficiency**: Free tier options available on Kaggle; saves local power and cooling costs
3. **Scalability**: Easier to experiment with larger batch sizes and longer training runs
4. **Reliability**: Long training sessions are more stable on cloud infrastructure
5. **Development Workflow**: Keep your local machine free for frontend/UI development while model trains in cloud

**Recommendation**: Use local setup for **development, testing, and inference**. Use Kaggle/Colab for **training and experimentation**.

---

## Local Development

LumiSign is designed for **local development and testing** only. Follow the **Quick Start** section above to run locally:

- Backend API runs on `http://localhost:8070`
- Frontend UI accessible at `http://localhost:8070`
- Perfect for testing, development, and documentation purposes
- Real-time feedback with hot-reload capabilities (npm dev mode)

### System Requirements

- **Disk Space**: At least 4-5 GB (model + dependencies + dataset)
- **RAM**: 8 GB minimum; 16 GB+ recommended for smooth concurrent requests
- **GPU**: Optional but recommended for faster inference (2-3s → <1s per request)
- **CPU**: Modern multi-core CPU for real-time performance

---

## Running on Cloud (Optional)

### Google Colab

For training on Colab GPUs, see `commands_colab.txt`. 

**Note**: Colab commands are experimental and may require adjustments for your Drive setup.

### Kaggle

For training on Kaggle GPUs, see `commands_kaggle.txt`.

**Note**: Kaggle commands are experimental and may require adjustments for dataset attachment and paths.

---

## Paper & Publication

This work has been published in a peer-reviewed journal. If you use LumiSign in your research, please cite our paper:

**"A Robust and Multidisciplinary Approach to Indian Sign Language Recognition"**

Available at:
- [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=AkTYhqMAAAAJ&citation_for_view=AkTYhqMAAAAJ:Tyk-4Ss8FVUC)
- [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.177004941.19376614/v1)
- [ResearchGate](https://www.researchgate.net/profile/Ravija-Vitharana/publication/399881048_A_Robust_and_Multidisciplinary_Approach_to_Indian_Sign_Language_Recognition/links/696e12f1abecff2489ecfe56/A-Robust-and-Multidisciplinary-Approach-to-Indian-Sign-Language-Recognition.pdf)

---

## Citation

If you use this work, please cite the INCLUDE dataset:

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

---

## Project Status

This is a research project. It has been successfully tested locally and provides a solid foundation for ISL recognition applications. The project is:

- ✅ Fully functional for local development and inference
- ✅ Documented with clear setup instructions and UI workflows
- ✅ Includes proof-of-concept screenshots for visual reference
- ✅ Tested on real hardware (ASUS TUF AN515-57)
- ⏳ Open for community contributions and improvements

Cloud training environments (Colab, Kaggle) may require environment-specific adjustments.
