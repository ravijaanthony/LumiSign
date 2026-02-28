# INCLUDE - Isolated Indian Sign Language Recognition with Web UI

This repository features an enhanced implementation of the **INCLUDE** dataset for Indian Sign Language (ISL) recognition. It introduces modern deep learning architectures, a comprehensive preprocessing pipeline, and a full-stack web interface for real-time video inference.

---

## 🚀 Getting Started

### 1. Environment Setup
Install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Web Interface Setup
The UI is built with React. To prepare the frontend:

```bash
cd ui
npm install
npm run build  # Builds the UI for production
```

For local development:

```bash
npm run dev
```

### 3. Running the Application
Launch the FastAPI server to start the web interface:

```bash
python app.py
```

Access the interface at http://localhost:8000.

---

## 📈 Training and Evaluation

### Extract Keypoints
Process the INCLUDE dataset to extract pose and hand keypoints:

```bash
python generate_keypoints.py --include_dir <path_to_dataset> --save_dir <path_to_save> --dataset include50
```

### Train a Model
To train a Transformer-based model:

```bash
python runner.py --dataset include50 --use_augs --model transformer --data_dir <keypoints_dir>
```

### Perform Inference
Run predictions on a folder of videos:

```bash
python evaluate.py --data_dir <path_to_videos>
```
## 🔬 Related Research
The architectural enhancements, preprocessing techniques, and full-stack implementation provided in this repository are discussed in detail in the following research paper:

* **[A Robust and Multidisciplinary Approach to Indian Sign Language Recognition](https://www.researchgate.net/profile/Ravija-Vitharana/publication/399881048_A_Robust_and_Multidisciplinary_Approach_to_Indian_Sign_Language_Recognition/links/696e12f1abecff2489ecfe56/A-Robust-and-Multidisciplinary-Approach-to-Indian-Sign-Language-Recognition.pdf)**

---

## 📝 Citation & Credits
This project is built upon the INCLUDE dataset and research by the following authors:

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

> **Note:** This repository includes custom UI development and model architectural enhancements beyond the original publication.
