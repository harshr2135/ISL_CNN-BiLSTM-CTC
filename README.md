
# 🤟 Indian Sign Language Recognition (CNN + BiLSTM + CTC)

This project implements an end-to-end system for **continuous Indian Sign Language (ISL)** recognition using a hybrid deep learning model that combines:
- **CNN** for spatial feature extraction
- **BiLSTM** for sequence modeling
- **CTC Loss** for alignment-free sequence decoding

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── raw_videos/              # Input videos organized in class-wise folders
│   └── processed_data/          # Preprocessed .npy frame sequences
│
├── models/
│   ├── isl_ctc_model.py         # CNN + BiLSTM model definition
│   └── ctc_trainer.py           # CTC wrapper for model training
│
├── utils/
│   ├── video_loader.py          # Frame extractor from videos
│   ├── label_utils.py           # Label map creation and loading
│   ├── dataset_loader.py        # Loads dataset into tf.data pipeline
│   ├── ctc_decoder.py           # Greedy decoding logic for CTC
│   └── metrics.py               # WER and accuracy computation
│
├── preprocess.py                # Converts raw videos to fixed-length .npy files
├── train_model.py               # Loads dataset, builds model, trains and evaluates
├── inference_realtime.py        # Captures webcam input and runs real-time inference
├── label_map.json               # Word-to-label and label-to-word mappings
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

---

## ✅ Key Features

- Processes `.mp4`, `.avi`, `.mov`, `.mkv` videos
- Frame-extraction, resizing, and normalization
- CNN-based spatial encoder
- BiLSTM for learning temporal dependencies
- CTC loss for variable-length gesture decoding
- Real-time webcam-based prediction
- Evaluation using WER and sequence accuracy

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place raw videos into class-named folders like:

```
data/raw_videos/
├── hello/
│   ├── hello1.mp4
│   ├── hello2.avi
│   ...
├── thank_you/
│   ├── thank1.mp4
│   ...
```

### 3. Preprocess Videos

```bash
python preprocess.py
```

> This saves frame sequences as `.npy` files in `data/processed_data/` and generates `label_map.json`.

---

## 🏋️ Model Training

```bash
python train_model.py
```

- Trains on 80% of the dataset, validates on 20%
- Automatically logs:
  - Best model in `models/`
  - TensorBoard logs in `logs/`
- Prints final:
  - ✅ Accuracy
  - 📝 Word Error Rate (WER)

---

## 🎥 Real-Time Inference

```bash
python inference_realtime.py
```

- Launches webcam
- Captures short sequence (up to 60 frames)
- Predicts sign sequence using the trained model
- Prints predicted phrase

> ⚠️ Update the `MODEL_PATH` in `inference_realtime.py` with your actual `.keras` model file path.

---

## 📊 Evaluation Metrics

- ✅ **Exact Match Accuracy**: Number of correct full-sequence predictions
- 📝 **Word Error Rate (WER)**: Levenshtein distance between predicted and true sequences
- 📈 **Top-k Accuracy** (optional extension)

---

## 📦 Requirements

```
tensorflow>=2.11.0
opencv-python
numpy
tqdm
scikit-learn
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 💡 Future Improvements

- Beam search decoder for better predictions
- Face and hand landmark fusion
- Multi-signer generalization
- Export to TensorRT/ONNX for Jetson deployment

---

## 🧠 Skills Applied

- Temporal Deep Learning (CNN + BiLSTM)
- Sequence Modeling with CTC Loss
- Computer Vision (OpenCV)
- Real-Time Inference
- TensorFlow & tf.data pipeline
- Evaluation metrics like WER

---

## 📝 License

MIT License. For academic and research use.

---

## 🙏 Acknowledgments

This system is part of a larger effort to build accessible communication tools for the deaf and hard-of-hearing community in India using state-of-the-art AI and computer vision techniques.
