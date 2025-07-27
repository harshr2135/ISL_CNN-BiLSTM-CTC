import cv2
import numpy as np
import tensorflow as tf
import json
from utils.video_loader import extract_frames
from utils.ctc_decoder import ctc_greedy_decode

# CONFIG
MODEL_PATH = "models/isl_ctc_50_words_<timestamp>.keras"  # update with actual
LABEL_MAP_PATH = "label_map.json"
MAX_FRAMES = 60
TARGET_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded.")

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    label_data = json.load(f)
    id_to_word = {int(k): v for k, v in label_data["label_to_word"].items()}
    blank_index = len(label_data["word_to_label"])  # last index for blank

# Capture webcam or load a test video
cap = cv2.VideoCapture(0)  # or replace with path to test .mp4
frames = []

print("🎥 Capturing... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, TARGET_SIZE)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frames.append(frame_norm)

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Pad or trim
if len(frames) > MAX_FRAMES:
    frames = frames[:MAX_FRAMES]
else:
    pad_len = MAX_FRAMES - len(frames)
    frames.extend([np.zeros_like(frames[0])] * pad_len)

sequence = np.expand_dims(np.array(frames), axis=0)  # shape: (1, T, H, W, 3)

# Predict
predictions = model.predict(sequence)[0]  # shape: (T, vocab+1)
decoded_ids = ctc_greedy_decode(predictions, blank_index)

# Convert to words
predicted_words = [id_to_word[i] for i in decoded_ids]
print("🧠 Predicted:", " ".join(predicted_words))
