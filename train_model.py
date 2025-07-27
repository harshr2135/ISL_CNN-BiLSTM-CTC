import os
import datetime
import json
import tensorflow as tf
from models.isl_ctc_model import build_isl_ctc_model
from models.ctc_trainer import CTCModel
from utils.dataset_loader import create_dataset, load_label_map
from utils.metrics import ctc_decode_batch, calculate_wer

# === CONFIG ===
DATA_DIR = "data/processed_data"
LABEL_MAP_PATH = "data/label_map.json"
LOG_DIR = "logs"
MODEL_DIR = "models"
EPOCHS = 30
BATCH_SIZE = 8
MAX_FRAMES = 60
BUFFER_SIZE = 1000  # For shuffling

# === PREP ===
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load label mappings
word_to_label, label_to_word = load_label_map(LABEL_MAP_PATH)
vocab_size = len(word_to_label)
blank_index = vocab_size  # For CTC blank

# === Load Dataset ===
full_dataset = create_dataset(
    data_dir=DATA_DIR,
    label_map_path=LABEL_MAP_PATH,
    max_frames=MAX_FRAMES
).shuffle(BUFFER_SIZE)

# Dynamically determine dataset size (optional)
TOTAL_SAMPLES = 600  # Replace with this if needed:
# TOTAL_SAMPLES = sum(1 for _ in full_dataset)

train_size = int(0.8 * TOTAL_SAMPLES)

# Split + Batch
train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Build Model ===
base_model = build_isl_ctc_model(
    input_shape=(MAX_FRAMES, 224, 224, 3),
    vocab_size=vocab_size
)

model = CTCModel(base_model)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# === CTC Loss ===
def custom_ctc_loss(y_true, y_pred):
    return tf.keras.backend.ctc_batch_cost(
        y_true["labels"], y_pred,
        y_true["input_length"], y_true["label_length"]
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=custom_ctc_loss
)

# === Save & Logging ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"isl_ctc_{vocab_size}_words_{timestamp}.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, model_name),
        save_best_only=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, f"ctc_logs_{timestamp}"))
]

# === Debug Preview ===
print("✅ Ready to train...")
print("Epochs:", EPOCHS)
for sample in train_dataset.take(1):
    print("Sample X shape:", sample[0].shape)
    print("Sample Y shape:", sample[1]["labels"].shape)

# === Train ===
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Evaluation ===
def evaluate_model(model, val_dataset, label_to_word):
    blank_index = len(label_to_word)
    total_correct = 0
    total_words = 0
    total_wer = 0
    num_samples = 0

    for x, y_true in val_dataset:
        y_true_labels = y_true["labels"].numpy()
        y_pred = model.predict(x)

        decoded_batch = ctc_decode_batch(y_pred, blank_index)

        for pred_seq, true_seq in zip(decoded_batch, y_true_labels):
            true_seq = true_seq[true_seq != -1].tolist()  # Remove padding
            if pred_seq == true_seq:
                total_correct += 1
            total_wer += calculate_wer(true_seq, pred_seq)
            total_words += len(true_seq)
            num_samples += 1

    accuracy = total_correct / num_samples
    avg_wer = total_wer / num_samples

    print("\n🧪 Final Evaluation on Validation Set:")
    print(f"✅ Exact Match Accuracy: {accuracy:.4f}")
    print(f"📝 Avg Word Error Rate (WER): {avg_wer:.4f}")

# === Run Evaluation ===
evaluate_model(model, val_dataset, label_to_word)
