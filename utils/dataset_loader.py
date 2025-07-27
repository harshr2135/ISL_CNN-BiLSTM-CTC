import os
import numpy as np
import tensorflow as tf
import json

def load_label_map(label_path):
    with open(label_path, 'r') as f:
        label_map = json.load(f)
    return label_map['word_to_label'], label_map['label_to_word']

def encode_label(label_str, word_to_label):
    return [word_to_label[word] for word in label_str.strip().split()]

def create_dataset(data_dir, label_map_path, max_frames=60, shuffle=True):
    word_to_label, _ = load_label_map(label_map_path)
    sample_paths, labels = [], []

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.npy'):
                sample_paths.append(os.path.join(class_dir, file))
                labels.append(class_name)

    def gen():
        for path, label in zip(sample_paths, labels):
            x = np.load(path).astype(np.float32)
            label_seq = encode_label(label, word_to_label)
            yield x, {
                "labels": np.array(label_seq, dtype=np.int32),
                "input_length": np.array([len(x)], dtype=np.int32),
                "label_length": np.array([len(label_seq)], dtype=np.int32)
            }

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(max_frames, 224, 224, 3), dtype=tf.float32),
            {
                "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "label_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            }
        )
    )

    if shuffle:
        dataset = dataset.shuffle(len(sample_paths))
    return dataset
