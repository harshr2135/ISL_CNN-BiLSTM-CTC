import os
import numpy as np
import json
from tqdm import tqdm
from utils.video_loader import extract_frames
from utils.label_utils import create_label_map

DATA_ROOT = 'C:/Users/Jyothi/Documents/GitHub/ISL_CNN/data/raw_videos'               # Input folder containing subfolders of videos
OUTPUT_ROOT = 'data/processed_data'       # Output folder to save .npy files
LABEL_MAP_PATH = os.path.join("data", "label_map.json") 
MAX_FRAMES = 60
TARGET_SIZE = (224, 224)

def preprocess_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    word_to_label = create_label_map(DATA_ROOT, LABEL_MAP_PATH)

    for label in tqdm(os.listdir(DATA_ROOT), desc="Processing classes"):
        input_dir = os.path.join(DATA_ROOT, label)
        output_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext not in ['.mp4', '.avi', '.mov', '.mkv']:
                continue

            video_path = os.path.join(input_dir, file)
            try:
                frames = extract_frames(video_path, target_size=TARGET_SIZE, max_frames=MAX_FRAMES)
                save_path = os.path.join(output_dir, file.replace(ext, '.npy'))
                np.save(save_path, frames)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    preprocess_dataset()
