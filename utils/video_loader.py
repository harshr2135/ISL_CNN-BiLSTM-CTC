import cv2
import numpy as np

SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

def extract_frames(video_path, target_size=(224, 224), max_frames=60):
    """
    Extract and resize up to `max_frames` from a video at evenly spaced intervals.
    Returns a NumPy array of shape (T, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return np.zeros((max_frames, *target_size, 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0  # Normalize to [0, 1]
            frames.append(frame)
    
    cap.release()
    
    # Pad if fewer than max_frames
    num_padding = max_frames - len(frames)
    if num_padding > 0:
        pad = np.zeros((num_padding, *target_size, 3), dtype=np.float32)
        frames.extend(pad)
    
    return np.array(frames, dtype=np.float32)
