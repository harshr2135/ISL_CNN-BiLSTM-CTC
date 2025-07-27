import os
import json

def create_label_map(data_root, output_path):
    """
    Create a label2id and id2label dictionary from folder names
    """
    class_names = sorted([f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))])
    word_to_label = {word: idx for idx, word in enumerate(class_names)}
    label_to_word = {idx: word for word, idx in word_to_label.items()}
    
    with open(output_path, 'w') as f:
        json.dump({
            'word_to_label': word_to_label,
            'label_to_word': label_to_word
        }, f, indent=2)

    print(f"Label map created with {len(class_names)} classes.")
    return word_to_label
