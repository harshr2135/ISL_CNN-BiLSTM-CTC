import numpy as np

def ctc_greedy_decode(predictions, blank_index):
    """
    Greedy decode CTC output.
    Args:
        predictions: np.array of shape (T, vocab_size + 1), logits
        blank_index: index of the CTC blank token
    Returns:
        Decoded list of label indices
    """
    pred_indices = np.argmax(predictions, axis=-1)
    decoded = []
    prev = -1
    for idx in pred_indices:
        if idx != prev and idx != blank_index:
            decoded.append(idx)
        prev = idx
    return decoded
