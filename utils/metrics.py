import numpy as np

def ctc_decode_batch(predictions, blank_index):
    batch_decoded = []
    for pred in predictions:
        pred_ids = np.argmax(pred, axis=-1)
        decoded = []
        prev = -1
        for idx in pred_ids:
            if idx != prev and idx != blank_index:
                decoded.append(idx)
            prev = idx
        batch_decoded.append(decoded)
    return batch_decoded

def calculate_wer(ref, hyp):
    """
    Compute Word Error Rate between two lists of word IDs
    """
    import editdistance
    return editdistance.eval(ref, hyp) / max(len(ref), 1)
