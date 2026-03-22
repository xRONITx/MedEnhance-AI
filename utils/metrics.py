import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _prepare_image_array(image):
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _ssim(reference, estimate):
    try:
        return structural_similarity(reference, estimate, channel_axis=-1, data_range=1.0)
    except TypeError:
        return structural_similarity(reference, estimate, multichannel=True, data_range=1.0)


def compute_classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_metrics_from_probabilities(y_true, positive_probabilities, threshold=0.5):
    y_true = np.asarray(y_true)
    positive_probabilities = np.asarray(positive_probabilities)
    predictions = (positive_probabilities >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, predictions)
    metrics["threshold"] = float(threshold)
    return metrics


def find_best_classification_threshold(y_true, positive_probabilities, step=0.01):
    y_true = np.asarray(y_true)
    positive_probabilities = np.asarray(positive_probabilities)

    best_threshold = 0.5
    best_metrics = compute_metrics_from_probabilities(y_true, positive_probabilities, threshold=0.5)
    best_score = (best_metrics["f1"], best_metrics["recall"], best_metrics["accuracy"])

    candidate_thresholds = np.arange(0.2, 0.81, step)
    for threshold in candidate_thresholds:
        metrics = compute_metrics_from_probabilities(y_true, positive_probabilities, threshold=float(threshold))
        score = (metrics["f1"], metrics["recall"], metrics["accuracy"])
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def compute_enhancement_metrics(reference, enhanced):
    reference = _prepare_image_array(reference)
    enhanced = _prepare_image_array(enhanced)
    return {
        "psnr": float(peak_signal_noise_ratio(reference, enhanced, data_range=1.0)),
        "ssim": float(_ssim(reference, enhanced)),
    }
