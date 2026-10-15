import numpy as np


def calculate_iou(y_true, y_pred):
    """Calculates Intersection over Union (IoU) for binary segmentation masks."""
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 1.0


def calculate_dice(y_true, y_pred):
    """Calculates Dice coefficient (F1 score for segmentation)."""
    intersection = np.logical_and(y_true, y_pred)
    total_pixels = np.sum(y_true) + np.sum(y_pred)
    return 2.0 * np.sum(intersection) / total_pixels if total_pixels != 0 else 1.0


def calculate_pixel_accuracy(y_true, y_pred):
    """Calculates the percentage of correctly classified pixels."""
    return np.sum(y_true == y_pred) / y_true.size
