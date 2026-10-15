import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from src.dataset import get_dataloaders
from src.utils import calculate_iou, calculate_dice, calculate_pixel_accuracy

def evaluate_split(model, dataset, split_name):
    """Runs the dataset through the model and computes all rubric metrics."""
    print(f"Evaluating on {split_name} Set...")
    ious, dices, pixel_accs = [], [], []
    all_true_classes, all_pred_classes = [], []

    for images, (true_masks, true_labels) in dataset:
        pred_masks, pred_labels = model.predict(images, verbose=0)
        
        pred_masks_bin = (pred_masks > 0.5).astype(np.float32)
        true_masks_bin = true_masks.numpy()
        
        all_pred_classes.extend(np.argmax(pred_labels, axis=1))
        all_true_classes.extend(true_labels.numpy())
        
        for i in range(len(true_masks_bin)):
            ious.append(calculate_iou(true_masks_bin[i], pred_masks_bin[i]))
            dices.append(calculate_dice(true_masks_bin[i], pred_masks_bin[i]))
            pixel_accs.append(calculate_pixel_accuracy(true_masks_bin[i], pred_masks_bin[i]))

    acc = np.sum(np.array(all_true_classes) == np.array(all_pred_classes)) / len(all_true_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_classes, all_pred_classes, average='macro', zero_division=0
    )

    return {
        "mIoU": np.mean(ious),
        "Dice": np.mean(dices),
        "PixAcc": np.mean(pixel_accs),
        "Class_Acc": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def print_results(model_name, split_name, metrics):
    print(f"\n--- {model_name} ({split_name}) ---")
    print(f"Segmentation -> mIoU: {metrics['mIoU']:.4f} | Dice: {metrics['Dice']:.4f} | Pixel Acc: {metrics['PixAcc']:.4f}")
    print(f"Classification -> Acc: {metrics['Class_Acc']:.4f} | Prec: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")

def main():
    print("Loading Data Splits...")
    train_ds, val_ds, _ = get_dataloaders('data/images/', 'data/annotations/trimaps/', batch_size=32)

    if os.path.exists('models/unet_best_model.keras'):
        print("Loading Baseline U-Net Model...")
        unet_model = tf.keras.models.load_model('models/unet_best_model.keras')
        val_metrics_unet = evaluate_split(unet_model, val_ds, "Validation")
        print_results("Baseline U-Net", "Validation", val_metrics_unet)

    if os.path.exists('models/attention_unet_best_model.keras'):
        print("Loading Attention U-Net Model...")
        att_model = tf.keras.models.load_model('models/attention_unet_best_model.keras')
        val_metrics_att = evaluate_split(att_model, val_ds, "Validation")
        print_results("Attention U-Net", "Validation", val_metrics_att)

if __name__ == "__main__":
    main()