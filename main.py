import os
import tensorflow as tf
from src.dataset import get_dataloaders
from src.models.attention_unet import build_attention_unet
from keras.callbacks import ModelCheckpoint, EarlyStopping


IMAGES_DIR = 'data/images/'
MASKS_DIR = 'data/annotations/trimaps/'
BATCH_SIZE = 32
TARGET_SIZE = (128, 128)
EPOCHS = 10

def main():
    print("Loading data...")
    train_ds, val_ds, class_names = get_dataloaders(
        IMAGES_DIR, MASKS_DIR, 
        batch_size=BATCH_SIZE, 
        target_size=TARGET_SIZE
    )
    print(f"Done. {len(class_names)} breeds found.")

    print("Building  Model...")
    model = build_attention_unet(input_shape=(128, 128, 3), num_classes=len(class_names))
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'mask_output': 'binary_crossentropy',
            'class_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'mask_output': 'accuracy',
            'class_output': 'accuracy'
        }
    )

    checkpoint_cb = ModelCheckpoint(
        "models/unet_best_model.keras", 
        save_best_only=True, 
        monitor='val_loss'
    )

    checkpoint_cb = ModelCheckpoint(
        "models/attention_unet_best_model.keras", 
        save_best_only=True, 
        monitor='val_loss'
    )

    early_stop_cb = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb]
    )
    
    print("Done. Model saved to models/")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()