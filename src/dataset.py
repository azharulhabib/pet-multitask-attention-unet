import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_paths_and_labels(images_dir, masks_dir):
    """
    Scans directories, matches images to masks, and generates integer labels.
    Returns: image_paths, mask_paths, labels, class_names
    """
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    image_paths = []
    mask_paths = []
    class_names_raw = []
    
    for file_name in image_filenames:
        img_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(masks_dir, file_name.replace('.jpg', '.png'))
        
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
            
            breed_name = "_".join(file_name.split('_')[:-1])
            class_names_raw.append(breed_name)
            
    class_names = sorted(list(set(class_names_raw)))
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = [class_to_idx[name] for name in class_names_raw]
    
    return image_paths, mask_paths, labels, class_names


def parse_function(img_path, mask_path, label, target_size=(128, 128)):
    """
    Reads files, decodes images/masks, normalizes, and fixes mask boundaries.
    """
    img_string = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_string, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size)
    
    mask_string = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_string, channels=1)
    mask = tf.cast(mask, tf.float32)
    mask = tf.image.resize(mask, target_size, method='nearest')
    
    mask = tf.where(mask == 2.0, 0.0, mask)
    mask = tf.where(mask == 3.0, 1.0, mask)
    
    return img, (mask, label)


def augment_data(img, targets):
    """
    Applies random horizontal flip to BOTH image and mask simultaneously.
    Applies slight random brightness to the image only.
    """
    mask, label = targets
    
    combined = tf.concat([img, mask], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    
    img = combined[:, :, :3]
    mask = combined[:, :, 3:]
    
    img = tf.image.random_brightness(img, max_delta=0.2)
    
    return img, (mask, label)


def get_dataloaders(images_dir, masks_dir, batch_size=32, target_size=(128, 128), test_split=0.2):
    """
    function to create the Train and Validation TensorFlow Datasets.
    """
    img_paths, mask_paths, labels, class_names = prepare_paths_and_labels(images_dir, masks_dir)
    
    X_img_train, X_img_val, X_mask_train, X_mask_val, y_train, y_val = train_test_split(
        img_paths, mask_paths, labels, test_size=test_split, random_state=42
    )
    
    def build_tf_dataset(imgs, masks, lbls, is_training):
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks, lbls))
        
        ds = ds.map(
            lambda i, m, l: parse_function(i, m, l, target_size), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(buffer_size=1000)
            
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_dataset = build_tf_dataset(X_img_train, X_mask_train, y_train, is_training=True)
    val_dataset = build_tf_dataset(X_img_val, X_mask_val, y_val, is_training=False)
    
    return train_dataset, val_dataset, class_names