import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, GlobalAveragePooling2D, Dense

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    """
    Adds 2 convolutional layers with ReLU activation and He normal initialization.
    """
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               kernel_initializer="he_normal", padding="same", activation="relu")(input_tensor)
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               kernel_initializer="he_normal", padding="same", activation="relu")(x)
    return x


def build_encoder(inputs):
    """
    Builds the downsampling path of the U-Net.
    Returns the encoder outputs for skip connections and the deep bottleneck.
    """

    c1 = conv2d_block(inputs, n_filters=16)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)


    c2 = conv2d_block(p1, n_filters=32)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)


    c3 = conv2d_block(p2, n_filters=64)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.2)(p3)


    c4 = conv2d_block(p3, n_filters=128)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(0.2)(p4)


    c5 = conv2d_block(p4, n_filters=256)
    
    return c1, c2, c3, c4, c5


def build_decoder(c1, c2, c3, c4, c5):
    """
    Builds the upsampling path of the U-Net using skip connections.
    """
  
    u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=128)


    u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=64)


    u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=32)


    u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=16)
    
    return c9


def build_unet(input_shape=(128, 128, 3), num_classes=37):
    """
    Assembles the full U-Net model with TWO output heads:
    1. mask_output: Segmentation Mask (Foreground/Background)
    2. class_output: Classification Label (37 Pet Breeds)
    """
    inputs = Input(input_shape)
    

    c1, c2, c3, c4, c5 = build_encoder(inputs)
    
    c9 = build_decoder(c1, c2, c3, c4, c5)
    

    mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(c9)
    
    class_pool = GlobalAveragePooling2D()(c5)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(class_pool)

    model = tf.keras.Model(inputs=[inputs], outputs=[mask_output, class_output], name="MultiTask_UNet")
    return model