import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, GlobalAveragePooling2D, Dense, multiply, add, Activation

from src.models.unet import build_encoder, conv2d_block 

def attention_block(x, g, inter_channel):
    """
    Attention Gate: Focuses on target structures (pets) and suppresses background.
    x: Skip connection from the encoder (High resolution, weak semantics)
    g: Gating signal from the decoder (Low resolution, strong semantics)
    """

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    

    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x, rate])
    return att_x

def build_attention_decoder(c1, c2, c3, c4, c5):
    """
    Builds the upsampling path using Attention Gates before concatenation.
    """

    u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    att4 = attention_block(x=c4, g=u6, inter_channel=128)
    u6 = concatenate([u6, att4])
    c6 = conv2d_block(u6, n_filters=128)


    u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
    att3 = attention_block(x=c3, g=u7, inter_channel=64)
    u7 = concatenate([u7, att3])
    c7 = conv2d_block(u7, n_filters=64)


    u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
    att2 = attention_block(x=c2, g=u8, inter_channel=32)
    u8 = concatenate([u8, att2])
    c8 = conv2d_block(u8, n_filters=32)


    u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c8)
    att1 = attention_block(x=c1, g=u9, inter_channel=16)
    u9 = concatenate([u9, att1], axis=3)
    c9 = conv2d_block(u9, n_filters=16)
    
    return c9

def build_attention_unet(input_shape=(128, 128, 3), num_classes=37):
    """
    Assembles the Multi-Task Attention U-Net.
    """
    inputs = Input(input_shape)
    
    c1, c2, c3, c4, c5 = build_encoder(inputs)
    
    c9 = build_attention_decoder(c1, c2, c3, c4, c5)
    
    mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(c9)
    
    class_pool = GlobalAveragePooling2D()(c5)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(class_pool)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[mask_output, class_output], name="MultiTask_Attention_UNet")
    return model