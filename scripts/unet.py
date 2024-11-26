import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(224, 224, 3), num_filters=64):
    """
    Build a UNet model for image enhancement/denoising
    
    Args:
        input_shape (tuple): Input image shape
        num_filters (int): Base number of filters in the first layer
    
    Returns:
        tf.keras.Model: Compiled UNet model
    """
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(num_filters * 16, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(num_filters * 16, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c9)
    
    return models.Model(inputs, outputs)