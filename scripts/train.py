import os
import tensorflow as tf
from data_pre import DataPreprocessor
from unet import build_unet

# Dataset paths
BASE_DIR = os.path.join('..', '..', 'dataset', 'sat_dataset')
CLEAN_DIR = os.path.join(BASE_DIR, 'clean')
NOISY_DIR = os.path.join(BASE_DIR, 'noisy')

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

def prepare_dataset(dataset):
    """Extract clean and noisy images from dataset"""
    def extract_images(batch):
        return batch['noisy'], batch['clean']
    
    return dataset.map(extract_images)

def psnr_metric(y_true, y_pred):
    """Custom PSNR metric"""
    max_pixel = 1.0
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)

def main():
    # Ensure output directories exist
    os.makedirs('models', exist_ok=True)

    # Initialize data preprocessor
    preprocessor = DataPreprocessor(CLEAN_DIR, NOISY_DIR, IMG_SIZE)
    train_ds, val_ds, test_ds = preprocessor.get_loaders(batch_size=BATCH_SIZE)

    # Prepare datasets
    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds)

    # Build U-Net model
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Compile model with custom metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=[
            'mae', 
            psnr_metric
        ]
    )

    # Callbacks
    # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    #     os.path.join('models', 'unet_best_model.keras'), 
    #     save_best_only=True, 
    #     monitor='val_loss', 
    #     mode='min'
    # )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('models', 'unet_epoch_{epoch:02d}.weights.h5'),  # Save weights with epoch number
        save_weights_only=True,  # Save only the weights
        save_best_only=False,    # Save after every epoch, not just the best
        verbose=1                # Print a message when saving
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, 
        restore_best_weights=True, 
        monitor='val_loss', 
        mode='min'
    )

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    # Save final model
    model.save(os.path.join('models', 'unet_final_model.h5'))

    # Evaluate on test data
    test_loss, test_mae, test_psnr = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test PSNR: {test_psnr}")

if __name__ == '__main__':
    main()