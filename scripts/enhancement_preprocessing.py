import tensorflow as tf
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SatelliteDataset:
    """Custom Dataset class for satellite image pairs with multiple noise types"""
    def __init__(self, clean_dir, noisy_dir, img_size):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.img_size = img_size
        self.image_pairs = self._get_image_pairs()
        
    def _get_image_pairs(self):
        """Get all matching clean and noisy image pairs"""
        clean_paths = sorted(glob.glob(os.path.join(self.clean_dir, '*.tif')))
        image_pairs = []
        
        for clean_path in clean_paths:
            base_name = os.path.splitext(os.path.basename(clean_path))[0]
            noisy_patterns = os.path.join(self.noisy_dir, f'*_{base_name}.tif')
            noisy_images = sorted(glob.glob(noisy_patterns))
            
            for noisy_path in noisy_images:
                image_pairs.append((clean_path, noisy_path))
        
        return image_pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def create_dataset(self, batch_size=32, shuffle=True):
        """Create a TensorFlow dataset"""
        # Convert image pairs to lists of clean and noisy paths
        clean_paths = [pair[0] for pair in self.image_pairs]
        noisy_paths = [pair[1] for pair in self.image_pairs]
        
        # Create path datasets
        clean_path_ds = tf.data.Dataset.from_tensor_slices(clean_paths)
        noisy_path_ds = tf.data.Dataset.from_tensor_slices(noisy_paths)
        
        # Combine paths into a single dataset
        path_ds = tf.data.Dataset.zip((clean_path_ds, noisy_path_ds))
        
        # Apply shuffling if requested
        if shuffle:
            # Make sure buffer_size is valid
            buffer_size = len(self.image_pairs)
            if buffer_size > 0:
                path_ds = path_ds.shuffle(buffer_size=buffer_size, 
                                        reshuffle_each_iteration=True)
        
        # Map to load and process images
        dataset = path_ds.map(
            lambda clean_path, noisy_path: tf.py_function(
                self._process_image_pair,
                [clean_path, noisy_path],
                [tf.float32, tf.float32, tf.string, tf.string, tf.string]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes explicitly
        dataset = dataset.map(
            lambda clean, noisy, noise_type, clean_path, noisy_path: {
                'clean': tf.ensure_shape(clean, [self.img_size, self.img_size, 3]),
                'noisy': tf.ensure_shape(noisy, [self.img_size, self.img_size, 3]),
                'noise_type': noise_type,
                'clean_path': clean_path,
                'noisy_path': noisy_path
            }
        )
        
        # Batch and prefetch
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def _process_image_pair(self, clean_path, noisy_path):
        """Process a single image pair"""
        # Decode paths from tensors
        clean_path = clean_path.numpy().decode('utf-8')
        noisy_path = noisy_path.numpy().decode('utf-8')
        
        # Read and process images
        clean_img = self._load_and_preprocess(clean_path)
        noisy_img = self._load_and_preprocess(noisy_path)
        
        # Get noise type from filename
        noise_type = os.path.basename(noisy_path).split('_')[0]
        
        return (
            clean_img,
            noisy_img,
            noise_type.encode('utf-8'),
            clean_path.encode('utf-8'),
            noisy_path.encode('utf-8')
        )
    
    def _load_and_preprocess(self, image_path):
        """Load and preprocess a single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img, dtype=np.float32) / 255.0
        return img

class DataPreprocessor:
    def __init__(self, clean_dir, noisy_dir, img_size):
        """
        Initialize the data preprocessor
        
        Args:
            clean_dir (str): Directory containing clean images
            noisy_dir (str): Directory containing noisy images
            img_size (int): Size to resize images to
        """
        # Verify directories exist
        if not os.path.exists(clean_dir):
            raise ValueError(f"Clean directory does not exist: {clean_dir}")
        if not os.path.exists(noisy_dir):
            raise ValueError(f"Noisy directory does not exist: {noisy_dir}")
            
        self.dataset = SatelliteDataset(clean_dir, noisy_dir, img_size)
        print(f"Found {len(self.dataset)} image pairs")
        
    def get_loaders(self, batch_size=32, train_ratio=0.8, val_ratio=0.1):
        """Create train, validation, and test data loaders"""
        # Calculate split sizes
        total_size = len(self.dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Create full dataset
        full_dataset = self.dataset.create_dataset(batch_size=batch_size)
        
        # Create splits using skip and take
        train_dataset = full_dataset.take(train_size)
        temp_dataset = full_dataset.skip(train_size)
        val_dataset = temp_dataset.take(val_size)
        test_dataset = temp_dataset.skip(val_size)
         
        print(f"Dataset splits created:")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Test samples: {test_size}")
        
        return train_dataset, val_dataset, test_dataset
    
    def visualize_batch(self, batch, num_samples=4):
        """Visualize a batch of images"""
        clean_images = batch['clean']
        noisy_images = batch['noisy']
        noise_types = batch['noise_type']
        
        plt.figure(figsize=(12, 6))
        for i in range(min(num_samples, len(clean_images))):
            # Clean image
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(clean_images[i])
            plt.axis('off')
            plt.title('Clean')
            
            # Noisy image
            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(noisy_images[i])
            plt.axis('off')
            plt.title(f'Noisy ({noise_types[i].numpy().decode("utf-8")})')
        
        plt.tight_layout()
        plt.show()