import numpy as np
from PIL import Image
import pywt
from scipy.stats import kurtosis, skew, norm, poisson
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from joblib import load
from scipy import ndimage
import math
from unet import build_unet
import tensorflow as tf

def load_model_weights(weights_path):
    """Load model weights"""
    model = build_unet(input_shape=(224, 224, 3))  # Ensure architecture matches weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    try:
        model.load_weights(weights_path)
    except Exception as e:
        raise ValueError(f"Error loading weights: {e}")
    return model

cnn_model = load_model('cnn_model.keras')
unet_model = load_model_weights('unet_final_model.h5')

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Ensure both images are in the same shape and type
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Assuming maximum pixel value is 255
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def extract_noise_component(image):
    """Extract the noise component from the image using multiple techniques"""
    # 1. High-frequency component extraction using wavelet decomposition
    coeffs = pywt.wavedec2(image, 'db1', level=1)
    # Reconstruct using only detail coefficients
    noise_wavelet = pywt.waverec2([np.zeros_like(coeffs[0]), coeffs[1]], 'db1')
    
    # 2. Local variation analysis
    # Estimate local structure using median filter
    structure = ndimage.median_filter(image, size=3)
    noise_median = image - structure
    
    # 3. High-pass filtering
    highpass_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]) / 8
    noise_highpass = ndimage.convolve(image, highpass_kernel)
    
    # Combine noise estimates (normalize and average)
    noise_components = np.stack([
        (noise_wavelet - noise_wavelet.mean()) / (noise_wavelet.std() + 1e-10),
        (noise_median - noise_median.mean()) / (noise_median.std() + 1e-10),
        (noise_highpass - noise_highpass.mean()) / (noise_highpass.std() + 1e-10)
    ])
    
    noise_estimate = np.mean(noise_components, axis=0)
    
    # Normalize to [-1, 1] range
    noise_estimate = (noise_estimate - noise_estimate.min()) / (noise_estimate.max() - noise_estimate.min() + 1e-10) * 2 - 1
    
    return noise_estimate

def extract_noise_specific_features(image):
    """Extract statistical features from the noise component"""
    noise_component = extract_noise_component(image)
    
    # Local variance analysis
    patches = np.lib.stride_tricks.sliding_window_view(noise_component, (8, 8))
    local_vars = np.var(patches, axis=(2, 3))
    var_of_vars = np.var(local_vars)
    local_var_skew = skew(local_vars.flatten())
    local_var_kurt = kurtosis(local_vars.flatten())
    
    # Distribution analysis of noise component
    noise_skew = skew(noise_component.flatten())
    noise_kurt = kurtosis(noise_component.flatten())
    
    # Power spectrum analysis
    fft_noise = np.abs(np.fft.fft2(noise_component))
    fft_mean = np.mean(fft_noise)
    fft_std = np.std(fft_noise)
    
    # Directional analysis (for pattern detection)
    gradients = np.gradient(noise_component)
    gradient_coherence = np.corrcoef(gradients[0].flatten(), gradients[1].flatten())[0,1]
    
    features = np.array([
        var_of_vars, local_var_skew, local_var_kurt,
        noise_skew, noise_kurt,
        fft_mean, fft_std,
        gradient_coherence
    ])
    
    return features

def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    try:
        # Load and preprocess the image
        image = Image.open(file_path).convert('L')  
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_normalized = image_array / 255.0

        image_color = Image.open(file_path).convert('RGB')
        image_color = image_color.resize((224,224))
        image_color_array = np.array(image_color)
        image_color_normalized = image_color_array/255.0

        # Calculate features
        noise_component = extract_noise_component(image_normalized)
        features = extract_noise_specific_features(image_normalized)

        enhanced_image_array = unet_model.predict(np.expand_dims(image_color_normalized, axis=0))
        enhanced_image_array = np.squeeze(enhanced_image_array)  # Remove the batch dimension
        enhanced_image_array = (enhanced_image_array * 255).astype(np.uint8) 

        # Calculate PSNR
        psnr_value = calculate_psnr(enhanced_image_array, image_color_array)


        noise_input = np.expand_dims(noise_component, axis=0)
        features_input = np.expand_dims(features, axis=0)

        # Predict probabilities using the CNN model
        probabilities_cnn = cnn_model.predict([noise_input, features_input])

        # Convert probabilities to a numpy array
        probabilities_cnn_array = np.array(probabilities_cnn)

        # Get class value based on max probability
        class_index = np.argmax(probabilities_cnn_array)

        print(probabilities_cnn)
        
        # Display result
        if class_index == 0:
            class_name = "Gaussian"
        elif class_index == 1:
            class_name = "Lognormal"
        elif class_index == 2:
            class_name = "Poisson"
        elif class_index == 3:
            class_name = "Rayleigh"
        elif class_index == 4:
            class_name = "Salt and Pepper"
        else:
            class_name = "unknown"

        import matplotlib.pyplot as plt

        # Create figure with custom gridspec
        fig = plt.figure(figsize=(15, 12))
        from matplotlib import gridspec
        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2])

        # Top row - smaller plots (2x4 grid)
        plt.subplot(gs[0, 0])
        plt.title("Original Image")
        plt.imshow(image_color_array, cmap='gray')
        plt.axis('off')

        plt.subplot(gs[0, 1])
        plt.title("Image Histogram")
        plt.hist(image_normalized.reshape(-1), bins=256, range=(0, 1), color='r')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.subplot(gs[0, 2])
        plt.title("Extracted noise component")
        plt.imshow(noise_component, cmap='gray')
        plt.axis('off')

        plt.subplot(gs[0, 3])
        plt.title("Noise histogram")
        plt.hist(noise_component.reshape(-1), bins=256, range=(0, 1), color='r')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Bottom row - large enhanced image (spans all columns)
        plt.subplot(gs[1, :])
        plt.title(f"Enhanced Image (PSNR: {psnr_value:.2f} dB)")
        plt.imshow(enhanced_image_array, cmap='gray')
        plt.axis('off')

        plt.suptitle(f"Predicted Class: {class_name}", fontsize=16, y=0.95)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the UI
root = tk.Tk()
root.title("Image Classification")

btn_classify = tk.Button(root, text="Upload Image and Classify", command=classify_image)
btn_classify.pack(pady=20)

root.mainloop()