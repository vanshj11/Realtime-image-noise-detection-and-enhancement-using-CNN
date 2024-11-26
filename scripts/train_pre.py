import numpy as np
import os
import cv2
import pywt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kurtosis, skew
from scipy import ndimage

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

def extract_features(image):
    """Extract features that specifically characterize noise patterns"""
    # 1. Local Variance Analysis
    patches = np.lib.stride_tricks.sliding_window_view(image, (8, 8))
    local_vars = np.var(patches, axis=(2, 3))
    var_of_vars = np.var(local_vars)
    local_var_skew = skew(local_vars.flatten())
    local_var_kurt = kurtosis(local_vars.flatten())
    
    # 2. High-frequency component analysis
    highpass_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]) / 8
    noise_component = ndimage.convolve(image, highpass_kernel)
    noise_skew = skew(noise_component.flatten())
    noise_kurt = kurtosis(noise_component.flatten())
    
    # 3. Coefficient of variation in local windows
    local_means = ndimage.uniform_filter(image, size=8)
    local_stds = np.sqrt(np.maximum(ndimage.uniform_filter(image**2, size=8) - local_means**2, 0))  
    cv_map = np.divide(local_stds, local_means, where=local_means!=0)
    cv_std = np.std(cv_map[~np.isnan(cv_map)])
    
    # 4. Edge coherence
    sobel_x = ndimage.sobel(image, axis=0)
    sobel_y = ndimage.sobel(image, axis=1)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_coherence = np.std(edge_mag) / np.mean(edge_mag) if np.mean(edge_mag) != 0 else 0
    
    # 5. Isolated pixel analysis
    diff_matrix = np.abs(image - ndimage.median_filter(image, size=3))
    isolated_pixels = np.sum(diff_matrix > 0.5 * np.std(image)) / image.size
    
    # 6. Wavelet noise analysis
    coeffs = pywt.wavedec2(image, 'db1', level=2)
    detail_coeffs = coeffs[1][0].flatten()
    wavelet_kurt = kurtosis(detail_coeffs)
    wavelet_skew = skew(detail_coeffs)
    wavelet_energy = np.mean(detail_coeffs**2)
    
    # 7. Rayleigh-specific features
    positive_values = image[image > 0]
    log_mean = np.mean(np.log(positive_values)) if len(positive_values) > 0 else 0
    log_var = np.var(np.log(positive_values)) if len(positive_values) > 0 else 0
    
    return np.array([
        var_of_vars, local_var_skew, local_var_kurt,
        noise_skew, noise_kurt, cv_std,
        edge_coherence, isolated_pixels,
        wavelet_kurt, wavelet_skew, wavelet_energy,
        log_mean, log_var
    ])

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

def preprocess_images(input_folder, output_size=(224, 224)):
    images = []
    features = []
    labels = []

    for noise_class in os.listdir(input_folder):
        class_folder = os.path.join(input_folder, noise_class)
        if os.path.isdir(class_folder): 
            for image_name in os.listdir(class_folder):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', 'tif')):
                    image_path = os.path.join(class_folder, image_name)
                    img = cv2.imread(image_path)

                    if img is None:
                        print(f"Warning: Could not read image {image_path}")
                        continue

                    img_resized = cv2.resize(img, output_size)
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    img_normalized = img_gray / 255.0

                    noise_component = np.array(extract_noise_component(img_normalized))

                    # Calculate image features
                    features_img = extract_noise_specific_features(img_normalized)
                    features_img = np.array(features_img)

                    images.append(noise_component)
                    features.append(features_img)

                    # Get class label
                    labels.append(noise_class)

    # Convert lists to numpy arrays
    images = np.array(images)
    features = np.array(features)
    labels = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    # Save to NPZ file
    np.savez_compressed(
        'train_data.npz',
        images=images,
        features=features,
        labels=y_encoded,
        classes=le.classes_,
    )


    print(images.shape)
    print(features.shape)
    print(labels.shape)

    print("Data saved to NPZ file.")

# run the preprocessing
input_folder = "../dataset/train_im"
preprocess_images(input_folder)