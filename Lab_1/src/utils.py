import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        # 2 images are identical
        return float("inf")
    
    return 10 * np.log10(255 ** 2 / mse)

def convert_rgb_to_color_space(images: np.ndarray, color_space: str) -> np.ndarray:
    """
    Convert RGB images to the specified color space (HSV, LAB, grayscale).

    Parameters:
        images (np.ndarray): Input images in RGB format with shape (N, H, W, 3).
        color_space (str): Target color space to convert to. Supported values: "HSV", "LAB", "gray".

    Returns:
        np.ndarray: Converted images in the specified color space.
    """
    
    if len(images.shape) != 4:
        raise ValueError("Input images should have shape (N, H, W, 3)")
    
    if images.shape[3] != 3:
        raise ValueError("Input images should have 3 channels in RGB formatt")

    if color_space == "RGB":
        return images

    r = images[:, :, :, 0]
    g = images[:, :, :, 1]
    b = images[:, :, :, 2]

    if color_space == "HSV":

        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        c_max = np.maximum(np.maximum(r, g), b)
        c_min = np.minimum(np.minimum(r, g), b)
        delta = c_max - c_min

        # value
        v = c_max

        # saturation
        s = np.zeros_like(r)
        mask = c_max != 0
        s[mask] = delta[mask] / c_max[mask]

        # hue
        h = np.zeros_like(r)

        mask = delta != 0

        idx = (r == c_max) & mask
        h[idx] = 60 * (g[idx] - b[idx]) / delta[idx]

        idx = (g == c_max) & mask
        h[idx] = 60 * (b[idx] - r[idx]) / delta[idx] + 120

        idx = (b == c_max) & mask
        h[idx] = 60 * (r[idx] - g[idx]) / delta[idx] + 240

        h = (h + 360) % 360

        h /= 2
        s *= 255
        v *= 255

        return np.astype(np.stack([h, s, v], axis=3), np.uint8)

    elif color_space == "LAB":
        r = r / 255.0 # (N, H, W)
        g = g / 255.0 # (N, H, W)
        b = b / 255.0 # (N, H, W)

        l = np.zeros_like(r)
        a = np.zeros_like(r)
        b = np.zeros_like(r)

        x = np.zeros_like(r) # (N, H, W)
        y = np.zeros_like(r) # (N, H, W)
        z = np.zeros_like(r) # (N, H, W)

        matrix = np.array([[0.412453, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]]) # (3, 3)
        
        x = matrix[0, 0] * r + matrix[0, 1] * g + matrix[0, 2] * b
        y = matrix[1, 0] * r + matrix[1, 1] * g + matrix[1, 2] * b
        z = matrix[2, 0] * r + matrix[2, 1] * g + matrix[2, 2] * b

        x /= 0.950456
        z /= 1.088754

        l[y > 0.008856] = 116 * (y[y > 0.008856] ** (1 / 3)) - 16
        l[y <= 0.008856] = 903.3 * y[y <= 0.008856]

        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > 0.008856, t ** (1 / 3), (7.787 * t) + (16 / 116))

        a = 500 * (f(x) - f(y)) + 128
        b = 200 * (f(y) - f(z)) + 128

        l = l * 255 / 100
        a += 128
        b += 128

        return np.astype(np.stack([l, a, b], axis=3), np.uint8)

    elif color_space == "gray":
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = np.expand_dims(gray, 3)
        return np.astype(gray, np.uint8)

    else:
        raise ValueError("Invalid color space")
    
def calculate_explained_variance(images: np.ndarray, pca_components: int=50) -> dict[str, float]:
    pca = PCA(n_components=pca_components)
    pca.fit(images.reshape((len(images), -1)))
    return np.sum(pca.explained_variance_ratio_)

def normalize_min_max(images: np.ndarray):
    """
    Normalize images pixels value to range [0, 1].

    Parameters:
        images (np.ndarray): Input images with shape (N, H, W, C).

    Returns:
        np.ndarray: Normalized images with pixels value in range [0, 1].
    """

    return images / 255.0

def normalize_min_max_(images: np.ndarray):
    """
    Normalize images pixels value to range [-1, 1].

    Parameters:
        images (np.ndarray): Input images with shape (N, H, W, C).

    Returns:
        np.ndarray: Normalized images with pixels value in range [-1, 1].
    """

    return images / 127.5 - 1

def z_score_all_channel(images: np.ndarray, mean_std: bool=False):
    """
    Standardize pixel values to have zero mean and unit variance across all channels.

    Apply global Z-score normalization using the mean and standard deviation computed over all pixels and channels.

    Parameters:
        images (np.ndarray): Input images with shape (N, H, W, C).
        mean_std (bool): Whether to return mean and standard deviation.

    Returns:
        np.ndarray: Normalized images with zero mean and unit variance across all channels.
        
        float: Mean.

        float: Standard deviation.
    """
    
    mean = np.mean(images)
    std = np.std(images)

    if mean_std:
        return (images - mean) / std, mean, std
    else:
        return (images - mean) / std

def z_score_per_channel(images: np.ndarray, mean_std: bool=False):
    """
    Standardize pixel values to have zero mean and unit variance per channel.

    Apply per-channel Z-score normalization using channel-wise mean and standard deviation.

    Parameters:
        images (np.ndarray): Input images with shape (N, H, W, C).
        mean_std (bool): Whether to return mean and standard deviation.

    Returns:
        np.ndarray: Normalized images with zero mean and unit variance per channel.

        float: Mean.

        float: Standard deviation.
    """
    
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    if mean_std:
        return (images - mean) / std, mean, std
    else:
        return (images - mean) / std
    
def hamming_distance(hash1, hash2):
    xor = np.bitwise_xor(hash1, hash2)
    return np.unpackbits(xor).sum()