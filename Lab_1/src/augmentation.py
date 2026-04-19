import numpy as np
import cv2
import random
from skimage.util import random_noise
from skimage import exposure
from skimage.transform import rotate

def random_flip(image: np.ndarray) -> np.ndarray:
    """
    Randomly flip image

    Parameters:
        image (np.ndarray): HxWxC or HxW image

    Returns:
        np.ndarray: Flipped image
    """

    mode = random.choice(["keep", "horizontal", "vertical", "both"])

    if mode == "horizontal":
        return cv2.flip(image, 1)
    elif mode == "vertical":
        return cv2.flip(image, 0)
    elif mode == "both":
        return cv2.flip(image, -1)
    else:
        return image
    
def random_rotate(image: np.ndarray) -> np.ndarray:
    """
    Randomly rotate image

    Parameters:
        image (np.ndarray): HxWxC or HxW image
    
    Returns:
        np.ndarray: Rotated image
    """

    img_float = image.astype(np.float64) / 255.0
    random_angle = np.random.randint(0, 360)
    return (rotate(img_float, random_angle) * 255).astype(np.uint8)

def random_crop(image: np.ndarray) -> np.ndarray:
    """
    Randomly crop image and resize back to the original size

    Parameters:
        image (np.ndarray): HxWxC or HxW image

    Returns:
        np.ndarray: Cropped image
    """
    h, w = image.shape[:2]

    random_crop_ratio = random.uniform(0.8, 1)

    ch = int(h * random_crop_ratio)
    cw = int(w * random_crop_ratio)

    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    cropped = image[top:top + ch, left:left + cw]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def random_add_gaussian_noise(image) -> np.ndarray:
    """
    Randomly add Gaussian noise to image

    Parameters:
        image (np.ndarray): HxWxC or HxW image

    Returns:
        np.ndarray: Noise-added image
    """

    img_float = image.astype(np.float64) / 255.0
    noise_img = random_noise(img_float)
    return (noise_img * 255).astype(np.uint8)

def random_change_brightness(image: np.ndarray) -> np.ndarray:
    """
    Randomly change image's brightness

    Parameters:
        image (np.ndarray): HxWxC or HxW image

    Returns:
        np.ndarray: Brightness-changed image
    """

    random_brightness = int(np.random.normal(0, 20))

    img = image.astype(np.int16)
    img = img + random_brightness
    img = np.clip(img, 0, 255).astype(image.dtype)

    return img

def random_change_contrast(image: np.ndarray) -> np.ndarray:
    """
    Randomly change image's contrast

    Parameters:
        image (np.ndarray): HxWxC or HxW image

    Returns:
        np.ndarray: Contrast-changed image
    """

    random_contrast = random.uniform(0.8, 1.2)
    
    img = image.astype(np.float32)
    
    if img.ndim == 3:
        mean = img.mean(axis=(0, 1), keepdims=True)
    else:
        mean = img.mean()

    img = (img - mean) * random_contrast + mean
    img = np.clip(img, 0, 255).astype(image.dtype)

    return img

class AugmentationPipeline:
    def __init__(self, apply_probability: float):
        self.apply_probability = np.clip(apply_probability, 0, 1).astype(np.float32)

    def _should_apply(self) -> bool:
        return random.random() < self.apply_probability

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()

        # flip
        if self._should_apply():
            img = random_flip(img)

        # rotate
        if self._should_apply():
            img = random_rotate(img)

        # brightness
        if self._should_apply():
            img = random_change_brightness(img)

        # contrast
        if self._should_apply():
            img = random_change_contrast(img)
        
        # noise
        if self._should_apply():
            img = random_add_gaussian_noise(img)

        # crop
        if self._should_apply():
            img = random_crop(img)

        return img
    
    def apply_batch(self, images: np.ndarray, n_copies: int=1) -> np.ndarray:
        """
        Augment a batch of NxHxWxC images

        Parameters:
            images (np.ndarray): Batch of images
            n_copies (int): Number of augmented copies produced per input image

        Returns:
            (np.ndarray): Batch of augmented images (length = len(images) * n_copies)
        """

        results = []
        for img in images:
            for _ in range(n_copies):
                results.append(self.apply(img))
        return np.array(results)