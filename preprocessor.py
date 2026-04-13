"""
Image preprocessing module
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class Preprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def resize(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
        """Resize image to specified dimensions"""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values"""
        return cv2.normalize(image, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    @staticmethod
    def blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur to image"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            # For color images, convert to HSV and equalize V channel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # For grayscale
            return cv2.equalizeHist(image)
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def adaptive_brightness(image: np.ndarray, clip_limit: float = 2.0,
                           tile_size: int = 8) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=(tile_size, tile_size))
        
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = clahe.apply(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return clahe.apply(image)
    
    @staticmethod
    def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """Apply non-local means denoising"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h=strength,
                                                   hForColorComponents=strength,
                                                   templateWindowSize=7,
                                                   searchWindowSize=21)
        else:
            return cv2.fastNlMeansDenoising(image, None, h=strength,
                                           templateWindowSize=7,
                                           searchWindowSize=21)
    
    @staticmethod
    def edge_detection(image: np.ndarray, threshold1: int = 50,
                      threshold2: int = 150) -> np.ndarray:
        """Detect edges using Canny method"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return cv2.Canny(gray, threshold1, threshold2)
    
    @staticmethod
    def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Reduce noise using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image
    
    @staticmethod
    def preprocessing_pipeline(image: np.ndarray, resize_width: int = 640,
                              resize_height: int = 480,
                              denoise: bool = False,
                              equalize: bool = False,
                              gamma: float = 1.0) -> np.ndarray:
        """
        Full preprocessing pipeline
        
        Args:
            image: Input image
            resize_width: Target width
            resize_height: Target height
            denoise: Apply denoising
            equalize: Apply histogram equalization
            gamma: Gamma correction value
        
        Returns:
            Preprocessed image
        """
        # Resize
        result = Preprocessor.resize(image, resize_width, resize_height)
        
        # Denoise if requested
        if denoise:
            result = Preprocessor.denoise(result)
        
        # Histogram equalization
        if equalize:
            result = Preprocessor.histogram_equalization(result)
        
        # Gamma correction
        if gamma != 1.0:
            result = Preprocessor.gamma_correction(result, gamma)
        
        return result
