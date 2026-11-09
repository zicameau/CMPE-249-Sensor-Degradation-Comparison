"""Camera degradations aligned with fusion docs/degradation.md.
Applied to original BGR images before any model resize.
"""
import cv2
import numpy as np

BRIGHTNESS = {1:0.5, 2:0.25, 3:0.1}
BLUR_RADIUS = {1:2, 2:5, 3:10}

def apply(image, corruption='clean', severity=0):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected HxWx3 image')
    if severity == 0 or corruption == 'clean':
        return image.copy()
    if corruption == 'brightness':
        if severity not in BRIGHTNESS: raise ValueError('brightness severity 1..3')
        return np.clip(image.astype(np.float32)*BRIGHTNESS[severity],0,255).astype(np.uint8)
    if corruption == 'blur':
        if severity not in BLUR_RADIUS: raise ValueError('blur severity 1..3')
        # Fusion docs say Gaussian blur radius (px); OpenCV sigma maps to radius approximately,
        # but exact equivalence requires checking fusion's implementation.
        sigma = BLUR_RADIUS[severity]
        return cv2.GaussianBlur(image, (0,0), sigmaX=sigma, sigmaY=sigma)
    if corruption == 'camera_drop':
        if severity != 1: raise ValueError('camera_drop severity 1 only for CAM_FRONT')
        return np.zeros_like(image)
    raise ValueError(f'Unknown corruption: {corruption}')
