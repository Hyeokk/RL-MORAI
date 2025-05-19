import numpy as np

def preprocess_image(image):
    image = image.astype(np.float32) / 255.0
    if image.ndim == 2:  # grayscale, no channel dim
        image = image[:, :, None]
    image = np.transpose(image, (2, 0, 1))
    return image