import cv2 as cv
import numpy as np

def image_from_tensor(image):
    """
    Converts image from CPU Torch tensor to OpenCV BGRA image.
    """
    image = image.numpy().transpose((1, 2, 0))
    image = (image * 255).astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    return image

def mask_from_array(mask):
    """
    Converts mask from NumPy array to OpenCV BGRA image.
    """
    rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
    for i, row in enumerate(mask):
        for j, val in enumerate(row):
            if val == 1: # Silhouette
                colour = (255, 0, 0, 255) # Red
            elif val == 2: # Ridge
                colour = (0, 255, 0, 255) # Green
            elif val == 3: # Ligament
                colour = (0, 0, 255, 255) # Blue
            if val != 0:
                rgba_mask[i][j] = colour
    bgra_mask = cv.cvtColor(rgba_mask, cv.COLOR_RGBA2BGRA)
    return bgra_mask

def image_mask_overlay(sample):
    """
    Takes dataset sample (tuple of image CPU tensor and mask NumPy array) a converts into an overlayed OpenCV image.
    """
    image, mask = sample
    image = image_from_tensor(image)
    mask = mask_from_array(mask)
    
    overlay = cv.addWeighted(image, 1, mask, 1, 0)
    return overlay

def sample_combo(sample):
    """
    Creates OpenCV image showing the image, the mask, and overlayed result.
    """
    image, mask = sample
    image = image_from_tensor(image)
    mask = mask_from_array(mask)
    overlay = image_mask_overlay(sample)

    stack = np.hstack((image, mask, overlay))
    return stack

def sample_predict_combo(sample, predict):
    """
    Creates OpenCV image showing the image with its ground truth mask, alongside the image with a given predicted mask.
    """
    truth = sample_combo(sample)
    guess = sample_combo(predict)
    stack = np.vstack((truth, guess))
    return stack