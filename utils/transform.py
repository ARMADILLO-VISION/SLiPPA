import cv2 as cv
import torch
import imgaug.augmenters as iaa

from skimage import transform
from imgaug.augmentables import SegmentationMapsOnImage

class Resize(object):
    """
    Resizes dataset sample to given dimensions.
    """
    def __init__(self, height, width):
        self.height, self.width = height, width
    
    def __call__(self, sample):
        image, mask = sample
        image = transform.resize(image, (self.height, self.width))
        mask = cv.resize(mask, (self.width, self.height), interpolation=cv.INTER_NEAREST)
        return image, mask

class ToTensor(object):
    """
    Converts dataset sample from NumPy arrays to PyTorch tensors.
    """
    def __call__(self, sample):
        image, mask = sample
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

class AugmentIT(object):
    """
    Augmentation pipeline used in initial training.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ], random_order=True)
    
    def __call__(self, sample):
        image, mask = sample
        mask = SegmentationMapsOnImage(mask, image.shape)
        image, mask = self.aug(image=image, segmentation_maps=mask)
        return image, mask.get_arr()

class AugmentFT(object):
    """
    Augmentation pipeline used in fine-tuning
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sharpen(0.0, 1.0),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-45, 45)),
            iaa.PerspectiveTransform()
        ], random_order=True)
    
    def __call__(self, sample):
        image, mask = sample
        mask = SegmentationMapsOnImage(mask, image.shape)
        image, mask = self.aug(image=image, segmentation_maps=mask)
        return image, mask.get_arr()