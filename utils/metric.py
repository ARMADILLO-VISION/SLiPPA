import numpy as np
import torch

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def true_positive(pred, truth, value):
    """
    NumPy binary mask of where pred is the same as truth, focusing on value.
    """
    result = np.zeros(pred.shape, np.uint8)
    result[np.logical_and(pred == truth, pred == value)] = 1
    return result

def false_positive(pred, truth, value):
    """
    NumPy binary mask of where pred is different to truth, focusing on value.
    """
    result = np.zeros(pred.shape, np.uint8)
    result[np.logical_and(pred != truth, pred == value)] = 1
    return result

def true_negative(pred, truth, value):
    """
    NumPy binary mask of where pred is the same as truth, focusing on everything but value.
    """
    result = np.zeros(pred.shape, np.uint8)
    result[np.logical_and(pred == truth, pred != value)] = 1
    return result

def false_negative(pred, truth, value):
    """
    NumPy binary mask of where pred is different to truth, focusing on everything but value.
    """
    result = np.zeros(pred.shape, np.uint8)
    result[np.logical_and(pred == truth, pred == value)] = 1
    return result

def precision(pred, truth, value):
    """
    Computes precision of a prediction [TP / (TP + FP)], focusing on value.
    """
    tp = np.count_nonzero(true_positive(pred, truth, value))
    fp = np.count_nonzero(false_positive(pred, truth, value))

    if np.count_nonzero(truth[truth == value]) == 0:
        # Invalid due to no positives in ground truth
        return -1
    elif tp + fp == 0:
        # No positives in prediction
        return 0
    return tp / (tp + fp)

def recall(pred, truth, value):
    """
    Computes recall of a prediction [TP / (TP + FN)], focusing on value.
    """
    tp = np.count_nonzero(true_positive(pred, truth, value))
    fn = np.count_nonzero(false_negative(pred, truth, value))

    if np.count_nonzero(truth[truth == value]) == 0:
        # Invalid due to no positives in ground truth
        return -1
    elif tp + fn == 0:
        return 0
    return tp / (tp + fn)

def dice_score(pred, truth, value):
    """
    Computes Dice similarity coefficient (DSC), focusing on value.
    """
    truth_class = (truth == value).astype(np.uint8) # GT binary mask
    if np.count_nonzero(truth_class) == 0:
        # No instance of value in ground truth
        return -1
    pred_class = (pred == value).astype(np.uint8)
    if np.count_nonzero(pred_class) == 0:
        return 0
    
    intersection = np.sum(truth_class * pred_class)
    union = np.sum(truth_class) + np.sum(pred_class)
    return (2 * intersection) / union

def cl_dice(pred, truth, value):
    """
    Computes centre-line Dice score (clDice), focusing on value.
    """
    def cl_score(a, b):
        return np.sum(a * b) / np.sum(a)
    
    truth_class = (truth == value).astype(np.uint8)
    if np.count_nonzero(truth_class) == 0:
        # No instance of value in ground truth
        return -1
    pred_class = (pred == value).astype(np.uint8)
    if np.count_nonzero(pred_class) == 0:
        return 0
    
    t_prec = cl_score(pred_class, skeletonize(truth_class))
    t_sens = cl_score(truth_class, skeletonize(pred_class))
    if t_prec + t_sens == 0:
        return 0
    return (2 * t_prec * t_sens) / (t_prec + t_sens)

# NOTE: In the paper, the François metric from the following repository was used:
# https://github.com/sharib-vision/P2ILF
def francois_distance(pred, truth, value):
    """
    Computes François symmetric distance, focusing on value.
    """
    def one_hot_encode(mask, value):
        binary = np.zeros_like(mask, np.uint8)
        binary[mask == value] = 1
        tensor = torch.from_numpy(binary).long()
        encoded = torch.nn.functional.one_hot(tensor, 2)
        if tensor.dim() == 3:
            tensor = encoded.permute(0, 3, 1, 2)
        elif tensor.dim() == 2:
            tensor = encoded.permute(2, 0, 1)
        return tensor
    
    def distance_transform(contour, threshold=255, normalise=True):
        contour = contour.numpy()
        if np.sum(contour) == 0:
            return threshold * np.ones_like(contour, np.float32)
        inversed = 1 - contour
        dt = np.float32(distance_transform_edt(inversed))
        dt[dt > threshold] = threshold
        if normalise:
            dt = dt / threshold
        return dt
    
    def distance_matching(a, b, d_max):
        total = a.sum().item()
        if total == 0:
            return 0.0, 0.0, 0.0, False
        
        b_dt = torch.from_numpy(distance_transform(b, d_max, False))
        dist_ab = a.to(torch.float64) * b_dt.to(torch.float64)
        mask_dmax = dist_ab >= d_max
        mask_match_and_zero = dist_ab < d_max

        num_non_match = float(mask_dmax.sum().item())
        num_match = total - num_non_match
        dist_total = dist_ab[mask_match_and_zero].sum().item()
        return dist_total, num_non_match, num_match, True
    
    d_max = 20

    a, b = one_hot_encode(pred, value), one_hot_encode(truth, value)
    dist_ab, miss_a, match_a, _ = distance_matching(a, b, d_max)
    dist_ba, miss_b, match_b, _ = distance_matching(b, a, d_max)

    num_pixels = a.shape[0] * a.shape[1]
    sum_b = miss_b + match_b
    
    dist_match = (dist_ab + dist_ba) / (2.0 * sum_b + 1e-6)
    dist_miss = d_max * miss_b / (sum_b + 1e-6)
    dist_outliers = d_max * miss_a / (num_pixels - (2.0 * d_max * sum_b))
    return dist_match + dist_miss + dist_outliers