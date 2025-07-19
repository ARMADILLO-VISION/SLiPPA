import torch
import numpy as np

def inverse_square_class_weights(dataset, num_classes, device):
    """
    Inverse square class weights function. Takes non-transformed dataset for time efficiency.
    """
    histogram = np.zeros(num_classes, np.float32)
    for _, mask in dataset:
        mask = mask.flatten()
        histogram += np.bincount(mask, minlength=num_classes)
    weights = 1 / (histogram**2 + 1e-6)
    weights = weights / (np.sum(weights) * num_classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def logarithmic_class_weights(dataset, num_classes, device):
    """
    Logarithmic class weights function. Takes non-transformed dataset for time efficiency.
    """
    histogram = np.zeros(num_classes, np.float32)
    for _, mask in dataset:
        mask = mask.flatten()
        histogram += np.bincount(mask, minlength=num_classes)
    weights = np.sum(histogram) / histogram
    weights = np.log(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def slippa_class_weights(dataset, num_classes, device):
    """
    SLiPPA bespoke logarithmic class weights function. Takes non-transformed dataset for time efficiency.
    """
    histogram = np.zeros(num_classes, np.float32)
    for _, mask in dataset:
        mask = mask.flatten()
        histogram += np.bincount(mask, minlength=num_classes)
    weights = np.sum(histogram) / histogram
    weights = np.log(weights)
    for i, w in enumerate(weights):
        weights[i] = max(w, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)