import torch
import tqdm
import os
import cv2 as cv
import argparse
import sys

from torchvision.transforms import Compose
from time import time_ns
from pathlib import Path
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus

from utils.transform import Resize, ToTensor
from utils.results import write_csv, scores_per_landmark
from utils.mask import generate_P2ILF_mask, resize_mask
from models.unet3p.model import Unet3Plus
from models.resunetpp.model import ResUnetPlusPlus
from utils.metric import precision, dice_score, francois_distance, cl_dice
from utils.postprocess import process_mask


def generate_predictions(model, device, weights, samples):
    """
    Generates a set of predictions from a model for a chosen sample set, also records prediction and processing time.
    """
    transform = Compose([
        Resize(320, 416),
        ToTensor()
    ])
    
    model.to(device)
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.eval()

    results = []
    times = []
    for sample in tqdm(samples):
        timings = []
        _, image = sample
        x, y = transform(image)
        with torch.no_grad():
            timings.append(time_ns()) # Before prediction
            prediction = model(x.to(device).unsqueeze(0))
            timings.append(time_ns) # After prediction

            mask = torch.argmax(prediction, dim=1).squeeze(0).cpu()
        timings.append(time_ns()) # Before processing
        mask = process_mask(image, mask)
        timings.append(time_ns()) # After processing
        
        results.append(mask)
        times.append(((timings[1] - timings[0]) / 1_000_000, (timings[3] - timings[2]) / 1_000_000))
    time_pred = 0
    time_proc = 0
    for t in times:
        time_pred += t[0]
        time_proc += t[1]
    time_pred /= len(times)
    time_proc /= len(times)
    return results, time_pred, time_proc

def load_P2ILF_test_samples():
    """
    Loads sample from the P2ILF test set into memory.
    """
    samples = []
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    test_path = Path(dir_path / "data" / "P2ILF" / "test")

    for patient in test_path.iterdir():
        for p in Path(patient / "2D-3D_contours").iterdir():
            image = p.parents[1] / "images" / f"{p.name[:-4]}.jpg"
            image = cv.imread(image)
            image = cv.convertScaleAbs(image, alpha=1.15, beta=0)
            mask = generate_P2ILF_mask(p)
            samples.append([p.name[:-4], (image, mask)])
    return samples

def evaluate_results(samples, results):
    scores = {
        "precision": [],
        "dsc": [],
        "francois": [],
        "cld": []
    }
    names = []

    for i, pred in enumerate(results):
        name = samples[i][0]
        names.append(name)
        pred = resize_mask(pred, 1080, 1920)
        truth = samples[i][1][1]
        truth = resize_mask(truth, 1080, 1920)

        scores["precision"].append(scores_per_landmark(pred, truth, precision))
        scores["dsc"].append(scores_per_landmark(pred, truth, dice_score))
        scores["francois"].append(scores_per_landmark(pred, truth, francois_distance))
        scores["cld"].append(scores_per_landmark(pred, truth, cl_dice))
    return names, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", help="model architecture identifier")
    parser.add_argument("--path", help="path to model state dictionary")
    args = parser.parse_args()
    if not args.path or not args.arch:
        print("--path and --arch are required, please read README for instructions.")
        sys.exit(1)

    args.path = Path(args.path)

    print("Loading samples...")
    samples = load_P2ILF_test_samples()
    
    MODELS = {
        "unet": Unet(classes=4),
        "unetpp": UnetPlusPlus(classes=4),
        "unet3p": Unet3Plus(classes=4),
        "resunetpp": ResUnetPlusPlus(classes=4),
        "deeplabv3p": DeepLabV3Plus(classes=4)
    }

    model = MODELS[args.arch]
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Generating predictions...")
    results, _, _ = generate_predictions(model, device, args.path, samples)
    print("Evaluating...")
    names, scores = evaluate_results(samples, results)

    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    Path.mkdir(dir_path / "results", exist_ok=True)
    write_csv(names, scores, Path(dir_path / "results" / f"{args.path.name[:-4]}.csv"))
    print(f"Written results to results/{args.path.name[:-4]}.csv")