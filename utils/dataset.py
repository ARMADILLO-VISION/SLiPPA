import cv2 as cv
import json

from torch.utils.data import Dataset
from pathlib import Path

from utils.mask import generate_L3D_mask, generate_P2ILF_mask

class L3D(Dataset):
    """
    Dataset class for handling the L3D dataset from the D2GPLand paper.
    """
    def __init__(self, subset="train", transform=None):
        self.path = Path(Path(__file__).resolve().parent / ".." / "data" / "L3D" / subset)
        self.transform = transform
        self.data = []

        for p in Path(self.path / "labels").iterdir():
            name = p.stem
            image = Path(self.path / "images" / f"{name}.jpg")
            self.data.append({
                "image": image,
                "labels": p
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv.imread(self.data[index]["image"])

        f = open(self.data[index]["labels"], 'r')
        mask_data = json.load(f)
        f.close()
        mask = generate_L3D_mask()

        sample = (image, mask)
        if self.transform:
            sample = self.transform(sample)
        return sample

class P2ILF(Dataset):
    """
    Dataset class for the P2ILF challenge dataset.
    """
    def __init__(self, subset="train", transform=None):
        self.path = Path(Path(__file__).resolve().parent / ".." / "data" / "P2ILF" / subset)
        self.transform = transform
        self.data = []
    
        for p in self.path.iterdir():
            contour_folder = Path(p / "2D-3D_contours")
            for f in contour_folder.iterdir():
                if f.suffix == ".xml":
                    self.data.append({
                        "patient": p,
                        "labels": f,
                        "image": Path(p / "images" / f"{f.stem}.jpg")
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv.imread(self.data[index]["image"])
        image = cv.convertScaleAbs(image, alpha=1.2, beta=10) # Increase visibility

        mask = generate_P2ILF_mask()

        sample = (image, mask)
        if self.transform:
            sample = self.transform(sample)
        return sample