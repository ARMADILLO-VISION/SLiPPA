import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

def generate_L3D_mask(data):
    """
    Generates mask from L3D dataset JSON file.
    """
    height, width = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((height, width), np.uint8)
    
    shapes = {}
    shapes["ridge"] = []
    shapes["silhouette"] = []
    shapes["ligament"] = []

    for shape in data["shapes"]:
        label = shape["label"] if shape["label"] != "rigde" else "ridge" # Handles a typo in dataset
        shapes[label].append(np.array(shape["points"], np.int32).reshape(-1, 1, 2))
    
    for key, landmark in shapes.items():
        if key == "silhouette":
            colour = 1
        elif key == "ridge":
            colour = 2
        else:
            colour = 3
        cv.polylines(mask, landmark, False, colour, 12)
    return mask

def generate_P2ILF_mask(path):
    """
    Generates mask from P2ILF dataset XML file.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    mask = np.zeros((1080, 1920), np.uint8)
    
    for contour in root.findall("contour"):
        label = contour.find("contourType").text.strip()
        x_str = contour.find("imagePoints").find('x').text
        y_str = contour.find("imagePoints").find('y').text
        x_pts = [ int(float(x.strip())) for x in x_str.split(',') if x.strip() ]
        y_pts = [ int(float(y.strip())) for y in y_str.split(',') if y.strip() ]
        if len(x_pts) != len(y_pts) or len(x_pts) == 0:
            continue
        pts = np.array([[ (x, y) for x, y in zip(x_pts, y_pts) ]], np.int32)
        if label == "Silhouette":
            colour = 1
        elif label == "Ridge":
            colour = 2
        else:
            colour = 3
        cv.polylines(mask, pts, False, colour, 12)
    return mask

def resize_mask(mask, height, width):
    """
    Resizes mask using OpenCV.
    """
    return cv.resize(mask, (width, height), interpolation=cv.INTER_NEAREST)
