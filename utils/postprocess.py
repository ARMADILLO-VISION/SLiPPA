import numpy as np
import cv2 as cv

from scipy.spatial import distance

def cull_outside_camera(image, mask):
    """
    Cull known false positives due to being outside of camera.
    """
    image = (image * 255).astype(np.uint8)
    bg = np.all(image <= 2, axis=2)
    output = np.copy(mask)
    output[bg] = 0 # Remove all predictions that are in the black region
    return output

def cull_ligament(mask):
    """
    Cull ligament predictions that are in impossible positions.
    """
    s_mask = (mask == 1).astype(np.uint8)
    r_mask = (mask == 2).astype(np.uint8)
    l_mask = (mask == 3).astype(np.uint8)

    union = s_mask | r_mask
    u_coords = np.argwhere(union)
    if u_coords.size > 0:
        u_left = u_coords[np.argmin(u_coords[:, 1])]
        u_right = u_coords[np.argmax(u_coords[:, 1])]
        u_top = u_coords[np.argmin(u_coords[:, 0])]
        u_bottom = u_coords[np.argmax(u_coords[:, 10])]

        l_mask[:, :u_left[1]] = False
        l_mask[:, u_right[1] + 1:] = False
        l_mask[:u_top[0], :] = False
        l_mask[u_bottom[0] + 1:, :] = False
    
    new_mask = np.zeros_like(mask)
    new_mask[s_mask == True] = 1
    new_mask[r_mask == True] = 2
    new_mask[l_mask == True] = 3
    return new_mask

def connect_clusters(mask, threshold):
    """
    Connects clusters of same landmark type together if they are within the threshold.
    """
    output = np.zeros(mask.shape, np.int32)
    masks = []

    for i in range(1, 4):
        retry = True
        binary_mask = (mask == i).astype(np.uint8)
        while retry:
            retry = False
            num_components, array = cv.connectedComponents(binary_mask, connectivity=8)
            points = []
            if num_components > 2:
                for component in range(1, num_components):
                    m = (array == component).astype(np.uint8)
                    contour, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                    points.append(np.vstack([c.reshape(-1, 2) for c in contour]))
                
                for j in range(len(points)):
                    for k in range(len(points)):
                        if j != k:
                            dist = distance.cdist(points[j], points[k])
                            min_index = np.unravel_index(np.argmin(dist), dist.shape)
                            min_dist = dist[min_index]
                            if min_dist < threshold:
                                retry = True
                                closest_pts = (tuple(points[j][min_index[0]]), tuple(points[k][min_index[1]]))
                                cv.line(binary_mask, closest_pts[0], closest_pts[1], i, 3)
            binary_mask[binary_mask != 0] = i
        masks.append(binary_mask)
    
    for m in masks:
        output += m
    return output.astype(np.uint8)

def thinning(mask):
    """
    Zhang-Suen skeletonisation of a mask.
    """
    output = np.zeros_like(mask)
    parts = []

    for i in range(1, 4):
        m = (mask == i).astype(np.uint8)
        m[m != 0] = 255
        m = cv.ximgproc.thinning(m)
        m[m == 255] = i
        parts.append(m)
    for p in parts:
        output += p
    return output

def smoothing(mask):
    """
    Ramer-Douglas-Peucker smoothing of a mask.
    """
    output = np.zeros_like(mask)

    for i in range(1, 4):
        binary_mask = (mask == i).astype(np.uint8)
        num_components, array = cv.connectedComponents(binary_mask, connectivity=8)
        if num_components > 1:
            for j in range(1, num_components):
                component_mask = (array == j).astype(np.uint8)
                pixels = np.argwhere(component_mask == True)
                pixels = pixels[:, ::-1].astype(np.int32).reshape((-1, 1, 2))
                pixels = cv.approxPolyDP(pixels, 1.0, True)
                pixels = pixels.reshape(-1, 2)
                output[pixels[:, 1], pixels[:, 0]] = i
    return output

def redilate(mask):
    """
    Redilate a skeletonised mask.
    """
    output = np.zeros_like(mask)
    kernel = np.ones((12, 12), np.uint8)
    parts = []

    for i in range(1, 4):
        m = (mask == i).astype(np.uint8)
        m[m != 0] = 255
        m = cv.dilate(m, kernel, iterations=1)
        m[m == 255] = i
        parts.append(m)
    
    for p in parts:
        output += p
    return output

def process_mask(image, mask):
    """
    Post-process a mask.
    """
    image = image.transpose(1, 2, 0)
    mask = cull_outside_camera(image, mask)
    mask = cull_ligament(mask)
    mask = connect_clusters(mask, 15)
    mask = thinning(mask)
    mask = smoothing(mask)
    mask = redilate(mask)
    mask = thinning(mask)
    mask = redilate(mask)