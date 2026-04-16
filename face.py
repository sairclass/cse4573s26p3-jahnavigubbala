'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    img_cpu = img.detach().cpu()

    if img_cpu.ndim == 3 and img_cpu.shape[0] == 3:
        img_cpu = img_cpu.permute(1, 2, 0)

    if img_cpu.dtype != torch.uint8:
        if img_cpu.max() <= 1.0:
            img_cpu = (img_cpu * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            img_cpu = img_cpu.clamp(0, 255).to(torch.uint8)

    img_np = img_cpu.contiguous().numpy()

    boxes = face_recognition.face_locations(
        img_np,
        number_of_times_to_upsample=2,
        model="hog"
    )

    H, W = img_cpu.shape[0], img_cpu.shape[1]

    for (top, right, bottom, left) in boxes:
        x = float(max(0, left))
        y = float(max(0, top))
        w = float(max(0, min(W, right) - max(0, left)))
        h = float(max(0, min(H, bottom) - max(0, top)))
        detection_results.append([x, y, w, h])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    img_names: List[str] = []
    encodings: List[torch.Tensor] = []

    for img_name in sorted(imgs.keys()):
        img = imgs[img_name].detach().cpu()

        if img.ndim == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)

        if img.dtype != torch.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                img = img.clamp(0, 255).to(torch.uint8)

        img_np = img.contiguous().numpy()
        face_encs = face_recognition.face_encodings(img_np)

        if len(face_encs) == 0:
            continue

        enc = torch.tensor(face_encs[0], dtype=torch.float32)
        enc = enc / (torch.norm(enc) + 1e-8)

        img_names.append(img_name)
        encodings.append(enc)

    if len(encodings) == 0:
        return cluster_results

    clusters = []
    for i in range(len(encodings)):
        clusters.append({
            "names": [img_names[i]],
            "features": [encodings[i]]
        })

    while len(clusters) > K:
        best_i = -1
        best_j = -1
        best_dist = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                total_dist = 0.0
                count = 0

                for f1 in clusters[i]["features"]:
                    for f2 in clusters[j]["features"]:
                        dist = torch.sum((f1 - f2) ** 2).item()
                        total_dist += dist
                        count += 1

                avg_dist = total_dist / count

                if best_dist is None or avg_dist < best_dist:
                    best_dist = avg_dist
                    best_i = i
                    best_j = j

        clusters[best_i]["names"].extend(clusters[best_j]["names"])
        clusters[best_i]["features"].extend(clusters[best_j]["features"])
        del clusters[best_j]

    for i in range(len(clusters)):
        clusters[i]["names"].sort()
        cluster_results[i] = clusters[i]["names"]
        
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
