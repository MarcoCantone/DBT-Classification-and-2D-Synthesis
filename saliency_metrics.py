import argparse
import sys
import numpy
import pandas as pd
import torchvision.transforms
from scipy.interpolate import griddata
import cv2 as cv
from customDatasets import OMITomoSlices
from customModels import *
import statistics
import math
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from itertools import product
from PIL import Image
from customTransform import ToTensorFloat32, MinMax, OneHot
from utils import model_rgb2gray
from torch import nn
import importlib
from utils import has_parameter
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import Rbf
import lpips
import cv2

from config import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def save_surfaces(dataset, config):

    patch_size = config["MODEL"]["params"]["patch_size"]
    ssim_list_score = []
    ssim_list_diff = []
    distances = []
    lpips_list_score = []
    device = config["TRAINING"]["device"]

    # 1. Initialize the LPIPS model (choose your backbone: 'alex', 'vgg', 'squeeze')
    #    Move the model to the same device as your image tensors.
    lpips_model = lpips.LPIPS(net='alex', version='0.1').to(device)
    lpips_model.eval()  # Set to evaluation mode for consistent results

    for k, (img, label, img_path, bbox, _) in enumerate(dataset):

        h = img.shape[1]
        w = img.shape[2]

        print(img_path)

        prediction, attn_weights = model(img[None, :, :, :], True)

        num_slices = attn_weights[0].shape[0]

        saliency = attn_weights[1].permute(1, 0, 2) * attn_weights[0][:, 0, 1:].reshape(num_slices, int(h / patch_size),
                                                                                        int(w / patch_size))

        num_points = config["MODEL"]["points"]
        flat_indices = np.argsort(saliency.detach().numpy(), axis=None)[::-1]
        positions_list = [np.unravel_index(idx, saliency.shape) for idx in flat_indices]

        dictionary = {i: positions_list[i] for i in range(len(positions_list))}

        unique_values = set()
        new_dict = {}
        for key, value in dictionary.items():
            if value[1:3] not in unique_values:
                new_dict[key] = value
                unique_values.add(value[1:3])

            if len(new_dict) == num_points:
                break

        data_points = list(new_dict.values())
        matrix = np.array(data_points)
        matrix[:, 1] = (matrix[:, 1] * patch_size) + patch_size / 2
        matrix[:, 2] = (matrix[:, 2] * patch_size) + patch_size / 2

        rbf_spline = Rbf(matrix[:, 2], matrix[:, 1], matrix[:, 0], function='thin_plate')
        x_range = np.arange(w)
        y_range = np.arange(h)
        X, Y = np.meshgrid(x_range, y_range)

        Z_spline = rbf_spline(X, Y)
        Z_spline = np.clip(Z_spline, 0, img.shape[0] - 1)
        Z_spline = np.round(Z_spline).astype(int)

        output_image = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                output_image[i, j] = (img[Z_spline[i, j], i, j])

        truth_point = [bbox[0], int(bbox[2] + bbox[4] / 2), int(bbox[1] + bbox[3] / 2)]
        dist_from_Z = abs(truth_point[0] - img.shape[0]//2)

        patch_origin = img[bbox[0], bbox[2]:bbox[2] + bbox[4], bbox[1]:bbox[1] + bbox[3]] * 255
        patch_synth = output_image[bbox[2]:bbox[2] + bbox[4], bbox[1]:bbox[1] + bbox[3]] * 255

        patch_origin = torch.Tensor(patch_origin).to(torch.uint8).cpu().numpy()
        patch_synth = torch.Tensor(patch_synth).to(torch.uint8).cpu().numpy()

        # Compute SSIM
        score = ssim(patch_synth, patch_origin)

        ssim_list_score.append(score)

        distances.append(dist_from_Z)

        resize_patch = torchvision.transforms.Resize((224,224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)

        patch_origin_LPIPS = (resize_patch((torch.Tensor(patch_origin).to(torch.uint8).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)))/255.0) * 2 - 1
        patch_synth_LPIPS = (resize_patch((torch.Tensor(patch_synth).to(torch.uint8).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)))/255.0) * 2 - 1

        patch_synth_LPIPS = patch_synth_LPIPS.to(device)
        patch_origin_LPIPS = patch_origin_LPIPS.to(device)

        # 2. Compute the LPIPS distance
        #    Use torch.no_grad() to save memory and speed up computation during inference.
        with torch.no_grad():
            lpips_score_tensor = lpips_model(patch_origin_LPIPS, patch_synth_LPIPS)

        # 3. Get the scalar value from the tensor and print
        lpips_score = lpips_score_tensor.item()

        print("SCORE LPIPS:" + str(lpips_score))

        lpips_list_score.append(lpips_score)

    # Calculate Average (Mean)
    average_stats = statistics.mean(ssim_list_score)
    print(f"Average SSIM: {average_stats}")

    average_stats = statistics.mean(distances)
    print(f"Average Distances: {average_stats}")

    average_stats = statistics.mean(lpips_list_score)
    print(f"Average LPIPS: {average_stats}")

    std_dev_stats = statistics.stdev(ssim_list_score)
    print(f"Standard Deviation SSIM: {std_dev_stats}")

    std_dev_stats = statistics.stdev(distances)
    print(f"Standard Deviation Distances: {std_dev_stats}")

    std_dev_stats = statistics.stdev(lpips_list_score)
    print(f"Standard Deviation LPIPS: {std_dev_stats}")

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default=None)
args = parser.parse_args()

config = load_config(args.config)

print("Config selected: " + str(args.config))

transform_list = []
for i in range(len(config["DATA_TRANSFORM"])):
    transform_list.append(create_object_from_dict(config["DATA_TRANSFORM"][i]))
data_transform = transforms.Compose(transform_list)

transform_list = []
for i in range(len(config["TARGET_TRANSFORM"])):
    transform_list.append(create_object_from_dict(config["TARGET_TRANSFORM"][i]))
target_transform = transforms.Compose(transform_list)

config["DATASET"]["params"].update({"data_transform": data_transform, "target_transform": target_transform})
config["DATASET"]["params"].update({"slices": None})
inference_dataset = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"],
                                                                  labels_file="/ssd2/dellascenza/datasets/BCS-DBT_cropped_boxes/info_bboxes.csv",
                                                                  boxes=True)

print("Dataset class selected: " + str(config["DATASET"]["class"]))

backbone = create_object_from_dict(config["BACKBONE"]["model"])
if config["BACKBONE"]["gray"]:
    model_rgb2gray(backbone)
if config["BACKBONE"]["layerToDiscard"] > 0:
    backbone = nn.Sequential(*list(backbone.children())[:-config["BACKBONE"]["layerToDiscard"]])
model_class = get_class_by_path(config["MODEL"]["class"])
if has_parameter(model_class, "backbone"):
    config["MODEL"]["params"].update({"backbone": backbone})
model = create_object_from_dict(config["MODEL"])

print("Model selected: " + str(config["MODEL"]["class"]))

model.load_state_dict(torch.load(os.path.join(config["TRAINING"]["workspace"], "best_model.pth"), map_location="cpu"))

print("Workspace path: " + str(config["TRAINING"]["workspace"]))
print("Number of points used:" + str(config["MODEL"]["points"]))

save_surfaces(inference_dataset, config)