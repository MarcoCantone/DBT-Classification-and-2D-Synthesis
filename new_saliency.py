import argparse
import sys

import numpy
from scipy.interpolate import griddata
import cv2
from customDatasets import OMITomoSlices
from customModels import *

from torchvision import transforms
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
from scipy.interpolate import Rbf

from config import *

def save_surfaces(save_path, dataset, config):
    for classe, tipo in product(["normal", "malignant"],
                                ["soft_attention", "attention_map", "attention_map_int", "surfaces"]):
        if not os.path.exists(save_path + classe + '/' + tipo):
            os.makedirs(save_path + classe + '/' + tipo)

    patch_size = config["MODEL"]["params"]["patch_size"]
    h = config["MODEL"]["params"]["img_size"][0]
    w = config["MODEL"]["params"]["img_size"][1]

    for k, (img, label, img_path) in enumerate(dataset):

        print(img_path)

        if label.tolist().index(1) == 0:
            save_path_images = save_path + "normal/"
        else:
            save_path_images = save_path + "malignant/"

        file_name = img_path[::-1].split('/')[0][::-1]
        relative_path_list = img_path[::-1].split('images'[::-1])[0][::-1].split('/')[1:3]
        relative_path = ''
        for string in relative_path_list:
            relative_path += '/' + string

        for tipo in ["soft_attention", "attention_map", "attention_map_int", "surfaces"]:
            if not os.path.exists(save_path_images + tipo + relative_path):
                os.makedirs(save_path_images + tipo + relative_path)

        if os.path.exists(save_path_images + "surfaces" + relative_path + '/' + file_name + ".png"):
            continue

        prediction, attn_weights = model(img[None, :, :, :], True)

        # plt.plot(x, attn_weights[1][0, :, 0].detach().numpy())
        # plt.savefig(save_path_images + "soft_attention/" + relative_path + '/' + file_name + ".png")

        num_slices = attn_weights[0].shape[0]

        saliency = attn_weights[1].permute(1, 0, 2) * attn_weights[0][:, 0, 1:].reshape(num_slices, int(h / patch_size),
                                                                                        int(w / patch_size))

        num_points = config["MODEL"]["points"]
        # Flatten del tensore e ottenimento delle posizioni ordinate
        flat_indices = np.argsort(saliency.detach().numpy(), axis=None)[::-1]  # Ordina gli indici in ordine decrescente
        # Creiamo una lista per le posizioni complete
        positions_list = [np.unravel_index(idx, saliency.shape) for idx in flat_indices]
        # Visualizziamo i risultati
        # for item in positions_list:
        #     print(item)

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

        # Interpolazione con Thin Plate Spline usando Rbf
        rbf_spline = Rbf(matrix[:, 2], matrix[:, 1], matrix[:, 0], function='thin_plate')
        x_range = np.arange(w)
        y_range = np.arange(h)
        X, Y = np.meshgrid(x_range, y_range)

        Z_spline = rbf_spline(X, Y)
        Z_spline = np.clip(Z_spline, 0, img.shape[0] - 1)
        Z_spline = np.round(Z_spline).astype(int)

        # plt.imshow(img[img.shape[0]//2, :, :], cmap='gray')
        # plt.title("Center Image")
        # plt.show()
        #
        # plt.imshow(Z_spline, cmap='jet')
        # plt.title("Saliency Map")
        # plt.show()

        # # 5. Genera un grafico 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # Aggiungi i punti di controllo originali nel grafico
        # ax.scatter(matrix[:,2], matrix[:,1], matrix[:,0], color='blue', label='Punti di Controllo')
        #
        # # Plot della superficie 3D
        # ax.plot_surface(X, Y, Z_spline, color='green', alpha=0.7)
        #
        # # Etichette e titolo
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_zlim(0, img.shape[0] - 1)  # Lim
        # ax.set_title('Interpolazione Thin Plate Spline')
        # plt.show()

        output_image = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                output_image[i, j] = (img[Z_spline[i, j], i, j])

        Image.fromarray(output_image * ((2 ** 8) - 1)).convert("RGB").save(
            save_path_images + "surfaces" + relative_path + '/' + file_name + ".png")

        # plt.imshow(output_image, cmap='gray')
        # plt.title(img_path.split('/')[6])
        # plt.show()

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default=None)
args = parser.parse_args()

config = load_config(args.config)

print("File di configurazione selezionato: " + str(args.config))

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
data = create_object_from_dict(config["DATASET"])

print("Classe del dataset selezionata: " + str(config["DATASET"]["class"]))

train_dataset, valtest_dataset = data.random_split(test_ratio=0.2, seed=config["TRAINING"]["split_seed"], savepath=None)
print(config["TRAINING"]["split_seed"])
val_dataset, test_dataset = valtest_dataset.random_split(test_ratio=0.5, seed=config["TRAINING"]["split_seed"], savepath=None)

backbone = create_object_from_dict(config["BACKBONE"]["model"])
if config["BACKBONE"]["gray"]:
    model_rgb2gray(backbone)
if config["BACKBONE"]["layerToDiscard"] > 0:
    backbone = nn.Sequential(*list(backbone.children())[:-config["BACKBONE"]["layerToDiscard"]])
model_class = get_class_by_path(config["MODEL"]["class"])
if has_parameter(model_class, "backbone"):
    config["MODEL"]["params"].update({"backbone": backbone})
model = create_object_from_dict(config["MODEL"])

print("Modello selezionato: " + str(config["MODEL"]["class"]))

model.load_state_dict(torch.load(os.path.join(config["TRAINING"]["workspace"], "best_model.pth"), map_location="cpu"))

print("Percorso del modello selezionato: " + str(config["TRAINING"]["workspace"]))

Omi_classes = {0: "Normal", 1: "Malignant"}

save_path_images = "/data/dellascenza/newtask/Tomo2Dsynt_softself/patch-256_numpoints-3/seed-43/"

print("Percorso in cui verranno salavate le superfici: " + save_path_images)

print("Numero di punti usato:" + str(config["MODEL"]["points"]))

save_surfaces(save_path_images + "training/", train_dataset, config)
save_surfaces(save_path_images + "validation/", val_dataset, config)
save_surfaces(save_path_images + "test/", test_dataset, config)