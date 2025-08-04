import os

import customModels
from customDatasets import BCS_DBT
from collections import Counter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from collections import Counter
from utils import model_rgb2gray, has_parameter
from train_functions import training
from config import load_config, write_config, create_object_from_dict, get_class_by_path
import loss
def count_parameters(model):
    frozen_params = 0
    trainable_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            layer_info.append((name, "trainable"))
        else:
            frozen_params += param.numel()
            layer_info.append((name, "frozen"))

    return trainable_params, frozen_params, layer_info

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default=None)
args = parser.parse_args()

config = load_config(args.config)

if config["TRAINING"]["deterministic"]:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print("Deterministic run")

seeds_split = config["TRAINING"]["seeds"]
test_mcc = []
test_acc = []
test_auc = []
test_f1 = []

for i in range(0, len(seeds_split)):

    transform_list = []
    for j in range(len(config["DATA_TRANSFORM"])):
        transform_list.append(create_object_from_dict(config["DATA_TRANSFORM"][j]))
    data_transform = transforms.Compose(transform_list)

    transform_list = []
    for j in range(len(config["TARGET_TRANSFORM"])):
        transform_list.append(create_object_from_dict(config["TARGET_TRANSFORM"][j]))
    target_transform = transforms.Compose(transform_list)

    print("Dataset path: " + config["DATASET"]["params"]["root"])
    print("Dataset class: " + config["DATASET"]["class"])

    print(data_transform)

    # FIXME: lambda function in yaml
    if i == 0:
        config["DATASET"]["params"].update({"slices": eval(config["DATASET"]["params"]["slices"])})

    if config["DATASET"]["class"] == 'customDatasets.BCS_DBT': # Inference mode
        test_dataset = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], data_transform=data_transform, target_transform=target_transform)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=config["TRAINING"]["batch_size_val"],
                                     num_workers=config["TRAINING"]["num_workers"],
                                     shuffle=False)
        train_dataset = None
        val_dataset = None

    elif config["DATASET"]["class"] == 'customDatasets.MamaMiaDataset':
        train_dataset = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], split="train_split",
                                                                     data_transform=data_transform,
                                                                     target_transform=target_transform)
        val_dataset = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], split="test_split",
                                                                     data_transform=data_transform,
                                                                     target_transform=target_transform)
        test_dataset = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], split="test_split",
                                                                     data_transform=data_transform,
                                                                     target_transform=target_transform)
    else:
        data = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"],
                                                             data_transform=data_transform,
                                                             target_transform=target_transform)

        print(len(data))
        train_dataset, valtest_dataset = data.random_split(test_ratio=0.2, seed=seeds_split[i], savepath=config["TRAINING"]["workspace"] + "run-" + str(i + 1) + "_seed-" + str(seeds_split[i]) + '/')
        print("Split seed: " + str(seeds_split[i]))
        val_dataset, test_dataset = valtest_dataset.random_split(test_ratio=0.5, seed=seeds_split[i], savepath=config["TRAINING"]["workspace"] + "run-" + str(i + 1) + "_seed-" + str(seeds_split[i]) + '/')

    train_dataloader = None
    if train_dataset is not None:
        print("Num training samples: " + str(len(train_dataset)))
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config["TRAINING"]["batch_size"],
                                      num_workers=config["TRAINING"]["num_workers"],
                                      shuffle=True)

    val_dataloader = None
    if val_dataset is not None:
        print("Num validation samples: " + str(len(val_dataset)))
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=config["TRAINING"]["batch_size_val"],
                                    num_workers=config["TRAINING"]["num_workers"],
                                    shuffle=False)

    test_dataloader = None
    if test_dataset is not None:
        print("Num test samples: " + str(len(test_dataset)))
        test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=config["TRAINING"]["batch_size_val"],
                                    num_workers=config["TRAINING"]["num_workers"],
                                    shuffle=False)

    print("Backbone used: " + config["BACKBONE"]["model"]["class"])
    print("Batch size training: " + str(config["TRAINING"]["batch_size"]))
    print("Batch size validation and test: " + str(config["TRAINING"]["batch_size_val"]))

    backbone = create_object_from_dict(config["BACKBONE"]["model"])
    if config["BACKBONE"]["gray"]:
        model_rgb2gray(backbone)
    if config["BACKBONE"]["layerToDiscard"] > 0:
        backbone = nn.Sequential(*list(backbone.children())[:-config["BACKBONE"]["layerToDiscard"]])

    print("Model used: " + config["MODEL"]["class"])
    model_class = get_class_by_path(config["MODEL"]["class"])

    if has_parameter(model_class, "backbone"):
        model = model_class(**config["MODEL"]["params"], backbone=backbone)
    else:
        model = model_class(**config["MODEL"]["params"])


    if has_parameter(model_class, "patch_size"):
        print("Patch size: " + str(config["MODEL"]["params"]["patch_size"]))

    frozen_perc = config["TRAINING"]["frozen_perc"]

    # Calculate the middle index to freeze the first half of the layers
    num_layers = len(list(model.children()))
    layer_idx = int(num_layers * frozen_perc)

    # Freeze the first half of the layers
    for idx, child in enumerate(model.children()):
        if idx < layer_idx:
            for param in child.parameters():
                param.requires_grad = False

    trainable, frozen, _ = count_parameters(model)

    print("Number of trainable parameters: " + str(trainable))
    print("Number of frozen parameters: " + str(frozen))

    device = torch.device(config["TRAINING"]["device"])

    model = model.to(device)

    #model = torch.nn.DataParallel(model, device_ids=[0, 1])

    #summary(model, input_size=(1, 1024, 512))  # Specify input size

    # create loss, optimizer and scheduler
    loss_class = get_class_by_path(config["LOSS"]["class"])
    loss = loss_class(**config["LOSS"]["params"])

    print("Loss used: " + config["LOSS"]["class"])

    optimizer = get_class_by_path(config["OPTIMIZER"]["class"])(**config["OPTIMIZER"]["params"], params=model.parameters())

    scheduler = get_class_by_path(config["SCHEDULER"]["class"])(**config["SCHEDULER"]["params"], optimizer=optimizer)

    print("Scheduler: " + config["SCHEDULER"]["class"])

    if config["TRAINING"]["grad_acc"]:
        print("Gradient Accumulation enabled")
    else:
        print("Gradient Accumulation disabled")

    if config["TRAINING"]["grad_scaler"]:
        print("AMP enabled")
    else:
        print("AMP disabled")

    # print("Number of backbone's parameters: " + str(sum(p.numel() for p in backbone.parameters())))
    # print("Number of model's parameters: " + str(sum(p.numel() for p in model.parameters())))

    res, test_mcc_run, test_acc_run, test_auc_run, test_f1_run = training(dataloader_train=train_dataloader,
                   dataloader_valid=val_dataloader,
                   dataloader_test=test_dataloader,
                   model=model,
                   criterion=loss,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   epochs=config["TRAINING"]["epochs"],
                   device=device,
                   resume=config["TRAINING"]["resume"],
                   workspace=config["TRAINING"]["workspace"] + "run-" + str(i + 1) + "_seed-" + str(seeds_split[i]),
                   print_per_epoch=20,
                   grad_acc=config["TRAINING"]["grad_acc"],
                   grad_scaler=config["TRAINING"]["grad_scaler"],
                   inference=config["TRAINING"]["inference"])

    test_mcc.append(test_mcc_run)
    test_acc.append(test_acc_run)
    test_auc.append(test_auc_run)
    test_f1.append(test_f1_run)

with open(config["TRAINING"]["workspace"] + "performance.txt",'w') as f:
    for idx, item in enumerate(test_mcc):
        f.write("Test MCC " + str(idx + 1) + ": " + str(item) + '\n')

    for idx, item in enumerate(test_acc):
        f.write("Test Accuracy " + str(idx + 1) + ": " + str(item) + '\n')

    for idx, item in enumerate(test_auc):
        f.write("Test AUC " + str(idx + 1) + ": " + str(item) + '\n')

    for idx, item in enumerate(test_f1):
        f.write("Test F1Score " + str(idx + 1) + ": " + str(item) + '\n')

    f.write("Avg MCC: " + str(sum(test_mcc)/len(test_mcc)))