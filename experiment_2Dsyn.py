import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import model_rgb2gray, has_parameter
from train_functions import training
from config import load_config, write_config, create_object_from_dict, get_class_by_path

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default=None)
args = parser.parse_args()

config = load_config(args.config)

print("File di configurazione selezionato: " + str(args.config))

if config["TRAINING"]["deterministic"]:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print("Deterministic run")

test_mcc = []
test_acc = []
test_auc = []
test_f1 = []

seeds = config["TRAINING"]["seeds"]

for i in range(0, len(seeds)):

    transform_list = []
    for j in range(len(config["DATA_TRANSFORM"])):
        transform_list.append(create_object_from_dict(config["DATA_TRANSFORM"][j]))
    data_transform = transforms.Compose(transform_list)

    transform_list = []
    for j in range(len(config["TARGET_TRANSFORM"])):
        transform_list.append(create_object_from_dict(config["TARGET_TRANSFORM"][j]))
    target_transform = transforms.Compose(transform_list)

    if config["DATASET"]["class"] != 'customDatasets.BCSTomoSurf':
        if i == 0:
            root_dataset_path = config["DATASET"]["params"]["root"]

        config["DATASET"]["params"]["root"] = root_dataset_path + "seed-" + str(seeds[i]) + "/"
        print(config["DATASET"]["params"]["root"])

    if config["DATASET"]["class"] == 'customDatasets.BCSTomoSurf':
        data_train = None
        data_test = None
        data_test = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"],
                                                                  transform=data_transform,
                                                                  transfrom_labels=target_transform)

        train_dataloader = None
        val_dataloader = None
        test_dataloader = DataLoader(dataset=data_test,
                                     batch_size=config["TRAINING"]["test_batch_size"],
                                     num_workers=config["TRAINING"]["num_workers"],
                                     shuffle=False)

        print("Num test samples: " + str(len(data_test)))

        if config["TRAINING"]["inference"] == False:
            train_dataset, valtest_dataset = data_test.random_split(test_ratio=0.2, seed=seeds[i],
                                                               savepath=config["TRAINING"]["workspace"]
                                                                + "seed-" + str(seeds[i]) + "-" + str(i + 1) + "/")
            print("Split seed: " + str(seeds[i]))
            val_dataset, test_dataset = valtest_dataset.random_split(test_ratio=0.5, seed=seeds[i],
                                                               savepath=config["TRAINING"]["workspace"]
                                                                + "seed-" + str(seeds[i]) + "-" + str(i + 1) + "/")

            print("Num training samples: " + str(len(train_dataset)))
            print("Num validation samples: " + str(len(val_dataset)))
            print("Num test samples: " + str(len(test_dataset)))

            # DATALOADERS
            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=config["TRAINING"]["train_batch_size"],
                                          num_workers=config["TRAINING"]["num_workers"],
                                          shuffle=True)

            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=config["TRAINING"]["test_batch_size"],
                                        num_workers=config["TRAINING"]["num_workers"],
                                        shuffle=False)

            test_dataloader = DataLoader(dataset=test_dataset,
                                         batch_size=config["TRAINING"]["test_batch_size"],
                                         num_workers=config["TRAINING"]["num_workers"],
                                         shuffle=False)
    else:
        data_train = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], type=0, transform=data_transform, transfrom_labels=target_transform)
        print(data_train)
        data_validation = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], type=1, transform=data_transform, transfrom_labels=target_transform)
        data_test = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], type=2, transform=data_transform, transfrom_labels=target_transform)
        # DATALOADERS
        train_dataloader = DataLoader(dataset=data_train,
                                      batch_size=config["TRAINING"]["train_batch_size"],
                                      num_workers=config["TRAINING"]["num_workers"],
                                      shuffle=True)

        val_dataloader = DataLoader(dataset=data_validation,
                                    batch_size=config["TRAINING"]["test_batch_size"],
                                    num_workers=config["TRAINING"]["num_workers"],
                                    shuffle=False)

        test_dataloader = DataLoader(dataset=data_test,
                                     batch_size=config["TRAINING"]["test_batch_size"],
                                     num_workers=config["TRAINING"]["num_workers"],
                                     shuffle=False)

        print("Num training samples: " + str(len(data_train)))
        print("Num validation samples: " + str(len(data_validation)))
        print("Num test samples: " + str(len(data_test)))


    print("Classe del dataset selezionata: " + str(config["DATASET"]["class"]))

    model_class = get_class_by_path(config["MODEL"]["class"])
    model = model_class(**config["MODEL"]["params"])

    print("Modello selezionato: " + str(config["MODEL"]["class"]))
    print("Percorso dataset: " + str(config["DATASET"]["params"]["root"]))

    if config["TRAINING"]["parallel"]:
        model = nn.DataParallel(model, device_ids=config["TRAINING"]["parallel"])
        device = config["TRAINING"]["parallel"][0]
    else:
        device = torch.device(config["TRAINING"]["device"])

    model = model.to(device)

    # create loss, optimizer and scheduler
    loss = create_object_from_dict(config["LOSS"])

    optimizer = get_class_by_path(config["OPTIMIZER"]["class"])(**config["OPTIMIZER"]["params"], params=model.parameters())

    scheduler = get_class_by_path(config["SCHEDULER"]["class"])(**config["SCHEDULER"]["params"], optimizer=optimizer)

    print("Percorso in cui verranno salvati i risultati: " + str(config["TRAINING"]["workspace"]))

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
                   workspace=config["TRAINING"]["workspace"] + "seed-" + str(seeds[i]) + "-" + str(i + 1),
                   inference=config["TRAINING"]["inference"],
                   print_per_epoch=20)

    test_mcc.append(test_mcc_run)
    test_acc.append(test_acc_run)
    test_auc.append(test_auc_run)
    test_f1.append(test_f1_run)

with open(config["TRAINING"]["workspace"] + "performance.txt", 'w') as f:
    for idx, item in enumerate(test_mcc):
        f.write("Test MCC " + str(idx + 1) + ": " + str(item) + '\n')

    for idx, item in enumerate(test_acc):
        f.write("Test Accuracy " + str(idx + 1) + ": " + str(item) + '\n')

    for idx, item in enumerate(test_auc):
        f.write("Test AUC " + str(idx + 1) + ": " + str(item) + '\n')

    for idx, item in enumerate(test_f1):
        f.write("Test F1Score " + str(idx + 1) + ": " + str(item) + '\n')

    f.write("Avg MCC: " + str(sum(test_mcc) / len(test_mcc)))
