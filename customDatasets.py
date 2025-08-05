import os
import numpy as np
import pydicom
import pandas as pd
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import random
import itertools
import customTransform
import re
from collections import Counter

class BCS_DBT(Dataset):

    def __init__(self, root, labels_file="labels_inference.txt",
                 data_transform=None,
                 target_transform=None,
                 slices=None,
                 boxes=False
                 ):
        super(BCS_DBT, self).__init__()

        self.root = root
        self.boxes = boxes

        self.subpath = ""
        data = pd.read_csv(os.path.join(root, labels_file))

        data.loc[data['Class'] == "normal", 'Class'] = '0'
        data.loc[data['Class'] == "benign", 'Class'] = '0'
        data = data[data['Class'] != "actionable"]
        # data.loc[data['Class'] == " Actionable", 'Class'] = '1'
        data.loc[data['Class'] == "cancer", 'Class'] = '1'

        data = data.reset_index(drop=True)
        # Count occurrences
        counts = data['Class'].value_counts()

        print(counts)
        self.data = data
        self.slices = slices

        self.image_transform = data_transform
        self.label_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = os.path.join(self.subpath, self.data.loc[i, "Path"])
        label = self.data.loc[i, "Class"]
        n = np.max([int(x[:-4]) for x in os.listdir(os.path.join(self.root, path))]) + 1
        slices = self.slices(n) if self.slices else range(n)
        slice_list = [cv.imread(os.path.join(os.path.join(self.root, path), str(i) + ".png"), cv.IMREAD_GRAYSCALE) for i in slices]
        img = np.stack(slice_list)
        if self.boxes == True:

            print(img.shape)

            slice = self.data.loc[i, "Slice"]
            x = int(self.data.loc[i, "X"])
            y = int(self.data.loc[i, "Y"])
            w = int(self.data.loc[i, "Width"])
            h = int(self.data.loc[i, "Height"])
            bbox = (slice, x, y, w, h)

            if self.image_transform is not None:
                _, H_orig, W_orig = img.shape
                _, H_new, W_new = self.image_transform(img).shape

                scale_x = W_new / W_orig
                scale_y = H_new / H_orig
                x = int(self.data.loc[i, "X"] * scale_x)
                y = int(self.data.loc[i, "Y"] * scale_y)
                w = int(self.data.loc[i, "Width"] * scale_x)
                h = int(self.data.loc[i, "Height"] * scale_y)
                bbox = (slice, x, y, w, h)

            if self.image_transform:
                img = self.image_transform(img)
            if self.label_transform:
                label = self.label_transform(label)

            info = [self.data.loc[i, "View"], self.data.loc[i,"Class"]]

            return img, label, path, bbox, info

        if self.image_transform:
            img = self.image_transform(img)
        if self.label_transform:
            label = self.label_transform(label)

        return img, label


class OMITomoSlices(Dataset):

    def __new__(cls,
                root,
                data_transform=None,
                target_transform=None,
                label_files=None,
                slices=None):

        obj = super().__new__(cls)

        return obj

    def __init__(self,
                 root,
                 data_transform=None,
                 target_transform=None,
                 label_files=None,
                 slices=None):

        if label_files is None:
            label_files = ["normal.txt", "malignant.txt"]

        self.root = root

        self.data_transform = data_transform
        self.target_transform = target_transform

        self.slices = slices

        self.image_list = []
        self.targets = []

        self.classes = ["Non malignant", "Malignant"]

        for i, label_file in enumerate(label_files):
            with open(os.path.join(self.root, label_file), "r") as f:
                tmp = [x[:-1] for x in f.readlines()]
            self.targets += [i] * len(tmp)
            self.image_list += tmp


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_list[idx])
        n = np.max([int(x[:-4]) for x in os.listdir(img_path)])+1
        slices = self.slices(n) if self.slices else range(n)
        slice_list = [cv.imread(os.path.join(img_path, str(i)+".png"), cv.IMREAD_GRAYSCALE) for i in slices]
        img = np.stack(slice_list)

        target = self.targets[idx]

        if self.data_transform:
            img = self.data_transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def random_split(self, test_ratio=0.2, seed=None, savepath=None):

        train = OMITomoSlices.__new__(OMITomoSlices, self.root)
        test = OMITomoSlices.__new__(OMITomoSlices, self.root)

        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)

        train.root = self.root
        test.root = self.root

        train.classes = self.classes
        test.classes = self.classes

        train.data_transform = self.data_transform
        test.data_transform = self.data_transform

        train.target_transform = self.target_transform
        test.target_transform = self.target_transform

        train.slices = self.slices
        test.slices = self.slices

        train.image_list = []
        train.targets = []
        test.image_list = []
        test.targets = []
        clients = []

        train.count = [0] * len(train.classes)
        test.count = [0] * len(test.classes)

        for i in range(len(self)):
            clients.append(self.image_list[i].split('/')[1])

        clients = list(dict.fromkeys(clients))
        random.seed(seed)
        clients_in_test = random.sample(clients, round(test_ratio*len(clients)))

        if savepath is not None:
            if test_ratio == 0.5:
                f_1 = open(savepath + "validation.txt", "w")
                f_2 = open(savepath + "test.txt", "w")
            else:
                f_1 = open(savepath + "training.txt", "w")
                f_2 = open(savepath + "validation.txt", "w")

        for i in range(len(self.image_list)):
            if self.image_list[i].split('/')[1] in clients_in_test:
                test.image_list.append(self.image_list[i])
                test.targets.append(self.targets[i])
                test.count[self.targets[i]] += 1
                if savepath is not None:
                    f_2.write(self.image_list[i].split('/')[1] + "\n")
            else:
                train.image_list.append(self.image_list[i])
                train.targets.append(self.targets[i])
                train.count[self.targets[i]] += 1
                if savepath is not None:
                    f_1.write(self.image_list[i].split('/')[1] + "\n")

        return train, test


class OMITomoImages(Dataset):

    def __new__(cls,
                root,
                data_transform=None,
                target_transform=None,
                label_files=None,
                slices=None):

        obj = super().__new__(cls)

        return obj

    def __init__(self,
                 root,
                 data_transform=None,
                 target_transform=None,
                 label_files=None,
                 slices=None):

        if label_files is None:
            label_files = ["normal.txt", "benign.txt", "malignant.txt"]

        self.root = root

        self.classes = [x[:-4] for x in label_files]

        self.data_transform = data_transform
        self.target_transform = target_transform

        self.slices = slices

        self.image_list = []
        self.targets = []

        for i, label_file in enumerate(label_files):
            with open(os.path.join(self.root, label_file), "r") as f:
                tmp = [x[:-1] for x in f.readlines()]
            self.targets += [i] * len(tmp)
            self.image_list += tmp

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_list[idx])
        n = np.max([int(x[:-4]) for x in os.listdir(img_path)])+1
        img = cv.imread(os.path.join(img_path, str(n//2)+".png"), cv.IMREAD_GRAYSCALE)
        target = self.targets[idx]

        if self.data_transform:
            img = self.data_transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def random_split(self, test_ratio=0.2, seed=None):

        train = OMITomoImages.__new__(OMITomoImages, self.root)
        test = OMITomoImages.__new__(OMITomoImages, self.root)

        train.root = self.root
        test.root = self.root

        train.classes = self.classes
        test.classes = self.classes

        train.data_transform = self.data_transform
        test.data_transform = self.data_transform

        train.target_transform = self.target_transform
        test.target_transform = self.target_transform

        train.slices = self.slices
        test.slices = self.slices

        print("Splitting dataset with ratio: " + str(test_ratio))

        train.image_list = []
        train.targets = []
        test.image_list = []
        test.targets = []
        clients = []
        for i in range(len(self)):
            clients.append(self.image_list[i].split('/')[1])

        clients = list(dict.fromkeys(clients))
        random.seed(seed)
        random.seed(seed)
        clients_in_test = random.sample(clients, round(test_ratio*len(clients)))
        for i in range(len(self.image_list)):
            if self.image_list[i].split('/')[1] in clients_in_test:
                test.image_list.append(self.image_list[i])
                test.targets.append(self.targets[i])
            else:
                train.image_list.append(self.image_list[i])
                train.targets.append(self.targets[i])

        return train, test

class DB:

    def __init__(self, labels_path, image_path=None, transform=None, type=0):

        if type == 1:
            self.list = []
            self.targets = []
            self.transform = transform
            return

        self.labels_path = labels_path
        self.normal_path = labels_path + "/normal.txt"
        self.benign_path = labels_path + "/benign.txt"
        self.malignant_path = labels_path + "/malignant.txt"
        self.transform = transform
        self.normal_list = []
        self.benign_list = []
        self.malignant_list = []
        self.list = []
        self.targets = []

        with open(self.normal_path) as file:
            for line in file:
                self.normal_list.append(image_path + line.strip())
                self.targets.append(0)

        with open(self.benign_path) as file:
            for line in file:
                self.benign_list.append(image_path + line.strip())
                self.targets.append(1)

        with open(self.malignant_path) as file:
            for line in file:
                self.malignant_list.append(image_path + line.strip())
                self.targets.append(2)

        self.list = list(itertools.chain(self.normal_list, self.benign_list, self.malignant_list))
        self.targets = torch.nn.functional.one_hot(torch.tensor(self.targets), 3).float()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, key):
        ds = torch.load(str(self.list[key]))
        if self.transform is not None:
            return [self.transform(ds), self.targets[key]]
        else:
            return ds, self.targets[key]


class OptimamTomoSurf:

    def __init__(self, root, type, transform=None, transfrom_labels=None):

        if type == 0: # train
            self.normal_path = root + "/training/normal/surfaces/"
            self.malignant_path = root + "/training/malignant/surfaces/"
        if type == 1: # val
            self.normal_path = root+ "/validation/normal/surfaces/"
            self.malignant_path = root + "/validation/malignant/surfaces/"
        if type == 2: # test
            self.normal_path = root + "/test/normal/surfaces/"
            self.malignant_path = root + "/test/malignant/surfaces/"

        self.transform = transform
        self.transform_labels = transfrom_labels
        self.normal_list = []
        self.malignant_list = []
        self.list = []
        self.targets = []
        self.normal_list = [os.path.join(dirpath, files) for (dirpath, dirnames, filenames) in os.walk(self.normal_path) for files in filenames]
        self.targets = [0] * len(self.normal_list)
        self.malignant_list = [os.path.join(dirpath, files) for (dirpath, dirnames, filenames) in os.walk(self.malignant_path) for files in filenames]
        self.targets += [1] * len(self.malignant_list)
        print(len(self.targets))
        self.list = list(itertools.chain(self.normal_list, self.malignant_list))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, key):
        img = Image.open(self.list[key])
        img = np.array(img).transpose(2,0,1)
        if self.transform is not None:
            return [self.transform(img), self.transform_labels(self.targets[key])]
        else:
            return img, self.targets[key]

class BCSTomoSurf:

    def __init__(self, root, transform=None, transfrom_labels=None):

        self.root = root

        self.subpath = "inference"
        self.normal_path = os.path.join(self.root, self.subpath, "normal")
        self.malignant_path = os.path.join(self.root, self.subpath, "malignant")
        self.normal_list = [os.path.join(dirpath, files) for (dirpath, dirnames, filenames) in os.walk(self.normal_path)
                            for files in filenames]

        self.targets = [0] * len(self.normal_list)
        self.malignant_list = [os.path.join(dirpath, files) for (dirpath, dirnames, filenames) in
                               os.walk(self.malignant_path) for files in filenames]
        self.targets += [1] * len(self.malignant_list)

        self.classes = ["Non malignant", "Malignant"]

        self.image_transform = transform
        self.label_transform = transfrom_labels
        print(len(self.targets))
        self.list = list(itertools.chain(self.normal_list, self.malignant_list))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, key):
        img = Image.open(self.list[key])
        img = np.array(img).transpose(2,0,1)
        if self.image_transform is not None:
            return [self.image_transform(img), self.label_transform(self.targets[key])]
        else:
            return img, self.targets[key]