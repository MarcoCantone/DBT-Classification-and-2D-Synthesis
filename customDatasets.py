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

def natural_key(s):
    # Split string into list of strings and ints
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class DukeMRIDataset(Dataset):

    def __init__(self, root, slices, split = 0, data_transform=None, target_transform=None):

        self.root = root
        self.slices = slices
        self.split = split
        self.image_transform = data_transform
        self.label_transform = target_transform
        self.image_dir = os.path.join(self.root, 'Duke-Breast-Cancer-MRI')
        self.label_filepath = os.path.join(self.root, 'pCR_info.csv')

        self.clinical_data = pd.read_csv(self.label_filepath)

        self.data_info = []
        self.patients = []
        self.labels = { 0.0 : 0, 1.0 : 1}

        self.patients = self.clinical_data["patient_id"].unique().tolist()

        # for index, row in self.clinical_data.iterrows():
        #     patient_path = str(row['classic_path'])
        #     patient_id = str(row['patient_id'])
        #
        #     class_id = int(self.labels[row['Histologic type']])
        #
        #     self.data_info.append((patient_path, class_id))
        #     self.patients.append(patient_id)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, i):
        patient_id = self.patients[i]
        label = self.clinical_data[self.clinical_data["patient_id"] == patient_id]
        label = self.labels[label["pCR"].unique().item()]
        series = self.clinical_data[self.clinical_data["patient_id"] == patient_id]
        path_pre = series[series["sequence"] == "pre"]
        path_post = series[series["sequence"] == "post_2"]
        path_pre = path_pre["classic_path"].item()
        path_post = path_post["classic_path"].item()
        path_pre = os.path.join(self.root, path_pre)
        path_post = os.path.join(self.root, path_post)
        pre_contrast = []
        post_contrast = []

        sorted_images = sorted(os.listdir(path_pre), key=natural_key)

        for image_name in sorted_images:
            image_path = path_pre + "/" + image_name
            dicom = pydicom.dcmread(image_path)
            pre_contrast.append(torch.Tensor(dicom.pixel_array))

        sorted_images = sorted(os.listdir(path_post), key=natural_key)
        for image_name in sorted_images:
            image_path = path_pre + "/" + image_name
            dicom = pydicom.dcmread(image_path)
            post_contrast.append(torch.Tensor(dicom.pixel_array))

        pre_contrast = [pre_contrast[i] for i in self.slices(len(pre_contrast))]
        post_contrast = [post_contrast[i] for i in self.slices(len(post_contrast))]

        pre_contrast = torch.stack(pre_contrast, dim=0)
        post_contrast = torch.stack(post_contrast, dim=0)

        img = torch.cat([pre_contrast, post_contrast], dim=0)

        # sequence_path = os.path.join(self.root, self.data_info[i][0])
        #
        # sorted_images = sorted(os.listdir(sequence_path), key=natural_key)
        # volume = []
        # for image_name in sorted_images:
        #     image_path = sequence_path + "/" + image_name
        #     dicom = pydicom.dcmread(image_path)
        #     volume.append(torch.Tensor(dicom.pixel_array))
        #
        # volume = [volume[i] for i in self.slices(len(volume))]
        # img = torch.stack(volume, dim=0)

        if self.image_transform:
            img = self.image_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        return img, label

    # def random_split_old(self, test_ratio=0.2, seed=None, savepath=None):
    #
    #     train = DukeMRIDataset.__new__(DukeMRIDataset)
    #     test = DukeMRIDataset.__new__(DukeMRIDataset)
    #
    #     train.root = self.root
    #     test.root = self.root
    #
    #     train.image_transform = self.image_transform
    #     test.image_transform = self.image_transform
    #
    #     train.label_transform = self.label_transform
    #     test.label_transform = self.label_transform
    #
    #     train.slices = self.slices
    #     test.slices = self.slices
    #
    #     train.data_info = []
    #     train.patients = []
    #     test.data_info = []
    #     test.patients = []
    #     clients = self.patients
    #
    #     clients = list(dict.fromkeys(clients))
    #     random.seed(seed)
    #     clients_in_test = random.sample(clients, round(test_ratio * len(clients)))
    #
    #     for i in range(len(self.data_info)):
    #         if self.patients[i] in clients_in_test:
    #             test.data_info.append(self.data_info[i])
    #             test.patients.append(self.patients[i])
    #         else:
    #             train.data_info.append(self.data_info[i])
    #             train.patients.append(self.patients[i])
    #
    #     return train, test

    def random_split(self, test_ratio=0.2, seed=None, savepath=None):

        train = DukeMRIDataset.__new__(DukeMRIDataset)
        test = DukeMRIDataset.__new__(DukeMRIDataset)

        train.root = self.root
        test.root = self.root

        train.image_transform = self.image_transform
        test.image_transform = self.image_transform

        train.label_transform = self.label_transform
        test.label_transform = self.label_transform

        train.slices = self.slices
        test.slices = self.slices

        clients = self.patients
        train.labels = self.labels
        test.labels = self.labels

        clients = list(dict.fromkeys(clients))
        random.seed(seed)
        clients_in_test = random.sample(clients, round(test_ratio * len(clients)))

        train.clinical_data = self.clinical_data[~self.clinical_data['patient_id'].isin(clients_in_test)]
        test.clinical_data = self.clinical_data[self.clinical_data['patient_id'].isin(clients_in_test)]
        train.patients = train.clinical_data["patient_id"].unique().tolist()
        test.patients = test.clinical_data["patient_id"].unique().tolist()

        return train, test


# class MamaMiaDataset(Dataset):
#
#     def __init__(self, root, split, slices, data_transform=None, target_transform=None):
#
#         self.root_dir = root
#         self.num_slices = slices()
#         self.image_dir = os.path.join(self.root_dir, 'images')
#         self.label_filepath = os.path.join(self.root_dir, 'clinical_info.csv')
#         df_split = pd.read_csv(os.path.join(self.root_dir, 'train_test_splits.csv'))
#         patients_list = df_split[split].to_list()
#         self.image_transform = data_transform
#         self.label_transform = target_transform
#
#         self.clinical_data = pd.read_csv(self.label_filepath)
#
#         self.clinical_data.loc[self.clinical_data['tumor_subtype'].isin(['luminal_a', 'luminal_b']), 'tumor_subtype'] = 'luminal'
#         # --- Multiclass Labeling Logic ---
#         # Get all unique tumor subtypes, EXCLUDING NaN values
#         # .dropna() removes NaN before getting unique values
#
#         unique_subtypes = self.clinical_data['tumor_subtype'].dropna().unique()
#
#         # Create a mapping from subtype string to integer ID
#         # It's good practice to sort them for consistent ID assignment
#         self.subtype_to_id = {subtype: i for i, subtype in enumerate(sorted(unique_subtypes))}
#         self.id_to_subtype = {i: subtype for subtype, i in self.subtype_to_id.items()}
#
#         self.data_info = []
#         # Prepare data_info list: (patient_id, label)
#         for index, row in self.clinical_data.iterrows():
#             patient_id = str(row['patient_id'])  # Ensure patient_id is string for path consistency
#             if patient_id not in patients_list:  # Check if patient is in the proper split
#                 continue
#
#             if pd.isna(row['tumor_subtype']): # Skip if there is no label
#                 continue
#
#             # Get the subtype string
#             subtype_str = str(row['tumor_subtype'])  # pd.notna() already handled by if-condition above
#
#             # Get the corresponding numerical class ID
#             class_id = self.subtype_to_id[subtype_str]
#
#             image_path_in_patient_folder = os.path.join(self.image_dir, patient_id)
#             # Check if the image file actually exists to avoid errors during iteration
#             if os.path.exists(image_path_in_patient_folder):
#                 self.data_info.append((patient_id, class_id))
#             else:
#                 print(f"Warning: Image file not found for patient '{patient_id}' at {image_path_in_patient_folder}. Skipping.")
#
#         list_try = []
#         for i in range(len(self.data_info)):
#             list_try.append(self.data_info[i][1])
#
#         class_counts = Counter(list_try)
#
#         print(class_counts)
#
#     def __len__(self):
#         return len(self.data_info)
#
#     def __getitem__(self, idx):
#
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         patient_id, label = self.data_info[idx]
#
#         patient_path = os.path.join(self.image_dir, patient_id)
#
#         # Load NIfTI image using nibabel
#         try:
#             # nib.load returns a Nifti1Image object. .get_fdata() extracts the data as a numpy array.
#             pre_contrast = torch.tensor(nib.load(os.path.join(patient_path, patient_id + "_0000.nii.gz")).get_fdata())
#             post_contrast = torch.tensor(nib.load(os.path.join(patient_path, patient_id + "_0002.nii.gz")).get_fdata())
#             pre_contrast = torch.permute(pre_contrast, [2, 0, 1])
#             # Generate 10 evenly spaced indices from 0 to 99
#             indices = torch.linspace(0, pre_contrast.size(0) - 1, steps=self.num_slices).long()
#             pre_contrast = pre_contrast[indices]
#             post_contrast = torch.permute(post_contrast, [2, 0, 1])
#             indices = torch.linspace(0, pre_contrast.size(0) - 1, steps=self.num_slices).long()
#             post_contrast = post_contrast[indices]
#             img = torch.cat([pre_contrast, post_contrast], dim=0)
#         except Exception as e:
#             raise RuntimeError(f"Failed to load image from patient {patient_id}: {e}")
#
#         if self.image_transform:
#             img = self.image_transform(img)
#
#         if self.label_transform:
#             label = self.label_transform(label)
#
#         return img, label

# data = MamaMiaDataset("/ssd2/dellascenza/datasets/MAMA-MIA/MAMA-MIA_data/", "train_split", slices=40)
# for sample in data:
#     print(sample[0].shape, sample[1])

class AMBLDataset(Dataset):

    def __init__(self, root,
                 data_transform=None,
                 target_transform=None,
                 slices=None):
        super(AMBLDataset, self).__init__()

        self.root = root

        self.data = pd.read_csv(self.root + "AMBL_data.csv")
        self.slices = slices

        blacklisted_patients = ["AMBL-592"]

        self.samples = []
        self.patients = []

        for _, row in self.data.iterrows():
            file_path = row["File Location"][2:]

            label_fields = [
                row.get("tumor/benign1", 0),
                row.get("tumor/benign2", 0),
                row.get("tumor/benign3", 0),
                row.get("tumor/benign4", 0),
                row.get("tumor/benign5", 0),
                row.get("tumor/benign6", 0),
            ]

            label_values = [float(x) for x in label_fields if pd.notna(x)]
            class_id = 1 if any(val == 1.0 for val in label_values) else 0

            if ("AX Sen Vibrant MultiPhase" in file_path
                    and "Registered" not in file_path
                    and row["BIRADS"] not in [-1, 0]
                    and row["Patient ID"] not in blacklisted_patients):
                self.samples.append((root + file_path, class_id))
                self.patients.append(row["Patient ID"])

        self.image_transform = data_transform
        self.label_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        pre_contrast = []
        post_contrast = []

        for item in os.listdir(file_path): # This way images will be stacked properly
            item_path = file_path + '/' + item
            if item[0] == '1':
                dicom = pydicom.dcmread(item_path)
                pre_contrast.append(torch.Tensor(dicom.pixel_array))
            if item[0] == '3':
                dicom = pydicom.dcmread(item_path)
                post_contrast.append(torch.Tensor(dicom.pixel_array))

        pre_contrast = [pre_contrast[i] for i in self.slices(len(pre_contrast))]
        pre_contrast = torch.stack(pre_contrast, dim=0)
        # print(pre_contrast.shape)

        post_contrast = [post_contrast[i] for i in self.slices(len(post_contrast))]
        post_contrast = torch.stack(post_contrast, dim=0)
        # print(post_contrast.shape)

        img = torch.cat([pre_contrast, post_contrast], dim=0)

        #print(img.shape)

        if self.image_transform:
            img = self.image_transform(img)
        if self.label_transform:
            label = self.label_transform(label)

        return img, label

    def random_split(self, test_ratio=0.2, seed=None, savepath=None):

        train = AMBLDataset.__new__(AMBLDataset)
        test = AMBLDataset.__new__(AMBLDataset)

        train.root = self.root
        test.root = self.root

        train.image_transform = self.image_transform
        test.image_transform = self.image_transform

        train.label_transform = self.label_transform
        test.label_transform = self.label_transform

        train.slices = self.slices
        test.slices = self.slices

        train.samples = []
        train.patients = []
        test.samples = []
        test.patients = []
        clients = self.patients

        clients = list(dict.fromkeys(clients))
        random.seed(seed)
        clients_in_test = random.sample(clients, round(test_ratio * len(clients)))

        for i in range(len(self.samples)):
            if self.patients[i] in clients_in_test:
                test.samples.append(self.samples[i])
                test.patients.append(self.patients[i])
            else:
                train.samples.append(self.samples[i])
                train.patients.append(self.patients[i])

        return train, test

class BCS_DBT(Dataset):

    def __init__(self, root, labels_file = "labels_inference.txt",
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
        # if not os.path.exists(os.path.join(self.root, path)):
        #     return None, None, None
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


    def random_split(self, test_ratio=0.2, seed=None, savepath=None):

        train = BCSTomoSurf.__new__(BCSTomoSurf)
        test = BCSTomoSurf.__new__(BCSTomoSurf)

        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)

        train.root = self.root
        test.root = self.root

        train.root = self.root
        test.root = self.root

        train.classes = self.classes
        test.classes = self.classes

        train.image_transform = self.image_transform
        test.image_transform = self.image_transform

        train.label_transform = self.label_transform
        test.label_transform = self.label_transform

        train.list = []
        train.targets = []
        test.list = []
        test.targets = []
        image_list = []

        for i in range(len(self)):
            image_list.append(self.list[i].split('/')[10])

        image_list = list(dict.fromkeys(image_list))
        random.seed(seed)
        clients_in_test = random.sample(image_list, round(test_ratio*len(image_list)))

        if savepath is not None:
            if test_ratio == 0.5:
                f_1 = open(savepath + "validation.txt", "w")
                f_2 = open(savepath + "test.txt", "w")
            else:
                f_1 = open(savepath + "training.txt", "w")
                f_2 = open(savepath + "validation.txt", "w")

        for i in range(len(self.list)):
            if self.list[i].split('/')[10] in clients_in_test:
                test.list.append(self.list[i])
                test.targets.append(self.targets[i])
                if savepath is not None:
                    f_2.write(self.list[i].split('/')[1] + "\n")
            else:
                train.list.append(self.list[i])
                train.targets.append(self.targets[i])
                if savepath is not None:
                    f_1.write(self.list[i].split('/')[1] + "\n")

        return train, test


def split_dataset(dataset_path, db, test_ratio=0.2, perc_of_data=1, shuffle=True, random_state=None):

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    client_list = []

    list_len = int(len(db.list) * perc_of_data)

    # Generazione lista di pazienti senza ripetizioni
    for i in range(0, list_len):
        client_list.append(db.list[i].split('/')[7])

    client_list = list(dict.fromkeys(client_list))

    num_test_clients = int(len(client_list) * test_ratio)

    if shuffle:
        random.seed(random_state)
        test_clients = random.sample(client_list, num_test_clients)
    else:
        test_clients = client_list

    db_train = DB(dataset_path, type=0, transform=db.transform)
    db_test = DB(dataset_path, type=1, transform=db.transform)

    for i in range(list_len):
        if db.list[i].split('/')[7] in test_clients:
            db_test.list.append(db.list[i])
            db_test.targets.append(db.targets[i])
        else:
            db_train.list.append(db.list[i])
            db_train.targets.append(db.targets[i])

    return db_train, db_test


