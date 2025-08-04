import os

import cv2

from customDatasets import BCS_DBT
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pandas as pd

def conditionalflip(image):

    img_left = image[:,:int(image.shape[1]/2)]
    img_right = image[:,int(image.shape[1]/2):]

    print(np.average(img_right))
    print(np.average(img_left))

    if np.average(img_right) < np.average(img_left):
        return True
    else:
        return False

def find_bounding_box(binary_image):
    # Find the indices of the non-zero elements
    coords = np.argwhere(binary_image)
    # If no non-zero elements found, return an empty bounding box
    if coords.shape[0] == 0:
        return None
    # Find the min and max indices along each dimension
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    # Return the bounding box coordinates
    return min_coords, max_coords

# ------- FIND 3D BOUNDING BOXES AND SAVE MINIATURES -------

data_root = "/ssd2/dellascenza/datasets/BCS-DBT_boxes/"

save_path = "/ssd2/dellascenza/datasets/BCS-DBT_cropped_boxes/"

with open(save_path + "crop_coords_new_final.csv", "w") as f:
    f.write("Path,min_x,min_y,min_z,max_x,max_y,max_z,View,Class,Slice,X,Y,Width,Height,flipped,img_width,img_height\n")
    data = BCS_DBT(root=data_root, labels_file="/ssd2/dellascenza/datasets/BCS-DBT_cropped_boxes/info.csv",
                                                              boxes=True)
    for i in range(len(data)):
        print(i)
        img, label, path, bbox, info = data[i]

        new_path = '/'.join(path.split('/')[1:])

        complete_path = os.path.join(save_path, path)
        original_path = os.path.join(data.root, path)

        img_to_check = img[int(img.shape[0]/2)]
        flip = conditionalflip(img_to_check)
        need_to_flip_coords = False
        if flip:
            need_to_flip_coords = True
            print("Flipping...")
            for k in range(0,img.shape[0]):
                img[k] = cv2.flip(img[k], 1)

        img = img.transpose(1, 2, 0)

        _, bin = cv.threshold(img, 30, 255, cv.THRESH_BINARY)

        min_coords, max_coords = find_bounding_box(bin)
        x_min, y_min, z_min, x_max, y_max, z_max = min_coords[1], min_coords[0], min_coords[2], max_coords[1], max_coords[0], max_coords[2]

        f.write(f'{new_path},{x_min},{y_min},{z_min},{x_max},{y_max},{z_max},{info[0]},{info[1]},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]},{need_to_flip_coords},{img.shape[1]},{img.shape[0]}\n')

        image_with_bbox = cv.cvtColor(img.max(2), cv.COLOR_GRAY2BGR)
        # cv.rectangle(image_with_bbox, (min_coords[1], min_coords[0]), (max_coords[1], max_coords[0]), (0, 255, 0), 20)
        #
        # image_with_bbox = cv.resize(image_with_bbox, (0, 0), fx=0.2, fy=0.2)
        #
        # x_min, y_min, z_min, x_max, y_max, z_max = min_coords[1], min_coords[0], min_coords[2], max_coords[1], max_coords[0], max_coords[2]
        for k in range(z_min, z_max):
            single_slice = cv.imread(os.path.join(original_path, str(k) + ".png"), cv.IMREAD_GRAYSCALE)

            if flip:
                single_slice = cv2.flip(single_slice, 1)

            cropped_slice = single_slice[y_min:y_max + 1, x_min:x_max + 1]
            os.makedirs(os.path.join(save_path, new_path), exist_ok=True)
            cv.imwrite(os.path.join(os.path.join(save_path, new_path), str(k)+ ".png"), cropped_slice)

# # ------- CREATE CROPPED DATASET -------
#
# import os
# import pandas as pd
# import time
#
# dest_dir = "/data/cantone/datasets/OMIDB.tomo.SlicesPNG.cropped/"
# root = "/data/cantone/datasets/OMIDB.tomo.SlicesPNG/"
# clients = os.listdir(root)
# tot = len(clients)
# coords = pd.read_csv("/home/cantone/Datasets/OMITomoSlices/crop_coords.csv")
#
# start = time.time()
# for j, client in enumerate(clients):
#     print(f'{client}  ({j}/{tot})')
#     if j>5:
#         elapsed = time.time()-start
#         eta = (elapsed/j)*(tot-j)
#         print(f'eta={eta}')
#     if os.path.isdir(os.path.join(dest_dir, client)):
#         print(f'{os.path.join(dest_dir, client)} already exist')
#         continue
#     if os.path.isfile(os.path.join(root, client)):
#         print(f'skipping {os.path.join(root, client)}')
#         continue
#     os.makedirs(os.path.join(dest_dir, client))
#     studies = os.listdir(os.path.join(root, client))
#     for study in studies:
#         os.makedirs(os.path.join(dest_dir, client, study))
#         images = os.listdir(os.path.join(root, client, study))
#         for img in images:
#             os.makedirs(os.path.join(dest_dir, client, study, img))
#             x_min, y_min, z_min, x_max, y_max, z_max = coords[coords["image"] == os.path.join(client, study, img)].values[0, 1:]
#             for i in range(z_min, z_max+1):
#                 slice = cv.imread(os.path.join(root, client, study, img, str(i)+".png"), cv.IMREAD_GRAYSCALE)
#                 cropped_slice = slice[y_min:y_max+1, x_min:x_max+1]
#                 cv.imwrite(os.path.join(dest_dir, client, study, img, str(i)+".png"), cropped_slice)

