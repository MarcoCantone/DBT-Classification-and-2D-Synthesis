import os
from customDatasets import BCS_DBT
import cv2
import numpy as np

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

data_root = "BCS-DBT_boxes/"

save_path = "BCS-DBT_cropped_boxes/"

with open(save_path + "crop_coords_new_final.csv", "w") as f:
    f.write("Path,min_x,min_y,min_z,max_x,max_y,max_z,View,Class,Slice,X,Y,Width,Height,flipped,img_width,img_height\n")
    data = BCS_DBT(root=data_root, labels_file="BCS-DBT_cropped_boxes/info.csv",
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

        _, bin = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

        min_coords, max_coords = find_bounding_box(bin)
        x_min, y_min, z_min, x_max, y_max, z_max = min_coords[1], min_coords[0], min_coords[2], max_coords[1], max_coords[0], max_coords[2]

        f.write(f'{new_path},{x_min},{y_min},{z_min},{x_max},{y_max},{z_max},{info[0]},{info[1]},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]},{need_to_flip_coords},{img.shape[1]},{img.shape[0]}\n')

        image_with_bbox = cv2.cvtColor(img.max(2), cv2.COLOR_GRAY2BGR)

        for k in range(z_min, z_max):
            single_slice = cv2.imread(os.path.join(original_path, str(k) + ".png"), cv2.IMREAD_GRAYSCALE)

            if flip:
                single_slice = cv2.flip(single_slice, 1)

            cropped_slice = single_slice[y_min:y_max + 1, x_min:x_max + 1]
            os.makedirs(os.path.join(save_path, new_path), exist_ok=True)
            cv2.imwrite(os.path.join(os.path.join(save_path, new_path), str(k) + ".png"), cropped_slice)