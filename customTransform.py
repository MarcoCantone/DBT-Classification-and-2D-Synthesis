import torch
from utils import BCS_DBT_classes


class DicomToNumpy(object):
    def __call__(self, dicom):
        return dicom.pixel_array.transpose(1, 2, 0).astype("float32")


class MinMax(object):
    def __call__(self, x):
        return (x-x.min())/(x.max()-x.min())

class Resize(object):
    def __call__(self, x):
        return torch.nn.functional.interpolate(torch.tensor(x).unsqueeze(0).unsqueeze(0), size=(1024, 512),
                                         mode='bilinear').squeeze(0)

class toMonaiShape(object):
    def __call__(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 2)
        return x

class NearestResize(object):
    def __init__(self, patch_size, rescale_factor=1, sampling_slices_step=1):
        self.p_w = patch_size[0]
        self.p_h = patch_size[1]
        self.rescale_factor = rescale_factor
        self.sampling_slices_step = sampling_slices_step

    def __call__(self, image3d):
        _, x, y = image3d.shape
        image3d = image3d[::self.sampling_slices_step]
        x, y = x//self.rescale_factor, y//self.rescale_factor
        resize = Resize((round(x/self.p_h)*self.p_h, round(y/self.p_w)*self.p_w))
        return resize(image3d)


class BCS_DBT_classification_label(object):
    def __call__(self, label):
        transformed_label = torch.zeros(4)
        transformed_label[BCS_DBT_classes.index(label[0])] = 1
        return transformed_label


class ToTensorFloat32:
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32)


class OneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        out = torch.zeros(self.num_classes, dtype=torch.float)
        out[int(x)] = 1
        return out

class Unsqueeze:
    def __call__(self, x):
        return x.unsqueeze(0)