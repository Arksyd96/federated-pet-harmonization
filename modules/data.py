import itertools
from typing import *
import random
import cv2

import numpy as np
import torch
from monai.transforms import RandSpatialCrop
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from scipy.ndimage import distance_transform_edt as edt


class IdentityDataset(torch.utils.data.Dataset):
    """
    Simple dataset that returns the same data (d0, d1, ..., dn)
    """

    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

def normalize(input_data, norm="centered-norm"):
    assert norm in [
        "centered-norm",
        "z-score",
        "min-max",
    ], "Invalid normalization method"

    if norm == "centered-norm":
        norm = lambda x: (2 * x - x.min() - x.max()) / (x.max() - x.min())
    elif norm == "z-score":
        norm = lambda x: (x - x.mean()) / x.std()
    elif norm == "min-max":
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
    return norm(input_data)
