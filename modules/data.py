from typing import *

import os
import numpy as np
import torch

import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
import nibabel as nib
from sklearn.model_selection import train_test_split


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


class MIPDataset(torch.utils.data.Dataset):
    """
    Dataset for 2D Maximum Intensity Projection (MIP) images stored as NIfTI files.
    """
    def __init__(
        self,
        root: str = None,
        paths: List[str] = None, # Optional list of file paths to use instead of discovering from root
        pattern: str = '_MIP',
        cache_after_load: bool = True,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        resize: Optional[Tuple[int, int]] = None,
        horizontal_flip: Optional[float] = None,
        vertical_flip: Optional[float] = None,
        random_crop_size: Optional[Tuple[int, int]] = None,
        dtype: str = 'float32'
    ):
        super().__init__()
        assert root is not None or paths is not None, "Either root directory or list of paths must be provided"
        self.root = root
        self.pattern = pattern.lower()
        self.cache_after_load = cache_after_load
        self.normalize = normalize
        self.transform = T.Compose(
            [
                T.Resize(resize) if resize is not None else nn.Identity(),
                (
                    T.RandomHorizontalFlip(p=horizontal_flip)
                    if horizontal_flip
                    else nn.Identity()
                ),
                (
                    T.RandomVerticalFlip(p=vertical_flip)
                    if vertical_flip
                    else nn.Identity()
                ),
                (
                    T.RandomCrop(random_crop_size)
                    if random_crop_size is not None
                    else nn.Identity()
                ),
                T.ConvertImageDtype(getattr(torch, dtype)),
            ]
        ) if transform is None else transform
        self.dtype = dtype
        
        # discover MIP files
        self.paths = self._discover_mip_files(self.root, pattern=self.pattern) if paths is None else paths
        if self.paths.__len__() == 0:
            raise RuntimeError(f"No MIP files found under root: {root} (pattern='{pattern}')")

        # cache for loaded images and per-image stats
        self._images: Dict[int, np.ndarray] = {}
        self._stats: Dict[int, Tuple[float, float]] = {}

    # ---------------- discovery / I/O ----------------
    @staticmethod
    def _discover_mip_files(root: str, pattern: str) -> List[str]:
        files = []
        for subject in sorted(os.listdir(root)):
            subject_dir = os.path.join(root, subject)
            if not os.path.isdir(subject_dir):
                continue
            for name in sorted(os.listdir(subject_dir)):
                low = name.lower()
                if pattern in low and (low.endswith('.nii') or low.endswith('.nii.gz')):
                    files.append(os.path.join(subject_dir, name))
        return files

    def _load_nifti_as_numpy(self, path: str) -> np.ndarray:
        """Load a 2D NIfTI using nibabel and return float32 numpy array (H, 1, W) [Coronal MIP] => (H, W)."""
        nii = nib.load(path)
        arr = nii.get_fdata().squeeze()
        arr = np.rot90(arr, k=1, axes=(0, 1))  # rotate the MIP to standard orientation (effect introduced by numpy z, y, x ordering)
        if arr.ndim != 2:
            raise RuntimeError(f"Expected 2D MIP image at {path}, got array shape {arr.shape}")
        return arr.astype(getattr(np, self.dtype))

    @staticmethod
    def _compute_mean_std(arr: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
        mean, std = np.mean(arr), np.max([np.std(arr), eps])
        assert mean is not None and std is not None
        return mean, std

    # ---------------- Dataset protocol ----------------
    def __len__(self) -> int:
        return self.paths.__len__()

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self.paths.__len__() + idx
        if idx < 0 or idx >= self.paths.__len__():
            raise IndexError(idx)

        # load or get from cache
        if idx in self._images:
            arr = self._images[idx]
        else:
            arr = self._load_nifti_as_numpy(self.paths[idx])
            if self.cache_after_load:
                self._images[idx] = arr

        # compute mean/std if needed and cache
        if idx in self._stats:
            mean, std = self._stats[idx]
        else:
            mean, std = self._compute_mean_std(arr)
            self._stats[idx] = (mean, std)

        # normalize (z-score per-image)
        if self.normalize:
            arr_norm = (arr - mean) / std
        else:
            arr_norm = arr

        # convert to tensor (1, H, W)
        tensor = torch.from_numpy(arr_norm).unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        sample = {
            'image': tensor,            # torch.FloatTensor shape (1, H, W)
            'mean': float(mean),        # float, used to denormalize
            'std': float(std),          # float
            'path': self.paths[idx],
        }
        return sample


class MIPDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        pattern: str = '_MIP',
        train_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 4,
        shuffle: bool = True,
        verbose: bool = True,
        **dataset_kwargs,
    ):
        super().__init__()
        self.root = root
        self.pattern = pattern.lower()
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.verbose = verbose
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset: Optional[MIPDataset] = None
        self.val_dataset: Optional[MIPDataset] = None

    def setup(self, stage: Optional[str] = None):
        paths = MIPDataset._discover_mip_files(self.root, pattern=self.pattern)
        train_paths, val_paths = train_test_split(
            paths,
            train_size=self.train_ratio,
            shuffle=self.shuffle
        )

        self.train_dataset = MIPDataset(paths=train_paths, **self.dataset_kwargs)
        self.val_dataset = MIPDataset(paths=val_paths, **self.dataset_kwargs)

        if self.verbose:
            print(f"Discovered {paths.__len__()} MIP files under {self.root} with pattern '{self.pattern}'")
            print(f"  Training samples: {self.train_dataset.__len__()}")
            print(f"  Validation samples: {self.val_dataset.__len__()}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
