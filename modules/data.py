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
from torch.utils.data import DataLoader

import torchio as tio

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    MapTransform
)
from monai.data import CacheDataset, list_data_collate


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


# EFFICIENT DATAMODULE FOR PET/EARL TRANSLATION TASK USING MONAI

# --- 1. TRANSFORMATION CUSTOM POUR LA NORMALISATION CONJOINTE ---
class JointZScoreNormalize(MapTransform):
    """
    Calcule Mean/Std sur l'image 'source_key' uniquement (en ignorant les zéros si demandé).
    Applique (X - mean) / std sur TOUTES les images (source et target).
    Sauvegarde mean et std dans le dictionnaire pour la reconstruction.
    """
    def __init__(self, keys, source_key="source", ignore_zeros=True, eps=1e-8, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.ignore_zeros = ignore_zeros
        self.eps = eps

    def __call__(self, data):
        d = dict(data)
        
        # 1. Récupération de l'image source
        img = d[self.source_key] # Ceci est un Tensor ou un Numpy array
        
        # 2. Calcul des stats sur la SOURCE uniquement
        if self.ignore_zeros:
            # Masque pour ne pas prendre en compte le fond noir infini dans la moyenne
            # Cela évite d'avoir une moyenne proche de 0 et un std énorme
            mask = img > 0
            if mask.sum() > 0:
                mean = img[mask].mean().item()
                std = img[mask].std().item()
            else:
                # Fallback si l'image est vide (rare)
                mean = 0.0
                std = 1.0
        else:
            mean = img.mean().item()
            std = img.std().item()
            
        # Sécurité
        std = max(std, self.eps)

        # 3. Sauvegarde des métadonnées (Important pour votre reconstruction !)
        # On les stocke sous forme de float simple pour l'instant
        d["norm_mean"] = np.array([mean], dtype=np.float32)
        d["norm_std"] = np.array([std], dtype=np.float32)

        # 4. Application de la normalisation (Même mu/sigma pour tout le monde)
        for key in self.key_iterator(d):
            d[key] = (d[key] - mean) / std

        return d
    

def robust_patch_normalization(src: torch.Tensor, tgt: torch.Tensor, percentiles=(0.5, 99.5), clone=True):
    src_out, tgt_out = src, tgt
    if clone:
        src_out = src.clone()
        tgt_out = tgt.clone()
    
    batch_size = src_out.shape[0]
    norm_factors = torch.zeros((batch_size, 2), device=src.device)  # min, max per patch

    for idx in range(batch_size):
        src_patch = src_out[idx]
        tgt_patch = tgt_out[idx]
        
        # Aplatir pour calculer les percentiles
        src_flat = src_out.view(-1)
        
        # Calcul des seuils (quantile attend une entrée float)
        # Note: quantile sur GPU est rapide
        p_min = torch.quantile(src_flat, percentiles[0] / 100.0)
        p_max = torch.quantile(src_flat, percentiles[1] / 100.0)
        
        if (p_max - p_min) < 1e-6:
            continue
            
        src_patch = 2 * (src_patch - p_min) / (p_max - p_min) - 1
        tgt_patch = 2 * (tgt_patch - p_min) / (p_max - p_min) - 1

        norm_factors[idx, 0] = p_min
        norm_factors[idx, 1] = p_max
        
        src_out[idx] = src_patch
        tgt_out[idx] = tgt_patch
        
    return src_out, tgt_out, norm_factors

def robust_patch_denormalization(src: torch.Tensor, tgt: torch.Tensor, norm_factors: torch.Tensor, clone=True):
    src_out, tgt_out = src, tgt
    if clone:
        src_out = src.clone()
        tgt_out = tgt.clone()
        
    batch_size = src_out.shape[0]
    
    for idx in range(batch_size):
        p_min = norm_factors[idx, 0]
        p_max = norm_factors[idx, 1]
        
        if (p_max - p_min) < 1e-6:
            continue
            
        src_patch = src_out[idx]
        tgt_patch = tgt_out[idx]
        
        src_patch = (src_patch + 1) * (p_max - p_min) / 2 + p_min
        tgt_patch = (tgt_patch + 1) * (p_max - p_min) / 2 + p_min
        
        src_out[idx] = src_patch
        tgt_out[idx] = tgt_patch
        
    return src_out, tgt_out


class Float32Lambda:
    def __init__(self):
        pass

    def __call__(self, subject):
        for image in subject.get_images(intensity_only=False):
            image.data = image.data.float()
        return subject
        
    

# --- LE LIGHTNING DATA MODULE ---
class PETTranslationDataModule(LightningDataModule):
    def __init__(
        self, 
        root_dir: str, 
        batch_size: int = 4, 
        train_ratio: float = 0.8,
        patch_size: tuple = (64, 64, 64),
        num_workers: int = 8,             # Augmentez si vous avez bcp de coeurs
        queue_max_length: int = 600,      
        samples_per_volume: int = 4, # On tire 4 patches par patient
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume

    def get_pt_earl_file_pairs(self, files: List[str]) -> List[Dict[str, str]]:
        #  --- Adaptez ces filtres à vos noms de fichiers exacts ---
        files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        pt_files = [f for f in files if f.startswith('PET') and 'MIP' not in f]
        earl_files = [f for f in files if f.startswith('EARL') and 'MIP' not in f]
        return pt_files[0], earl_files[0]

    def setup(self, stage=None):
        # --- Listing des fichiers ---
        all_subjects = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        tio_subjects = list()
        for subj_name in all_subjects:
            subj_path = os.path.join(self.root_dir, subj_name)
            files = os.listdir(subj_path)
            pt_file, earl_file = self.get_pt_earl_file_pairs(files)
            
            if pt_file and earl_file:
                subject = tio.Subject(
                    source=tio.Image(os.path.join(subj_path, pt_file), type=tio.INTENSITY),
                    target=tio.Image(os.path.join(subj_path, earl_file), type=tio.INTENSITY),
                    subject_id=subj_name
                )

                tio_subjects.append(subject)
        
        # --- Split train/val ---
        np.random.shuffle(tio_subjects) # Mélange avant split
        split_idx = int(len(tio_subjects) * self.train_ratio)
        self.train_subjects, self.val_subjects = tio_subjects[:split_idx], tio_subjects[split_idx:]

        # just for the record (in case needed)
        self.train_subj_paths, self.val_subj_paths = all_subjects[:split_idx], all_subjects[split_idx:]
        
        print(f"[TorchIO] {len(self.train_subjects)} Train, {len(self.val_subjects)} Val.")

        # --- Pipelines de Transformation ---        
        self.transform = tio.Compose([
            Float32Lambda(),
            tio.ToCanonical(),
            tio.RandomFlip(axes=(0, 1, 2), p=0.5),
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=True, p=0.5),
        ])

    def train_dataloader(self):
        if self.train_subjects.__len__() > 0:
            train_dataset = tio.SubjectsDataset(self.train_subjects, transform=self.transform)
            sampler = tio.data.UniformSampler(self.patch_size)

            patches_queue = tio.Queue(
                subjects_dataset=train_dataset,
                max_length=self.queue_max_length,
                samples_per_volume=self.samples_per_volume,
                sampler=sampler,
                num_workers=self.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True
            )
            return tio.SubjectsLoader(
                patches_queue,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True
            )
        return None


    def val_dataloader(self):
        if self.val_subjects.__len__() > 0:
            val_dataset = tio.SubjectsDataset(self.val_subjects, transform=self.transform)
            sampler = tio.data.UniformSampler(self.patch_size)

            patches_queue = tio.Queue(
                subjects_dataset=val_dataset,
                max_length=100,
                samples_per_volume=1,
                sampler=sampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False
            )
            
            return tio.SubjectsLoader(
                patches_queue,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True
            )
        return None
    
    def test_dataloader(self):
        if self.val_subjects.__len__() > 0:
            val_dataset = tio.SubjectsDataset(self.val_subjects, transform=tio.Compose([Float32Lambda(), tio.ToCanonical()]))
            return tio.SubjectsLoader(
                val_dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=True,
                shuffle=False
            )
