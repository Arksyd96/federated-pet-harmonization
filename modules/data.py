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
    

class MinMaxPercentileNormalize(MapTransform):
    def __init__(self, keys, source_key="source", percentile=99.5, min_max_suv=3.0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.percentile = percentile
        self.min_max_suv = min_max_suv # Sécurité pour les patients très faibles

    def __call__(self, data):
        d = dict(data)
        img = d[self.source_key] # Image Source (Standard)
        
        mask = img > 0
        if mask.sum() > 0:
            valid_voxels = img[mask]
            robust_max = np.percentile(valid_voxels, self.percentile)
        else:
            robust_max = 1.0

        final_max = max(robust_max, self.min_max_suv)
        d["norm_max"] = np.array([final_max], dtype=np.float32)

        for key in self.key_iterator(d):
            x = d[key]
            
            if isinstance(x, np.ndarray):
                x_clipped = np.clip(x, 0, final_max)
            else:
                x_clipped = x.clamp(0, final_max)
            
            d[key] = (x_clipped / final_max) * 2.0 - 1.0

        return d

# --- 2. LE LIGHTNING DATA MODULE ---
class PETTranslationDataModule(LightningDataModule):
    def __init__(
        self, 
        root_dir: str, 
        batch_size: int = 4, 
        train_ratio: float = 0.8,
        patch_size: tuple = (64, 64, 64),
        spacing: tuple = (2.0, 2.0, 2.0), # Resampling isotrope recommandé
        num_workers: int = 8,             # Augmentez si vous avez bcp de coeurs
        cache_rate: float = 1.0           # 1.0 = Charge tout le dataset en RAM
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.patch_size = patch_size
        self.spacing = spacing
        self.num_workers = num_workers
        self.cache_rate = cache_rate

    # à définir selon l'arborescence de vos données
    def get_pt_earl_file_pairs(self, files: List[str]) -> List[Dict[str, str]]:
        # Adaptez ces filtres à vos noms de fichiers exacts
        files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz') and 'MIP' not in f]
        pt_files = [f for f in files if f.startswith('PT') and 'EARL' not in f]
        earl_files = [f for f in files if f.startswith('PT') and 'EARL' in f]
        return pt_files[0], earl_files[0]

    def setup(self, stage=None):
        # --- A. Listing des fichiers ---
        data_dicts = []
        subjects = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        for subj in subjects:
            subj_path = os.path.join(self.root_dir, subj)
            files = os.listdir(subj_path)
            pt_file, earl_file = self.get_pt_earl_file_pairs(files)
            
            if pt_file and earl_file:
                data_dicts.append({
                    "source": os.path.join(subj_path, pt_file),
                    "target": os.path.join(subj_path, earl_file),
                    "subject_id": subj # Toujours utile pour le debug
                })
        
        # Split Train/Val (80/20)
        np.random.shuffle(data_dicts) # Mélange avant split
        split_idx = int(len(data_dicts) * self.train_ratio)
        train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
        
        print(f"[DataModule] Setup complet. Train: {len(train_files)} | Val: {len(val_files)}")
        print(f"[DataModule] Cache Rate: {self.cache_rate} (RAM usage estimé ~130Go pour 1000 patients)")

        # --- B. Pipelines de Transformation ---
        
        # NOTE IMPORTANTE SUR LE CACHING :
        # CacheDataset va exécuter les transformations JUSQU'À la première transformation aléatoire (Rand*).
        # C'est pourquoi nous mettons le chargement, le resampling et la normalisation AVANT le crop.
        # Ainsi, le volume en RAM sera déjà propre et normalisé Z-Score.
        
        self.train_transforms = Compose([
            LoadImaged(keys=["source", "target"]),
            EnsureChannelFirstd(keys=["source", "target"]),
            Orientationd(keys=["source", "target"], axcodes="RAS"),
            
            # Resampling isotrope (Source et Target restent alignées)
            Spacingd(keys=["source", "target"], pixdim=self.spacing, mode=("bilinear", "bilinear")),
            
            # NOTRE NORMALISATION CUSTOM (Déterministe -> Sera cachée)
            # JointZScoreNormalize(keys=["source", "target"], source_key="source", ignore_zeros=True),
            MinMaxPercentileNormalize(keys=["source", "target"], source_key="source", percentile=99.5, min_max_suv=3.0),
            
            # --- À partir d'ici, transformations ALÉATOIRES (Calculées à la volée sur le CPU) ---
            
            # Crop 3D synchronisé sur source et target
            RandSpatialCropd(
                keys=["source", "target"], 
                roi_size=self.patch_size, 
                random_center=True, 
                random_size=False
            ),
            
            # Data Augmentation géométrique
            RandFlipd(keys=["source", "target"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["source", "target"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["source", "target"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["source", "target"], prob=0.5, max_k=3),
            
            # Conversion finale
            # Notez qu'on inclut "norm_mean" et "norm_std" pour qu'ils deviennent des Tensors dans le batch
            ToTensord(keys=["source", "target", "norm_max"]),
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=["source", "target"]),
            EnsureChannelFirstd(keys=["source", "target"]),
            Orientationd(keys=["source", "target"], axcodes="RAS"),
            Spacingd(keys=["source", "target"], pixdim=self.spacing, mode=("bilinear", "bilinear")),
            JointZScoreNormalize(keys=["source", "target"], source_key="source", ignore_zeros=True),
            
            # En validation, on crop aussi (ou on utilise SlidingWindowInferer dans le modèle)
            RandSpatialCropd(keys=["source", "target"], roi_size=self.patch_size, random_center=True, random_size=False),
            
            ToTensord(keys=["source", "target", "norm_mean", "norm_std"]),
        ])

        # --- C. Création des Datasets avec Cache ---
        self.train_ds = CacheDataset(
            data=train_files, 
            transform=self.train_transforms, 
            cache_rate=self.cache_rate, 
            num_workers=self.num_workers
        )
        
        self.val_ds = CacheDataset(
            data=val_files, 
            transform=self.val_transforms, 
            cache_rate=self.cache_rate, 
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            collate_fn=list_data_collate, # Obligatoire pour MONAI (gère les dicts)
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )  

