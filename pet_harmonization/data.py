from typing import *

import os
import numpy as np
import torch
import multiprocessing

import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    

def robust_patch_normalization(src: torch.Tensor, tgt: torch.Tensor, percentiles=(0.1, 99.9), clone=True):
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
        src_flat = src_patch.view(-1)
        
        # Calcul des seuils (quantile attend une entrée float)
        # Note: quantile sur GPU est rapide
        p_min = torch.quantile(src_flat, percentiles[0] / 100.0)
        p_max = torch.quantile(src_flat, percentiles[1] / 100.0)
        
        if (p_max - p_min) < 1e-6:
            continue

        src_patch = src_patch.clip(min=p_min, max=p_max)
        tgt_patch = tgt_patch.clip(min=p_min, max=p_max)
            
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
        for image in subject.get_images(intensity_only=True):
            image.data = image.data.float()
        return subject

# Définition des types pour plus de clarté
SplitConfigType = Union[
    float, 
    Tuple[int, int], 
    List[Union[float, Tuple[int, int]]]
]

# class Float32Lambda:
#     def __call__(self, sample):
#         return sample
    
class CropToVOI(tio.Transform):
    def __init__(self, min_shape, **kwargs):
        super().__init__(**kwargs)
        self.min_shape = min_shape

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if 'voi' not in subject:
            return subject
        
        # Récupération des données du masque (Shape: C, W, H, D)
        mask_data = subject[ 'voi' ].data
        nonzero = torch.nonzero(mask_data > 0)
        
        if nonzero.numel() == 0:
            return subject  # Masque vide, on renvoie le volume entier
        
        # Les dimensions spatiales sont aux index 1, 2, 3 du tenseur nonzero
        min_w, max_w = nonzero[:, 1].min().item(), nonzero[ :, 1 ].max().item()
        min_h, max_h = nonzero[:, 2].min().item(), nonzero[ :, 2 ].max().item()
        min_d, max_d = nonzero[:, 3].min().item(), nonzero[ :, 3 ].max().item()
        
        # Récupération des dimensions max de l'image
        w, h, d = subject[ 'source' ].spatial_shape
        target_w, target_h, target_d = self.min_shape

        # Fonction interne pour étendre la Bounding Box de manière sécurisée
        def expand_dim(c_min, c_max, max_limit, target_size):
            size = c_max - c_min + 1
            if size < target_size:
                diff = target_size - size
                pad_before = diff // 2
                pad_after = diff - pad_before
                
                c_min -= pad_before
                c_max += pad_after
                
                # Si on déborde à gauche/en bas (ex: lésion près de la peau)
                if c_min < 0:
                    c_max += abs(c_min)
                    c_min = 0
                # Si on déborde à droite/en haut (ex: sommet du crâne)
                if c_max >= max_limit:
                    excess = c_max - max_limit + 1
                    c_min -= excess
                    c_max = max_limit - 1
                    
                # Sécurité ultime si l'image entière est plus petite que target_size
                c_min = max(0, c_min)
                c_max = min(max_limit - 1, c_max)
                
            return c_min, c_max

        # 🎯 Élargissement dynamique pour atteindre la taille minimum
        min_w, max_w = expand_dim(min_w, max_w, w, target_w)
        min_h, max_h = expand_dim(min_h, max_h, h, target_h)
        min_d, max_d = expand_dim(min_d, max_d, d, target_d)
        
        # Calcul du nombre de voxels à supprimer sur chaque face (6-tuple requis par TorchIO)
        crop_left = min_w
        crop_right = w - (max_w + 1)
        crop_top = min_h
        crop_bottom = h - (max_h + 1)
        crop_front = min_d
        crop_back = d - (max_d + 1)
        
        # Application du recadrage physique (met à jour la matrice affine !)
        cropper = tio.Crop((crop_left, crop_right, crop_top, crop_bottom, crop_front, crop_back))
        cropped_subject = cropper(subject)
        
        return cropped_subject

# # # --- LE LIGHTNING DATA MODULE ---
# class PETTranslationDataModule(LightningDataModule):
#     def __init__(
#         self, 
#         root_dir: str, 
#         batch_size: int = 4, 
#         train_ratio: float = 0.8,
#         patch_size: tuple = (64, 64, 64),
#         num_workers: int = 8,             # Augmentez si vous avez bcp de coeurs
#         queue_max_length: int = 600,      
#         samples_per_volume: int = 4, # On tire 4 patches par patient
#     ):
#         super().__init__()
#         self.root_dir = root_dir
#         self.batch_size = batch_size
#         self.train_ratio = train_ratio
#         self.patch_size = patch_size
#         self.num_workers = num_workers
#         self.queue_max_length = queue_max_length
#         self.samples_per_volume = samples_per_volume

#     def get_pt_earl_files(self, files: List[str]) -> List[Dict[str, str]]:
#         #  --- Adaptez ces filtres à vos noms de fichiers exacts ---
#         files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
#         pt_files = [f for f in files if f.startswith('PET') and 'MIP' not in f]
#         earl_files = [f for f in files if f.startswith('EARL') and 'MIP' not in f]
#         sampling_files = [f for f in files if f.startswith('body') and (f.endswith('.nii') or f.endswith('.nii.gz'))]
#         return pt_files[0], earl_files[0], sampling_files[0]

#     def setup(self, stage=None):
#         # --- Listing des fichiers ---
#         all_subjects = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
#         tio_subjects = list()
#         for subj_name in all_subjects:
#             subj_path = os.path.join(self.root_dir, subj_name)
#             files = os.listdir(subj_path)
#             pt_file, earl_file, sampling_file = self.get_pt_earl_files(files)
            
#             if pt_file and earl_file:
#                 subject = tio.Subject(
#                     source=tio.Image(os.path.join(subj_path, pt_file), type=tio.INTENSITY),
#                     target=tio.Image(os.path.join(subj_path, earl_file), type=tio.INTENSITY),
#                     sampling_map=tio.Image(os.path.join(subj_path, sampling_file), type=tio.LABEL), # Utilisé pour le sampling
#                     subject_id=subj_name
#                 )

#                 tio_subjects.append(subject)
        
#         # --- Split train/val ---
#         np.random.shuffle(tio_subjects) # Mélange avant split
#         split_idx = self.train_ratio
#         if isinstance(self.train_ratio, float) and 0.0 <= self.train_ratio <= 1.0:
#             split_idx = int(len(tio_subjects) * self.train_ratio)
#         self.train_subjects, self.val_subjects = tio_subjects[:split_idx], tio_subjects[split_idx:]

#         # just for the record (in case needed)
#         self.train_subj_paths, self.val_subj_paths = all_subjects[:split_idx], all_subjects[split_idx:]
        
#         print(f"[TorchIO] {len(self.train_subjects)} Train, {len(self.val_subjects)} Val.")

#         # --- Pipelines de Transformation ---        
#         self.transform = tio.Compose([
#             # Float32Lambda(),
#             tio.ToCanonical(),
#             tio.RandomFlip(axes=(0, 1, 2), p=0.5)
#         ])

#     def train_dataloader(self):
#         if self.train_subjects.__len__() > 0:
#             train_dataset = tio.SubjectsDataset(self.train_subjects, transform=self.transform)
#             # sampler = tio.data.UniformSampler(self.patch_size)
#             sampler = tio.LabelSampler(
#                 patch_size=self.patch_size,
#                 label_name='sampling_map',
#                 label_probabilities={
#                     0: 0.05,  # 5% de chance de prendre un patch centré sur l'air (pour la robustesse)
#                     1: 0.95   # 95% de chance de prendre un patch centré sur le patient
#                 }
#             )

#             patches_queue = tio.Queue(
#                 subjects_dataset=train_dataset,
#                 max_length=self.queue_max_length,
#                 samples_per_volume=self.samples_per_volume,
#                 sampler=sampler,
#                 num_workers=self.num_workers,
#                 shuffle_subjects=True,
#                 shuffle_patches=True
#             )

#             return tio.SubjectsLoader(
#                 patches_queue,
#                 batch_size=self.batch_size,
#                 num_workers=0,
#                 pin_memory=True
#             )
        
#         return None

#     def val_dataloader(self):
#         if self.val_subjects.__len__() > 0:
#             val_dataset = tio.SubjectsDataset(self.val_subjects, transform=self.transform)
#             sampler = tio.LabelSampler(
#                 patch_size=self.patch_size,
#                 label_name='sampling_map',
#                 label_probabilities={
#                     0: 0.05,  # 5% de chance de prendre un patch centré sur l'air (pour la robustesse)
#                     1: 0.95   # 95% de chance de prendre un patch centré sur le patient
#                 }
#             )

#             patches_queue = tio.Queue(
#                 subjects_dataset=val_dataset,
#                 max_length=300,
#                 samples_per_volume=32,
#                 sampler=sampler,
#                 num_workers=4,
#                 shuffle_subjects=False,
#                 shuffle_patches=False
#             )
            
#             return tio.SubjectsLoader(
#                 patches_queue,
#                 batch_size=self.batch_size,
#                 num_workers=0,
#                 pin_memory=True
#             )
        
#         return None
    
#     def test_dataloader(self):
#         if self.val_subjects.__len__() > 0:
#             val_dataset = tio.SubjectsDataset(self.val_subjects, transform=tio.Compose([Float32Lambda(), tio.ToCanonical()]))
#             return tio.SubjectsLoader(
#                 val_dataset,
#                 batch_size=1,
#                 num_workers=multiprocessing.cpu_count() - 1,
#                 pin_memory=True,
#                 shuffle=False
#             )


class BasePETDataModule(LightningDataModule):
    def __init__(
        self, 
        root_dir: str, 
        batch_size: int = 4, 
        train_ratio: float = 0.8,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        num_workers: int = 8,             
        queue_max_length: int = 600,      
        samples_per_volume: int = 4, 
        voi_filename: Optional[str] = None, # Nom du fichier de la VOI (ex: 'voi.nii.gz') ou None si pas de VOI
    ):
        super().__init__()
        # Sauvegarde des hyperparamètres pour PyTorch Lightning
        self.save_hyperparameters()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        
        self.train_subjects: List[tio.Subject] = []
        self.val_subjects: List[tio.Subject] = []
        self.transform: Optional[tio.Compose] = None
        self.voi_filename = voi_filename

    def get_pt_earl_files(self, files: List[str]) -> Dict[str, Optional[str]]:
        raise NotImplementedError("La méthode get_pt_earl_files doit être implémentée par les sous-classes.")

    def setup(self, stage: Optional[str] = None):
        # --- Listing des fichiers ---
        all_subjects = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        tio_subjects = list()
        for subj_name in all_subjects:
            subj_path = os.path.join(self.root_dir, subj_name)
            files = os.listdir(subj_path)
            self.file_paths = self.get_pt_earl_files(files)
            
            # Vérification minimale (doit au moins avoir le PET source)
            if not self.file_paths.get('source'):
                print(f"Avertissement: Sujet {subj_name} ignoré (PET source manquant).")
                continue
            
            # Création de l'objet tio.Subject
            subject_dict = {
                'source': tio.Image(os.path.join(subj_path, self.file_paths['source']), type=tio.INTENSITY),
                'sampling_map': tio.Image(os.path.join(subj_path, self.file_paths['sampling_map']), type=tio.LABEL),
                'subject_id': subj_name
            }
            
            if self.voi_filename:
                matched_voi = next(f for f in files if f == self.voi_filename or f == f"{self.voi_filename}.nii.gz")
                if matched_voi:
                    subject_dict["voi"] = tio.Image(os.path.join(subj_path, matched_voi), type=tio.LABEL)
                else:
                    print(f"[{subj_name}] Masque VOI '{self.voi_filename}' introuvable.")
            
            # Ajout des cibles (targets)
            for key, file_name in self.file_paths.items():
                if key.startswith('target') and file_name:
                    subject_dict[key] = tio.Image(os.path.join(subj_path, file_name), type=tio.INTENSITY)
            
            # Création du sujet TorchIO
            subject = tio.Subject(**subject_dict)
            tio_subjects.append(subject)
        
        # --- Split train/val ---
        np.random.shuffle(tio_subjects) # Mélange avant split
        
        split_idx = self.train_ratio
        if isinstance(self.train_ratio, float) and 0.0 <= self.train_ratio <= 1.0:
            split_idx = int(len(tio_subjects) * self.train_ratio)
            
        self.train_subjects, self.val_subjects = tio_subjects[:split_idx], tio_subjects[split_idx:]
        
        print(f"[TorchIO] {len(self.train_subjects)} Train, {len(self.val_subjects)} Val.")

        # --- Pipelines de Transformation ---        
        self.transform = tio.Compose([
            Float32Lambda(), # Assure la conversion en float32
            tio.ToCanonical(),
            tio.RandomFlip(axes=(0, 1, 2), p=0.5)
        ])

    def _create_dataloader(
        self, subjects: List[tio.Subject], shuffle_subjects: bool, shuffle_patches: bool, is_validation: bool = False):
        """Méthode utilitaire pour créer les DataLoaders (train et val)"""
        if not subjects:
            return None
        
        dataset = tio.SubjectsDataset(subjects, transform=self.transform)
        
        # Le sampler reste le même pour les deux versions
        sampler = tio.LabelSampler(
            patch_size=self.patch_size,
            label_name='sampling_map',
            label_probabilities={
                0: 0.05,  # 5% de chance de prendre un patch centré sur l'air
                1: 0.95   # 95% de chance de prendre un patch centré sur le patient
            }
        )
        
        # Paramètres spécifiques à la validation
        max_length = 1000 if is_validation else self.queue_max_length
        samples_per_volume = 32 if is_validation else self.samples_per_volume
        num_workers = 4 if is_validation else self.num_workers
        
        patches_queue = tio.Queue(
            subjects_dataset=dataset,
            max_length=max_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

        # TorchIO recommande num_workers=0 pour le SubjectsLoader si la Queue est utilisée
        return tio.SubjectsLoader(
            patches_queue,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_subjects, shuffle_subjects=True, shuffle_patches=True, is_validation=False)

    def val_dataloader(self):
        return self._create_dataloader(self.val_subjects, shuffle_subjects=False, shuffle_patches=False, is_validation=True)
    
    def test_dataloader(self):
        if self.val_subjects:
            # Pour le test, on charge le volume entier (batch_size=1)
            test_dataset = tio.SubjectsDataset(self.val_subjects, transform=tio.Compose([Float32Lambda(), tio.ToCanonical()]))
            return tio.SubjectsLoader(
                test_dataset,
                batch_size=1,
                num_workers=multiprocessing.cpu_count() - 1,
                pin_memory=True,
                shuffle=False
            )
        return None
    
    def voi_dataloader(self, min_voi_crop_shape):
        if not self.voi_filename:
            raise ValueError("Impossible d'appeler voi_dataloader sans avoir défini 'voi_filename' à l'initialisation.")
            
        if self.val_subjects:
            # Construction de la pipeline de transformation dédiée à la VOI (Pas de data augmentation !)
            voi_transform = tio.Compose([
                Float32Lambda(), 
                tio.ToCanonical(),
                CropToVOI(min_shape=min_voi_crop_shape) 
            ])
            
            voi_dataset = tio.SubjectsDataset(self.val_subjects, transform=voi_transform)
            
            return tio.SubjectsLoader(
                voi_dataset, 
                batch_size=1, 
                num_workers=multiprocessing.cpu_count() // 2, 
                pin_memory=True, 
                shuffle=False
            )
        return None  
    
    
class SingleTargetPETDataModule(BasePETDataModule):
    def get_pt_earl_files(self, files: List[str]) -> Dict[str, Optional[str]]:
        """
        Filtre pour 1 PET, 1 EARL (target_1), 1 Mask (sampling_map).
        """
        # --- Adaptez ces filtres à vos noms de fichiers exacts ---
        nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        pt_files = [f for f in nii_files if f.startswith('PET') and 'MIP' not in f]
        earl_files = [f for f in nii_files if f.startswith('EARL') and 'MIP' not in f]
        sampling_files = [f for f in nii_files if f.startswith('body') and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        
        # La cible unique est nommée 'target_1' pour la cohérence avec la classe mère
        return {
            'source': pt_files[0] if pt_files else None,
            'target': earl_files[0] if earl_files else None,
            'sampling_map': sampling_files[0] if sampling_files else None,
        }

class MultiTargetPETDataModule(BasePETDataModule):
    def get_pt_earl_files(self, files: List[str]) -> Dict[str, Optional[str]]:
        """
        Filtre pour 1 PET, 2 EARL (target_1=EARL1, target_2=EARL2), 1 Mask.
        """
        # --- Adaptez ces filtres à vos noms de fichiers exacts ---
        nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        pt_files = [f for f in nii_files if f.startswith('PET') and 'MIP' not in f]
        sampling_files = [f for f in nii_files if f.startswith('body') and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        
        # Filtres spécifiques pour EARL1 et EARL2
        earl1_files = [f for f in nii_files if 'EARL1' in f and 'MIP' not in f]
        earl2_files = [f for f in nii_files if 'EARL2' in f and 'MIP' not in f]
        
        # Si un fichier est manquant, on utilise None comme demandé
        return {
            'source': pt_files[0] if pt_files else None,
            'target_1': earl1_files[0] if earl1_files else None, # EARL 1
            'target_2': earl2_files[0] if earl2_files else None, # EARL 2
            'sampling_map': sampling_files[0] if sampling_files else None,
        }


class MultiDomainUnlearningDataModule(LightningDataModule):
    def __init__(
        self, 
        root_dir: str, 
        split_config: SplitConfigType = 0.8, # NOUVEAU PARAMÈTRE
        batch_size: int = 4, 
        patch_size: tuple = (64, 64, 64),
        num_workers: int = 8,
        queue_max_length: int = 600,      
        samples_per_volume: int = 4,
        voi_filename: Optional[str] = None, # pour le recadrage à la VOI si besoin en inférence
        seed: int = 42 # Ajout d'une seed pour garantir la reproductibilité du split
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_config = split_config
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        self.voi_filename = voi_filename
        self.seed = seed
        
        self.domain_to_id = {}
        self.train_subjects = []
        self.val_subjects = []

    def get_pet_body_files(self, files: List[str]):
        """Filtre les fichiers PET et body mask selon la nomenclature os."""
        pet_files = [f for f in files if f.startswith('PET') and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        body_files = [f for f in files if f.startswith('body') and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        
        pet_file = pet_files[0] if len(pet_files) > 0 else None
        body_file = body_files[0] if len(body_files) > 0 else None
        
        return pet_file, body_file

    def _create_tio_subjects(self, subj_names: List[str], domain_path: str, domain_name: str, domain_id: int) -> List[tio.Subject]:
        """Fonction utilitaire pour charger une liste de sujets en objets TorchIO."""
        tio_subjects = []
        for subj_name in subj_names:
            subj_path = os.path.join(domain_path, subj_name)
            files = os.listdir(subj_path)
            
            pet_file, body_file = self.get_pet_body_files(files)
            
            if pet_file and body_file:
                # Configuration de base du sujet
                subject_kwargs = {
                    "source": tio.Image(os.path.join(subj_path, pet_file), type=tio.INTENSITY),
                    "sampling_map": tio.Image(os.path.join(subj_path, body_file), type=tio.LABEL),
                    "domain_id": torch.tensor(domain_id).long(),
                    "subject_name": subj_name,
                    "domain_name": domain_name
                }
                
                if self.voi_filename:
                    # Recherche exacte ou partielle du fichier spécifié
                    matched_voi = next(f for f in files if f == self.voi_filename or f == f"{self.voi_filename}.nii.gz")
                    if matched_voi:
                        subject_kwargs["voi"] = tio.Image(os.path.join(subj_path, matched_voi), type=tio.LABEL)
                    else:
                        print(f"[{subj_name}] Masque VOI '{self.voi_filename}' introuvable.")
                
                subject = tio.Subject(**subject_kwargs)
                tio_subjects.append(subject)
        return tio_subjects

    def setup(self, stage=None):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Le dossier racine {self.root_dir} n'existe pas.")

        # 1. Lister et trier les domaines
        domain_names = sorted([d for d in sorted(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.domain_to_id = {name: i for i, name in enumerate(domain_names)}
        print(f"Mapping Domaines: {self.domain_to_id}")

        self.train_subjects = []
        self.val_subjects = []
        
        rng = np.random.RandomState(self.seed) # Générateur aléatoire isolé

        for i, domain_name in enumerate(domain_names):
            domain_id = self.domain_to_id[domain_name]
            domain_path = os.path.join(self.root_dir, domain_name)
            
            # Lister et mélanger les sujets du domaine
            subjects_in_domain = sorted([s for s in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, s))])
            rng.shuffle(subjects_in_domain)
            total_subj = len(subjects_in_domain)

            # 2. Déterminer la configuration de split pour CE domaine
            if isinstance(self.split_config, list):
                if len(self.split_config) != len(domain_names):
                    raise ValueError(f"La liste split_config ({len(self.split_config)}) doit correspondre au nombre de domaines ({len(domain_names)}).")
                current_config = self.split_config[i]
            else:
                current_config = self.split_config

            # 3. Calculer les index de coupure (train_count et test_count)
            if isinstance(current_config, float):
                # Cas 1 : Pourcentage
                train_count = int(total_subj * current_config)
                test_count = total_subj - train_count
            elif isinstance(current_config, (tuple, list)) and len(current_config) == 2:
                # Cas 2 : Nombre exact (train, test)
                train_count = current_config[0]
                test_count = current_config[1]
                
                # Sécurité si on demande plus de données qu'il n'y en a
                if train_count + test_count > total_subj:
                    print(f"⚠️ Avertissement : Le domaine {domain_name} n'a que {total_subj} patients. " 
                          f"Impossible d'extraire {train_count} train et {test_count} test. Plafonnement appliqué.")
                    train_count = min(train_count, total_subj)
                    test_count = min(test_count, total_subj - train_count)
            else:
                raise TypeError(f"Format de split_config invalide pour le domaine {domain_name}: {current_config}")

            # 4. Découpage
            train_names = subjects_in_domain[:train_count]
            val_names = subjects_in_domain[train_count:train_count + test_count]

            print(f"Domaine {domain_name:12} -> Train: {len(train_names):3d}, Test: {len(val_names):3d} (Total: {total_subj})")

            # 5. Création des objets TorchIO
            self.train_subjects.extend(self._create_tio_subjects(train_names, domain_path, domain_name, domain_id))
            self.val_subjects.extend(self._create_tio_subjects(val_names, domain_path, domain_name, domain_id))

        # Mélange global final des listes de sujets
        rng.shuffle(self.train_subjects)
        rng.shuffle(self.val_subjects)
            
        print(f"\n[TorchIO] Total Global : {len(self.train_subjects)} Train, {len(self.val_subjects)} Val (Test set).")
        
        self.transform = tio.Compose([
            Float32Lambda(),
            tio.ToCanonical(),
            tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        ])

    def _create_dataloader(self, subjects: List[tio.Subject], shuffle_subjects: bool, shuffle_patches: bool, is_validation: bool = False):
        if not subjects:
            return None
        
        dataset = tio.SubjectsDataset(subjects, transform=self.transform)
        
        sampler = tio.LabelSampler(
            patch_size=self.patch_size,
            label_name='sampling_map',
            label_probabilities={
                0: 0.05, 
                1: 0.95
            }
        )
        
        max_length = 1000 if is_validation else self.queue_max_length
        samples_per_volume = 32 if is_validation else self.samples_per_volume
        num_workers = 4 if is_validation else self.num_workers
        
        patches_queue = tio.Queue(
            subjects_dataset=dataset,
            max_length=max_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

        return tio.SubjectsLoader(
            patches_queue, 
            batch_size=self.batch_size, 
            num_workers=0, 
            pin_memory=True
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_subjects, shuffle_subjects=True, shuffle_patches=True, is_validation=False)

    def val_dataloader(self):
        return self._create_dataloader(self.val_subjects, shuffle_subjects=False, shuffle_patches=False, is_validation=True)
    
    def test_dataloader(self):
        if self.val_subjects:
            test_dataset = tio.SubjectsDataset(self.val_subjects, transform=tio.Compose([Float32Lambda(), tio.ToCanonical()]))
            return tio.SubjectsLoader(
                test_dataset, 
                batch_size=1, 
                num_workers=multiprocessing.cpu_count() // 2, 
                pin_memory=True, 
                shuffle=False
            )
        return None 
    
    def voi_dataloader(self):
        if not self.voi_filename:
            raise ValueError("Impossible d'appeler voi_dataloader sans avoir défini 'voi_filename' à l'initialisation.")
            
        if self.val_subjects:
            # Construction de la pipeline de transformation dédiée à la VOI (Pas de data augmentation !)
            voi_transform = tio.Compose([
                Float32Lambda(), 
                tio.ToCanonical(),
                CropToVOI() # Application de notre cropper intelligent
            ])
            
            voi_dataset = tio.SubjectsDataset(self.val_subjects, transform=voi_transform)
            
            return tio.SubjectsLoader(
                voi_dataset, 
                batch_size=1, 
                num_workers=multiprocessing.cpu_count() // 2, 
                pin_memory=True, 
                shuffle=False
            )
        return None     
    

class InMemoryVolumeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subjects: List[dict],          
        patch_size: Tuple[int, int, int],
        augment: bool = False,
        label_prob_body: float = 0.95,  # probabilité de cropper dans le corps
    ):
        self._subjects       = subjects
        self.patch_size      = patch_size
        self.augment         = augment
        self.label_prob_body = label_prob_body

        self._flip = tio.RandomFlip(axes=(0, 1, 2), p=0.5) if augment else None

    def __len__(self) -> int:
        return len(self._subjects)

    def _random_label_crop(self, source: torch.Tensor, body: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, D, H, W = source.shape
        pd, ph, pw = self.patch_size

        # Candidats de départ valides (le patch doit tenir dans le volume)
        d_max = max(D - pd, 0)
        h_max = max(H - ph, 0)
        w_max = max(W - pw, 0)

        # Avec probabilité label_prob_body, on centre le crop dans le corps
        if torch.rand(1).item() < self.label_prob_body:
            coords = torch.nonzero(body[0], as_tuple=False)  # (N, 3)
            if len(coords) > 0:
                idx = torch.randint(len(coords), (1,)).item()
                cd, ch, cw = coords[idx].tolist()
                # Centrer le patch sur ce voxel, clampé dans les bornes
                d0 = int(np.clip(cd - pd // 2, 0, d_max))
                h0 = int(np.clip(ch - ph // 2, 0, h_max))
                w0 = int(np.clip(cw - pw // 2, 0, w_max))
            else:
                d0 = torch.randint(0, d_max + 1, (1,)).item()
                h0 = torch.randint(0, h_max + 1, (1,)).item()
                w0 = torch.randint(0, w_max + 1, (1,)).item()
        else:
            d0 = torch.randint(0, d_max + 1, (1,)).item()
            h0 = torch.randint(0, h_max + 1, (1,)).item()
            w0 = torch.randint(0, w_max + 1, (1,)).item()

        src_patch  = source[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
        body_patch = body[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
        return src_patch, body_patch

    def __getitem__(self, idx: int) -> dict:
        subj = self._subjects[idx]

        source = subj["source"].clone()       # (1, D, H, W)
        body   = subj["sampling_map"].clone() # (1, D, H, W)

        # Crop guidé par le body label
        source, body = self._random_label_crop(source, body)

        # Augmentation
        if self.augment and self._flip is not None:
            tmp = tio.Subject(
                source=tio.Image(tensor=source, type=tio.INTENSITY),
                sampling_map=tio.Image(tensor=body, type=tio.LABEL),
            )
            tmp    = self._flip(tmp)
            source = tmp["source"][tio.DATA]
            body   = tmp["sampling_map"][tio.DATA]

        return {
            "source":       {tio.DATA: source},
            "sampling_map": {tio.DATA: body},
            "domain_id":    subj["domain_id"],
            "subject_name": subj["subject_name"],
            "domain_name":  subj["domain_name"],
        }


class InMemoryUnlearningDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        split_config: SplitConfigType = 0.8,
        batch_size: int = 4,
        patch_size: tuple = (5, 64, 64),
        num_workers: int = 8,
        queue_max_length: int = 600,    # ignoré, conservé pour compatibilité YAML
        samples_per_volume: int = 4,    # ignoré, conservé pour compatibilité YAML
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir   = root_dir
        self.split_config = split_config
        self.batch_size = batch_size
        self.patch_size = tuple(patch_size)
        self.num_workers = num_workers
        self.seed = seed

        self.domain_to_id  = {}
        self._train_data:  List[dict] = []
        self._val_data:    List[dict] = []
        self._val_subjects: List[tio.Subject] = []   # pour test_dataloader

    def get_pet_body_files(self, files):
        pet_files  = [f for f in files if f.startswith('PET')  and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        body_files = [f for f in files if f.startswith('body') and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        return (pet_files[0] if pet_files else None,
                body_files[0] if body_files else None)

    def _load_subjects_into_ram(
        self,
        subj_names: List[str],
        domain_path: str,
        domain_name: str,
        domain_id: int,
        desc: str,
    ) -> List[dict]:
        """
        Charge chaque volume en RAM via tio (ToCanonical garantit l'orientation),
        et retourne une liste de dicts de tenseurs prêts à l'emploi.
        """
        transform = tio.Compose([Float32Lambda(), tio.ToCanonical()])
        result    = []

        for name in tqdm(subj_names, desc=desc, unit="vol"):
            path = os.path.join(domain_path, name)
            pet_file, body_file = self.get_pet_body_files(os.listdir(path))
            if not (pet_file and body_file):
                continue

            subject = tio.Subject(
                source=tio.Image(os.path.join(path, pet_file),  type=tio.INTENSITY),
                sampling_map=tio.Image(os.path.join(path, body_file), type=tio.LABEL),
                domain_id=torch.tensor(domain_id).long(),
                subject_name=name,
                domain_name=domain_name,
            )
            loaded = transform(subject)

            result.append({
                "source":       loaded["source"][tio.DATA].float(),       # (1, D, H, W)
                "sampling_map": loaded["sampling_map"][tio.DATA].float(), # (1, D, H, W)
                "domain_id":    loaded["domain_id"],
                "subject_name": loaded["subject_name"],
                "domain_name":  loaded["domain_name"],
            })

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # setup
    # ──────────────────────────────────────────────────────────────────────────

    def setup(self, stage=None):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dossier racine introuvable : {self.root_dir}")

        domain_names = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        self.domain_to_id = {name: i for i, name in enumerate(domain_names)}
        print(f"Mapping domaines : {self.domain_to_id}\n")

        rng = np.random.RandomState(self.seed)

        for i, domain_name in enumerate(domain_names):
            domain_id   = self.domain_to_id[domain_name]
            domain_path = os.path.join(self.root_dir, domain_name)

            subjects_in_domain = sorted([
                s for s in os.listdir(domain_path)
                if os.path.isdir(os.path.join(domain_path, s))
            ])
            rng.shuffle(subjects_in_domain)
            total_subj = len(subjects_in_domain)

            current_config = (
                self.split_config[i]
                if isinstance(self.split_config, list)
                else self.split_config
            )

            if isinstance(current_config, float):
                train_count = int(total_subj * current_config)
                test_count  = total_subj - train_count
            elif isinstance(current_config, (tuple, list)) and len(current_config) == 2:
                train_count, test_count = current_config
                if train_count + test_count > total_subj:
                    print(f"⚠️  {domain_name} : seulement {total_subj} sujets, plafonnement.")
                    train_count = min(train_count, total_subj)
                    test_count  = min(test_count, total_subj - train_count)
            else:
                raise TypeError(f"split_config invalide pour {domain_name} : {current_config}")

            train_names = subjects_in_domain[:train_count]
            val_names   = subjects_in_domain[train_count:train_count + test_count]
            print(f"Domaine {domain_name:12} → Train: {len(train_names):3d}, Val: {len(val_names):3d} (Total: {total_subj})")

            self._train_data.extend(self._load_subjects_into_ram(
                train_names, domain_path, domain_name, domain_id,
                desc=f"  [RAM] {domain_name} train",
            ))
            val_data = self._load_subjects_into_ram(
                val_names, domain_path, domain_name, domain_id,
                desc=f"  [RAM] {domain_name} val  ",
            )
            self._val_data.extend(val_data)

        rng.shuffle(self._train_data)
        rng.shuffle(self._val_data)

        # Reconstruire des tio.Subject pour test_dataloader (volumes entiers)
        self._val_subjects = [
            tio.Subject(
                source=tio.Image(tensor=d["source"], type=tio.INTENSITY),
                sampling_map=tio.Image(tensor=d["sampling_map"], type=tio.LABEL),
                domain_id=d["domain_id"],
                subject_name=d["subject_name"],
                domain_name=d["domain_name"],
            )
            for d in self._val_data
        ]

        print(
            f"\n[RAM] Chargement terminé — "
            f"{len(self._train_data)} volumes train, "
            f"{len(self._val_data)} volumes val en mémoire."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────────────────────

    def train_dataloader(self):
        dataset = InMemoryVolumeDataset(
            self._train_data,
            patch_size=self.patch_size,
            augment=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        dataset = InMemoryVolumeDataset(
            self._val_data,
            patch_size=self.patch_size,
            augment=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, self.num_workers),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Volumes entiers en mémoire pour l'inférence patch-wise."""
        if not self._val_subjects:
            return None
        test_dataset = tio.SubjectsDataset(self._val_subjects, transform=None)
        return tio.SubjectsLoader(
            test_dataset,
            batch_size=1,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            shuffle=False,
        )