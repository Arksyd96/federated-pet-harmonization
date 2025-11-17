import random
import numpy as np

import torch
import pytorch_lightning as pl

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            False  # Disable cuDNN benchmarking for consistent results
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = (
            True  # Enable for better performance if determinism is not required
        )

    pl.seed_everything(seed)

    print(f"Seed set to {seed} with deterministic={deterministic}")

def parse_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    elif dtype == "int16":
        return torch.int16
    elif dtype == "int32":
        return torch.int32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError("Invalid dtype")

def kl_gaussians(mean1, logvar1, mean2, logvar2):
    """ Compute the KL divergence between two gaussians."""
    return 0.5 * (logvar2-logvar1 + torch.exp(logvar1 - logvar2) + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2)-1.0)

def dice_score(pred_mask, true_mask):
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    dice = (2 * intersection + 1e-8) / (union + 1e-8)
    return dice

def root_mean_squared_error(pred, true):
    """ Compute the RMSE between two tensors."""
    return torch.sqrt(torch.mean((pred - true) ** 2))

def continuous_nll_loss(prediction, target):
    # Vérifier que les probabilités sont valides (entre 0 et 1)
    assert torch.all((prediction >= 0) & (prediction <= 1)), "Les prédictions doivent être des probabilités (entre 0 et 1)"
    assert torch.all((target >= 0) & (target <= 1)), "Les cibles doivent être des probabilités (entre 0 et 1)"

    # Ajouter un epsilon pour éviter log(0)
    log_prediction = torch.log(prediction + 1e-10)

    # Appliquer la NLL continue : - P_target * log(P_pred)
    loss = - target * log_prediction

    return loss.mean()  # Moyenne sur tous les pixels

def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon=1e-8):
    # Ensure P and Q are valid probability distributions by normalizing them
    p = p / (p.sum(dim=(-2, -1), keepdim=True) + epsilon)
    q = q / (q.sum(dim=(-2, -1), keepdim=True) + epsilon)

    # Apply epsilon for numerical stability
    q = q.clamp(min=epsilon)
    p = p.clamp(min=epsilon)

    # Compute KL divergence
    kl = (p * (p.log() - q.log())).sum(dim=(-2, -1))

    # If batch input, return mean over batch
    return kl.mean() if kl.dim() > 0 else kl