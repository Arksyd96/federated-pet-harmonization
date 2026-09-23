import os
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.unlearning_vae import UnlearningVAE
from pet_harmonization.utils import set_seed

def process_subject(
    model: UnlearningVAE,
    batch: dict,
    device: torch.device,
    curr_idx: int,
    length_loader: int,
    spatial_dims: int = 3,
    alpha: float = 0.0,
    patch_size: tuple = (16, 64, 64),
    patch_overlap: tuple = (8, 16, 16),
    override: bool = False
):
    subject = tio.utils.get_subjects_from_batch(batch)[0]
    subject_name = subject['subject_name']
    subj_out_dir = batch['subject_path'][0]
    
    filename = "3d-vae-sandbox"
    pred_path = os.path.join(subj_out_dir, f"harmonized-pet-{filename}.nii.gz")
    
    print(f"\nTreating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    if os.path.exists(pred_path):
        if not override:
            print(f"⏩ Skip : Le fichier {os.path.basename(pred_path)} existe déjà.")
            return
        else:
            print(f"⚠️ Override : Le fichier {os.path.basename(pred_path)} sera écrasé.")

    print(f"Patch size: {patch_size}, Overlap: {patch_overlap}")
    
    grid_sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=4, num_workers=0) 

    aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode='hann')

    with torch.inference_mode():
        for patch_batch in tqdm(patch_loader, desc=f"Inférence (UnlearningVAE)"):
            locations = patch_batch[tio.LOCATION]
            
            # shape issue de TorchIO : (B, 1, D, H, W)
            patch_tio = patch_batch['source'][tio.DATA].to(device)
            
            if spatial_dims == 2:
                patch_src = patch_tio.squeeze(1) # devient (B, D, H, W)
            else:
                patch_src = patch_tio            # reste (B, 1, D, H, W)

            if patch_src.mean() < 1e-3: 
                patch_pred_tio = torch.zeros_like(patch_tio)
                aggregator.add_batch(patch_pred_tio, locations)
                continue
            
            # Normalisation (utilise la méthode interne du modèle)
            patch_norm = model._normalize(patch_src)
            
            # Appel de la nouvelle fonction harmonize que nous avons ajoutée
            # L'absence de style injecte nativement le style source, annulé par l'alpha
            patch_pred_norm = model.vae.harmonize(
                patch_norm, 
                x_style_ref=None, 
                z_style_fixed=None, 
                alpha_style=alpha
            )
            
            # Dénormalisation
            patch_pred = model._denormalize(patch_pred_norm)
            
            if spatial_dims == 2:
                patch_pred_tio = patch_pred.unsqueeze(1) # revient à (B, 1, D, H, W)
            else:
                patch_pred_tio = patch_pred              # reste (B, 1, D, H, W)
            
            aggregator.add_batch(patch_pred_tio, locations)

    recon_tensor = aggregator.get_output_tensor()
    output_image = tio.ScalarImage(tensor=recon_tensor, affine=subject['source'].affine)
    output_image.save(pred_path)
    
    print(f"✅ Whole-body prediction saved correctly at: {pred_path}")


def main():
    # --- Configuration Inférence Bac à Sable ---
    cfg = {
        'SEED': 101,
        "num_domains": 5,
        "spatial_dims": 3,
        "suv_global_log_max": 6.0,
        
        # Chemin vers le checkpoint que vous avez retrouvé
        'ckpt_path': "runs/sandbox-unlearn-vae/stagestage=2-epoch=epoch=147-rec=val/rec_loss=0.0177-style=val/style_acc=0.966-content=val/content_acc=0.219.ckpt", 
        
        # Alpha contrôle l'intensité de la signature (0.0 = harmonisation totale avec un vecteur de style nul)
        'alpha': 0.5,           
        
        'patch_size': (16, 64, 64),
        'patch_overlap': (10, 16, 16),
        'override': True,

        # Modèle
        "input_shape": (16, 64, 64),
        "in_channels": 1,               # 1 = image seule, 2 = image + FFT
        "out_channels": 1,
        "hidden_channels": [32, 64, 128, 256],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
        "num_residual_blocks": 1,
        "latent_channels": 64,
        "style_channels": 256,
        "use_fft": False,       # Concatène les features FFT en entrée
        "fft_sigma": 7.5,
        "fft_learnable": False, # False = filtre fixe (pas de gradient)
        
        # Configuration du Datamodule pour trouver les images à traiter
        'datamodule': {
            "root_dir": "./data/PET-EARL/",
            "split_config": [[0, 50], [0, 50], [0, 50], [0, 50], [0, 50], [0, 50]],
            "batch_size": 16,
            "patch_size": [16, 64, 64],
            "num_workers": 24,
            "queue_max_length": 4096,
            "samples_per_volume": 64,
        }
    }
    # ------------------------------------------

    set_seed(cfg['SEED'], workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initialisation de l'inférence Sandbox | Device : {device}")

    if not os.path.exists(cfg['ckpt_path']):
        raise FileNotFoundError(f"Checkpoint introuvable : {cfg['ckpt_path']}\nModifiez cfg['ckpt_path'] !")

    # Chargement du modèle (UnlearningVAE s'occupe d'instancier DisentangledVAE grâce à save_hyperparameters)
    model = UnlearningVAE.load_from_checkpoint(cfg['ckpt_path'], strict=True)
    model.to(device)
    model.eval()

    print(f"⚠️ Harmonisation avec un style neutre (alpha={cfg['alpha']}).")

    # Préparation des données
    datamodule = MultiDomainUnlearningDataModule(**cfg['datamodule'])
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.test_dataloader() # ou val_dataloader() selon ce que vous voulez inférer
    
    for idx, batch in enumerate(loader):
        process_subject(
            model=model,
            batch=batch,
            device=device,
            curr_idx=idx + 1,
            length_loader=len(loader),
            spatial_dims=cfg['spatial_dims'],
            alpha=cfg['alpha'],
            patch_size=cfg['patch_size'],
            patch_overlap=cfg['patch_overlap'],
            override=cfg['override']
        )

if __name__ == "__main__":
    main()
