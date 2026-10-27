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
    harmonization_mode: str = "gaussian",  # "gaussian", "zero", "original", ou mode style extrait
    alpha: float = 0.5,
    patch_size: tuple = (16, 64, 64),
    patch_overlap: tuple = (8, 16, 16),
    override: bool = False,
    z_style_fixed: torch.Tensor = None
):
    subject = tio.utils.get_subjects_from_batch(batch)[0]
    subject_name = subject['subject_name']
    subj_out_dir = batch['subject_path'][0]
    
    filename = f"vae-3d-sandbox-{harmonization_mode}"
    pred_path = os.path.join(subj_out_dir, f"harmonized-pet-{filename}.nii.gz")
    
    print(f"\nTreating subject: {subject_name} ({curr_idx}/{length_loader}) [Mode: {harmonization_mode}]")
    
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

    style_channels = getattr(model.hparams, 'style_channels', 256)

    # Préparation du vecteur de style :
    # Si un style fixe (ex: chargé depuis styles.pt ou global) est fourni, il a priorité absolue
    if z_style_fixed is not None:
        current_z_style = z_style_fixed
        current_alpha = 1.0
    elif harmonization_mode == "gaussian":
        current_z_style = torch.randn(1, style_channels, device=device)
        current_alpha = 1.0
    elif harmonization_mode == "zero":
        current_z_style = torch.zeros(1, style_channels, device=device)
        current_alpha = 1.0
    elif harmonization_mode == "original":
        current_z_style = None
        current_alpha = alpha
    else:
        raise ValueError(f"Mode d'harmonisation non reconnu : {harmonization_mode}")

    with torch.inference_mode():
        for patch_batch in tqdm(patch_loader, desc=f"Inférence ({harmonization_mode})"):
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
            
            # Harmonisation selon le style préparé
            patch_pred_norm = model.vae.harmonize(
                patch_norm, 
                x_style_ref=None, 
                z_style_fixed=current_z_style, 
                alpha_style=current_alpha
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
        "num_domains": 1,
        "spatial_dims": 3,
        "suv_global_log_max": 6.0,

        # ── Style externe pré-calculé (depuis styles.pt) ──
        # Mettre 'style_key' pour activer (ex: "global_median", "center_0", "center_1")
        # Si None : utilise les modes classiques ci-dessous ('harmonization_mode')
        "styles_dict_path": "./data/style_embeddings/styles.pt",
        "style_key": None,  # Options : "global_median" | "center_0" | "center_1" ... ou None
        
        # ── Mode d'harmonisation classique (actif si style_key est None) ──
        # Options :
        #   - "gaussian" : injecte un bruit N(0,1) fixe (harmonisation dans le Typical Set)
        #   - "zero"     : injecte un vecteur de zéros (harmonisation centre absolu)
        #   - "original" : conserve le style patient atténué par 'alpha'
        "harmonization_mode": "gaussian",
        "global_gaussian_style": True,  # Si True en mode "gaussian", un seul vecteur aléatoire est partagé pour toute la cohorte
        
        # Checkpoints disponibles documentés :
        # 1. Modèle sans dropout (à utiliser avec mode="original" et alpha=0.5)
        # 'ckpt_path': "runs/sandbox-unlearn-vae/working_ckpt/stagestage=2-epoch=epoch=147-rec=val/rec_loss=0.0177-style=val/style_acc=0.966-content=val/content_acc=0.219.ckpt", 
        
        # 2. Modèle entraîné avec style Gaussien (à utiliser avec mode="gaussian") [ACTIF PAR DÉFAUT]
        # 'ckpt_path': "runs/sandbox-unlearn-vae/2026_10_06_160101_random_style/checkpoints/2-epoch=149-rec=0.0190-style=0.966-content=0.186.ckpt",
        'ckpt_path': "runs/sandbox-unlearn-vae/2026_10_23_162550/checkpoints/2-epoch=127-rec=0.0201-style=0.960-content=0.172.ckpt",

        # 3. Modèle entraîné avec style Zéro (à utiliser avec mode="zero")
        # 'ckpt_path': "runs/sandbox-unlearn-vae/2026_10_07_182104/checkpoints/2-epoch=149-rec=0.0234-style=0.961-content=0.224.ckpt",
        
        # Alpha (utilisé uniquement si harmonization_mode == "original")
        'alpha': 1.0,           
        
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
            # "split_config": [[0, 4]],
            "batch_size": 16,
            "patch_size": [16, 64, 64],
            "num_workers": 4,
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

    style_channels = getattr(model.hparams, 'style_channels', 256)
    active_mode = cfg['harmonization_mode']
    global_z_style = None

    # 1. Priorité au style extrait si style_key est renseigné
    if cfg.get('style_key') is not None:
        dict_path = cfg.get('styles_dict_path', "./data/style_embeddings/styles.pt")
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Fichier de styles introuvable : {dict_path}\nLancez d'abord extract_center_styles.py !")

        styles_dict = torch.load(dict_path, map_location=device)
        key = cfg['style_key']
        if key not in styles_dict:
            raise KeyError(f"Clé '{key}' introuvable dans {dict_path}. Clés disponibles : {list(styles_dict.keys())}")

        global_z_style = styles_dict[key].to(device)
        if global_z_style.dim() == 1:
            global_z_style = global_z_style.unsqueeze(0)

        active_mode = f"style_{key}"
        print(f"📦 Style extrait activé depuis {dict_path} [clé='{key}'] (Norme = {global_z_style.norm():.3f})")

    else:
        # 2. Mode classique (gaussian, zero, original)
        print(f"⚙️ Harmonisation configurée en mode classique : '{cfg['harmonization_mode']}'")
        if cfg['harmonization_mode'] == 'gaussian' and cfg.get('global_gaussian_style', True):
            global_z_style = torch.randn(1, style_channels, device=device)
            print(f"🎲 Style Gaussien global N(0,1) généré pour toute la cohorte (dim={style_channels}).")
        elif cfg['harmonization_mode'] == 'zero':
            global_z_style = torch.zeros(1, style_channels, device=device)
            print(f"🛑 Style neutre Zéro activé pour toute la cohorte.")
        else:
            global_z_style = None

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
            harmonization_mode=active_mode,
            alpha=cfg['alpha'],
            patch_size=cfg['patch_size'],
            patch_overlap=cfg['patch_overlap'],
            override=cfg['override'],
            z_style_fixed=global_z_style
        )

if __name__ == "__main__":
    main()
