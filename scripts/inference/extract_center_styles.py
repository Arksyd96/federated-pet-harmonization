import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchio as tio

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.unlearning_vae import UnlearningVAE
from pet_harmonization.utils import set_seed


def extract_patient_style_vectors(
    model: UnlearningVAE,
    subject: tio.Subject,
    device: torch.device,
    patch_size: tuple = (16, 64, 64),
    patch_overlap: tuple = (8, 16, 16),
    patches_per_patient: int = 100,
    min_patch_mean: float = 1e-3,
    batch_size: int = 16,
    spatial_dims: int = 3,
    rng: np.random.RandomState = None
) -> torch.Tensor:
    """
    Extrait 'patches_per_patient' vecteurs de style mu_s représentatifs
    en parcourant le volume complet et en filtrant les patchs vides.
    """
    grid_sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size, num_workers=0)

    all_valid_mu_s = []

    with torch.inference_mode():
        for patch_batch in patch_loader:
            patch_tio = patch_batch['source'][tio.DATA].to(device)

            if spatial_dims == 2:
                patch_src = patch_tio.squeeze(1)
            else:
                patch_src = patch_tio

            # Masque booléen pour ne garder que les patchs avec du tissu
            means = patch_src.view(patch_src.shape[0], -1).mean(dim=1)
            valid_mask = means >= min_patch_mean

            if not valid_mask.any():
                continue

            valid_patches = patch_src[valid_mask]
            patch_norm = model._normalize(valid_patches)

            # Encodage du style
            _, _, mu_s, _ = model.vae.encode(patch_norm)
            all_valid_mu_s.append(mu_s.detach().cpu())

    if not all_valid_mu_s:
        return None

    all_mu_s_tensor = torch.cat(all_valid_mu_s, dim=0)
    total_valid = all_mu_s_tensor.shape[0]

    # Sélection uniforme de 'patches_per_patient' pour couvrir toute l'anatomie
    if total_valid > patches_per_patient:
        indices = rng.choice(total_valid, size=patches_per_patient, replace=False)
        selected_mu_s = all_mu_s_tensor[indices]
    else:
        selected_mu_s = all_mu_s_tensor

    return selected_mu_s


def main():
    # ── Configuration de l'extraction ──
    cfg = {
        'SEED': 101,
        'ckpt_path': "runs/sandbox-unlearn-vae/2026_10_23_162550/checkpoints/2-epoch=127-rec=0.0201-style=0.960-content=0.172.ckpt",
        'spatial_dims': 3,
        'suv_global_log_max': 6.0,

        # Échantillonnage de patchs par patient
        'patches_per_patient': 100,
        'patch_size': (16, 64, 64),
        'patch_overlap': (8, 16, 16),
        'min_patch_mean': 1e-3,
        'batch_size_encoder': 16,

        # Sortie : fichier unique dictionnaire contenant TOUS les styles
        'output_path': "./data/style_embeddings/styles.pt",

        # Datamodule : le nombre de patients par centre est fixé directement dans split_config !
        'datamodule': {
            "root_dir": "./data/PET-EARL/",
            "split_config": [[0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 30]],
            "batch_size": 16,
            "patch_size": [16, 64, 64],
            "num_workers": 4,
            "queue_max_length": 4096,
            "samples_per_volume": 64,
        }
    }

    set_seed(cfg['SEED'], workers=True)
    rng = np.random.RandomState(cfg['SEED'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initialisation de l'extraction des styles | Device : {device}")

    if not os.path.exists(cfg['ckpt_path']):
        raise FileNotFoundError(f"Checkpoint introuvable : {cfg['ckpt_path']}")

    # Chargement du modèle
    model = UnlearningVAE.load_from_checkpoint(cfg['ckpt_path'], strict=True)
    model.to(device)
    model.eval()

    # Préparation des données
    datamodule = MultiDomainUnlearningDataModule(**cfg['datamodule'])
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.test_dataloader()

    domain_to_id = datamodule.domain_to_id
    id_to_domain = {v: k for k, v in domain_to_id.items()}
    print(f"📋 Domaines détectés ({len(domain_to_id)}) : {domain_to_id}")

    # Collecte des patchs par centre (domain_id -> list of tensors)
    center_patches = defaultdict(list)
    patients_per_center_count = defaultdict(int)

    for idx, batch in enumerate(tqdm(loader, desc="Extraction des styles")):
        subject = tio.utils.get_subjects_from_batch(batch)[0]
        domain_id = int(batch['domain_id'][0])
        domain_name = id_to_domain.get(domain_id, f"center_{domain_id}")
        subj_name = subject['subject_name']

        patient_mu_s = extract_patient_style_vectors(
            model=model,
            subject=subject,
            device=device,
            patch_size=cfg['patch_size'],
            patch_overlap=cfg['patch_overlap'],
            patches_per_patient=cfg['patches_per_patient'],
            min_patch_mean=cfg['min_patch_mean'],
            batch_size=cfg['batch_size_encoder'],
            spatial_dims=cfg['spatial_dims'],
            rng=rng
        )

        if patient_mu_s is not None and len(patient_mu_s) > 0:
            center_patches[domain_id].append(patient_mu_s)
            patients_per_center_count[domain_id] += 1

    # ── Calcul des Médianes ──
    print("\n📊 Calcul des styles médians...")
    center_medians = {}
    styles_dict = {}

    for d_id, d_name in id_to_domain.items():
        if d_id not in center_patches or len(center_patches[d_id]) == 0:
            print(f"⚠️ Aucun patch pour {d_name} (ID {d_id}).")
            continue

        # Tous les patchs de ce centre : Shape (N_patchs, 256)
        all_center_mu_s = torch.cat(center_patches[d_id], dim=0)

        # Médiane dimension par dimension : Shape (1, 256)
        c_median = torch.median(all_center_mu_s, dim=0).values.unsqueeze(0)
        center_medians[d_id] = c_median

        # Clés d'accès directes
        styles_dict[f"center_{d_id}"] = c_median
        styles_dict[d_name] = c_median

        print(f"  • {d_name} (ID {d_id}) : {patients_per_center_count[d_id]} patients, {all_center_mu_s.shape[0]} patchs | Norme = {c_median.norm():.3f}")

    # Médiane globale hiérarchique (médiane des médianes de centres)
    if center_medians:
        stacked_medians = torch.cat(list(center_medians.values()), dim=0) # (num_centers, 256)
        global_median = torch.median(stacked_medians, dim=0).values.unsqueeze(0) # (1, 256)
        styles_dict["global_median"] = global_median
        print(f"\n🌐 Style Médian Global (clé 'global_median') : Norme = {global_median.norm():.3f}")

    # ── Sauvegarde unique sous format dictionnaire ──
    os.makedirs(os.path.dirname(cfg['output_path']), exist_ok=True)
    torch.save(styles_dict, cfg['output_path'])
    print(f"\n💾 Dictionnaire unique sauvegardé dans : {cfg['output_path']}")
    print(f"🔑 Clés prêtes à l'emploi dans ce fichier : {list(styles_dict.keys())}")


if __name__ == "__main__":
    main()
