import argparse
import os
import math
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from omegaconf import OmegaConf

# Tes modules personnalisés
from pet_harmonization.data import SingleTargetPETDataModule
from pet_harmonization.models.unet import TranslationUNet
from pet_harmonization.utils import set_seed

from torch.utils.data import DataLoader


def save_prediction(recon_volume: torch.Tensor, source_path: str, pred_path: str):
    """Fonction modulaire pour formater et sauvegarder le volume SimpleITK."""
    final_prediction = recon_volume.squeeze().permute(2, 1, 0).numpy()
    final_prediction = np.flip(final_prediction, axis=2) # Flip Z
    final_prediction = np.flip(final_prediction, axis=1) # Flip Y (Correction orientation)
    final_prediction = final_prediction.astype(np.float32) # ensure float32 for SimpleITK

    output_sitk = sitk.GetImageFromArray(final_prediction)
    output_sitk.CopyInformation(sitk.ReadImage(source_path))
    sitk.WriteImage(output_sitk, pred_path)
    print(f"✅ Prediction saved at: {pred_path}")


def process_subject(
    model, 
    batch, 
    device, 
    filename, 
    output_dir, 
    override, 
    include_only,
    curr_idx, 
    length_loader,
    num_standards,
    patch_size,
    overlap
    ):
    SUV_LOG_MAX = model.hparams.suv_global_log_max
    ALPHA = model.hparams.alpha
    
    subject_name = batch['subject_id'][0]
    print(f"Treating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    if include_only is not None and subject_name not in include_only:
        print(f"⚠️  Subject {subject_name} not in include_only list. Skipping...")
        return
    
    # Check if files already exist
    subj_out_dir = os.path.join(output_dir, subject_name)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    # Détermination dynamique des noms de fichiers
    out_filenames = [f"{filename}.nii.gz"] if num_standards == 1 else [f"{filename}{i + 1}.nii.gz" for i in range(num_standards)]
    all_exist = all(os.path.exists(os.path.join(subj_out_dir, f)) for f in out_filenames)
    
    if not override and all_exist:
        print(f"⚠️  Predictions already exist for {subject_name}. Skipping...")
        return

    subject = tio.utils.get_subjects_from_batch(batch)[0]
    
    grid_sampler = tio.data.GridSampler(subject, patch_size, tuple(overlap))
    patch_loader = DataLoader(grid_sampler, batch_size=4, num_workers=0) 

    aggregators = [tio.data.GridAggregator(grid_sampler, overlap_mode='hann') for _ in range(num_standards)]

    with torch.inference_mode():
        for patch_batch in tqdm(patch_loader, desc="Patch-wise inference"):
            locations = patch_batch[tio.LOCATION]
            
            patch_tio = patch_batch['source'][tio.DATA].float().to(device)
            if patch_tio.ndim == 5:
                patch_src = patch_tio.squeeze(1) # Passage à (B, D, H, W)
            else:
                patch_src = patch_tio

            if patch_src.mean() < 1e-3: 
                patch_pred_tio = torch.zeros_like(patch_tio)
                for agg in aggregators:
                    agg.add_batch(patch_pred_tio, locations)
                continue
            
            log_source = torch.log1p(patch_src)
            normalized_log_source = 2.0 * (log_source / SUV_LOG_MAX) - 1.0
            
            # Prédiction du/des résidus (peut être 5 canaux ou 10 canaux)
            predicted_residual = model.forward(normalized_log_source)

            # Duplication de la source pour l'addition si on a plusieurs standards (ex: (1, 10, 64, 64))
            src_repeated = normalized_log_source.repeat(1, num_standards, 1, 1)

            # Reconstruction inverse
            normalized_log_prediction = src_repeated + (predicted_residual / ALPHA)
            log_prediction = 0.5 * (normalized_log_prediction + 1.0) * SUV_LOG_MAX
            suv_prediction = torch.expm1(log_prediction)

            # Split des canaux prédits selon les standards (morceaux de taille 5)
            chunks = torch.chunk(suv_prediction, num_standards, dim=1)

            # Accumulation séparée pour chaque standard
            for i, chunk in enumerate(chunks):
                patch_pred_tio = chunk.unsqueeze(1) # Retour à (B, 1, D, H, W)
                aggregators[i].add_batch(patch_pred_tio, locations)
    
    source_path = batch['source']['path'][0]
    
    for i, out_filename in enumerate(out_filenames):
        recon_tensor = aggregators[i].get_output_tensor()
        recon_volume = recon_tensor.cpu()
        pred_path = os.path.join(subj_out_dir, out_filename)
        save_prediction(recon_volume, source_path, pred_path)

        
def predict_patch_wise_earl(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initialisation | Modèle : TranslationUNET | Device : {device}")
    
    model = TranslationUNet.load_from_checkpoint(args.ckpt_path)
    model.to(device)
    model.eval()
    print('Model loaded successfully.')

    print(f"⚙️  Configuration : Génération de {args.num_standards} standard(s) cible(s).")

    # On utilise toujours le SingleTargetPETDataModule pour l'inférence
    datamodule_kwargs = config.get('datamodule', {})
    datamodule = SingleTargetPETDataModule(**datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    
    if loader is None:
        print("⚠️ No data found in test_dataloader. Check your dataset configuration.")
        return
        
    patch_size = datamodule_kwargs.get('patch_size', (5, 64, 64))

    for idx, batch in enumerate(loader):
        process_subject(
            model=model, 
            batch=batch, 
            device=device, 
            filename=args.filename, 
            output_dir=args.output, 
            include_only=args.include_only,
            override=args.override,
            curr_idx=idx + 1, 
            length_loader=len(loader),
            num_standards=args.num_standards,
            patch_size=patch_size,
            overlap=args.overlap
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction Translation UNet (pseudo-EARL)")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--output', '-o', type=str, required=True, help='ex: outputs/pseudoEARL.')
    parser.add_argument('--include-only', '-i', type=str, nargs='*', default=None, help='List of subject IDs to include (default: all).')
    parser.add_argument('--filename', '-f', type=str, required=False, default='pseudo-earl', help='Filename to process.')
    parser.add_argument('--override', '-r', action='store_true', help='Whether to override existing predictions.')
    parser.add_argument('--num-standards', '-n', type=int, default=1, help='Number of target standards to generate (ex: 1 or 2).')
    parser.add_argument('--overlap', type=int, nargs=3, default=[1, 2, 2], help='Overlap along z, y, x axes (default: 1 2 2) to avoid artifacts.')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)
