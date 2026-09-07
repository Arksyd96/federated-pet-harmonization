import torch
import logging

from pet_harmonization.models.harmonization_vae import DisentangledHarmonizationVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Initializing DisentangledHarmonizationVAE for 3D...")
    
    # Configuration correspondant à vae.yaml
    config = {
        "input_shape": [16, 64, 64],
        "fft_sigma": 7.5,
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": [32, 64, 128, 256],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
        "latent_channels": 8,
        "style_channels": 256,
        "style_embedding_dim": 256,
        "num_residual_blocks": 1,
        "spatial_dims": 3,
        "normalization": ["group", {"num_groups": 32, "affine": True}],
        "activation": ["swish", {}],
        "dropout": 0.0,
        "use_residual_block": True,
        "learnable_interpolation": True,
        "attention_type": "none",
        "use_contour_skip": False
    }

    # Initialisation du modèle
    model = DisentangledHarmonizationVAE(**config)
    model.eval()

    # Création d'un tenseur factice (Batch, Channels, Z, Y, X)
    batch_size = 2
    x = torch.randn(batch_size, 1, 16, 64, 64)
    logger.info(f"Input shape (x): {tuple(x.shape)}")

    with torch.no_grad():
        # Lancement de l'encodeur pour vérifier les dimensions latentes
        mu_content, logvar_content, mu_style, logvar_style = model.encode(x)
        
        logger.info("--- Encodeur ---")
        logger.info(f"mu_content shape : {tuple(mu_content.shape)} -> Espace latent Z: {tuple(mu_content.shape[2:])} avec {mu_content.shape[1]} canaux.")
        logger.info(f"mu_style shape   : {tuple(mu_style.shape)}")

        # Verification des dimensions attendues
        assert tuple(mu_content.shape) == (batch_size, 8, 8, 8, 8), f"Erreur de dimension sur z_content ! Obtenu: {mu_content.shape}"
        assert tuple(mu_style.shape) == (batch_size, 256, 8, 8, 8), f"Erreur de dimension sur z_style ! Obtenu: {mu_style.shape}"

        # Lancement complet (forward) pour vérifier la reconstruction
        out = model(x)
        x_hat = out[0] if isinstance(out, tuple) else out
        
        logger.info("--- Décodeur ---")
        logger.info(f"Reconstruction (x_hat) shape : {tuple(x_hat.shape)}")
        
        if tuple(x_hat.shape) != tuple(x.shape):
            logger.error(f"❌ Erreur critique : L'image d'entrée est {tuple(x.shape)} mais le décodeur recrache {tuple(x_hat.shape)} !")
        else:
            logger.info("✅ Test réussi ! L'architecture encode et décode correctement.")

if __name__ == "__main__":
    run_test()
