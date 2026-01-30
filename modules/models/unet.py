from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid
import wandb

from monai.networks.blocks import UnetOutBlock
from modules.models.base import (
    BasicBlock,
    UnetBasicBlock,
    UnetResBlock,
    save_add,
    BasicDown,
    BasicUp,
    SequentialEmb,
)
from modules.models.attention import Attention, zero_module
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        emb_dim=16,
        downscale_freq_shift=1,
        max_period=1000,
        flip_sin_to_cos=False,
        rescale_to_max=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos = flip_sin_to_cos
        self.rescale = rescale_to_max

    def forward(self, x):
        device = x.device
        if self.rescale:
            x = x * self.max_period

        half_dim = self.emb_dim // 2
        emb = np.log(self.max_period) / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb * torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        spatial_dims: int = 3,
        hid_chs: List[int] = [256, 256, 512, 1024],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        temb_channels: int = 128,
        max_period: int = 1000,
        scale_shift_norm: bool = True,
        act_name: Tuple[str, Dict] = ('swish', {}),
        norm_name: Tuple[str, Dict] = ('group', {'num_groups': 32, 'affine': True}),
        cond_embedder: Optional[nn.Module] = None,
        deep_supervision: bool = False,
        use_res_block: bool = True,
        estimate_variance: bool = False,
        use_self_conditioning: bool = False,
        dropout: float = 0.0,
        learnable_interpolation: bool = True,
        use_attention: Union[str, List[str]] = 'none',
        num_res_blocks: int = 2,
        **kwargs
    ):
        super().__init__()
        assert (
            len(hid_chs) == len(kernel_sizes) == len(strides)
        ), "The length of hidden_channels, kernel_sizes, and strides must be the same."

        use_attention = (
            use_attention
            if isinstance(use_attention, list)
            else [use_attention] * len(strides)
        )
        self.use_self_conditioning = use_self_conditioning
        self.use_res_block = use_res_block
        self.depth = strides.__len__()
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_ch
        self.out_channels = out_ch

        # ------------- Time-Embedder-----------
        self.time_embedder = None
        if temb_channels is not None:
            self.time_embedder = nn.Sequential(
                SinusoidalPosEmb(emb_dim=temb_channels, max_period=max_period),
                nn.Linear(temb_channels, temb_channels * 4),
                nn.SiLU(),
                nn.Linear(temb_channels * 4, temb_channels),
            )

        # ------------- Condition-Embedder-----------
        self.cond_embedder = None
        if cond_embedder is not None:
            self.cond_embedder = cond_embedder

        # ----------- In-Convolution ---------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        in_ch = in_ch * 2 if self.use_self_conditioning else in_ch
        self.in_conv = BasicBlock(
            spatial_dims,
            in_ch,
            hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
        )

        # ----------- Encoder ------------
        in_blocks = []
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = []
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i - 1 if k == 0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=temb_channels,
                        scale_shift_norm=scale_shift_norm,
                    )
                )

                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i] // 8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=temb_channels,
                        attention_type=use_attention[i],
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))

            if i < self.depth - 1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation,
                    )
                )

        self.in_blocks = nn.ModuleList(in_blocks)

        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=temb_channels,
                scale_shift_norm=scale_shift_norm,
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1] // 8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=temb_channels,
                attention_type=use_attention[-1],
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=temb_channels,
                scale_shift_norm=scale_shift_norm,
            ),
        )

        # ------------ Decoder ----------
        out_blocks = []
        for i in range(1, self.depth):
            for k in range(num_res_blocks + 1):
                seq_list = []
                out_channels = hid_chs[i - 1 if k == 0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i] + hid_chs[i - 1 if k == 0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=temb_channels,
                        scale_shift_norm=scale_shift_norm,
                    )
                )

                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels // 8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=temb_channels,
                        attention_type=use_attention[i],
                    )
                )

                if (i > 1) and k == 0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation,
                        )
                    )

                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)

        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch * 2 if estimate_variance else out_ch

        self.outc = zero_module(
            UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None)
        )
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth - 2 if deep_supervision else 0

        self.outc_ver = nn.ModuleList(
            [
                zero_module(
                    UnetOutBlock(
                        spatial_dims, hid_chs[i] + hid_chs[i - 1], out_ch, dropout=None
                    )
                )
                for i in range(2, deep_supervision + 2)
            ]
        )

    def forward(self, x_t, t=None, condition=None, self_cond=None):
        # x_t [B, C, *]
        # t [B,]
        # condition [B,]
        # self_cond [B, C, *]

        # -------- Time Embedding (Gloabl) -----------
        if t is None:
            time_emb = None
        else:
            time_emb = self.time_embedder(t)  # [B, C]

        # -------- Condition Embedding (Gloabl) -----------
        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None
        else:
            cond_emb = self.cond_embedder(condition)  # [B, C]

        emb = save_add(
            time_emb, cond_emb
        )  # treating the condition as a global condition

        # ---------- Self-conditioning-----------
        if self.use_self_conditioning:
            self_cond = torch.zeros_like(x_t) if self_cond is None else x_t
            x_t = torch.cat([x_t, self_cond], dim=1)

        # --------- Encoder --------------
        x = [self.in_conv(x_t)]
        for i in range(len(self.in_blocks)):
            x.append(self.in_blocks[i](x[i], emb))

        # ---------- Middle --------------
        h = self.middle_block(x[-1], emb)

        # -------- Decoder -----------
        y_ver = []
        for i in range(len(self.out_blocks), 0, -1):
            h = torch.cat([h, x.pop()], dim=1)

            depth, j = i // (self.num_res_blocks + 1), i % (self.num_res_blocks + 1) - 1
            (
                y_ver.append(self.outc_ver[depth - 1](h))
                if (len(self.outc_ver) >= depth > 0) and (j == 0)
                else None
            )

            h = self.out_blocks[i - 1](h, emb)

        # ---------Out-Convolution ------------
        y = self.outc(h)

        return y

class TranslationUNet(LightningModule):
    def __init__(
        self,
        in_ch: int,          
        out_ch: int,         
        spatial_dims: int, 
        hid_chs: list = [64, 128, 256, 512],
        kernel_sizes: list = [3, 3, 3, 3],
        strides: list = [1, 2, 2, 2],
        
        # Hyperparamètres d'entraînement
        alpha: float = 10.0,  # Facteur d'amplification du résidu
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        suv_global_log_max: float = 6.0, # Notre constante de normalisation clinique
        loss_fn = nn.functional.l1_loss
    ):
        super().__init__()
        self.suv_global_log_max = suv_global_log_max
        self.alpha = alpha
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)

        # --- 1. Instanciation du UNet ---
        # On passe temb_channels=None pour désactiver l'embedding temporel (mode Regression)
        self.model = UNet(
            in_ch=in_ch,
            out_ch=out_ch,
            spatial_dims=spatial_dims,
            hid_chs=hid_chs,
            kernel_sizes=kernel_sizes,
            strides=strides,
            temb_channels=None,
            use_attention='none', # Peut être activé si besoin
            num_res_blocks=2
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x, t=None, condition=None)

    def _common_step(self, batch, batch_idx, stage):
        # 1. Récupération des données
        suv_source = batch['source'][tio.DATA].float() # PET
        suv_targets = {
            key: batch[key][tio.DATA].float() for key in ['target_1', 'target_2']
            if key in batch
        }
        
        if not suv_targets:
            raise ValueError("Aucune cible (target_1, target_2) trouvée dans le batch.")

        # Gestion des dimensions pour le "2D Channel Wise"
        if suv_source.ndim == 5 and self.hparams.spatial_dims == 2:
            suv_source = suv_source.squeeze(1) # Réduire la dimension du canal
            for key in suv_targets:
                suv_targets[key] = suv_targets[key].squeeze(1)

        # normalisation Globale
        log_source = torch.log1p(suv_source)
        concat_suv_targets = torch.cat([suv_targets[key] for key in sorted(suv_targets.keys())], dim=1)
        log_target = torch.log1p(concat_suv_targets)

        # scale to [-1, 1+eps]
        normalized_log_source = 2.0 * (log_source.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0
        normalized_log_target = 2.0 * (log_target.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

        # target_Residual = EARL_norm - PET_norm
        target_residual = (normalized_log_target - normalized_log_source.repeat(1, 2, 1, 1)) * self.alpha
        predicted_residual = self.forward(normalized_log_source)

        # normalized residual loss
        residual_loss = self.loss_fn(predicted_residual, target_residual)

        # reconstruction (log norm space)
        normalized_log_prediction = normalized_log_source.repeat(1, 2, 1, 1) + (predicted_residual / self.alpha)

        # reconstruction (suv space)
        log_prediction = 0.5 * (normalized_log_prediction + 1.0) * self.suv_global_log_max
        suv_prediction = torch.expm1(log_prediction)

        # residual suv loss
        loss_suv = self.loss_fn(suv_prediction, concat_suv_targets)
        
        # ssim loss range [0, 1]
        nlp_01 = (normalized_log_prediction + 1.0) / 2.0
        nlt_01 = (normalized_log_target + 1.0) / 2.0
        nlp_01 , nlt_01 = torch.clamp(nlp_01, 0.0, 1.0), torch.clamp(nlt_01, 0.0, 1.0)

        ssim_score = self.ssim_loss(nlp_01, nlt_01)
        loss_ssim = 1 - ssim_score

        # global loss
        loss = residual_loss + (0.1 * loss_suv) + (.5 * loss_ssim)

        # 6. Logging Metrics
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, 
                    sync_dist=True, batch_size=suv_source.size(0))
        self.log(f"{stage}/loss_residual", residual_loss, on_step=True, on_epoch=True, prog_bar=False,
                    sync_dist=True, batch_size=suv_source.size(0))
        self.log(f"{stage}/loss_suv", loss_suv, on_step=True, on_epoch=True, prog_bar=False,
                    sync_dist=True, batch_size=suv_source.size(0))
        self.log(f"{stage}/ssim_score", ssim_score, on_step=True, on_epoch=True, prog_bar=True, 
                    sync_dist=True, batch_size=suv_source.size(0))

        return loss, (
            normalized_log_source, 
            normalized_log_target, 
            normalized_log_prediction,
            suv_source,
            concat_suv_targets,
            suv_prediction,
            predicted_residual,
            target_residual
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, meta = self._common_step(batch, batch_idx, "val")
        suv_source, suv_target, suv_prediction = meta[3], meta[4], meta[5]
        
        # MSE en espace normalisé
        mse_val = F.mse_loss(suv_prediction, suv_target)
        l1_val = F.l1_loss(suv_prediction, suv_target)

        self.log("val/L2", mse_val, on_epoch=True, on_step=False, 
                 sync_dist=True, prog_bar=False, batch_size=suv_target.size(0))
        self.log("val/L1", l1_val, on_epoch=True, on_step=False, 
                 sync_dist=True, prog_bar=False, batch_size=suv_target.size(0))

        # Log Images (seulement sur le premier batch pour ne pas spammer)
        if batch_idx == 0:
            self.log_images(suv_source, suv_target, suv_prediction)
            
        return loss

    def log_images(self, source, target, prediction):
        """ Affiche les images dans WandB/Tensorboard """
        # On prend la slice centrale pour l'affichage (index 2 sur 5)
        mid = source.shape[1] // 2 
        
        # Extraction slice centrale (B, 1, H, W)
        src_slice = source[:, mid:mid + 1, :, :]
        tgt_slice = target[:, mid:mid + 1, :, :]
        pred_slice = prediction[:, mid:mid + 1, :, :]
        
        # Clipping pour affichage (0 à 5 SUV pour le contraste clinique)
        display_max = max(
            5.0, 
            src_slice.max().item(), 
            tgt_slice.max().item(), 
            pred_slice.max().item()
        )
        
        imgs = torch.cat([src_slice, tgt_slice, pred_slice], dim=3) # Stack horizontal
        imgs = (imgs / display_max).clamp(0, 1) # Normalisation visuelle
        
        grid = make_grid(imgs, nrow=1, padding=2)
        
        # Légende
        caption = f"Left: PET (Input) | Mid: EARL (Target) | Right: Reconstructed (PET+Delta)\n" + \
                    f"MSE (SUV space): {F.mse_loss(pred_slice, tgt_slice).item():.4f}"
        
        wandb_image = wandb.Image(
                grid.permute(1, 2, 0).cpu().numpy(), 
                caption=caption
            )

        wandb.log({"Validation/Reconstruction": wandb_image})


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }