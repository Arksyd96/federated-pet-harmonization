from typing import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid
import wandb

from monai.networks.blocks import UnetOutBlock
from pet_harmonization.models.base import (
    BasicBlock,
    UnetBasicBlock,
    UnetResBlock,
    save_add,
    BasicDown,
    BasicUp,
    SequentialEmb,
)
from pet_harmonization.models.attention import Attention, zero_module
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

    # def _common_step(self, batch, batch_idx, stage):
    #     # 1. Récupération des données
    #     suv_source = batch['source'][tio.DATA].float() # PET
    #     suv_targets = {
    #         key: batch[key][tio.DATA].float() for key in ['target_1', 'target_2']
    #         if key in batch
    #     }
        
    #     if not suv_targets:
    #         raise ValueError("Aucune cible (target_1, target_2) trouvée dans le batch.")

    #     # Gestion des dimensions pour le "2D Channel Wise"
    #     if suv_source.ndim == 5 and self.hparams.spatial_dims == 2:
    #         suv_source = suv_source.squeeze(1) # Réduire la dimension du canal
    #         for key in suv_targets:
    #             suv_targets[key] = suv_targets[key].squeeze(1)

    #     # normalisation Globale
    #     log_source = torch.log1p(suv_source)
    #     concat_suv_targets = torch.cat([suv_targets[key] for key in sorted(suv_targets.keys())], dim=1)
    #     log_target = torch.log1p(concat_suv_targets)

    #     # scale to [-1, 1+eps]
    #     normalized_log_source = 2.0 * (log_source.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0
    #     normalized_log_target = 2.0 * (log_target.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

    #     # target_Residual = EARL_norm - PET_norm
    #     target_residual = (normalized_log_target - normalized_log_source.repeat(1, 2, 1, 1)) * self.alpha
    #     predicted_residual = self.forward(normalized_log_source)

    #     # normalized residual loss
    #     residual_loss = self.loss_fn(predicted_residual, target_residual)

    #     # reconstruction (log norm space)
    #     normalized_log_prediction = normalized_log_source.repeat(1, 2, 1, 1) + (predicted_residual / self.alpha)

    #     # reconstruction (suv space)
    #     log_prediction = 0.5 * (normalized_log_prediction + 1.0) * self.suv_global_log_max
    #     suv_prediction = torch.expm1(log_prediction)

    #     # residual suv loss
    #     loss_suv = self.loss_fn(suv_prediction, concat_suv_targets)
        
    #     # ssim loss range [0, 1]
    #     nlp_01 = (normalized_log_prediction + 1.0) / 2.0
    #     nlt_01 = (normalized_log_target + 1.0) / 2.0
    #     nlp_01 , nlt_01 = torch.clamp(nlp_01, 0.0, 1.0), torch.clamp(nlt_01, 0.0, 1.0)

    #     ssim_score = self.ssim_loss(nlp_01, nlt_01)
    #     loss_ssim = 1 - ssim_score

    #     # global loss
    #     loss = residual_loss + (0.1 * loss_suv) + (.5 * loss_ssim)

    #     # 6. Logging Metrics
    #     self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, 
    #                 sync_dist=True, batch_size=suv_source.size(0))
    #     self.log(f"{stage}/loss_residual", residual_loss, on_step=True, on_epoch=True, prog_bar=False,
    #                 sync_dist=True, batch_size=suv_source.size(0))
    #     self.log(f"{stage}/loss_suv", loss_suv, on_step=True, on_epoch=True, prog_bar=False,
    #                 sync_dist=True, batch_size=suv_source.size(0))
    #     self.log(f"{stage}/ssim_score", ssim_score, on_step=True, on_epoch=True, prog_bar=True, 
    #                 sync_dist=True, batch_size=suv_source.size(0))

    #     return loss, (
    #         normalized_log_source, 
    #         normalized_log_target, 
    #         normalized_log_prediction,
    #         suv_source,
    #         concat_suv_targets,
    #         suv_prediction,
    #         predicted_residual,
    #         target_residual
    #     )


    def _common_step(self, batch, batch_idx, stage):
        # 1. Récupération des données
        suv_source = batch['source'][tio.DATA].float() # PET
        suv_target = batch['target'][tio.DATA].float() # EARL
        
        if suv_target is None:
            raise ValueError("Aucune cible trouvée dans le batch.")

        # Gestion des dimensions pour le "2D Channel Wise"
        if suv_source.ndim == 5 and self.hparams.spatial_dims == 2:
            suv_source = suv_source.squeeze(1) # Réduire la dimension du canal
            suv_target = suv_target.squeeze(1)

        # normalisation Globale
        log_source = torch.log1p(suv_source)
        log_target = torch.log1p(suv_target)

        # scale to [-1, 1+eps]
        normalized_log_source = 2.0 * (log_source.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0
        normalized_log_target = 2.0 * (log_target.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

        # target_Residual = EARL_norm - PET_norm
        target_residual = (normalized_log_target - normalized_log_source) * self.alpha
        predicted_residual = self.forward(normalized_log_source)

        # normalized residual loss
        residual_loss = self.loss_fn(predicted_residual, target_residual)

        # reconstruction (log norm space)
        normalized_log_prediction = normalized_log_source + (predicted_residual / self.alpha)

        # reconstruction (suv space)
        log_prediction = 0.5 * (normalized_log_prediction + 1.0) * self.suv_global_log_max
        suv_prediction = torch.expm1(log_prediction)

        # residual suv loss
        loss_suv = self.loss_fn(suv_prediction, suv_target)
        
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
            suv_target,
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

class UNetWithIntermediateFeatures(UNet):
    def __init__(self, 
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
        **kwargs):
        super().__init__(in_ch, out_ch, spatial_dims, hid_chs, kernel_sizes, strides,
                         temb_channels, max_period, scale_shift_norm, act_name, norm_name,
                         cond_embedder, deep_supervision, use_res_block, estimate_variance,
                         use_self_conditioning, dropout, learnable_interpolation,
                         use_attention, num_res_blocks, **kwargs)

        self._feature_indices: List[int] = []  
        cursor = 0
        for i in range(1, self.depth):
            # Index du dernier res block de ce niveau
            last_res_idx = cursor + self.num_res_blocks - 1
            self._feature_indices.append(last_res_idx)
            cursor += self.num_res_blocks
            if i < self.depth - 1:
                cursor += 1  # BasicDown compte aussi

    def forward_with_features(
        self,
        x_t: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # --- Embeddings (time + condition) ---
        if t is None:   time_emb = None
        else:           time_emb = self.time_embedder(t)

        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None
        else:
            cond_emb = self.cond_embedder(condition)

        emb = save_add(time_emb, cond_emb)

        # --- Self-conditioning ---
        if self.use_self_conditioning:
            self_cond = torch.zeros_like(x_t) if self_cond is None else x_t
            x_t = torch.cat([x_t, self_cond], dim=1)

        # --- Encodeur ---
        x = [self.in_conv(x_t)]
        encoder_features: List[torch.Tensor] = [x[0]]  # level_0

        for i in range(len(self.in_blocks)):
            feat = self.in_blocks[i](x[i], emb)
            x.append(feat)
            # On capture la feature si c'est la fin d'un niveau
            if i in self._feature_indices:
                encoder_features.append(feat)

        # --- Middle block (espace latent) ---
        h = self.middle_block(x[-1], emb)
        encoder_features.append(h)  # latent = dernier élément

        # --- Décodeur (identique à forward()) ---
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

        y = self.outc(h)
        return y, encoder_features
    


class UnlearningUNet(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        domain_classifier: nn.Module,
        feature_extractor: nn.Module,
        num_domains: int = 3,
        warmup_epochs: int = 5,
        beta_confusion: float = 1.0,
        lambda_composite: float = 1.0,    # poids du terme domain_acc dans le score composite
        k_conf_steps : int = 1,           # nombre de steps de confusion par step principal
        max_epochs: int = 15,            # nécessaire pour le scheduler
        lr_main: float = 1e-4,
        lr_dm: float = 1e-4,
        lr_dm_final: float = 1e-6, 
        lr_conf: float = 1e-5,  
        lr_conf_final: float = 1e-6, 
        weight_decay: float = 1e-5,
        suv_global_log_max: float = 6.0,
        recon_loss_fn=F.l1_loss,
    ):
        super().__init__()

        self.model              = model
        self.domain_classifier  = domain_classifier
        self.feature_extractor  = feature_extractor
        self.num_domains        = num_domains
        self.warmup_epochs      = warmup_epochs
        self.beta               = beta_confusion 
        self.k_conf_steps       = k_conf_steps
        self.lambda_composite   = lambda_composite
        self.max_epochs         = max_epochs
        self.lrs                = {
            'main': lr_main, 
            'dm':   lr_dm,      'dm_final': lr_dm_final,
            'conf': lr_conf,    'conf_final': lr_conf_final
        }
        
        self.weight_decay       = weight_decay
        self.suv_global_log_max = suv_global_log_max
        self.loss_fn            = recon_loss_fn
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Optimisation manuelle obligatoire (multi-optimiseurs adversariaux)
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["model", "domain_classifier", "loss_fn", "feature_extractor"])

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    
    def _confusion_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Loss de Dinsdale : -mean(log(softmax(logits))).
        Pousse l'encodeur vers l'incertitude maximale sur les domaines.
        """
        p = F.softmax(logits, dim=1)
        return -torch.log(p + 1e-8).mean()

    def _apply_patch_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.15,
        patch_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x_masked = x.clone()

        n_patches_h = H // patch_size
        n_patches_w = W // patch_size
        n_patches_total = n_patches_h * n_patches_w
        n_masked = max(1, int(n_patches_total * mask_ratio))

        for b in range(B):
            # Tirage aléatoire des indices de patches à masquer
            indices = torch.randperm(n_patches_total, device=x.device)[:n_masked]
            for idx in indices:
                ph = (idx // n_patches_w) * patch_size
                pw = (idx %  n_patches_w) * patch_size
                x_masked[b, :, ph:ph + patch_size, pw:pw + patch_size] = 0.0

        return x_masked, x

    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        """SUV → espace log normalisé [-1, 1]."""
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0
    
    def _denormalize(self, norm_log: torch.Tensor) -> torch.Tensor:
        """Espace log normalisé [-1, 1] → SUV."""
        log = 0.5 * (norm_log.clamp(-1, 1) + 1.0) * self.suv_global_log_max
        return torch.expm1(log)
    
    def _is_warmup(self) -> bool:
        """
        warmup_epochs : int   → nombre d'époques entières de warmup
                        float → fraction d'une époque (ex: 0.1 = 10% des itérations)
        """
        if isinstance(self.warmup_epochs, float):
            iters_per_epoch = self.trainer.num_training_batches
            warmup_iters    = int(self.warmup_epochs * iters_per_epoch)
            return self.global_step < warmup_iters
        else:
            return self.current_epoch < self.warmup_epochs
    

    # ──────────────────────────────────────────────────────────────────────────
    # Schedulers cosine
    # ──────────────────────────────────────────────────────────────────────────
    
    def _stage2_progress(self) -> float:
        t = max(0, self.current_epoch - self.warmup_epochs)
        T = max(1, self.max_epochs - self.warmup_epochs)
        return min(t / T, 1.0)
    
    def _cosine_lr(self, lr_init: float, lr_final: float) -> float:
        p = self._stage2_progress()
        return lr_final + (lr_init - lr_final) * (1 + math.cos(math.pi * p)) / 2
        
    def on_train_epoch_start(self):
        if self._is_warmup():
            return

        # Mise à jour du LR du classifieur de domaine directement dans l'optimiseur
        _, opt_dm, opt_conf = self.optimizers()
        
        new_lr_dm = self._cosine_lr(self.lrs["dm"], self.lrs["dm_final"])
        for pg in opt_dm.param_groups:
            pg["lr"] = new_lr_dm

        new_lr_conf = self._cosine_lr(self.lrs["conf"], self.lrs["conf_final"])
        for pg in opt_conf.param_groups:
            pg["lr"] = new_lr_conf

        # Log des valeurs schedulées (pratique pour vérifier dans WandB)
        self.log("debug/lr_dm",   new_lr_dm,   on_step=False, on_epoch=True)
        self.log("debug/lr_conf", new_lr_conf, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # configure_optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # opt_main : encodeur + décodeur complet (tâche principale)
        opt_main = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.feature_extractor.parameters()),
            lr=self.lrs["main"]
        )

        # opt_dm : classifieurs de domaine seulement
        opt_dm = torch.optim.AdamW(
            self.domain_classifier.parameters(),
            lr=self.lrs["dm"]
        )

        # opt_conf : encodeur seulement (désapprentissage)
        opt_conf = torch.optim.AdamW(
            self.feature_extractor.parameters(),
            lr=self.lrs["conf"]
        )

        return [opt_main, opt_dm, opt_conf]

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt_main, opt_dm, opt_conf = self.optimizers()

        # --- Données ---
        suv_source = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]  # LongTensor [B], valeurs dans {0, 1, 2}

        if suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)  # (B,1,D,H,W) → (B,D,H,W)

        x          = self._normalize(suv_source)  # [-1, 1]
        bs         = x.shape[0]
        is_stage_1 = self._is_warmup()
        

        # ==============================================================
        # STAGE 1 — Warmup
        # ==============================================================
        if is_stage_1:
            # Forward complet (tâche + features)
            feature_map = self.feature_extractor(x)  # [B, C, H', W']
            reconstruction = self.model.forward(feature_map, t=None)
            
            task_loss = self.loss_fn(reconstruction, x)

            logits = self.domain_classifier(feature_map)
            loss_dm = F.cross_entropy(logits, domain_labels) 

            total_loss = loss_dm + task_loss  # on peut aussi pondérer si besoin

            # Update encodeur + décodeur + classifieurs
            opt_main.zero_grad()
            opt_dm.zero_grad()
            self.manual_backward(total_loss)
            opt_main.step()
            opt_dm.step()

            # Logs
            self._log_dict({
                "train/task_loss":   task_loss,
                "train/domain_loss": loss_dm
            }, bs)

        # ==============================================================
        # STAGE 2 — Unlearning Adversarial
        # ==============================================================
        else:
            feature_map = self.feature_extractor(x)  # [B, C, H', W']
            reconstruction = self.model.forward(feature_map, t=None)
            
            task_loss = self.loss_fn(reconstruction, x)

            opt_main.zero_grad()
            self.manual_backward(task_loss)
            opt_main.step()

            # Étape B : mise à jour classifieur (encodeur GELÉ)
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

            with torch.no_grad():
                detached_feature_map = self.feature_extractor(x)

            logits_dm = self.domain_classifier(detached_feature_map)
            loss_dm   = F.cross_entropy(logits_dm, domain_labels)

            opt_dm.zero_grad()
            self.manual_backward(loss_dm)
            torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=1.0)
            opt_dm.step()

            # ----------------------------------------------------------
            # Étape C : confusion loss (encodeur seul)
            # ----------------------------------------------------------
            self.feature_extractor.train()
            for p in self.feature_extractor.parameters():
                p.requires_grad = True

            for _ in range(self.k_conf_steps):
                feature_map_conf = self.feature_extractor(x)
                logits_conf      = self.domain_classifier(feature_map_conf)
                loss_conf        = self.beta * self._confusion_loss(logits_conf)

                opt_conf.zero_grad()
                self.manual_backward(loss_conf)
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=1.0)
                opt_conf.step()

            # Logs
            self._log_dict({
                "train/task_loss":      task_loss,
                "train/domain_loss":    loss_dm,
                "train/confusion_loss": loss_conf,
            }, bs)

    # ------------------------------------------------------------------
    # validation_step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        suv_source = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]

        if suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)

        x = self._normalize(suv_source)
        bs = x.shape[0]

        # Forward complet (tâche + features)
        feature_map = self.feature_extractor(x)  # [B, C, H', W']
        reconstruction = self.model.forward(feature_map, t=None)
        
        task_loss = self.loss_fn(reconstruction, x)

        logits = self.domain_classifier(feature_map)
        loss_dm = F.cross_entropy(logits, domain_labels)
        loss_conf = self._confusion_loss(logits)
        
        # SSIM
        x_01 = (x.clamp(-1, 1) + 1.0) / 2.0
        r_01 = (reconstruction.clamp(-1, 1) + 1.0) / 2.0
        ssim_score = self.ssim(r_01, x_01)

        # Accuracy du classifieur (doit descendre en stage 2)
        # On prend la prédiction du niveau le plus profond (le latent)
        # preds = logits[-1].argmax(dim=1)
        preds = logits.argmax(dim=1)
        acc = (preds == domain_labels).float().mean()

        # Score composite : bon modèle = faible reconstruction + classifieur confus
        # domain_acc_excess = combien le classifieur performe au-dessus du hasard
        chance_level       = 1.0 / self.num_domains
        domain_acc_excess  = (acc - chance_level).clamp(min=0.0)
        composite_score    = task_loss + self.lambda_composite * domain_acc_excess

        self._log_dict({
            "val/task_loss":       task_loss,
            "val/domain_loss":     loss_dm,
            "val/confusion_loss":  loss_conf,
            "val/ssim":            ssim_score,
            "val/domain_acc":      acc, 
            "val/composite_score": composite_score,
        }, bs)

        # Visualisation (premier batch uniquement)
        if batch_idx == 0:
            self._log_images(x, reconstruction, suv_source)

        return task_loss

    # ------------------------------------------------------------------
    # Utilitaires de log
    # ------------------------------------------------------------------

    def _log_dict(self, d: dict, batch_size: int):
        for k, v in d.items():
            self.log(k, v, prog_bar=True, sync_dist=True, on_step=True,
                     on_epoch=True, batch_size=batch_size)

    def _log_images(self, x_norm, recon_norm, suv_source):
        """Log WandB : slice centrale de source vs reconstruction."""
        if self.trainer.global_rank != 0:
            return

        # Dénormalisation vers SUV pour l'affichage
        log_pred = 0.5 * (recon_norm.clamp(-1, 1) + 1.0) * self.suv_global_log_max
        suv_pred = torch.expm1(log_pred)

        mid = suv_source.shape[1] // 2
        src_slice  = suv_source[:, mid:mid+1, :, :]
        pred_slice = suv_pred[:, mid:mid+1, :, :]

        display_max = max(5.0, src_slice.max().item(), pred_slice.max().item())
        imgs = torch.cat([src_slice, pred_slice], dim=3)
        imgs = (imgs / display_max).clamp(0, 1)

        grid = make_grid(imgs, nrow=1, padding=2)
        wandb.log({
            "Validation/Reconstruction": wandb.Image(
                grid.permute(1, 2, 0).cpu().numpy(),
                caption="Left: PET input | Right: PET harmonisée"
            )
        })


