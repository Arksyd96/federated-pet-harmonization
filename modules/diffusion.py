"""
    Code inspired by medfusion public repo : https://github.com/mueller-franzes/medfusion
    And official Open AI repo : https://github.com/openai/guided-diffusion (https://arxiv.org/abs/2212.07501)
    The code is modified to fit the needs of the project
"""

from typing import Union, Tuple, Any
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torchvision.utils import make_grid
import pytorch_lightning as pl
import wandb

# from modules.models.base import BasicModel
from modules.ema import EMAModel  # TODO
from modules.utils import kl_gaussians  # TODO

from modules.models.unet import UNet
from modules.scheduler import GaussianNoiseScheduler
from modules.models.base import BasicModel
from modules.data import robust_patch_normalization, robust_patch_denormalization

class DiffusionPipeline(BasicModel):
    def __init__(
        self,
        noise_scheduler,
        noise_estimator,
        latent_embedder = None, # If used on latent variables
        estimator_objective = 'x_T',  # Predicting x_T / x_0 (Noise or denoised image)
        estimate_variance = False,
        use_self_conditioning = False,
        classifier_free_guidance_dropout = 0.0,
        clip_x0 = False,
        use_ema = False,
        ema_kwargs = {},
        optimizer = torch.optim.AdamW,
        optimizer_kwargs = {'lr': 1e-4},  # stable-diffusion ~ 1e-4
        lr_scheduler = None,  # stable-diffusion - LambdaLR
        lr_scheduler_kwargs = {},
        loss = torch.nn.L1Loss,
        loss_kwargs = {},
        std_norm = None, # If used on latent variables
        sample_every_n_steps = 500,
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.noise_scheduler = noise_scheduler
        self.noise_estimator = noise_estimator
        self.latent_embedder = latent_embedder

        if self.latent_embedder is not None:
            self.latent_embedder.freeze()
            self.latent_embedder.requires_grad_(False)

        self.loss_fct = loss(**loss_kwargs)
        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.estimate_variance = estimate_variance        
        self.clip_x0 = clip_x0
        self.std_norm = std_norm
        self.sample_every_n_steps = sample_every_n_steps

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)

        self.save_hyperparameters(ignore=['latent_embedder'])

    def _step(self, batch, batch_idx, state, step):
        x_0, target, results = batch, {}

        if self.latent_embedder is not None:
            x_0 = self.latent_embedder.encode(x_0, None)

        if self.std_norm is not None:
            x_0 = x_0.div(self.std_norm)

        if self.clip_x0:
            x_0 = torch.clamp(x_0, -1, 1)

        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0,T-1] and compute x_t (noisy version of x_0 at t)
            x_t, x_T, t = self.noise_scheduler.sample(x_0)

        # Use EMA Model
        if self.use_ema and (state != "train"):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Re-estimate x_T or x_0, self-conditioned on previous estimate
        self_cond = None
        if self.use_self_conditioning:
            with torch.no_grad():
                pred = noise_estimator(x_t, t, condition, None)
                
                if self.estimate_variance:
                    pred, _ = pred.chunk(2, dim=1)  # Seperate actual prediction and variance estimation

                if self.estimator_objective == "x_T":  # self condition on x_0
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0":  # self condition on x_T
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # Classifier free guidance
        if torch.rand(1) <= self.classifier_free_guidance_dropout:
            condition = None

        # Run Denoise
        pred = noise_estimator(x_t, t, condition, None)
        pred_vertical = []

        # Separate variance (scale) if it was learned
        if self.estimate_variance:
            pred, pred_var = pred.chunk(2, dim=1)  # Separate actual prediction and variance estimation

        # Specify target
        if self.estimator_objective == "x_T":
            target = x_T
        elif self.estimator_objective == "x_0":
            target = x_0
        else:
            raise NotImplementedError(
                f"Option estimator_target={self.estimator_objective} not supported."
            )

        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        loss = 0
        weights = [
            1 / 2**i for i in range(1 + len(pred_vertical))
        ]  # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w / tot_weight for w in weights]

        # ----------------- MSE/L1, ... ----------------------
        loss += self.loss_fct(pred, target) * weights[0]

        # ----------------- Variance Loss --------------
        if self.estimate_variance:
            # var_scale = var_scale.clamp(-1, 1) # Should not be necessary
            var_scale = (pred_var + 1) / 2  # Assumed to be in [-1, 1] -> [0, 1]
            pred_logvar = self.noise_scheduler.estimate_variance_t(
                t, x_t.ndim, log=True, var_scale=var_scale
            )
            # pred_logvar = pred_var  # If variance is estimated directly

            if self.estimator_objective == "x_T":
                pred_x_0 = self.noise_scheduler.estimate_x_0(
                    x_t, x_T, t, clip_x0=self.clip_x0
                )
            elif self.estimator_objective == "x_0":
                pred_x_0 = pred
            else:
                raise NotImplementedError()

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, x_0, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(
                    t, x_t.ndim, log=True, var_scale=0
                )

            kl_loss = torch.mean(
                kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar),
                dim=list(range(1, x_0.ndim)),
            )
            nnl_loss = torch.mean(
                F.gaussian_nll_loss(
                    pred_x_0, x_0, torch.exp(pred_logvar), reduction="none"
                ),
                dim=list(range(1, x_0.ndim)),
            )
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss

            results["variance_scale"] = torch.mean(var_scale)
            results["variance_loss"] = var_loss

        # ----------------------------- Deep Supervision -------------------------
        for i, pred_i in enumerate(pred_vertical):
            target_i = F.interpolate(
                target,
                size=pred_i.shape[2:],
                mode=interpolation_mode,
                align_corners=None,
            )
            loss += self.loss_fct(pred_i, target_i) * weights[i + 1]
        results["loss"] = loss

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            results["L2"] = F.mse_loss(pred, target)
            results["L1"] = F.l1_loss(pred, target)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(
                f"{state} - {metric_name}",
                metric_val,
                batch_size=x_0.shape[0],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        if (self.global_step + 1) % self.sample_every_n_steps == 0:
            self.log_samples(num_samples=1)

        return loss

    def log_samples(self, num_samples: int, *kwargs: Tuple[Any, ...]):
        if self.trainer.global_rank == 0:
            pass # à coder

    def forward(
        self,
        x_t,
        t,
        condition=None,
        self_cond=None,
        guidance_scale=1.0,
        cold_diffusion=False,
        un_cond=None,
    ):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition is not None) and (guidance_scale != 1.0):
            pred_uncond = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond)
            pred_cond   = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond = pred_uncond.chunk(2, dim=1)
                pred_cond, pred_var_cond = pred_cond.chunk(2, dim=1)
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)

        else:
            pred = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var = pred.chunk(2, dim=1)

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5  # [-1, 1] -> [0, 1]
            pred_var_value = pred_var
        else:
            pred_var_scale = 0
            pred_var_value = None


        if self.estimator_objective == "x_0":
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(
                x_t,
                t,
                pred,
                clip_x0=self.clip_x0,
                var_scale=pred_var_scale,
                cold_diffusion=cold_diffusion,
            )
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T

        elif self.estimator_objective == "x_T":
            if x_t.shape[1] != pred.shape[1]:
                x_t = x_t[:, : pred.shape[1]]
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(
                x_t,
                t,
                pred,
                clip_x0=self.clip_x0,
                var_scale=pred_var_scale,
                cold_diffusion=cold_diffusion,
            )
            x_T = pred
            self_cond = x_0
        else:
            raise ValueError("Unknown Objective")

        return x_t_prior, x_0, x_T, self_cond

    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition=None, use_ddim=False, **kwargs):
        self_cond = None

        # ---------- run denoise loop ---------------
        steps = self.noise_scheduler.timesteps if steps is None else steps
        if use_ddim:
            timesteps_array = torch.linspace(
                0,
                self.noise_scheduler.T - 1,
                steps,
                dtype=torch.long,
                device=x_t.device,
            )  # [0, 1, 2, ..., T-1] if steps = T
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[
                slice(0, steps)
            ]  # [0, ...,T-1] (target time not time of x_t)

        for i, t in tqdm(enumerate(reversed(timesteps_array))):
            # UNet prediction
            x_t, x_0, x_T, self_cond = self.forward(
                x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, **kwargs
            )
            self_cond = self_cond if self.use_self_conditioning else None

            if use_ddim and (steps - i - 1 > 0):
                t_next = timesteps_array[steps - i - 2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = (
                    kwargs.get("eta", 1)
                    * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                )
                c = (1 - alpha_next - sigma**2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # ------ Eventually decode from latent space into image space--------
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)

        return x_t  # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        img_size,
        condition=None,
        steps=None,
        use_ddim=False,
        **kwargs,
    ):
        template = torch.zeros((num_samples, *img_size), device=self.device)
        x_T = self.noise_scheduler.x_final(template)
        x_0 = self.denoise(
            x_T,
            steps=steps,
            condition=(
                condition.to(self.device, dtype=self.dtype)
                if condition is not None
                else None
            ),
            use_ddim=use_ddim,
            **kwargs,
        )
        return x_0

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.noise_estimator.parameters(), **self.optimizer_kwargs
        )
        if self.lr_scheduler is not None:
            lr_scheduler = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]


class TranslationDiffusionPipeline(BasicModel):
    def __init__(
        self,
        noise_scheduler,
        noise_estimator,
        estimator_objective="x_T",  # 'x_T' or 'x_0'
        estimate_variance=False,
        use_self_conditioning=False,
        classifier_free_guidance_dropout=0.0,
        clip_x0=False,  # Has only an effect during traing if use_self_conditioning=True, import for inference/sampling
        use_ema=False,
        ema_kwargs={},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4},  # stable-diffusion ~ 1e-4
        lr_scheduler=None,  # stable-diffusion - LambdaLR
        lr_scheduler_kwargs={},
        loss=torch.nn.L1Loss,
        loss_kwargs={},
        sample_every_n_steps=500,
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.loss_fct = loss(**loss_kwargs)
        self.noise_scheduler = noise_scheduler
        self.noise_estimator = noise_estimator
        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0
        self.sample_every_n_steps = sample_every_n_steps

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)

        self.save_hyperparameters(ignore=['noise_estimator', 'noise_scheduler'])

    def _step(self, batch, batch_idx, state, step):
        results = {}
        source = batch['source'][tio.DATA]
        target = batch['target'][tio.DATA]

        source, target = source.squeeze(1), target.squeeze(1)  # Remove extra dim introduced by torchIO
        norm_source, norm_target, norm_factors = robust_patch_normalization(source, target, percentiles=(0.0, 99.9), clone=True)
        target_delta = (norm_target - norm_source) * 5.0 # => [-2, 2]

        # if self.clip_x0:
        #     norm_target = torch.clamp(norm_target, -1, 1)

        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0, T-1] and compute x_t (noisy version of x_0 at t)
            target_placeholder = torch.clone(target_delta)
            x_t, x_T, t = self.noise_scheduler.sample(target_placeholder)
            condition = norm_source

        # Use EMA Model
        if self.use_ema and (state != "train"):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Classifier free guidance
        # if torch.rand(1) <= self.classifier_free_guidance_dropout:
        #     condition = torch.zeros_like(condition)

        x_t = torch.cat([x_t, condition], dim=1)  # condition is the second half of the tensor
        prediction = noise_estimator(x_t, t, condition)

        # Separate variance (scale) if it was learned
        if self.estimate_variance:
            prediction, variance = prediction.chunk(2, dim=1)

        # Specify target
        if self.estimator_objective == "x_T":
            objective = x_T
        elif self.estimator_objective == "x_0":
            objective = target_delta
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # ------------------------- Compute Loss ---------------------------
        loss = self.loss_fct(prediction, objective)

        # ----------------- Variance Loss --------------
        if self.estimate_variance:
            # var_scale = var_scale.clamp(-1, 1) # Should not be necessary
            var_scale = (variance + 1) / 2  # Assumed to be in [-1, 1] -> [0, 1]
            pred_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=var_scale)
            # pred_logvar = pred_var  # If variance is estimated directly

            if self.estimator_objective == "x_T":
                pred_x_0 = self.noise_scheduler.estimate_x_0(x_t, x_T, t, clip_x0=self.clip_x0)
            elif self.estimator_objective == "x_0":
                pred_x_0 = prediction
            else:
                raise NotImplementedError()

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, target, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(
                    t, x_t.ndim, log=True, var_scale=0
                )

            kl_loss = torch.mean(
                kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar),
                dim=list(range(1, target.ndim)),
            )
            nnl_loss = torch.mean(
                F.gaussian_nll_loss(
                    pred_x_0, target, torch.exp(pred_logvar), reduction="none"
                ),
                dim=list(range(1, target.ndim)),
            )
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss

            results["variance_scale"] = torch.mean(var_scale)
            results["variance_loss"] = var_loss

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            results["L2"] = F.mse_loss(prediction, objective)
            results["L1"] = F.l1_loss(prediction, objective)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(
                f"{state}/{metric_name}",
                metric_val,
                batch_size=objective.shape[0],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        if (self.global_step + 1) % self.sample_every_n_steps == 0:
            self.log_samples(norm_source[:8], norm_target[:8], norm_factors[:8])

        return loss

    # @torch.no_grad()
    # def log_samples(self, source: torch.FloatTensor, target: torch.FloatTensor = None):
    #     if self.trainer.global_rank == 0:
    #         self.noise_estimator.eval()
    #         with torch.no_grad():
    #             prediction = self.sample(
    #                 source,
    #                 condition=None, # Ou votre condition si nécessaire
    #                 # steps=self.noise_scheduler.timesteps,
    #                 steps=50,
    #                 use_ddim=True,
    #             )
    #         self.noise_estimator.train()

    #         # On suppose le format (B, C, D, H, W). On coupe au milieu de la profondeur D.
    #         mid_slice_idx = source.shape[2] // 2  # Index 32 pour une taille de 64
            
    #         # On récupère les slices 2D: (B, 1, 64, 64)
    #         source_slice = source[:, :, mid_slice_idx, :, :]
    #         pred_slice = prediction[:, :, mid_slice_idx, :, :]
            
    #         # Liste des images à afficher côte à côte
    #         images_to_stack = [source_slice, pred_slice]
            
    #         # Si on a la vérité terrain (Target EARL), on l'ajoute aussi
    #         if target is not None:
    #             target_slice = target[:, :, mid_slice_idx, :, :]
    #             images_to_stack.append(target_slice)

    #         # Shape résultante par patient : (1, 64, 128) ou (1, 64, 192)
    #         comparison_batch = torch.cat(images_to_stack, dim=3)

    #         # 4. Dénormalisation ([-1, 1] -> [0, 255])
    #         # On applique clamp avant et après pour éviter les artefacts visuels bizarres
    #         comparison_batch = (
    #             comparison_batch
    #             .clamp(-1, 1)   # S'assure qu'on ne dépasse pas les bornes théoriques
    #             .add(1)         # [-1, 1] -> [0, 2]
    #             .div(2)         # [0, 2]  -> [0, 1]
    #             .mul(255)       # [0, 1]  -> [0, 255]
    #             .clamp(0, 255)  # Sécurité finale
    #             .to(torch.uint8)
    #         )

    #         # make_grid gère automatiquement l'agencement des batchs (B patients)
    #         # nrow=1 force une colonne verticale de patients
    #         grid = make_grid(comparison_batch, nrow=1, padding=2, normalize=False)

    #         # WandB attend (H, W, C), donc on permute car PyTorch est (C, H, W)
    #         wandb_image = wandb.Image(
    #             grid.permute(1, 2, 0).cpu().numpy(), 
    #             caption=f"Epoch {self.current_epoch} | Step {self.global_step} | (Left: Input, Mid: Pred, Right: Target)"
    #         )

    #         wandb.log({"Validation/Reconstruction_Slices": wandb_image})

    @torch.no_grad()
    def log_samples(self, source: torch.FloatTensor, target: torch.FloatTensor = None, norm_factors=None):
        if self.trainer.global_rank == 0:
            self.noise_estimator.eval()
            with torch.no_grad():
                pred_delta = self.sample(
                    source,
                    condition=None, 
                    steps=50,
                    use_ddim=True,
                )
            self.noise_estimator.train()

            prediction = source + (pred_delta / 5.0)  # Reconstruction finale
            source, target = robust_patch_denormalization(source, target, norm_factors)
            prediction, _ = robust_patch_denormalization(prediction, prediction, norm_factors)
            # => SUV [p_min, p_max]
            
            self.log('val/mse', F.mse_loss(prediction, target), 
                batch_size=prediction.shape[0],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True
            )

            # Format actuel : (B, C, H, W) avec C=3
            # On veut visualiser le canal du milieu (la slice centrale du contexte)
            mid_chan_idx = source.shape[1] // 2  # Pour 3 canaux -> index 1
            
            # On extrait ce canal tout en gardant la dimension pour avoir (B, 1, H, W)
            source_slice = source[:, mid_chan_idx:mid_chan_idx + 1, :, :]
            pred_slice = prediction[:, mid_chan_idx:mid_chan_idx + 1, :, :]
            target_slice = target[:, mid_chan_idx:mid_chan_idx + 1, :, :]
            
            images_to_stack = [source_slice, pred_slice, target_slice]
        
            # Concaténation sur l'axe de la largeur (W est la dimension 3 dans B,C,H,W)
            batch_stack = torch.cat(images_to_stack, dim=3)
            SUV_DISPLAY_MAX = batch_stack.max().item()  # Pour normalisation [0, 1]
            display_grid = (batch_stack / SUV_DISPLAY_MAX)

            # 4. Dénormalisation ([-1, 1] -> [0, 255])
            # comparison_batch = (
            #     comparison_batch
            #     .clamp(-1, 1)
            #     .add(1)
            #     .div(2)
            #     .mul(255)
            #     .clamp(0, 255)
            #     .to(torch.uint8)
            # )

            # make_grid gère l'agencement
            grid = make_grid(display_grid, nrow=1, padding=2, normalize=False)

            # WandB attend (H, W, C), permutation nécessaire
            wandb_image = wandb.Image(
                grid.permute(1, 2, 0).cpu().numpy(), 
                caption=f"Epoch {self.current_epoch} | Step {self.global_step} | (Left: Input Mid-Ch, Mid: Pred Mid-Ch, Right: Target)"
            )

            wandb.log({"Validation/Reconstruction_Slices": wandb_image})


    def forward(
        self,
        x_t,
        t,
        condition=None,
        self_cond=None,
        guidance_scale=1.0,
        cold_diffusion=False,
        un_cond=None,
        **kwargs
    ):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition is not None) and (guidance_scale != 1.0):
            pred_uncond = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond)
            pred_cond   = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond = pred_uncond.chunk(2, dim=1)
                pred_cond, pred_var_cond = pred_cond.chunk(2, dim=1)
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            pred = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var = pred.chunk(2, dim=1)

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5  # [-1, 1] -> [0, 1]
            pred_var_value = pred_var
        else:
            pred_var_scale = 0
            pred_var_value = None

        # pred_var_scale = pred_var_scale.clamp(0, 1)

        if self.estimator_objective == "x_0":
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(
                x_t,
                t,
                pred,
                clip_x0=self.clip_x0,
                var_scale=pred_var_scale,
                cold_diffusion=cold_diffusion,
            )
            x_T = self.noise_scheduler.estimate_x_T(
                x_t, x_0=pred, t=t, clip_x0=self.clip_x0
            )
            self_cond = x_T

        elif self.estimator_objective == "x_T":
            if x_t.shape[1] != pred.shape[1]:
                x_t = x_t[:, :pred.shape[1]]
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(
                x_t,
                t,
                pred,
                clip_x0=self.clip_x0,
                var_scale=pred_var_scale,
                cold_diffusion=cold_diffusion,
            )
            x_T = pred
            self_cond = x_0
        else:
            raise ValueError("Unknown Objective")

        return x_t_prior, x_0, x_T, self_cond

    @torch.no_grad()
    def denoise(self, x_t, source, steps=None, condition=None, use_ddim=False, **kwargs):
        self_cond = None

        # ---------- run denoise loop ---------------
        steps = self.noise_scheduler.timesteps if steps is None else steps
        if use_ddim:
            timesteps_array = torch.linspace(
                0,
                self.noise_scheduler.T - 1,
                steps,
                dtype=torch.long,
                device=x_t.device,
            )  # [0, 1, 2, ..., T-1] if steps = T
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)] 

        verbose = kwargs.get("verbose", True) 

        # for i, t in tqdm(enumerate(reversed(timesteps_array))):
        iterator = enumerate(reversed(timesteps_array))
        if verbose:
            iterator = tqdm(iterator, total=steps, desc="Denoising", leave=False, position=0)
        
        for i, t in iterator:
            # UNet prediction
            x_t = torch.cat([x_t, source], dim=1)
            x_t, x_0, x_T, self_cond = self.forward(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, **kwargs)

            self_cond = self_cond if self.use_self_conditioning else None

            if use_ddim and (steps - i - 1 > 0):
                t_next = timesteps_array[steps - i - 2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = (
                    kwargs.get("eta", 1)
                    * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                )
                c = (1 - alpha_next - sigma**2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        return x_t  # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, source, condition=None, steps=None, use_ddim=False, **kwargs):
        # creating noise
        template = torch.zeros(source.shape, device=self.device)
        x_T = self.noise_scheduler.x_final(template)

        x_0 = self.denoise(
            x_T,
            source,
            steps=steps,
            condition=condition,
            use_ddim=use_ddim,
            **kwargs,
        )

        return x_0

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            list(self.noise_estimator.parameters()),
            **self.optimizer_kwargs,
        )

        if self.lr_scheduler is not None:
            lr_scheduler = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]



# On suppose que BasicModel gère l'héritage LightningModule
class UnlearningHarmonizationDiffusionPipeline(BasicModel):
    def __init__(
        self,
        iffn_encoder,             # Ton module IFFN (Encodeur F)
        domain_classifier,        # Ton module Classifieur (Mouchard)
        noise_scheduler,
        noise_estimator,          # Ton UNet de diffusion
        
        # --- Paramètres Dinsdale / Unlearning ---
        warmup_epochs=10,         # Correspond au 'epoch_stage_1' de Dinsdale
        beta_confusion=1.0,       # Poids de la confusion (args.beta)
        
        # Learning Rates séparés (Comme dans le papier)
        lr_main=1e-4,             # Encoder + UNet (Tâche)
        lr_dm=1e-4,               # Domain Classifier (Mouchard)
        lr_conf=1e-6,             # Unlearning (Encoder seulement, souvent plus faible)
        
        # --- Paramètres Diffusion existants ---
        estimator_objective="x_T",
        estimate_variance=False,
        use_self_conditioning=False,
        classifier_free_guidance_dropout=0.0,
        clip_x0=False,
        use_ema=False,
        ema_kwargs={},
        loss=torch.nn.L1Loss,
        loss_kwargs={},
        sample_every_n_steps=500,
        **kwargs
    ):
        # On passe None au parent car on gère les optimizers manuellement
        super().__init__(optimizer=None, optimizer_kwargs=None, lr_scheduler=None)
        
        self.iffn_encoder = iffn_encoder
        self.domain_classifier = domain_classifier
        self.noise_estimator = noise_estimator
        self.noise_scheduler = noise_scheduler
        
        # Hyperparams
        self.warmup_epochs = warmup_epochs
        self.beta = beta_confusion
        self.lrs = {"main": lr_main, "dm": lr_dm, "conf": lr_conf}
        
        self.loss_fct = loss(**loss_kwargs)
        self.estimator_objective = estimator_objective
        self.estimate_variance = estimate_variance
        self.use_self_conditioning = use_self_conditioning
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.clip_x0 = clip_x0
        self.sample_every_n_steps = sample_every_n_steps

        # EMA
        self.use_ema = use_ema
        if use_ema:
            # On applique l'EMA sur l'Encodeur et le UNet (la partie générative)
            self.ema_model = EMAModel(
                nn.ModuleList([self.iffn_encoder, self.noise_estimator]), 
                **ema_kwargs
            )

        # IMPORTANT : Optimisation Manuelle requise pour Dinsdale
        self.automatic_optimization = False
        
        self.save_hyperparameters(ignore=['noise_estimator', 'noise_scheduler', 'iffn_encoder', 'domain_classifier'])

    def configure_optimizers(self):
        """
        Configuration stricte des 3 optimizers de Dinsdale.
        """
        # 1. Main Optimizer : Tâche de reconstruction (Encoder + UNet)
        opt_main = torch.optim.Adam(
            list(self.iffn_encoder.parameters()) + list(self.noise_estimator.parameters()),
            lr=self.lrs["main"]
        )
        
        # 2. Domain Optimizer : Mouchard (Classifier seulement)
        opt_dm = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=self.lrs["dm"]
        )
        
        # 3. Confusion Optimizer : Unlearning (Encoder seulement)
        opt_conf = torch.optim.Adam(
            self.iffn_encoder.parameters(),
            lr=self.lrs["conf"]
        )
        
        return [opt_main, opt_dm, opt_conf]

    def training_step(self, batch, batch_idx):
        opt_main, opt_dm, opt_conf = self.optimizers()
        
        # Données
        source = batch['source'][tio.DATA]
        target = batch['target'][tio.DATA]
        source_domain = batch['domain_id'] # Label réel du scanner (Input pour la loss domaine)
        
        # Normalisation
        source, target, _ = robust_patch_normalization(source, target, percentiles=(0.5, 99.5))
        if self.clip_x0: target = torch.clamp(target, -1, 1)

        # Flag de phase : Stage 1 (Warmup) vs Stage 2 (Unlearning)
        is_stage_1 = self.current_epoch < self.warmup_epochs

        # ============================================================
        # PARTIE 1 : TACHE PRINCIPALE (Diffusion / Reconstruction)
        # ============================================================
        # Dans Dinsdale :
        # - Stage 1 : On update Encoder + UNet
        # - Stage 2 (Step 1) : On update Encoder + UNet
        # -> C'est le même code, seule la backprop change un peu à la fin
        
        # A. Forward Encodeur
        features = self.iffn_encoder(source)

        # B. Forward Diffusion (Task)
        target_placeholder = torch.clone(target)
        x_t, x_T, t = self.noise_scheduler.sample(target_placeholder)
        
        # Conditionnement du UNet :
        # - Contenu : Features F (via concaténation ou cross-attn)
        # - Style/Domaine : ID Cible (EARL). Si absent, on suppose 0 ou fix.
        target_domain_id = batch.get('target_domain_id', torch.zeros_like(source_domain))
        
        # Classifier Free Guidance logic
        if torch.rand(1) <= self.classifier_free_guidance_dropout:
            cond_input = torch.zeros_like(features)
            class_labels = None
        else:
            cond_input = features
            class_labels = target_domain_id

        # On suppose que ton UNet prend x_t concaténé avec F, et le label de classe
        unet_input = torch.cat([x_t, cond_input], dim=1)
        prediction = self.noise_estimator(unet_input, t, class_labels=class_labels)
        
        if self.estimate_variance: prediction, _ = prediction
            
        objective = x_T if self.estimator_objective == "x_T" else target
        task_loss = self.loss_fct(prediction, objective)

        # ============================================================
        # BRANCHEMENT : STAGE 1 (Apprentissage) vs STAGE 2 (Unlearning)
        # ============================================================
        
        if is_stage_1:
            # --- STAGE 1 : L'encodeur DOIT apprendre le domaine ---
            # On entraîne tout ensemble (Task + Domain Loss)
            
            # Forward Domain (On garde le graphe, pas de detach)
            domain_pred = self.domain_classifier(features)
            loss_dm = F.cross_entropy(domain_pred, source_domain)
            
            total_loss = task_loss + loss_dm
            
            # Update combinée (Encoder + UNet + Classifier)
            # On utilise opt_main et opt_dm
            opt_main.zero_grad()
            opt_dm.zero_grad()
            self.manual_backward(total_loss)
            opt_main.step()
            opt_dm.step()
            
            self.log("train/stage1_total_loss", total_loss, prog_bar=True)

        else:
            # --- STAGE 2 : UNLEARNING (Les 3 étapes dissociées) ---
            
            # 1. Update Task (Encoder + UNet) sur la Loss de reconstruction
            opt_main.zero_grad()
            self.manual_backward(task_loss)
            opt_main.step()
            
            # 2. Update Domain Classifier (Seul)
            # Dinsdale : "optimizer_dm.zero_grad(), output_dm = domain_predictor(features.detach())"
            # On doit détacher features pour ne pas toucher à l'encodeur ici
            for p in self.iffn_encoder.parameters(): p.requires_grad = False
            with torch.no_grad():
                self.iffn_encoder.eval()
                features_detached = self.iffn_encoder(source)
            
            domain_pred = self.domain_classifier(features_detached)
            loss_dm = F.cross_entropy(domain_pred, source_domain)
            
            opt_dm.zero_grad()
            self.manual_backward(loss_dm)
            opt_dm.step()
            
            # 3. Update Encoder (Confusion)
            # Dinsdale : "optimizer_conf.zero_grad(), loss_conf = beta * conf_criterion..."
            # On doit refaire un forward partiel car le graphe a été consommé
            for p in self.iffn_encoder.parameters(): p.requires_grad = True
            self.iffn_encoder.train()

            features_conf = self.iffn_encoder(source)
            domain_pred_conf = self.domain_classifier(features_conf)
            
            # Confusion Loss : KL Divergence vers distribution uniforme
            n_classes = domain_pred_conf.shape[1]
            uniform_target = torch.full_like(domain_pred_conf, 1.0 / n_classes)
            
            loss_conf = self.beta * F.kl_div(
                F.log_softmax(domain_pred_conf, dim=1),
                uniform_target,
                reduction='batchmean'
            )
            
            opt_conf.zero_grad()
            self.manual_backward(loss_conf)
            opt_conf.step()
            
            # Logs Stage 2
            self.log_dict({
                "train/task_loss": task_loss,
                "train/domain_loss_critic": loss_dm,
                "train/confusion_loss_encoder": loss_conf
            }, prog_bar=True)

        # --- Fin du Step : EMA & Logs communs ---
        if self.use_ema:
            # On update l'EMA de l'encodeur et du UNet
            self.ema_model.step(self.iffn_encoder)
            self.ema_model.step(self.noise_estimator)

        # Visualization Périodique
        if (self.global_step + 1) % self.sample_every_n_steps == 0:
            # On force la cible (ex: EARL=0) pour la visu
            vis_domain = torch.zeros(1, device=self.device).long()
            self.log_samples(source, target, vis_domain)

    # ---------------------------------------------------------
    # Méthodes Helper (Sample, Log, Forward Inférence)
    # ---------------------------------------------------------

    def forward(self, x_t, t, condition_features, target_domain_label=None, **kwargs):
        """Utilisé pour l'inférence/sampling"""
        # Sélection du modèle (EMA ou courant)
        if self.use_ema:
            # EMA contient une ModuleList [Encoder, UNet]
            unet = self.ema_model.averaged_model[1]
        else:
            unet = self.noise_estimator
            
        # Concaténation Features + Image Bruitée
        unet_input = torch.cat([x_t, condition_features], dim=1)
        
        # Prédiction avec conditionnement de classe
        return unet(unet_input, t, class_labels=target_domain_label)

    @torch.no_grad()
    def sample(self, source, target_domain_label=None, steps=None, use_ddim=False):
        # 1. Extraction Features
        enc = self.ema_model.averaged_model[0] if self.use_ema else self.iffn_encoder
        enc.eval()
        features = enc(source)
        
        # 2. Noise Loop
        template = torch.zeros_like(source)
        x_T = self.noise_scheduler.x_final(template)
        
        # On passe 'features' comme 'condition_features' au forward
        # et 'target_domain_label' comme 'target_domain_label'
        x_0 = self.denoise(
            x_T, 
            source=None, # Plus besoin de source brute dans denoise car intégrée dans forward via features
            steps=steps, 
            condition_features=features, # <--- Passage des features IFFN
            target_domain_label=target_domain_label,
            use_ddim=use_ddim
        )
        return x_0

    @torch.no_grad()
    def denoise(self, x_t, source, steps=None, condition_features=None, target_domain_label=None, use_ddim=False, **kwargs):
        # Réécriture légère de denoise pour appeler self.forward avec les bons args
        steps = self.noise_scheduler.timesteps if steps is None else steps
        if use_ddim:
            timesteps_array = torch.linspace(0, self.noise_scheduler.T - 1, steps, dtype=torch.long, device=x_t.device)
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)]

        for i, t in tqdm(enumerate(reversed(timesteps_array)), leave=False):
            t_tensor = t.expand(x_t.shape[0]).to(self.device)
            
            # Appel au forward adapté
            model_output = self.forward(
                x_t, 
                t_tensor, 
                condition_features=condition_features, 
                target_domain_label=target_domain_label
            )
            
            if self.estimate_variance: model_output, _ = model_output.chunk(2, dim=1)
            
            # ... Logique Scheduler standard (identique à ta classe originale) ...
            # Placeholder : x_t = scheduler.step(...)
            # Pour l'exemple, on suppose une fonction step simple :
            x_t = self.noise_scheduler.step(model_output, t, x_t) # Pseudo-code
            
        return x_t

    @torch.no_grad()
    def log_samples(self, source, target, target_domain_label):
        if self.trainer.global_rank == 0:
            prediction = self.sample(source, target_domain_label=target_domain_label, steps=50, use_ddim=True)
            
            # Logique d'affichage (Slices, WandB...)
            # Copie-colle ici ton code de visualisation slice 2D
            mid_slice = source.shape[2] // 2
            src_s = source[:, :, mid_slice, :, :]
            tgt_s = target[:, :, mid_slice, :, :]
            pred_s = prediction[:, :, mid_slice, :, :]
            
            grid = make_grid(torch.cat([src_s, pred_s, tgt_s], dim=3), nrow=1, normalize=True)
            wandb.log({"Validation/Samples": wandb.Image(grid.permute(1,2,0).cpu().numpy())})



