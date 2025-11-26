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
import pytorch_lightning as pl
import wandb

# from modules.models.base import BasicModel
from modules.ema import EMAModel  # TODO
from modules.utils import kl_gaussians  # TODO

from modules.models.unet import UNet
from modules.scheduler import GaussianNoiseScheduler
from modules.models.base import BasicModel

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
        input, target, results = batch, {}

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
        common_feature_estimator,
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
        self.common_feature_estimator = common_feature_estimator
        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0
        self.sample_every_n_steps = sample_every_n_steps

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)

        self.save_hyperparameters()

    def _step(self, batch, batch_idx, state, step):
        condition = None  # condition is made channel-wise
        (masked_x_0, x_0), results = batch, {}
        targets = torch.clone(x_0)

        if self.clip_x0:
            x_0 = torch.clamp(x_0, -1, 1)

        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0, T-1] and compute x_t (noisy version of x_0 at t)
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
                pred, pred_vertical = noise_estimator(x_t, t, condition, None)
                if self.estimate_variance:
                    pred, _ = pred.chunk(
                        2, dim=1
                    )  # Seperate actual prediction and variance estimation
                if self.estimator_objective == "x_T":  # self condition on x_0
                    self_cond = self.noise_scheduler.estimate_x_0(
                        x_t, pred, t=t, clip_x0=self.clip_x0
                    )
                elif self.estimator_objective == "x_0":  # self condition on x_T
                    self_cond = self.noise_scheduler.estimate_x_T(
                        x_t, pred, t=t, clip_x0=self.clip_x0
                    )
                else:
                    raise NotImplementedError(
                        f"Option estimator_target={self.estimator_objective} not supported."
                    )

        # Classifier free guidance
        if torch.rand(1) <= self.classifier_free_guidance_dropout:
            condition = None

        # Run Denoise: (1) predicting common feature map
        masked_x_0_features = self.common_feature_estimator(
            masked_x_0
        )  # => one channel feature map
        x_t = torch.cat(
            [x_t, masked_x_0_features], dim=1
        )  # condition is the second half of the tensor
        pred = noise_estimator(x_t, t, condition)
        pred_vertical = []

        # Separate variance (scale) if it was learned
        if self.estimate_variance:
            pred, pred_var = pred.chunk(
                2, dim=1
            )  # Separate actual prediction and variance estimation

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
        interpolation_mode = "area"
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
            self.log_samples(targets[0, None, ...])

        return loss

    def log_samples(self, images: torch.FloatTensor, *kwargs: Tuple[Any, ...]):
        if self.trainer.global_rank == 0:
            outputs = self.sample(
                images,
                condition=None,
                steps=self.noise_scheduler.timesteps,
                use_ddim=False,
            )

            spatial_stack = lambda x: torch.cat(
                [
                    torch.vstack([img for img in x[:, idx, ...]])
                    for idx in range(x.shape[1])
                ],
                dim=0,
            )

            outputs = (
                outputs.clamp(-1, 1)
                .add(1)
                .div(2)
                .mul(255)
                .clamp(0, 255)
                .to(torch.uint8)
            )
            images = images.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
            sample = spatial_stack(torch.cat([images, outputs], dim=1))

            wandb.log(
                {
                    "Reconstruction examples": wandb.Image(
                        sample.detach().cpu().numpy(),
                        caption="({}) [{} - {}]".format(
                            self.trainer.global_step, sample.min(), sample.max()
                        ),
                    )
                }
            )

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
            # Model prediction
            pred_uncond = noise_estimator(
                x_t, t, condition=un_cond, self_cond=self_cond
            )
            pred_cond = noise_estimator(
                x_t, t, condition=condition, self_cond=self_cond
            )
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond = pred_uncond.chunk(2, dim=1)
                pred_cond, pred_var_cond = pred_cond.chunk(2, dim=1)
                pred_var = pred_var_uncond + guidance_scale * (
                    pred_var_cond - pred_var_uncond
                )
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
    def denoise(
        self, x_t, features, steps=None, condition=None, use_ddim=False, **kwargs
    ):
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
            x_t = torch.cat([x_t, features], dim=1)
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

        return x_t  # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, images, condition=None, steps=None, use_ddim=False, **kwargs):
        # creating noise
        template = torch.zeros(images.shape, device=self.device)
        x_T = self.noise_scheduler.x_final(template)

        # extracting common feature map
        images = self.channel_masking(images, masking_ratio=0.5)
        masked_x_0_features = self.common_feature_estimator(images)

        x_0 = self.denoise(
            x_T,
            masked_x_0_features,
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
            list(self.noise_estimator.parameters())
            + list(self.common_feature_estimator.parameters()),
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

    def channel_masking(
        self, image: torch.FloatTensor, masking_ratio
    ) -> torch.FloatTensor:
        for n_idx in range(image.shape[0]):
            for c_idx in range(image.shape[1]):
                if np.random.rand() < masking_ratio:
                    image[n_idx, c_idx] = -1
        return image
