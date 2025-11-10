from typing import Optional, Sequence, Tuple, Union, Type

from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl

from monai.networks.blocks.dynunet_block import get_padding, get_output_padding
from monai.networks.layers import Pool, Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer, get_dropout_layer
from monai.utils.misc import ensure_tuple_rep

from modules.models.attention import Attention
from modules.models.attention import Attention, zero_module


class AdaGroupNorm(nn.Module):
    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[bool] = False, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn:
            self.act = nn.SiLU()
        else:
            self.act = None

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x

class ResnetBlockCondNorm2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        conv_shortcut_bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        conv_cls = nn.Conv2d

        self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups, eps=eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states, temb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, temb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor
    

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        time_embedding_norm: str = "default",  # default, scale_shift,
        output_scale_factor: float = 1.0,
        conv_shortcut_bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        linear_cls = nn.Linear
        conv_cls = nn.Conv2d

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor = None) -> torch.FloatTensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None and temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor

# ------------------------------------
    
class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        norm: bool = False,
        eps: float = 1e-6,
        elementwise_affine: bool = None,
        bias: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        conv_cls = nn.Conv2d

        if norm is None:
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        else:
            self.norm = None

        if use_conv:
            self.conv = conv_cls(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride)


    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        hidden_states = self.conv(hidden_states)
        return hidden_states
    

class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 512,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default", # or "scale_shift"
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if downsample:
            self.downsampler = Downsample2D(
                out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding
            )
        else:
            self.downsampler = None

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)

        return hidden_states
    
# ------------------------------------
    
class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding: int = 1,
        norm: bool = False,
        eps: float = 1e-6,
        elementwise_affine = None,
        bias: bool = True,
        interpolate: bool = True
    ):
        super().__init__()
        assert not (use_conv and use_conv_transpose), "Only one of `use_conv` and `use_conv_transpose` can be True"
        assert not (interpolate and use_conv_transpose), "interpolate and use_conv_transpose cannot be True at the same time"
        assert (use_conv_transpose or interpolate), "At least one of `use_conv_transpose` and `interpolate` should be True"

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.interpolate = interpolate
        conv_cls = nn.Conv2d

        if norm:
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)

        else:
            self.norm = None

        self.conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)


    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


    
class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 512,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # "scale_shift"
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        upsample: bool = True
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if upsample:
            self.upsampler = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsampler = None


    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states

    
# ------------------------------------
    
class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_num_heads: int = 8,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                output_scale_factor=output_scale_factor
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        spatial_dims=2,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        ch_per_head=in_channels // attention_num_heads,
                        num_heads=attention_num_heads,
                        norm_name=("GROUP", {'num_groups': resnet_groups, "affine": True}),
                        dropout=dropout,
                        emb_dim=temb_channels,
                        attention_type='linear'
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states
    

####################################################
########### Adapted from medfusion + monai #########
####################################################


class VeryBasicModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._step_train = 0
        self._step_val = 0
        self._step_test = 0
        self.validation_step_outputs = []

    def forward(self, x_in):
        raise NotImplementedError

    def _step(self, batch, batch_idx, split, step):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch, batch_idx):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val)

    def test_step(self, batch, batch_idx):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def _epoch_end(self, outputs, split):
        return

    def on_training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / "best_checkpoint.json", "w") as f:
            json.dump({"best_model_epoch": Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, version=0, **kwargs):
        path_version = "lightning_logs/version_" + str(version)
        with open(
            Path(path_checkpoint_dir) / path_version / "best_checkpoint.json", "r"
        ) as f:
            path_rel_best_checkpoint = Path(json.load(f)["best_model_epoch"])
        return Path(path_checkpoint_dir) / path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, version=0, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(
            path_checkpoint_dir, version
        )
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get("filter", lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {
            key: value for key, value in pretrained_weights.items() if filter(key)
        }
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self


class BasicModel(VeryBasicModel):
    def __init__(
        self,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-2},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin,
            nin,
            kernel_size=kernel_size,
            padding=padding,
            groups=nin,
            bias=bias,
            stride=stride,
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def save_add(*args):
    args = [arg for arg in args if arg is not None]
    return sum(args) if len(args) > 0 else None


class SequentialEmb(nn.Sequential):
    def forward(self, input, emb):
        for module in self:
            input = module(input, emb)
        return input


class BasicDown(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        learnable_interpolation=True,
        use_res=False,
    ) -> None:
        super().__init__()

        if learnable_interpolation:
            Convolution = Conv[Conv.CONV, spatial_dims]
            self.down_op = Convolution(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=get_padding(kernel_size, stride),
                dilation=1,
                groups=1,
                bias=True,
            )

            if use_res:
                self.down_skip = nn.PixelUnshuffle(
                    2
                )  # WARNING: Only supports 2D, , out_channels == 4*in_channels

        else:
            Pooling = Pool["avg", spatial_dims]
            self.down_op = Pooling(
                kernel_size=kernel_size,
                stride=stride,
                padding=get_padding(kernel_size, stride),
            )

    def forward(self, x, emb=None):
        y = self.down_op(x)
        if hasattr(self, "down_skip"):
            y = y + self.down_skip(x)
        return y


class BasicUp(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        learnable_interpolation=True,
        use_res=False,
    ) -> None:
        super().__init__()
        self.learnable_interpolation = learnable_interpolation
        if learnable_interpolation:
            # TransConvolution = Conv[Conv.CONVTRANS, spatial_dims]
            # padding = get_padding(kernel_size, stride)
            # output_padding = get_output_padding(kernel_size, stride, padding)
            # self.up_op = TransConvolution(
            #     in_channels,
            #     out_channels,
            #     kernel_size=kernel_size,
            #     stride=stride,
            #     padding=padding,
            #     output_padding=output_padding,
            #     groups=1,
            #     bias=True,
            #     dilation=1
            # )

            self.calc_shape = lambda x: tuple(
                (np.asarray(x) - 1) * np.atleast_1d(stride)
                + np.atleast_1d(kernel_size)
                - 2 * np.atleast_1d(get_padding(kernel_size, stride))
            )
            Convolution = Conv[Conv.CONV, spatial_dims]
            self.up_op = Convolution(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
            )

            if use_res:
                self.up_skip = nn.PixelShuffle(
                    2
                )  # WARNING: Only supports 2D, out_channels == in_channels/4
        else:
            self.calc_shape = lambda x: tuple(
                (np.asarray(x) - 1) * np.atleast_1d(stride)
                + np.atleast_1d(kernel_size)
                - 2 * np.atleast_1d(get_padding(kernel_size, stride))
            )

    def forward(self, x, emb=None):
        if self.learnable_interpolation:
            new_size = self.calc_shape(x.shape[2:])
            x_res = F.interpolate(x, size=new_size, mode="nearest-exact")
            y = self.up_op(x_res)
            if hasattr(self, "up_skip"):
                y = y + self.up_skip(x)
            return y
        else:
            new_size = self.calc_shape(x.shape[2:])
            return F.interpolate(x, size=new_size, mode="nearest-exact")


class BasicBlock(nn.Module):
    """
    A block that consists of Conv-Norm-Drop-Act, similar to blocks.Convolution.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        zero_conv: zero out the parameters of the convolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int] = 1,
        norm_name: Union[Tuple, str, None] = None,
        act_name: Union[Tuple, str, None] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        zero_conv: bool = False,
    ):
        super().__init__()
        Convolution = Conv[Conv.CONV, spatial_dims]
        conv = Convolution(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=get_padding(kernel_size, stride),
            dilation=1,
            groups=1,
            bias=True,
        )
        self.conv = zero_module(conv) if zero_conv else conv

        if norm_name is not None:
            self.norm = get_norm_layer(
                name=norm_name, spatial_dims=spatial_dims, channels=out_channels
            )
        if dropout is not None:
            self.drop = get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        if act_name is not None:
            self.act = get_act_layer(name=act_name)

    def forward(self, inp):
        out = self.conv(inp)
        if hasattr(self, "norm"):
            out = self.norm(out)
        if hasattr(self, "drop"):
            out = self.drop(out)
        if hasattr(self, "act"):
            out = self.act(out)
        return out


class BasicResBlock(nn.Module):
    """
        A block that consists of Conv-Act-Norm + skip.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        zero_conv: zero out the parameters of the convolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int] = 1,
        norm_name: Union[Tuple, str, None] = None,
        act_name: Union[Tuple, str, None] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        zero_conv: bool = False,
    ):
        super().__init__()
        self.basic_block = BasicBlock(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm_name,
            act_name,
            dropout,
            zero_conv,
        )
        Convolution = Conv[Conv.CONV, spatial_dims]
        self.conv_res = (
            Convolution(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=get_padding(1, stride),
                dilation=1,
                groups=1,
                bias=True,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inp):
        out = self.basic_block(inp)
        residual = self.conv_res(inp)
        out = out + residual
        return out


class UnetBasicBlock(nn.Module):
    """
    A modified version of monai.networks.blocks.UnetBasicBlock with additional embedding

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        emb_channels: Number of embedding channels
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int] = 1,
        norm_name: Union[Tuple, str] = None,
        act_name: Union[Tuple, str] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        emb_channels: int = None,
        sccale_shift_norm: bool = True,
        blocks=2,
    ):
        super().__init__()
        self.scale_shift_norm = sccale_shift_norm
        self.block_seq = nn.ModuleList(
            [
                BasicBlock(
                    spatial_dims,
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    norm_name,
                    act_name,
                    dropout,
                    i == blocks - 1,
                )
                for i in range(blocks)
            ]
        )

        if emb_channels is not None:
            self.local_embedder = nn.Sequential(
                get_act_layer(name=act_name),
                nn.Linear(emb_channels, out_channels * 2 if sccale_shift_norm else out_channels),
            )

    def forward(self, x, emb=None):
        # ------------ Embedding ----------
        if emb is not None:
            emb = self.local_embedder(emb)
            b, c, *_ = emb.shape
            sp_dim = x.ndim - 2
            emb = emb.reshape(b, c, *((1,) * sp_dim))

        # ----------- Convolution ---------
        n_blocks = len(self.block_seq)
        for i, block in enumerate(self.block_seq):
            x = block(x)
            if (emb is not None) and i < n_blocks:
                if self.scale_shift_norm:
                    scale, shift = emb.chunk(2, dim=1)
                    x = x * (scale + 1) + shift
                else:
                    x += emb
        return x


class UnetResBlock(nn.Module):
    """
    A modified version of monai.networks.blocks.UnetResBlock with additional skip connection and embedding

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        emb_channels: Number of embedding channels
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int] = 1,
        norm_name: Union[Tuple, str] = None,
        act_name: Union[Tuple, str] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        emb_channels: int = None,
        scale_shift_norm: bool = True,
        blocks: int = 2,
    ):
        super().__init__()
        self.scale_shift_norm = scale_shift_norm
        self.block_seq = nn.ModuleList(
            [
                BasicResBlock(
                    spatial_dims,
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    norm_name,
                    act_name,
                    dropout,
                    i == blocks - 1,
                )
                for i in range(blocks)
            ]
        )

        if emb_channels is not None:
            self.local_embedder = nn.Sequential(
                get_act_layer(name=act_name),
                nn.Linear(emb_channels, out_channels * 2 if scale_shift_norm else out_channels),
            )

    def forward(self, x, emb=None):
        # ------------ Embedding ----------
        if emb is not None:
            emb = self.local_embedder(emb)
            b, c, *_ = emb.shape
            sp_dim = x.ndim - 2
            emb = emb.reshape(b, c, *((1,) * sp_dim))

        # ----------- Convolution ---------
        n_blocks = self.block_seq.__len__()
        for i, block in enumerate(self.block_seq):
            x = block(x)
            if (emb is not None) and i < n_blocks - 1:
                if self.scale_shift_norm:
                    scale, shift = emb.chunk(2, dim=1)
                    x = x * (scale + 1) + shift
                else:
                    x += emb
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        downsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_res_block: bool = False,
        learnable_interpolation: bool = True,
        use_attention: str = "none",
        emb_channels: int = None,
    ):
        super(DownBlock, self).__init__()
        enable_down = ensure_tuple_rep(stride, spatial_dims) != ensure_tuple_rep(
            1, spatial_dims
        )
        down_out_channels = (
            out_channels if learnable_interpolation and enable_down else in_channels
        )

        # -------------- Down ----------------------
        self.down_op = (
            BasicDown(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=downsample_kernel_size,
                stride=stride,
                learnable_interpolation=learnable_interpolation,
                use_res=False,
            )
            if enable_down
            else nn.Identity()
        )

        # ---------------- Attention -------------
        self.attention = Attention(
            spatial_dims=spatial_dims,
            in_channels=down_out_channels,
            out_channels=down_out_channels,
            num_heads=8,
            ch_per_head=down_out_channels // 8,
            depth=1,
            norm_name=norm_name,
            dropout=dropout,
            emb_dim=emb_channels,
            attention_type=use_attention,
        )

        # -------------- Convolution ----------------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.conv_block = ConvBlock(
            spatial_dims,
            down_out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            emb_channels=emb_channels,
        )

    def forward(self, x, emb=None):
        # ----------- Down ---------
        x = self.down_op(x)

        # ----------- Attention -------------
        if self.attention is not None:
            x = self.attention(x, emb)

        # ------------- Convolution --------------
        x = self.conv_block(x, emb)

        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_res_block: bool = False,
        learnable_interpolation: bool = True,
        use_attention: str = "none",
        emb_channels: int = None,
        skip_channels: int = 0,
    ):
        super(UpBlock, self).__init__()
        enable_up = ensure_tuple_rep(stride, spatial_dims) != ensure_tuple_rep(
            1, spatial_dims
        )
        skip_out_channels = (
            out_channels
            if learnable_interpolation and enable_up
            else in_channels + skip_channels
        )
        self.learnable_interpolation = learnable_interpolation

        # -------------- Up ----------------------
        self.up_op = (
            BasicUp(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=upsample_kernel_size,
                stride=stride,
                learnable_interpolation=learnable_interpolation,
                use_res=False,
            )
            if enable_up
            else nn.Identity()
        )

        # ---------------- Attention -------------
        self.attention = Attention(
            spatial_dims=spatial_dims,
            in_channels=skip_out_channels,
            out_channels=skip_out_channels,
            num_heads=8,
            ch_per_head=skip_out_channels // 8,
            depth=1,
            norm_name=norm_name,
            dropout=dropout,
            emb_dim=emb_channels,
            attention_type=use_attention,
        )

        # -------------- Convolution ----------------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.conv_block = ConvBlock(
            spatial_dims,
            skip_out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            emb_channels=emb_channels,
        )

    def forward(self, x_enc, x_skip=None, emb=None):
        # ----------- Up -------------
        x = self.up_op(x_enc)

        # ----------- Skip Connection ------------
        if x_skip is not None:
            if (
                self.learnable_interpolation
            ):  # Channel of x_enc and x_skip are equal and summation is possible
                x = x + x_skip
            else:
                x = torch.cat((x, x_skip), dim=1)

        # ----------- Attention -------------
        if self.attention is not None:
            x = self.attention(x, emb)

        # ----------- Convolution ------------
        x = self.conv_block(x, emb)

        return x
