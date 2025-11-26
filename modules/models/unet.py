from typing import *
import numpy as np
import contextlib
import torch
import torch.nn as nn
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
        sample_size: Union[Tuple[int, int], int] = 128,
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
        self.sample_size = sample_size

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

