from typing import *
import numpy as np
import contextlib
import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock

from .base import BasicBlock, BasicDown, BasicUp, UnetResBlock, UnetBasicBlock, zero_module
from .attention import Attention


class Interp1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
            assert len(vec.shape) <= 2, "interp1d: all inputs must be " "at most 2-D."
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, "All parameters must be on the same device."
        device = device[0]

        # Checking for the dimensions
        assert v["x"].shape[1] == v["y"].shape[1] and (
            v["x"].shape[0] == v["y"].shape[0]
            or v["x"].shape[0] == 1
            or v["y"].shape[0] == 1
        ), (
            "x and y must have the same number of columns, and either "
            "the same number of row or one of them having only one "
            "row."
        )

        reshaped_xnew = False
        if (
            (v["x"].shape[0] == 1)
            and (v["y"].shape[0] == 1)
            and (v["xnew"].shape[0] > 1)
        ):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v["xnew"].shape
            v["xnew"] = v["xnew"].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v["x"].shape[0], v["xnew"].shape[0])
        shape_ynew = (D, v["xnew"].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v["xnew"].shape[0] == 1:
            v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes
        torch.searchsorted(
            v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind
        )

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ["x", "y", "xnew"]:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [
                    None,
                ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat["slopes"] = is_flat["x"]
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
                eps + (v["x"][:, 1:] - v["x"][:, :-1])
            )

            # now build the linear interpolation
            ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        # self.save_for_backward(ynew, *saved_inputs)
        return ynew
      

class FFTHighPassFilter(nn.Module):
    def __init__(self, input_shape, sigma: float = 10.0, dim=(2, 3)):
        super(FFTHighPassFilter, self).__init__()
        self.dim = dim
        self.center = (input_shape[0] // 2, input_shape[1] // 2)
        self.mask = self.create_gaussian_disc_mask(input_shape, sigma)
        self.interp = Interp1d()

    def create_gaussian_disc_mask(self, input_shape, sigma):
        """Create a Gaussian disc high-pass filter mask."""
        mask = torch.zeros(input_shape)
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                # Calculate the distance from the center of the frequency domain
                dist = ((i - self.center[0]) ** 2 + (j - self.center[1]) ** 2) ** 0.5
                # Gaussian disc mask: smooth transition across the cutoff
                mask[i, j] = 1 - np.exp(-0.5 * (dist / sigma) ** 2)
        return mask

    def calculate_2dft(self, input: torch.FloatTensor):
        """Compute 2D FFT of the input."""
        ft = torch.fft.ifftshift(input, dim=self.dim)
        ft = torch.fft.fft2(ft, dim=self.dim)
        return torch.fft.fftshift(ft, dim=self.dim)

    def calculate_2dift(self, input: torch.FloatTensor):
        """Compute inverse 2D FFT of the input."""
        ift = torch.fft.ifftshift(input, dim=self.dim)
        ift = torch.fft.ifft2(ift, dim=self.dim)
        ift = torch.fft.fftshift(ift, dim=self.dim)
        return ift.real

    def histogram_equalization(self, x: torch.FloatTensor):
        hist = torch.histc(x, bins=256, min=0, max=1)
        bins = torch.linspace(0, 1, 256).to(x.device)
        cdf = hist.cumsum(0).to(x.device)
        cdf_normalized = cdf * hist.max() / cdf.max()
        return self.interp.forward(bins, cdf_normalized, x, out=None)

    def min_max_normalization(self, x: torch.FloatTensor):
        return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x

    def forward(self, x: torch.FloatTensor):
        """Apply the Gaussian high-pass filter to the input."""
        fft_image = self.calculate_2dft(x)
        mask_repeated = self.mask.repeat(x.shape[0], x.shape[1], 1, 1).to(
            fft_image.device
        )
        fft_images = torch.abs(self.calculate_2dift(fft_image * mask_repeated))

        for b_idx in range(fft_images.shape[0]):
            for c_idx in range(fft_images.shape[1]):
                fft_images[b_idx, c_idx] = self.min_max_normalization(
                    fft_images[b_idx, c_idx]
                )
                fft_images[b_idx, c_idx] = self.histogram_equalization(
                    fft_images[b_idx, c_idx]
                )

        return fft_images


        
class ImageFrequencyFusionModel(nn.Module):
    def __init__(
        self,
        input_shape,
        input_channels: int,
        output_channels: int,
        spatial_dims: int,
        hidden_channels: List[int],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [2, 1, 1, 1],
        use_residual_block=True,
        learnable_interpolation=True,
        use_attention="none",
        num_residual_blocks=2,
        **kwargs
    ):
        super().__init__()
        assert (
            len(hidden_channels) == len(kernel_sizes) == len(strides)
        ), "The length of hidden_channels, kernel_sizes, and strides must be the same."

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_residual_block = use_residual_block
        self.num_residual_blocks = num_residual_blocks

        self.depth = strides.__len__()
        use_attention = (
            use_attention
            if isinstance(use_attention, list)
            else [use_attention] * len(strides)
        )
        norm = ("GROUP", {"num_groups": 32, "affine": True})
        act_name = ("SWISH", {})

        convolution = UnetResBlock if use_residual_block else UnetBasicBlock
        self.high_pass_filter = FFTHighPassFilter(input_shape=input_shape, sigma=7.5)

        convolution = UnetResBlock if self.use_residual_block else UnetBasicBlock

        # Architecture
        # strides are taking into acount in the down and up blocks (encoder and decoder)
        self.in_conv = BasicBlock(
            spatial_dims,
            self.input_channels,
            hidden_channels[0],
            kernel_size=kernel_sizes[0],
            stride=1,
        )

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            for k in range(self.num_residual_blocks):
                self.encoders.append(
                    nn.Sequential(
                        convolution(
                            spatial_dims=spatial_dims,
                            in_channels=hidden_channels[
                                i - 1 if k == 0 and i > 0 else i
                            ],
                            out_channels=hidden_channels[i],
                            kernel_size=kernel_sizes[i],
                            stride=1,
                            norm_name=norm,
                            act_name=act_name,
                        ),
                        Attention(
                            spatial_dims=spatial_dims,
                            in_channels=hidden_channels[i],
                            out_channels=hidden_channels[i],
                            num_heads=8,
                            ch_per_head=hidden_channels[i] // 8,
                            depth=1,
                            norm_name=norm,
                            attention_type=use_attention[i],
                        ),
                    )
                )

            if strides[i] > 1:
                self.encoders.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation,
                    )
                )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            convolution(
                spatial_dims=spatial_dims,
                in_channels=hidden_channels[-1],
                out_channels=hidden_channels[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm,
                act_name=act_name,
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hidden_channels[-1],
                out_channels=hidden_channels[-1],
                num_heads=8,
                ch_per_head=hidden_channels[-1] // 8,
                depth=1,
                norm_name=norm,
                attention_type=use_attention[-1],
            ),
            convolution(
                spatial_dims=spatial_dims,
                in_channels=hidden_channels[-1],
                out_channels=hidden_channels[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm,
                act_name=act_name,
            ),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(0, self.depth)):
            if strides[i] > 1:
                self.decoders.append(
                    BasicUp(
                        spatial_dims=spatial_dims,
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i],
                        kernel_size=strides[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation,
                    )
                )

            for k in range(self.num_residual_blocks):
                out_channels = hidden_channels[
                    i - 1 if k == self.num_residual_blocks - 1 and i > 0 else i
                ]
                self.decoders.append(
                    nn.Sequential(
                        convolution(
                            spatial_dims=spatial_dims,
                            in_channels=hidden_channels[i]
                            + hidden_channels[i]
                            + self.input_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_sizes[i],
                            stride=1,
                            norm_name=norm,
                            act_name=act_name,
                        ),
                        Attention(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            num_heads=8,
                            ch_per_head=out_channels // 8,
                            depth=1,
                            norm_name=norm,
                            attention_type=use_attention[i],
                        ),
                    )
                )

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(
                hidden_channels[0],
                hidden_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            zero_module(
                UnetOutBlock(
                    spatial_dims, hidden_channels[0], self.output_channels, dropout=None
                )
            ),
        )

    def forward(self, x):
        fft_x = self.high_pass_filter(x)

        h = self.in_conv(x)

        feature_maps = []
        for encoder in self.encoders:
            h = encoder(h)
            if not isinstance(encoder, BasicDown):
                feature_maps.append(h)

        h = self.bottleneck(h)

        for decoder in self.decoders:
            if not isinstance(decoder, BasicUp):
                fft_x = torch.nn.functional.interpolate(
                    fft_x, size=h.shape[-2:], mode="bilinear", align_corners=False
                )
                h = torch.cat([h, feature_maps.pop(), fft_x], dim=1)
            h = decoder(h)

        return self.output(h)