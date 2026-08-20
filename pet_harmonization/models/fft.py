from typing import Tuple
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, input_shape, in_channels, learnable=True, sigma: float = 7.5, dim=(2, 3)):
        super(FFTHighPassFilter, self).__init__()
        self.dim = dim
        self.center = (input_shape[0] // 2, input_shape[1] // 2)
        self.interp = Interp1d()
        
        if learnable:
            H, W      = input_shape
            freq_h    = H
            freq_w    = W // 2 + 1      # taille de la sortie rfft2

            # Masque gaussien passe-haut pour initialisation (même logique que FFTHighPassFilter)
            cy, cx    = H / 2, W / 2
            y         = torch.arange(H).float() - cy
            x         = torch.arange(freq_w).float()    # rfft → demi-spectre
            yy, xx    = torch.meshgrid(y, x, indexing='ij')
            dist      = torch.sqrt(yy**2 + xx**2)
            gauss_mask = torch.exp(-dist**2 / (2 * sigma**2))
            hp_mask   = 1.0 - gauss_mask                # passe-haut

            # W : (C, H, freq_w) — un filtre réel par canal, initialisé avec hp_mask
            # On travaille avec un poids réel (magnitude) appliqué sur le spectre complexe
            # → plus stable à entraîner qu'un poids complexe
            self.W = nn.Parameter(
                hp_mask.unsqueeze(0).expand(in_channels, -1, -1).clone()
            )   # (C, H, freq_w)
        else:
            self.W = self.create_gaussian_disc_mask(input_shape, sigma)

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
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-9)
        return self.interp.forward(bins, cdf_normalized, x, out=None)

    def min_max_normalization(self, x: torch.FloatTensor):
        """Normalise x vers [0, 1]. Si constant → retourne x tel quel."""
        x_min = x.min()
        x_max = x.max()
        if x_max == x_min:
            return x
        return (x - x_min) / (x_max - x_min)

    # @torch.compiler.disable
    def forward(self, x: torch.FloatTensor):
        """Apply the Gaussian high-pass filter to the input."""
        fft_image = self.calculate_2dft(x)
        mask_repeated = self.W.repeat(x.shape[0], x.shape[1], 1, 1).to(fft_image.device)
        fft_images = torch.abs(self.calculate_2dift(fft_image * mask_repeated))

        for b_idx in range(fft_images.shape[0]):
            for c_idx in range(fft_images.shape[1]):
                fft_images[b_idx, c_idx] = self.min_max_normalization(fft_images[b_idx, c_idx])
                fft_images[b_idx, c_idx] = self.histogram_equalization(fft_images[b_idx, c_idx])

        return fft_images * 2.0 - 1.0  # rescale to [-1, 1]


class LearnableFFTHighPassFilter(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        in_channels: int,
        learnable: bool = True,
        sigma: float = 7.5,
        dim: Tuple[int, int] = (2, 3),
    ):
        super().__init__()
        self.dim = dim
        H, W = input_shape

        # Masque gaussien passe-haut centré — cohérent avec fft2 + fftshift
        # DC au centre → dist calculée depuis (H/2, W/2)
        cy, cx = H / 2.0, W / 2.0
        y = torch.arange(H).float() - cy
        x = torch.arange(W).float() - cx       # spectre COMPLET (H, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist      = torch.sqrt(yy ** 2 + xx ** 2)
        gauss     = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        hp_mask   = 1.0 - gauss                 # passe-haut : supprime les basses fréq

        if learnable:
            # (C, H, W) — un filtre réel par canal
            # Initialisé avec hp_mask, le réseau peut affiner le filtrage
            self.W = nn.Parameter(
                hp_mask.unsqueeze(0).expand(in_channels, -1, -1).clone()
            )
        else:
            self.register_buffer('W', hp_mask)  # (H, W) fixe, pas de gradient

        self.learnable = learnable
        self.interp    = Interp1d()

    def calculate_2dft(self, x: torch.Tensor) -> torch.Tensor:
        ft = torch.fft.ifftshift(x, dim=self.dim)
        ft = torch.fft.fft2(ft, dim=self.dim)
        return torch.fft.fftshift(ft, dim=self.dim)     # (B, C, H, W) complexe, centré

    def calculate_2dift(self, x: torch.Tensor) -> torch.Tensor:
        ift = torch.fft.ifftshift(x, dim=self.dim)
        ift = torch.fft.ifft2(ift, dim=self.dim)
        return torch.fft.fftshift(ift, dim=self.dim).real

    def histogram_equalization(self, x: torch.Tensor) -> torch.Tensor:
        hist = torch.histc(x, bins=256, min=0, max=1)
        bins = torch.linspace(0, 1, 256, device=x.device)
        cdf  = hist.cumsum(0)
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-9)
        return self.interp(bins, cdf_normalized, x, out=None)

    def min_max_normalization(self, x: torch.Tensor) -> torch.Tensor:
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return x
        return (x - x_min) / (x_max - x_min)

    # @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        fft_x = self.calculate_2dft(x)         # (B, C, H, W) complexe

        # Broadcast du masque (C, H, W) → (1, C, H, W) — pas de repeat inutile
        mask = self.W.unsqueeze(0)              # (1, C, H, W)

        filtered = torch.abs(self.calculate_2dift(fft_x * mask))   # (B, C, H, W) réel

        # Normalisation par batch et canal
        out = torch.zeros_like(filtered)
        for b in range(filtered.shape[0]):
            for c in range(filtered.shape[1]):
                img = self.min_max_normalization(filtered[b, c])
                out[b, c] = self.histogram_equalization(img)

        return out * 2.0 - 1.0     # [0, 1] → [-1, 1]
    
    