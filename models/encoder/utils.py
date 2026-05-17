import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

class ButterBandpassFilter(nn.Module):
    def __init__(self,fs, lowcut=.5, highcut=40.,  order=5):
        super().__init__()
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # Precompute filter coefficients
        sos = butter(
            order,
            [lowcut, highcut],
            fs=fs,
            btype="band",
            output="sos",
        )
        self.register_buffer("sos", torch.tensor(sos, dtype=torch.float32))

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T) = (batch, leads, time)
        """
        B, C, T = x.shape

        # Move to CPU for scipy (required)
        x_np = x.detach().cpu().numpy()

        y = np.zeros_like(x_np)

        # Apply filter per batch + channel
        for b in range(B):
            for c in range(C):
                y[b, c] = sosfiltfilt(
                    self.sos.cpu().numpy(),
                    x_np[b, c],
                )

        # Back to torch
        y = torch.from_numpy(y).to(x.device)

        return y


class ECGNormalize(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T)
        Normalizes each (B, C) signal over time dimension T
        """
        mean = x.mean(dim=-1, keepdim=True)   # (B, C, 1)
        std = x.std(dim=-1, keepdim=True)     # (B, C, 1)

        x_norm = (x - mean) / (std + self.eps)
        return x_norm


class ECGDatasetNormalize(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", None)
        self.register_buffer("std", None)

    def set_stats(self, mean, std):
        """
        mean, std: tensors shaped (1, C, 1) or broadcastable
        """
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("Dataset mean/std not set")

        return (x - self.mean) / (self.std + self.eps)

class HeartLangNormalize(nn.Module):
    def __init__(self, new_min=-3.0, new_max=3.0, eps=1e-6):
        super().__init__()
        self.new_min = new_min
        self.new_max = new_max
        self.eps = eps

    def forward(self, x):
        # x: [B, C, L]

        B = x.shape[0]

        # flatten per sample
        x_flat = x.view(B, -1)

        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)

        x = (x - min_val) / (max_val - min_val + self.eps)
        x = x * (self.new_max - self.new_min) + self.new_min

        return x

class ECGInterpolator(nn.Module):
    def __init__(self, target_fs, duration_sec=10.0, mode="linear"):
        super().__init__()
        self.target_fs = target_fs
        self.duration_sec = duration_sec
        self.target_len = int(target_fs * duration_sec)
        self.mode = mode

    def forward(self, x):
        """
        x: Tensor (B, C, T)
        returns: (B, C, target_len)
        """
        # F.interpolate expects (B, C, T) for 1D already
        x_interp = F.interpolate(
            x,
            size=self.target_len,
            mode=self.mode,
            align_corners=True if self.mode in ["linear"] else None,
        )
        return x_interp
