import torch
import torch.nn as nn

from benchmark.models.encoder.resnet_merl import MerlResNet18
from benchmark.models.encoder.utils import (
    ButterBandpassFilter,
    ECGDatasetNormalize,
    ECGInterpolator,
    ECGNormalize,
)
from benchmark.models.encoder.vit_merl import vit_middle


class RandomEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pipeline = []
        self.Hz = None
        self.dataset_norm = None  # 👈 keep reference

        # -------------------
        # Sampling rate
        # -------------------
        if "100Hz" in args.model:
            self.Hz = 100
        elif "250Hz" in args.model:
            self.Hz = 250
        elif "500Hz" in args.model:
            self.Hz = 500
        else:
            raise ValueError("define Hz for random")

        # Interpolator
        if "Hz" in args.model:
            self.pipeline.append(ECGInterpolator(self.Hz))

        # Bandpass FIRST
        if "Add_Bandpass" in args.model:
            self.pipeline.append(ButterBandpassFilter(self.Hz))

        # Normalization
        if "Z_score_sample" in args.model:
            self.pipeline.append(ECGNormalize())

        elif "Z_score_dataset" in args.model:
            self.dataset_norm = ECGDatasetNormalize()  # 👈 keep it
            self.pipeline.append(self.dataset_norm)

        # Wrap preprocessing
        self.preprocess = nn.Sequential(*self.pipeline)

        # -------------------
        # Encoder
        # -------------------
        if "Resnet18" in args.model:
            self.encoder = MerlResNet18()
        elif "Vit" in args.model:
            self.encoder = vit_middle(12, seq_len=self.Hz * 10)
        else:
            raise ValueError("Unknown encoder")

    def set_stats(self, mean, std):
        """
        Set dataset normalization stats
        """
        if self.dataset_norm is None:
            raise ValueError("Model does not use dataset normalization")

        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)

        if mean.ndim == 1:
            mean = mean.view(1, -1, 1)
        if std.ndim == 1:
            std = std.view(1, -1, 1)

        self.dataset_norm.set_stats(mean, std)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        return x
