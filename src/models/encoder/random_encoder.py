import torch
import torch.nn as nn

from src.models.encoder.resnet_merl import MerlResNet18
from src.models.encoder.utils import (
    ButterBandpassFilter,
    ECGDatasetNormalize,
    ECGInterpolator,
    ECGNormalize,
)
from src.models.encoder.vit_merl import vit_middle


class RandomEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.pipeline = []
        self.Hz = None
        self.dataset_norm = None  # 👈 keep reference

        # -------------------
        # Sampling rate
        # -------------------
        if "100Hz" in model_name:
            self.Hz = 100
        elif "250Hz" in model_name:
            self.Hz = 250
        elif "500Hz" in model_name:
            self.Hz = 500
        else:
            raise ValueError("define Hz for random")

        # Interpolator
        if "Hz" in model_name:
            self.pipeline.append(ECGInterpolator(self.Hz))

        # Bandpass FIRST
        if "Add_Bandpass" in model_name:
            self.pipeline.append(ButterBandpassFilter(self.Hz))

        # Normalization
        if "Z_score_sample" in model_name:
            self.pipeline.append(ECGNormalize())

        elif "Z_score_dataset" in model_name:
            self.dataset_norm = ECGDatasetNormalize() 
            self.pipeline.append(self.dataset_norm)

        # Wrap preprocessing
        self.preprocess = nn.Sequential(*self.pipeline)

        # -------------------
        # Encoder
        # -------------------
        if "Resnet18" in model_name:
            self.encoder = MerlResNet18()
        elif "Vit" in model_name:
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
