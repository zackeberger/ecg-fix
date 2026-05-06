import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class MimicIVMemmapContrastiveDataset(Dataset):
    """
    Fast memmap-backed dataset for MIMIC-IV ECG.

    Returns exactly:
      frame_views, label, pid, modality, dataset, true_index
    """

    def __init__(
        self,
        basepath_to_data: str,
        phase: str,
        leads_dir: str = "leads_all",
        task: str = "contrastive_msml",
        fraction: float = 1.0,
        trial: str = "CMSMLC",
        nviews: int = 12,
        input_perturbed: bool = False,
        perturbation: str = "Gaussian",
        dataset_name: str = "mimiciv",
    ):
        if "train" in phase:
            phase = "train"
        elif "val" in phase:
            phase = "val"

        self.phase = phase
        self.task = task
        self.trial = trial
        self.nviews = nviews
        self.input_perturbed = input_perturbed
        self.perturbation = perturbation
        self.dataset_name = dataset_name

        self.modality = "ecg"
        self.dataset_tag = dataset_name

        path = os.path.join(basepath_to_data, "preprocessed_mimic4ecg_clocs", "patient_data", "contrastive_msml", leads_dir)

        # read index (same file name as your old pipeline expects)
        with open(os.path.join(path, "frames_phases_mimiciv.pkl"), "rb") as f:
            index = pickle.load(f)

        node = index["ecg"][fraction]
        if phase == "train":
            info = node["train"]["labelled"]
        else:
            info = node["val"]

        assert info["format"] == "npy_memmap_v1", f"Unsupported format: {info['format']}"

        self.frames_path = os.path.join(path, info["frames"])
        self.labels_path = os.path.join(path, info["labels"])
        self.pids_path   = os.path.join(path, info["pids"])
        self.length = int(info["length"])

        # lazy-open memmaps inside each worker process
        self._frames = np.load(self.frames_path, mmap_mode="r")  # (N,5000,12) float16
        self._labels = np.load(self.labels_path, mmap_mode="r")  # (N,)
        self._pids   = np.load(self.pids_path,   mmap_mode="r")  # (N,)



    def __len__(self):
        return self.length

    # ---- Augmentation helpers (torch, vectorized) ----
    def _normalize_per_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T,V) normalize each view independently to [0,1]
        """
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        return (x - x_min) / (x_max - x_min + 1e-8)

    def _gaussian_noise(self, shape, device, variance_factor: float):
        return torch.randn(shape, device=device) * variance_factor

    # def _gaussian_noise(self, x, sigma=0.02):
    #     """
    #     x: torch.Tensor
    #     sigma: standard deviation of noise
    #     """
    #     return x + sigma * torch.randn_like(x)

    def __getitem__(self, index: int):

        frame_np = self._frames[index]           # float16 (5000,12)
      #  label = int(self._labels[index])         # int
       # pid   = int(self._pids[index])           # int

     #   true_index = index

        # float32 torch for compute
        frame = torch.tensor(frame_np, dtype=torch.float32)

        # -------------------------------
        # Match your original output API
        # -------------------------------
        if self.trial == "CMSC":
            seg1 = frame[:2500, :]    # (2500,12)
            seg2 = frame[2500:, :]    # (2500,12)

                # independent noise per view
        #    seg1 = self._gaussian_noise(seg1, sigma=0.02)
       #    seg2 = self._gaussian_noise(seg2, sigma=0.02)

            # stack as two views
            frame_views = torch.stack([seg1, seg2], dim=-1)  # (2500,12,2)
        #    frame_views = frame_views.unsqueeze(0)            # (1,2500,12,2)

        #    label_t = torch.tensor(label, dtype=torch.float32)


        elif self.trial == "CMSMLC":
            # Original code returns: (1,2500,nviews*2)
            # Order: for each lead -> seg0 then seg1
            # frame: (5000,12) -> (2,2500,12) -> (12,2,2500) -> (24,2500) -> (2500,24)
            segs = frame.view(2, 2500, 12).permute(2, 0, 1).reshape(12 * 2, 2500).transpose(0, 1)

            # apply perturbation + normalize per view (same distribution as looped version)
            if self.input_perturbed and "Gaussian" in self.perturbation:
                # mimic scaling: keep a reasonable default noise for MIMIC (you can tune)
                variance_factor = 10.0
                segs = segs + self._gaussian_noise(segs.shape, segs.device, variance_factor)

            segs = self._normalize_per_view(segs)
            frame_views = segs.unsqueeze(0)  # (1,2500,24)

            label_t = torch.tensor(label, dtype=torch.float32)

        elif self.trial == "CMLC":
            # Original: frame_views (1,2500,nviews) from (2500,12) single segment
            seg = frame[:2500, :]  # (2500,12)

            if self.input_perturbed and "Gaussian" in self.perturbation:
                variance_factor = 10.0
                seg = seg + self._gaussian_noise(seg.shape, seg.device, variance_factor)

            seg = self._normalize_per_view(seg)
            frame_views = seg.unsqueeze(0)  # (1,2500,12)
            label_t = torch.tensor(label, dtype=torch.float32)

        elif self.trial in ["CMC", "SimCLR"]:
            # Original: (1,nsamples,nviews) where each view is a perturbed copy
            # For mimic multi-lead, most people don't use this branch, but we keep API.
            nsamples = frame.shape[0]
            # collapse leads if needed (match your old behavior: expects 1D)
            # pick lead II as default if multi-lead
            if frame.ndim == 2:
                one = frame[:, 1]  # II
            else:
                one = frame

            views = []
            for _ in range(self.nviews):
                x = one.clone()

                if self.input_perturbed and "Gaussian" in self.perturbation:
                    variance_factor = 10.0
                    x = x + self._gaussian_noise(x.shape, x.device, variance_factor)

                # normalize [0,1]
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                views.append(x)

            frame_views = torch.stack(views, dim=1).unsqueeze(0)  # (1,nsamples,nviews)
            label_t = torch.tensor(label, dtype=torch.float32)

        elif self.trial in ["Linear", "Fine-Tuning", "Random"]:
            # Original: frame (1,5000) then unsqueeze(2) => (1,5000,1)
            # pick lead II for linear heads
            x = frame[:, 1]  # II
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            x = x.unsqueeze(0)          # (1,5000)
            frame_views = x.unsqueeze(2) # (1,5000,1)
            label_t = torch.tensor(label, dtype=torch.float32)

        else:
            # fallback: behave like Linear on lead II
            x = frame[:, 1]
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            frame_views = x.unsqueeze(0).unsqueeze(2)
            label_t = torch.tensor(label, dtype=torch.float32)

        return frame_views #, label_t, pid, true_index
