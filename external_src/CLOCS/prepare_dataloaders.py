from torch.utils.data import DataLoader
from prepare_dataset import MimicIVMemmapContrastiveDataset
import os
import wandb

def load_initial_data_contrastive(
    basepath_to_data,
    phases,
    fraction,
    inferences,
    batch_size,
    modality,
    acquired_indices,
    acquired_labels,
    modalities,
    dataset_name,
    input_perturbed=False,
    perturbation='Gaussian',
    leads='ii',
    labelled_fraction=1,
    unlabelled_fraction=1,
    downstream_task='contrastive',
    class_pair='',
    trial='CMC',
    nviews=1,
):
    """
    Backward-compatible wrapper.
    Uses fast memmap loader ONLY for MIMIC-IV contrastive_msml.
    """

    # --------------------------------------------------
    # FAST PATH: MIMIC-IV + memmap dataset
    # --------------------------------------------------

    shuffles = {
        'train1': True,
        'train2': False,
        'train': True,
        'val': False,
        'test': False
    }

    num_workers = 8

    dataset = {}
    for phase in phases:
        dataset[phase] = MimicIVMemmapContrastiveDataset(
            basepath_to_data=basepath_to_data,
            phase=phase,
            task=downstream_task,
            leads_dir="leads_all",
            fraction=1.0,
            trial=trial,
            nviews=nviews,
            input_perturbed=input_perturbed,
            perturbation=perturbation,
            dataset_name="mimiciv",
        )

    dataloader = {
        phase: DataLoader(
            dataset[phase],
            batch_size=batch_size,
            shuffle=shuffles.get(phase, False),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )
        for phase in phases
    }

    operations = {
        'resize': False,
        'affine': False,
        'rotation': False,
        'color': False,
        'perform_cutout': False
    }
    for phase in phases:
        wandb.log({phase: len(dataset[phase])})

    return dataloader, operations
