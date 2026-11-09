"""
Train a no-mirroring HD Cascade with normalized erosion maps for 2,000 epochs.

This adapter is new release glue, used for the exported inference checkpoints.
The network architecture and optimizer are inherited from nnU-Net.
"""
import torch

from nnunetv2.training.lr_scheduler.hd_cascade_polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHDNoMirroring import (
    nnUNetTrainerHDNoMirroring,
)


class nnUNetTrainerHDCascade2000(nnUNetTrainerHDNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000

    def configure_optimizers(self):
        optimizer, _ = super().configure_optimizers()
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
