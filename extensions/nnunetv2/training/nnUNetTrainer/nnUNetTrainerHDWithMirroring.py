"""
Use standard nnU-Net mirroring augmentation with the existing HD loss.

``nnUNetTrainerHDNoMirroring`` inherits from ``nnUNetTrainerNoMirroring``,
which disables mirroring augmentation. This class preserves its loss and
epoch-dependent HD-weight schedule while using the base nnUNetTrainer
implementation for mirroring configuration.

"""

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHDNoMirroring import (
    nnUNetTrainerHDNoMirroring,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerHDWithMirroring(nnUNetTrainerHDNoMirroring):
    """Combine HD loss with standard nnU-Net mirroring augmentation."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnU-Net records constructor argument names in checkpoints. Match the
        # base trainer's explicit signature instead of using *args and **kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Extend the default 1,000 epochs to 2,000 to accommodate the original
        self.num_epochs = 2000

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # Bypass NoMirroring and use the base nnU-Net mirroring configuration.
        return nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(
            self
        )
