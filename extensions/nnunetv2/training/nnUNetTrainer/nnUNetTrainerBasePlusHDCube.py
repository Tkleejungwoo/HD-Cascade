"""
Base nnU-Net plus the normalized cube HD loss, with 2,000 training epochs.

The base trainer builds Dice/CE (including batch_dice, compilation and DDP)
and deep supervision unchanged. Only an HD term is added at each eligible scale.
The HD implementation and zero-based weight schedule retain the established training settings.
"""

import math

import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.hausdorff_erosion_loss import HausdorffErosionLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def hd_weight_for_epoch(epoch: int, *, start_epoch: int = 1001,full_epoch: int = 2000, max_weight: float = 0.1) -> float:
    # Compute the HD weight from explicit zero-based schedule settings.
    if start_epoch < 0 or full_epoch <= start_epoch:
        raise ValueError('HD schedule requires 0 <= start_epoch < full_epoch')
    if not math.isfinite(max_weight) or max_weight < 0:
        raise ValueError('HD max_weight must be finite and nonnegative')
    if epoch < start_epoch:
        return 0.0
    if epoch < full_epoch:
        return max_weight * (epoch - start_epoch) / (full_epoch - start_epoch)
    return max_weight


class BaseDiceCEPlusHDCube(nn.Module):
    
    # Retain the exact Base loss module; add the normalized HD component only.
    def __init__(self, base_loss, ignore_label=None):
        super().__init__()
        # Reject ignore labels because the HD loss does not exclude ignored regions.
        if ignore_label is not None:
            raise ValueError('This Base+HD ablation requires a dataset without ignore labels')
        
        self.base_loss = base_loss
        self.hd = HausdorffErosionLoss(
            alpha=2.0, erosion_steps=10, kernel_size=3, kernel_type='cube'
        )
        self.weight_hd = 0.0

    def forward(self, net_output, target):
        dc_ce = self.base_loss(net_output, target)
        if self.weight_hd == 0:
            return dc_ce
        if net_output.shape[2] < 32 or net_output.shape[3] < 32:
            return dc_ce
        return dc_ce + self.weight_hd * self.hd(net_output, target)


# Standard mirroring/architecture/optimizer/split; Dice+CE plus cube HD.
class nnUNetTrainerBasePlusHDCube(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        if configuration != '3d_fullres':
            raise ValueError('This ablation is defined for 3d_fullres, without a cascade')
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
        self.hd_start_epoch = 1001
        self.hd_full_epoch = 2000
        self.hd_max_weight = 0.1

    def _build_loss(self):
        loss = super()._build_loss()
        if isinstance(loss, DeepSupervisionWrapper):
            component = BaseDiceCEPlusHDCube(loss.loss, self.label_manager.ignore_label)
            loss.loss = component
        else:
            component = BaseDiceCEPlusHDCube(loss, self.label_manager.ignore_label)
            loss = component
        self.base_hd_component = component
        return loss

    def on_epoch_start(self):
        super().on_epoch_start()
        self.base_hd_component.weight_hd = hd_weight_for_epoch(
            self.current_epoch,
            start_epoch=self.hd_start_epoch,
            full_epoch=self.hd_full_epoch,
            max_weight=self.hd_max_weight,
        )
        if self.current_epoch == 0:
            self.print_to_log_file(
                f'epochs={self.num_epochs}; '
                f'HD schedule: start={self.hd_start_epoch}, '
                f'full={self.hd_full_epoch}, max={self.hd_max_weight}; '
                f'batch_dice={self.configuration_manager.batch_dice}'
            )
        if self.current_epoch % 10 == 0 or self.current_epoch == self.num_epochs - 1:
            self.print_to_log_file(
                f'Base+HD epoch_index={self.current_epoch}: '
                f'HD weight={self.base_hd_component.weight_hd:.10f}'
            )
