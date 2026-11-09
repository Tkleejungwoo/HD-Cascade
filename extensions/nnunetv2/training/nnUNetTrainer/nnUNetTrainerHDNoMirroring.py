import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.hausdorff_erosion_loss import DC_and_CE_and_HD_loss
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring
    
class nnUNetTrainerHDNoMirroring(nnUNetTrainerNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.hd_start_epoch = 1001
        self.hd_full_epoch = 2000
        self.hd_max_weight = 0.1

    def _build_loss(self):
        """
            Build the loss and retain it as self.my_custom_loss for weight updates.
        """

        if self.label_manager.ignore_label is not None:
            raise ValueError('The HD loss requires a dataset without ignore labels')

        if torch.distributed.is_initialized():
            do_batch_dice = self.configuration_manager.batch_dice
        else:
            do_batch_dice = False

        # Keep a reference to the loss so its HD weight can be updated each epoch.
        self.my_custom_loss = DC_and_CE_and_HD_loss(
            # Dice and cross-entropy settings.
            soft_dice_kwargs={'batch_dice': do_batch_dice, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={},
            hd_kwargs={'alpha': 2.0}, 
            
            # on_epoch_start updates the initially zero HD weight.
            weight_ce=1, 
            weight_dice=1, 
            weight_hd=0,  # Start with HD disabled.
            ignore_label=self.label_manager.ignore_label
        )
        
        # The deep-supervision wrapper shares this loss instance, so epoch-level
        # HD-weight updates apply to every eligible supervision scale.
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(self.my_custom_loss, weights)
        else:
            loss = self.my_custom_loss
            
        return loss

    def on_epoch_start(self):
        """
        Update the HD loss weight at the beginning of each epoch.
        """
        super().on_epoch_start()

        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBasePlusHDCube import hd_weight_for_epoch
        self.my_custom_loss.weight_hd = hd_weight_for_epoch(
            self.current_epoch, start_epoch=self.hd_start_epoch,
            full_epoch=self.hd_full_epoch, max_weight=self.hd_max_weight,
        )
        if self.current_epoch == 0:
            self.print_to_log_file(
                f'Normalized HD: epochs={self.num_epochs}; '
                f'start={self.hd_start_epoch}, full={self.hd_full_epoch}, '
                f'max={self.hd_max_weight}; cube=3x3x3; '
                'threshold=0.5; alpha=2; erosion_steps=10; '
                'spatial min-max normalization per sample/class'
            )

        # Log the current HD weight every ten epochs.
        if self.current_epoch % 10 == 0:
            self.print_to_log_file(f"Current HD Loss Weight: {self.my_custom_loss.weight_hd:.4f}")
