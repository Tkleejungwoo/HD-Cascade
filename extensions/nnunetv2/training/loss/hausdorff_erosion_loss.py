"""Erosion-based Hausdorff surrogate and Dice + CE + HD compound loss."""

import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class HausdorffErosionLoss(nn.Module):
    def __init__(self, alpha=2.0, erosion_steps=10, kernel_size=3, kernel_type='cube'):
        super(HausdorffErosionLoss, self).__init__()
        self.alpha = alpha
        self.erosion_steps = erosion_steps
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type

        # Construct the 3x3x3 convolution kernel.
        if kernel_type == 'cube':
            # Include the center and all 26 neighbors, including diagonals.
            kernel_3d = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
            
        else:
            # The cross kernel includes the center and six face neighbors.
            kernel_3d = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
            center = 1
            kernel_3d[0, 0, center, center, center] = 1.0       # Center.
            kernel_3d[0, 0, center, center, center - 1] = 1.0   # D-
            kernel_3d[0, 0, center, center, center + 1] = 1.0   # D+
            kernel_3d[0, 0, center, center - 1, center] = 1.0   # H-
            kernel_3d[0, 0, center, center + 1, center] = 1.0   # H+
            kernel_3d[0, 0, center - 1, center, center] = 1.0   # W-
            kernel_3d[0, 0, center + 1, center, center] = 1.0   # W+
        
        # Normalize the kernel to unit sum.
        self.kernel = kernel_3d / kernel_3d.sum()

        print(f"\n[HD Loss Setup] Initialized with kernel type: {self.kernel_type.upper()}")
        print(f"[HD Loss Setup] Alpha: {self.alpha}, Erosion Steps: {self.erosion_steps}\n")


    @staticmethod
    def _normalize_error_map(error):
        # Min-max normalize spatial values independently per sample and class.
        spatial_dims = tuple(range(2, error.ndim))
        minimum = error.amin(dim=spatial_dims, keepdim=True)
        span = error.amax(dim=spatial_dims, keepdim=True) - minimum

        has_range = span > 0
        denominator = torch.where(has_range, span, torch.ones_like(span))
        normalized = (error - minimum) / denominator
        
        # Preserve constant maps and avoid division by zero in both branches.
        return torch.where(has_range, normalized, error)


    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        with torch.no_grad():
            target_onehot = torch.zeros_like(pred)
            target_onehot.scatter_(1, target.long(), 1)
        
        # Base error map: (p - q)^2 
        current_error = (pred - target_onehot) ** 2
        loss = 0
        
        # Move the kernel to the prediction device.
        if self.kernel.device != pred.device:
            self.kernel = self.kernel.to(pred.device)

        # Expand the kernel for grouped convolution over (B, C, D, H, W).
        num_channels = pred.shape[1]
        weight = self.kernel.expand(num_channels, 1, 3, 3, 3)

        for k in range(1, self.erosion_steps + 1):
            # Apply convolution-based soft erosion.
            # 1. Convolution with normalized kernel
            current_error = F.conv3d(
                current_error, 
                weight, 
                padding=1, 
                groups=num_channels
            )
            
            # 2. Soft Thresholding at 0.50 
            # Soft-threshold the averaged error: max(x - 0.5, 0).
            current_error = F.relu(current_error - 0.5)
            current_error = self._normalize_error_map(current_error)
            # Weight the normalized result of the kth erosion.
            loss += current_error.mean() * (k ** self.alpha)
        return loss
    

class DC_and_CE_and_HD_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, hd_kwargs=None, 
                 weight_ce=1, weight_dice=1, weight_hd=0, ignore_label=None, 
                 dice_class=MemoryEfficientSoftDiceLoss, kernel_type='cube'):
        
        super(DC_and_CE_and_HD_loss, self).__init__(
            soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label, 
            dice_class=dice_class
        )

        # Preserve the existing setting of ten erosion iterations.
        self.weight_hd = weight_hd
        self.hd = HausdorffErosionLoss(alpha=2.0, erosion_steps=10, kernel_type=kernel_type)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # Compute the base Dice + cross-entropy loss.
        dc_ce_loss = super().forward(net_output, target)
        
        # Skip HD computation while its weight is zero.
        if self.weight_hd == 0:
            return dc_ce_loss

        # Preserve the existing HD scale-selection rule for deep supervision.
        if net_output.shape[2] < 32 or net_output.shape[3] < 32:
            return dc_ce_loss

        if self.ignore_label is not None:
            target_hd = torch.clone(target)
            target_hd[target == self.ignore_label] = 0
        else:
            target_hd = target

        hd_loss = self.hd(net_output, target_hd)
        return dc_ce_loss + (self.weight_hd * hd_loss)
