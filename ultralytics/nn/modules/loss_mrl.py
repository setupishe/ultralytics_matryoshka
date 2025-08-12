# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss


class MatryoshkaDetectionLoss:
    """
    Wrapper for detection loss calculation with support for Matryoshka learning.
    It uses V8DetectionLoss internally.
    """

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the MatryoshkaDetectionLoss class with the given model."""
        self.device = next(model.parameters()).device
        self.hyp = model.args
        self.v8_loss = v8DetectionLoss(model)

    def __call__(self, preds, batch):
        """
        Calculates the loss for detection models.
        """
        total_loss = 0.0
        loss_items = torch.zeros(3, device=self.device)

        # Determine the list of feature maps to process
        feats_list = []
        if (
            isinstance(preds, list) and preds and isinstance(preds[0], list)
        ):  # Matryoshka training
            feats_list = preds
        else:  # Validation or standard training
            # The original loss function expects the feature maps, which are the second element in validation's tuple output
            feats = preds[1] if isinstance(preds, tuple) and len(preds) == 2 else preds
            feats_list = [feats]

        # Calculate and accumulate loss for each set of feature maps
        for f in feats_list:
            # v8_loss returns (scalar loss for backprop, tensor of 3 losses for display)
            scalar_loss, detached_losses = self.v8_loss(f, batch)
            total_loss += scalar_loss / len(feats_list)
            loss_items += detached_losses / len(feats_list)

        return total_loss, loss_items
