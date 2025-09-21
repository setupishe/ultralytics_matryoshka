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

        # Build weights (default to equal weights if none provided)
        num_sets = len(feats_list)
        weights = getattr(self.hyp, "matryoshka_weights", None)
        if weights is None:
            weights_tensor = torch.ones(num_sets, device=self.device) / max(num_sets, 1)
        else:
            # Convert to tensor and normalize to sum to 1; fall back to equal if shape mismatch
            try:
                weights_tensor = torch.as_tensor(
                    weights, dtype=torch.float32, device=self.device
                )
                if weights_tensor.numel() != num_sets:
                    weights_tensor = torch.ones(num_sets, device=self.device) / max(
                        num_sets, 1
                    )
                else:
                    s = weights_tensor.sum()
                    if s <= 0:
                        weights_tensor = torch.ones(num_sets, device=self.device) / max(
                            num_sets, 1
                        )
                    else:
                        weights_tensor = weights_tensor / s
            except Exception:
                weights_tensor = torch.ones(num_sets, device=self.device) / max(
                    num_sets, 1
                )

        # Calculate and accumulate loss for each set of feature maps
        for i, f in enumerate(feats_list):
            # v8_loss returns (scalar loss for backprop, tensor of 3 losses for display)
            scalar_loss, detached_losses = self.v8_loss(f, batch)
            w = weights_tensor[i]
            total_loss += scalar_loss * w
            loss_items += detached_losses * w

        return total_loss, loss_items
