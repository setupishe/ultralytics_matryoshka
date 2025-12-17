# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


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

    def _loss_with_shared_assign(
        self,
        feats,
        batch_size,
        anchor_points,
        stride_tensor,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """Compute v8DetectionLoss given precomputed assignment from (typically) full-width predictions."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.v8_loss.no, -1) for xi in feats], 2
        ).split((self.v8_loss.reg_max * 4, self.v8_loss.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype

        # Cls loss (same targets for all granularities)
        loss[1] = (
            self.v8_loss.bce(pred_scores, target_scores.to(dtype)).sum()
            / target_scores_sum
        )

        # Bbox/DFL loss
        pred_bboxes = self.v8_loss.bbox_decode(
            anchor_points, pred_distri
        )  # xyxy, (b, h*w, 4)
        if fg_mask.sum():
            # Avoid in-place modification of shared target_bboxes
            target_bboxes_scaled = target_bboxes / stride_tensor
            loss[0], loss[2] = self.v8_loss.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes_scaled,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        # Apply gains
        loss[0] *= self.v8_loss.hyp.box
        loss[1] *= self.v8_loss.hyp.cls
        loss[2] *= self.v8_loss.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

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
        shared_assign = bool(getattr(self.hyp, "matryoshka_shared_assign", False))
        if shared_assign and num_sets > 1:
            # Compute assignment ONCE from the full-width predictions (last granularity).
            feats_full = feats_list[-1]
            pred_distri_f, pred_scores_f = torch.cat(
                [
                    xi.view(feats_full[0].shape[0], self.v8_loss.no, -1)
                    for xi in feats_full
                ],
                2,
            ).split((self.v8_loss.reg_max * 4, self.v8_loss.nc), 1)
            pred_scores_f = pred_scores_f.permute(0, 2, 1).contiguous()
            pred_distri_f = pred_distri_f.permute(0, 2, 1).contiguous()

            dtype = pred_scores_f.dtype
            batch_size = pred_scores_f.shape[0]
            imgsz = (
                torch.tensor(feats_full[0].shape[2:], device=self.device, dtype=dtype)
                * self.v8_loss.stride[0]
            )  # image size (h,w)
            anchor_points, stride_tensor = make_anchors(
                feats_full, self.v8_loss.stride, 0.5
            )

            targets = torch.cat(
                (
                    batch["batch_idx"].view(-1, 1),
                    batch["cls"].view(-1, 1),
                    batch["bboxes"],
                ),
                1,
            )
            targets = self.v8_loss.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            pred_bboxes_f = self.v8_loss.bbox_decode(
                anchor_points, pred_distri_f
            )  # xyxy
            _, target_bboxes, target_scores, fg_mask, _ = self.v8_loss.assigner(
                pred_scores_f.detach().sigmoid(),
                (pred_bboxes_f.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            target_scores_sum = max(target_scores.sum(), 1)

            for i, f in enumerate(feats_list):
                scalar_loss, detached_losses = self._loss_with_shared_assign(
                    f,
                    batch_size,
                    anchor_points,
                    stride_tensor,
                    target_bboxes,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )
                w = weights_tensor[i]
                total_loss += scalar_loss * w
                loss_items += detached_losses * w
        else:
            for i, f in enumerate(feats_list):
                # v8_loss returns (scalar loss for backprop, tensor of 3 losses for display)
                scalar_loss, detached_losses = self.v8_loss(f, batch)
                w = weights_tensor[i]
                total_loss += scalar_loss * w
                loss_items += detached_losses * w

        return total_loss, loss_items
