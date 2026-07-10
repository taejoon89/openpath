# Copyright (c) 2026 OpenPath authors.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Clean-room re-implementation of the gram-anchoring loss (technique from DINOv3,
# Siméoni et al., 2025). Written from the mathematical description only — the MSE
# between the (optionally L2-normalized) patch-token Gram / self-similarity matrices
# of the student and a frozen anchor — with no reference to the original DINOv3
# source, so this file carries the same Apache-2.0 license as the rest of the fork.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """Gram-anchoring loss.

    For patch-token features from the student and a frozen anchor, build each model's
    per-image Gram matrix (token-by-token self-similarity) and penalize their squared
    difference. This anchors the student's *relational* (dense/patch) structure to the
    anchor's while DINO/iBOT keep optimizing the global CLS representation.

    Args:
        apply_norm: L2-normalize tokens along the feature dim before the Gram product,
            so each Gram entry is a cosine similarity in [-1, 1].
        img_level: build one Gram per image, shape (B, N, N) (default path).
        remove_neg: clamp negative similarities to 0 before the comparison.
        remove_only_teacher_neg: when clamping, clamp the anchor's Gram only.
    """

    def __init__(self, apply_norm=True, img_level=True, remove_neg=True,
                 remove_only_teacher_neg=False):
        super().__init__()
        self.apply_norm = apply_norm
        self.img_level = img_level
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

    @staticmethod
    def _gram(x):
        # x: (..., N, D) -> (..., N, N) self-similarity
        return x @ x.transpose(-1, -2)

    def forward(self, student_tokens, teacher_tokens, img_level=None):
        img_level = self.img_level if img_level is None else img_level

        xs = student_tokens.float()
        xt = teacher_tokens.float()
        if self.apply_norm:
            xs = F.normalize(xs, dim=-1)
            xt = F.normalize(xt, dim=-1)

        if not img_level:
            # collapse the batch into one token set (rarely used)
            xs = xs.reshape(-1, xs.shape[-1])
            xt = xt.reshape(-1, xt.shape[-1])

        gs = self._gram(xs)
        gt = self._gram(xt)

        if self.remove_neg:
            gt = gt.clamp(min=0.0)
            if not self.remove_only_teacher_neg:
                gs = gs.clamp(min=0.0)

        return F.mse_loss(gs, gt)
