"""R2-Net v2: two-stage point-cloud denoising model."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..ops import (
    farthest_point_sampling,
    knn_group,
    knn_points,
    random_rotation_matrix,
    rotate_points,
)
from .feature import FeatureExtraction
from .score import ScoreNet


class DenoiseNet(nn.Module):
    """Feature extractor + conditional residual refinement network."""

    def __init__(
        self,
        frame_knn: int = 32,
        num_train_points: int = 128,
        num_clean_nbs: int = 4,
        score_net_hidden_dim: int = 128,
        score_net_num_blocks: int = 4,
        conv_channels: int = 24,
        num_convs: int = 4,
        num_fc_layers: int = 3,
        growth_rate: int = 12,
        conv_knn: int = 16,
        use_oc: bool = True,
        use_ace: bool = True,
        candidate_ratio: float = 1.5,
        ace_reduction: int = 8,
    ):
        super().__init__()
        self.frame_knn = int(frame_knn)
        self.num_train_points = int(num_train_points)
        self.num_clean_nbs = int(num_clean_nbs)
        self.feature_net = FeatureExtraction(
            conv_channels=conv_channels,
            num_convs=num_convs,
            num_fc_layers=num_fc_layers,
            growth_rate=growth_rate,
            knn=conv_knn,
            use_oc=use_oc,
            use_ace=use_ace,
            candidate_ratio=candidate_ratio,
            ace_reduction=ace_reduction,
        )
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            dim=3,
            out_dim=3,
            hidden_size=score_net_hidden_dim,
            num_blocks=score_net_num_blocks,
        )

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------
    def get_supervised_loss(
        self,
        pcl_noisy: torch.Tensor,
        pcl_clean: torch.Tensor,
        *,
        loss_type: str = "weighted_huber",
        rbf_sigma: float = 0.02,
        alpha: float = 0.5,
        huber_delta: float = 1.0,
        loss_scale: float = 100.0,
    ) -> torch.Tensor:
        """Denoising score/displacement matching loss (paper Eq. 15-19).

        The target is ``-v_ij`` where ``v_ij`` is the mean noisy-to-clean
        displacement.  For the weighted variant ``w = exp(-||dp||^2/(2 sigma^2))``
        and ``w_tilde = alpha + (1-alpha) w`` (paper Eq. 16-17).
        ``loss_scale`` is a constant optimisation scale; it does not change the
        minimiser of the objective.
        """
        B, N_noisy, d = pcl_noisy.shape
        N_clean = pcl_clean.size(1)
        if d != 3:
            raise ValueError("point clouds must be xyz")
        n_centres = min(self.num_train_points, N_noisy)
        pnt_idx = torch.randperm(N_noisy, device=pcl_noisy.device)[:n_centres]

        feat_all = self.feature_net(pcl_noisy)             # (B,N,F)
        feat = feat_all[:, pnt_idx, :]                     # (B,n,F)
        F = feat.size(-1)

        # Local noisy frame and clean nearest neighbours.
        _, _, frames = knn_points(
            pcl_noisy[:, pnt_idx, :], pcl_noisy,
            K=self.frame_knn, return_nn=True,
        )                                                  # (B,n,K,3)
        frames_centered = frames - pcl_noisy[:, pnt_idx, :].unsqueeze(2)
        _, _, clean_nbs = knn_points(
            frames.reshape(-1, self.frame_knn, d),
            pcl_clean.unsqueeze(1).expand(B, n_centres, N_clean, d).reshape(-1, N_clean, d),
            K=self.num_clean_nbs, return_nn=True,
        )                                                  # (B*n,K,C,3)
        clean_nbs = clean_nbs.reshape(B, n_centres, self.frame_knn, self.num_clean_nbs, d)
        v = (frames.unsqueeze(3) - clean_nbs).mean(dim=3)  # (B,n,K,3)

        grad_pred = self.score_net(
            x=frames_centered.reshape(-1, self.frame_knn, d),
            c=feat.reshape(-1, F),
        ).reshape(B, n_centres, self.frame_knn, d)

        diff = grad_pred + v
        abs_diff = diff.abs()
        huber = torch.where(
            abs_diff <= huber_delta,
            0.5 * diff.pow(2),
            huber_delta * abs_diff - 0.5 * huber_delta ** 2,
        )                                                  # (B,n,K,3)

        if loss_type == "plain_huber":
            weighted = huber.sum(dim=-1)
        elif loss_type == "weighted_huber":
            # w depends on ||p_j - p_i||^2 of the noisy local frame.
            dist2 = frames_centered.pow(2).sum(dim=-1)     # (B,n,K)
            w = torch.exp(-dist2 / (2.0 * max(float(rbf_sigma), 1e-8) ** 2))
            w_tilde = float(alpha) + (1.0 - float(alpha)) * w
            weighted = huber.sum(dim=-1) * w_tilde
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        loss = weighted.mean()
        if loss_scale is not None and loss_scale > 0:
            loss = loss * float(loss_scale)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def denoise_langevin_dynamics(
        self,
        pcl_noisy: torch.Tensor,
        *,
        step_size: float = 0.2,
        denoise_knn: int = 4,
        step_decay: float = 0.95,
        num_steps: int = 30,
        noise_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, list]:
        """Iterative conditional refinement.

        ``noise_scale=0`` gives deterministic displacement refinement, which is
        the v2 default because it gives better CD in our experiments.
        ``noise_scale=1`` adds the paper's ``sqrt(2 s_t) eps`` Langevin term.
        """
        if pcl_noisy.dim() != 3:
            raise ValueError("pcl_noisy must be (B,N,3)")
        B, N, d = pcl_noisy.shape
        self.eval()
        feat = self.feature_net(pcl_noisy)                 # (B,N,F)
        F = feat.size(-1)
        pcl_next = pcl_noisy.clone()
        trajectory: list = []

        for step in range(int(num_steps)):
            # Query the original noisy points and take neighbours from the
            # current refined cloud.  This is the original Score-Denoise
            # inference construction, with per-receiving-point averaging.
            _, nn_idx, frames = knn_points(
                pcl_noisy, pcl_next, K=denoise_knn, return_nn=True,
            )
            frames_centered = frames - pcl_noisy.unsqueeze(2)
            grad_pred = self.score_net(
                x=frames_centered.reshape(-1, denoise_knn, d),
                c=feat.reshape(-1, F),
            ).reshape(B, N, denoise_knn, d)

            flat_idx = nn_idx.reshape(B, -1)
            flat_grad = grad_pred.reshape(B, -1, d)
            acc = torch.zeros_like(pcl_next)
            acc.scatter_add_(1, flat_idx.unsqueeze(-1).expand_as(flat_grad), flat_grad)
            counts = torch.zeros(B, N, 1, device=pcl_next.device, dtype=pcl_next.dtype)
            counts.scatter_add_(
                1, flat_idx.unsqueeze(-1),
                torch.ones_like(flat_idx, dtype=pcl_next.dtype).unsqueeze(-1),
            )
            acc = acc / counts.clamp_min(1.0)

            s = float(step_size) * (float(step_decay) ** step)
            pcl_next = pcl_next + s * acc
            if noise_scale is not None and float(noise_scale) > 0.0:
                pcl_next = pcl_next + math.sqrt(max(2.0 * s, 0.0)) * float(noise_scale) * torch.randn_like(pcl_next)
            if step == int(num_steps) - 1:
                trajectory.append(pcl_next.detach().cpu())
        return pcl_next, trajectory

    @torch.no_grad()
    def infer_cloud(
        self,
        pcl_noisy: torch.Tensor,
        *,
        patch_size: int = 1000,
        seed_k: float = 3.0,
        patch_batch_size: int = 4,
        denoise_knn: int = 4,
        num_steps: int = 30,
        step_size: float = 0.2,
        step_decay: float = 0.95,
        noise_scale: float = 0.0,
        return_indices: bool = False,
    ):
        """Patch-based inference for one point cloud (N,3).

        Overlapping patch predictions are merged by a Gaussian centre weight
        and by the original point identity, so no point is duplicated or
        dropped.  Uncovered points are retained unchanged.
        """
        if pcl_noisy.dim() != 2:
            raise ValueError("pcl_noisy must be (N,3)")
        device = next(self.parameters()).device
        x = pcl_noisy.to(device=device, dtype=torch.float32)
        n = x.size(0)
        n_seeds = max(1, int(round(float(seed_k) * n / max(1, int(patch_size)))))
        n_seeds = min(n_seeds, n)
        seed_idx = farthest_point_sampling(x, n_seeds)
        # Query patch centers; patches are exact points from the input cloud.
        _, idx, patches = knn_points(
            x[seed_idx].unsqueeze(0), x.unsqueeze(0),
            K=min(int(patch_size), n), return_nn=True,
        )
        idx = idx[0]                                       # (P,K)
        patches = patches[0]                               # (P,K,3)
        # Gaussian weights based on distance to the patch centre.
        d2 = ((patches - patches[:, :1, :]) ** 2).sum(-1)  # (P,K)
        d2_max = d2[:, -1:].clamp_min(1e-12)
        weights = torch.exp(-d2 / d2_max)                  # (P,K)

        accumulated = torch.zeros_like(x)
        weight_sum = torch.zeros(n, 1, device=device, dtype=x.dtype)
        for start in range(0, patches.size(0), int(patch_batch_size)):
            batch = patches[start : start + int(patch_batch_size)]
            out, _ = self.denoise_langevin_dynamics(
                batch,
                step_size=step_size,
                denoise_knn=denoise_knn,
                step_decay=step_decay,
                num_steps=num_steps,
                noise_scale=noise_scale,
            )
            ids = idx[start : start + batch.size(0)].reshape(-1)
            wt = weights[start : start + batch.size(0)].reshape(-1, 1)
            accumulated.index_add_(0, ids, out.reshape(-1, 3) * wt)
            weight_sum.index_add_(0, ids, wt)
        result = accumulated / weight_sum.clamp_min(1e-12)
        uncovered = weight_sum[:, 0] == 0
        result[uncovered] = x[uncovered]
        if return_indices:
            return result, dict(seed_idx=seed_idx, patch_idx=idx, patch_weights=weights)
        return result

    @torch.no_grad()
    def infer_cloud_tta(
        self,
        pcl_noisy: torch.Tensor,
        *,
        n_rotations: int = 4,
        seed: int = 2020,
        **infer_kwargs,
    ) -> torch.Tensor:
        """Rotation test-time augmentation.

        Each rotation is denoised independently and the per-point predictions
        are averaged in the original coordinate frame.  This is optional and
        roughly multiplies inference time by ``n_rotations``.
        """
        device = next(self.parameters()).device
        x = pcl_noisy.to(device=device, dtype=torch.float32)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        acc = torch.zeros_like(x)
        for i in range(int(n_rotations)):
            if i == 0:
                rot = torch.eye(3, device=device)
            else:
                rot = random_rotation_matrix(device=device, generator=gen)
            # rotate: x @ R^T, so inverse is x @ R
            rotated = x @ rot.T
            denoised = self.infer_cloud(rotated, **infer_kwargs)
            acc += denoised @ rot
        return acc / float(max(1, n_rotations))
