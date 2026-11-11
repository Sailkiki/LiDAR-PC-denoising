"""Pure-PyTorch point-cloud operators used by R2-Net v2.

No compiled extensions are required.  The functions intentionally mirror the
subset of PyTorch3D that the original implementation used (squared kNN
distances, neighbour indices and neighbour coordinates).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def knn_points(
    x: torch.Tensor,
    y: torch.Tensor,
    k: Optional[int] = None,
    K: Optional[int] = None,
    return_nn: bool = True,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """K nearest neighbours.

    Args:
        x: (B, N, D) query points.
        y: (B, M, D) reference points.
        k: number of neighbours.
        return_nn: return neighbour coordinates (B, N, k, D).
        chunk_size: optional query chunk size to cap peak memory.

    Returns:
        dist2: (B, N, k) squared Euclidean distances.
        idx:   (B, N, k) indices into y.
        nn:    (B, N, k, D) neighbour coordinates, or None.
    """
    if k is None:
        k = K
    if k is None:
        raise ValueError("knn_points requires k or K")
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("knn_points expects (B,N,D) and (B,M,D)")
    if x.size(0) != y.size(0):
        raise ValueError("batch sizes must match")
    B, N, D = x.shape
    M = y.size(1)
    k_eff = min(int(k), M)
    # Use float32 for index search even under AMP.  Indices are not
    # differentiable; gradients for OC-Conv offsets flow through the attention
    # logits, not through the integer neighbour indices.
    xf = x.detach().float()
    yf = y.detach().float()
    if chunk_size is None:
        # Keep peak memory below ~256 MB for the distance matrix.
        chunk_size = max(1, int(32e6 // max(1, M)))
    dists_out = []
    idxs_out = []
    for start in range(0, N, chunk_size):
        xc = xf[:, start : start + chunk_size]
        d2 = torch.cdist(xc, yf, p=2.0).pow(2)
        vals, inds = torch.topk(d2, k_eff, dim=-1, largest=False, sorted=True)
        dists_out.append(vals)
        idxs_out.append(inds)
    dist2 = torch.cat(dists_out, dim=1)
    idx = torch.cat(idxs_out, dim=1)
    nn = None
    if return_nn:
        # Gather in a loop over batch to avoid a B*N*k*D expansion.
        nn_list = []
        for b in range(B):
            nn_list.append(y[b][idx[b]])
        nn = torch.stack(nn_list, dim=0)
    return dist2, idx, nn


def knn_idx(x: torch.Tensor, y: torch.Tensor, k: int, offset: int = 0) -> torch.Tensor:
    _, idx, _ = knn_points(x, y, k=k + offset, return_nn=False)
    return idx[:, :, offset:]


def knn_group(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbours.

    x:   (B, N, D)
    idx: (B, M, k)
    out: (B, M, k, D)
    """
    B, N, D = x.shape
    if idx.dim() != 3:
        raise ValueError("idx must be (B,M,k)")
    M, k = idx.shape[1], idx.shape[2]
    # Advanced indexing is much cheaper than expanding x to (B,M,N,D).
    offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N
    flat_idx = (idx + offsets).reshape(-1)
    out = x.reshape(B * N, D)[flat_idx]
    return out.reshape(B, M, k, D)


@torch.no_grad()
def farthest_point_sampling(points: torch.Tensor, n_samples: int, random_start: bool = False) -> torch.Tensor:
    """Farthest point sampling for a single cloud (N,D) -> indices (n_samples,)."""
    if points.dim() != 2:
        raise ValueError("points must be (N,D)")
    N = points.size(0)
    n_samples = int(min(max(1, n_samples), N))
    device = points.device
    idx = torch.empty(n_samples, dtype=torch.long, device=device)
    dist = torch.full((N,), float("inf"), dtype=torch.float32, device=device)
    farthest = torch.randint(0, N, (1,), device=device).item() if random_start else 0
    pf = points.float()
    for i in range(n_samples):
        idx[i] = farthest
        centroid = pf[farthest]
        d = ((pf - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = int(torch.argmax(dist).item())
    return idx


@torch.no_grad()
def normalize_unit_sphere(points: torch.Tensor):
    """Bounding-sphere normalisation used by the paper.

    The centre is the bounding-box centre and the radius is the maximum
    distance from that centre.  This is an enclosing sphere; it is not the
    exact minimum enclosing sphere, but it is the convention used in the
    original R2-Net implementation.
    """
    if points.dim() != 2:
        raise ValueError("points must be (N,3)")
    p_min = points.min(dim=0, keepdim=True).values
    p_max = points.max(dim=0, keepdim=True).values
    centre = (p_min + p_max) * 0.5
    shifted = points - centre
    radius = shifted.norm(dim=-1, keepdim=True).max().clamp_min(1e-12)
    return shifted / radius, centre, radius


def random_rotation_matrix(device=None, dtype=torch.float32, generator=None) -> torch.Tensor:
    """Uniform random rotation from a QR decomposition of a Gaussian matrix."""
    a = torch.randn(3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    # Make Q proper (det=+1).
    q = q * torch.sign(torch.det(q)).unsqueeze(0).unsqueeze(0)
    return q


def rotate_points(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """points (B,N,3), rotation (B,3,3) or (3,3)."""
    if rotation.dim() == 2:
        return points @ rotation.T
    return torch.bmm(points, rotation.transpose(1, 2))


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """PCA normals with a KD-tree neighbourhood, as described in the paper."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    k_eff = min(k + 1, n)
    idx = cKDTree(points).query(points, k=k_eff, workers=4)[1]
    if k_eff == 1:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    idx = np.atleast_2d(idx)[:, 1:]
    normals = np.empty((n, 3), dtype=np.float64)
    # Chunk to keep memory bounded.
    for start in range(0, n, 4096):
        stop = min(n, start + 4096)
        nbr = points[idx[start:stop]]
        centred = nbr - nbr.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centred, centred) / max(1, nbr.shape[1] - 1)
        normals[start:stop] = np.linalg.eigh(cov)[1][:, :, 0]
    return normals


def cd_rmse_nc(gen: np.ndarray, ref: np.ndarray, normal_k: int = 30) -> dict:
    """Chamfer distance, RMSE and normal consistency.

    CD is the sum of the two mean squared nearest-neighbour distances (paper
    Eq. 22), RMSE is the root mean squared gen->ref distance (paper Eq. 24),
    and NC is the mean absolute normal dot product at nearest neighbours
    (paper Eq. 23).
    """
    from scipy.spatial import cKDTree

    gen = np.asarray(gen, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    tree_ref = cKDTree(ref)
    tree_gen = cKDTree(gen)
    d_gen_ref = tree_ref.query(gen, workers=4)[0]
    d_ref_gen = tree_gen.query(ref, workers=4)[0]
    cd = float(np.mean(d_gen_ref ** 2) + np.mean(d_ref_gen ** 2))
    rmse = float(np.sqrt(np.mean(d_gen_ref ** 2)))
    n_gen = estimate_normals(gen, normal_k)
    n_ref = estimate_normals(ref, normal_k)
    match = tree_ref.query(gen, workers=4)[1]
    nc = float(np.abs(np.einsum("ij,ij->i", n_gen, n_ref[match])).mean())
    return {"cd": cd, "rmse": rmse, "nc": nc}
