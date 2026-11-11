"""Data loading and on-the-fly patch sampling for PlantN-R."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from .ops import knn_points, normalize_unit_sphere, random_rotation_matrix


@dataclass
class Cloud:
    name: str
    points: int
    split: str
    data: torch.Tensor  # normalised (N,3), float32

    def __len__(self) -> int:
        return self.data.size(0)


class CloudStore:
    """Keeps all clouds of a split on one device."""

    def __init__(self, clouds: Sequence[Cloud], device: torch.device, balance_species: bool = False):
        self.clouds = list(clouds)
        self.device = device
        self.balance_species = bool(balance_species)
        for c in self.clouds:
            c.data = c.data.to(device=device, dtype=torch.float32)
        if self.balance_species:
            counts: Dict[str, int] = {}
            for c in self.clouds:
                species = c.name.split("_")[0]
                counts[species] = counts.get(species, 0) + 1
            weights = []
            for c in self.clouds:
                species = c.name.split("_")[0]
                weights.append(1.0 / (len(counts) * counts[species]))
            total = float(sum(weights))
            self.sampling_weights = [w / total for w in weights]
        else:
            self.sampling_weights = None

    @classmethod
    def from_split(
        cls,
        root: str | Path,
        split: str,
        points: Iterable[int],
        device: torch.device,
        normalize: bool = True,
        balance_species: bool = False,
    ) -> "CloudStore":
        root = Path(root)
        clouds: List[Cloud] = []
        for pts in points:
            d = root / str(int(pts)) / split
            if not d.exists():
                continue
            for f in sorted(d.glob("*.npy")):
                arr = np.load(f, allow_pickle=False).astype(np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(f"{f}: expected (N,3), got {arr.shape}")
                t = torch.from_numpy(arr)
                if normalize:
                    t, _, _ = normalize_unit_sphere(t)
                clouds.append(Cloud(name=f.stem, points=int(pts), split=split, data=t))
        if not clouds:
            raise FileNotFoundError(f"No clouds found for split={split}, points={list(points)} under {root}")
        return cls(clouds, device=device, balance_species=balance_species)

    def sample_patch_pair(
        self,
        *,
        patch_size: int = 1000,
        clean_patch_size: int = 1200,
        noise_min: float = 0.005,
        noise_max: float = 0.025,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        rotate: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Sample one noisy/clean patch pair and metadata."""
        if generator is None:
            generator = torch.default_generator
        if self.sampling_weights is not None:
            idx = random.choices(range(len(self.clouds)), weights=self.sampling_weights, k=1)[0]
        else:
            idx = random.randrange(len(self.clouds))
        cloud = self.clouds[idx]
        clean = cloud.data
        sigma = float(torch.empty(1, device=self.device).uniform_(noise_min, noise_max, generator=generator).item())
        noisy = clean + torch.randn(clean.shape, device=self.device, generator=generator) * sigma
        centre_idx = int(torch.randint(0, clean.size(0), (1,), device=self.device, generator=generator).item())
        centre = noisy[centre_idx : centre_idx + 1]
        pn = knn_points(centre.unsqueeze(0), noisy.unsqueeze(0), K=min(patch_size, noisy.size(0)), return_nn=True)[2][0, 0]
        pc = knn_points(centre.unsqueeze(0), clean.unsqueeze(0), K=min(clean_patch_size, clean.size(0)), return_nn=True)[2][0, 0]
        scale = float(torch.empty(1, device=self.device).uniform_(scale_min, scale_max, generator=generator).item())
        pn = pn * scale
        pc = pc * scale
        if rotate:
            rot = random_rotation_matrix(device=self.device, dtype=torch.float32)
            pn = pn @ rot.T
            pc = pc @ rot.T
        meta = {"name": cloud.name, "points": cloud.points, "sigma": sigma, "scale": scale}
        return pn.contiguous(), pc.contiguous(), meta

    def sample_batch(
        self,
        batch_size: int,
        *,
        patch_size: int = 1000,
        clean_patch_size: int = 1200,
        noise_min: float = 0.005,
        noise_max: float = 0.025,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        rotate: bool = True,
        generator: torch.Generator | None = None,
    ):
        pns, pcs, metas = [], [], []
        for _ in range(int(batch_size)):
            pn, pc, meta = self.sample_patch_pair(
                patch_size=patch_size,
                clean_patch_size=clean_patch_size,
                noise_min=noise_min,
                noise_max=noise_max,
                scale_min=scale_min,
                scale_max=scale_max,
                rotate=rotate,
                generator=generator,
            )
            pns.append(pn)
            pcs.append(pc)
            metas.append(meta)
        return torch.stack(pns, dim=0), torch.stack(pcs, dim=0), metas


def load_manifest(root: str | Path) -> Dict:
    root = Path(root)
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
