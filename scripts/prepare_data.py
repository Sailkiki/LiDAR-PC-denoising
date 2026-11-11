#!/usr/bin/env python3
"""Extract PlantN-R point clouds from the original submission archive.

The archive contains .ply (binary little-endian, Open3D) and .xyz files.  This
script keeps only xyz coordinates, stores them as float32 .npy arrays, and
writes a manifest that makes the training/evaluation split explicit.

Usage:
  python scripts/prepare_data.py \
      --zip /path/to/Botany3D_LiDAR_backup.zip \
      --splits-dir /path/to/splits \
      --out data/plantn_r \
      --points 70000 90000 110000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np


def read_ply_xyz(raw: bytes) -> np.ndarray:
    header_end = raw.find(b"end_header\n")
    if header_end < 0:
        raise ValueError("PLY header end not found")
    header_bytes = raw[: header_end + len(b"end_header\n")]
    payload = raw[header_end + len(b"end_header\n") :]
    lines = header_bytes.decode("ascii", errors="replace").splitlines()
    fmt = None
    vertex_count = None
    properties = []
    in_vertex = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            # property <type> <name>
            properties.append((parts[1], parts[2]))
    if vertex_count is None:
        raise ValueError("PLY vertex element not found")
    names = [p[1] for p in properties]
    if not all(n in names for n in ("x", "y", "z")):
        raise ValueError(f"PLY missing xyz properties: {names}")
    if fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        np_types = {
            "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
        }
        dtype = np.dtype([(name, endian + np_types[t]) for t, name in properties])
        arr = np.frombuffer(payload, dtype=dtype, count=vertex_count)
        xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    elif fmt == "ascii":
        data = np.loadtxt(io.BytesIO(payload), dtype=np.float64, max_rows=vertex_count)
        data = np.atleast_2d(data)
        idx = [names.index(n) for n in ("x", "y", "z")]
        xyz = data[:, idx].astype(np.float32)
    else:
        raise ValueError(f"Unsupported PLY format: {fmt}")
    return xyz


def read_xyz(raw: bytes) -> np.ndarray:
    arr = np.loadtxt(io.BytesIO(raw), dtype=np.float64, ndmin=2)
    if arr.shape[1] < 3:
        raise ValueError(f"XYZ has {arr.shape[1]} columns")
    return arr[:, :3].astype(np.float32)


def read_member_xyz(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    raw = zf.read(name)
    suffix = Path(name).suffix.lower()
    if suffix == ".ply":
        return read_ply_xyz(raw)
    if suffix in (".xyz", ".txt"):
        return read_xyz(raw)
    raise ValueError(f"Unsupported point-cloud format: {name}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/plantn_r"))
    ap.add_argument("--points", nargs="+", type=int, default=[70000, 90000, 110000])
    ap.add_argument("--splits-dir", required=True, type=Path,
                    help="Directory containing benchmark_files.csv and corrected_{train,validation,test}.txt")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source_zip": str(args.zip.resolve()), "point_counts": args.points,
                "splits": {}, "files": []}

    with (args.splits_dir / "benchmark_files.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # benchmark_files.csv has blank lines; DictReader handles them as empty rows.
    rows = [r for r in rows if r.get("sample_id")]
    by_key = {(r["sample_id"], int(r["point_count"])): r for r in rows}

    # Only xyz coordinates are stored.  Colour and other PLY attributes are
    # intentionally dropped because the denoising models operate on geometry.
    with zipfile.ZipFile(args.zip) as zf:
        for split in ("train", "validation", "test"):
            names = (args.splits_dir / f"corrected_{split}.txt").read_text(encoding="utf-8").split()
            if split == "validation":
                split_for_benchmark = "validation"
            else:
                split_for_benchmark = split
            manifest["splits"][split] = names
            for points in args.points:
                dst_dir = out / f"{points}" / split
                dst_dir.mkdir(parents=True, exist_ok=True)
                for name in names:
                    key = (name, int(points))
                    if key not in by_key:
                        raise KeyError(f"Missing benchmark row for {key}")
                    row = by_key[key]
                    member = row["benchmark_relative_path"]
                    xyz = read_member_xyz(zf, member)
                    if xyz.shape != (points, 3):
                        raise ValueError(f"{member}: expected {(points,3)}, got {xyz.shape}")
                    if not np.isfinite(xyz).all():
                        raise ValueError(f"{member}: non-finite coordinates")
                    dst = dst_dir / f"{name}.npy"
                    np.save(dst, xyz.astype(np.float32), allow_pickle=False)
                    manifest["files"].append({
                        "split": split,
                        "sample_id": name,
                        "point_count": points,
                        "source_member": member,
                        "npy": str(dst.relative_to(out)),
                        "sha256": sha256_file(dst),
                        "source_sha256": row.get("sha256", ""),
                    })
                    print(f"[prepare] {split:10s} {points:6d} {name}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {out} ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()
