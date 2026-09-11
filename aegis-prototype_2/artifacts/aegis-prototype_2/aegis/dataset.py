"""Dataset construction and caching."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import config as C
from .scene import SceneGenerator, Tile, tile_features


@dataclass
class TileSet:
    cubes: np.ndarray          # (N, 6, 64, 64) float32
    cloud_frac: np.ndarray     # (N,) float32
    event_lbl: np.ndarray      # (N,) int64
    event_mask: np.ndarray     # (N, 64, 64) uint8
    severity: np.ndarray       # (N,) float32
    feats: np.ndarray          # (N, 32) float32
    usable: np.ndarray         # (N,) bool  - ground-truth "worth keeping"
    true_event: np.ndarray     # (N,) bool  - ground-truth observable event

    def __len__(self) -> int:
        return len(self.cloud_frac)

    def subset(self, idx: np.ndarray) -> "TileSet":
        return TileSet(
            self.cubes[idx], self.cloud_frac[idx], self.event_lbl[idx],
            self.event_mask[idx], self.severity[idx], self.feats[idx],
            self.usable[idx], self.true_event[idx],
        )


def tiles_to_set(tiles: List[Tile]) -> TileSet:
    n = len(tiles)
    cubes = np.empty((n, C.N_BANDS, C.TILE_PX, C.TILE_PX), dtype=np.float32)
    cf = np.empty(n, dtype=np.float32)
    lbl = np.empty(n, dtype=np.int64)
    masks = np.empty((n, C.TILE_PX, C.TILE_PX), dtype=np.uint8)
    sev = np.empty(n, dtype=np.float32)
    feats = np.empty((n, 32), dtype=np.float32)
    usable = np.empty(n, dtype=bool)
    true_ev = np.empty(n, dtype=bool)
    for i, t in enumerate(tiles):
        cubes[i] = t.cube
        cf[i] = t.cloud_fraction
        lbl[i] = t.event_label
        masks[i] = t.event_mask.astype(np.uint8)
        sev[i] = t.severity
        feats[i] = tile_features(t.cube)
        usable[i] = t.is_usable
        true_ev[i] = t.is_true_event
    return TileSet(cubes, cf, lbl, masks, sev, feats, usable, true_ev)


def _cache_path(name: str) -> str:
    return os.path.join(C.ARTIFACT_DIR, f"dataset_{name}.npz")


def save_set(ts: TileSet, name: str) -> str:
    os.makedirs(C.ARTIFACT_DIR, exist_ok=True)
    p = _cache_path(name)
    np.savez_compressed(
        p, cubes=ts.cubes.astype(np.float16), cloud_frac=ts.cloud_frac,
        event_lbl=ts.event_lbl, event_mask=ts.event_mask, severity=ts.severity,
        feats=ts.feats, usable=ts.usable, true_event=ts.true_event,
    )
    return p


def load_set(name: str) -> TileSet:
    d = np.load(_cache_path(name))
    return TileSet(
        d["cubes"].astype(np.float32), d["cloud_frac"], d["event_lbl"],
        d["event_mask"], d["severity"], d["feats"], d["usable"], d["true_event"],
    )


def build_datasets(cfg: C.TrainConfig = C.DEFAULT_TRAIN, force: bool = False,
                   verbose: bool = True) -> Dict[str, TileSet]:
    """Train/val use class-balanced sampling; test uses the natural prior.

    Evaluating on the natural prior is the point: a model that looks great on
    a balanced test set can still be useless when events are 4% of traffic.
    """
    specs = [
        ("train", cfg.n_train, True, cfg.seed),
        ("val", cfg.n_val, True, cfg.seed + 1),
        # Natural-prior validation set, used *only* to choose the operating
        # point. Calibrating on the balanced set would pick a threshold for a
        # world where 60% of tiles contain an event; calibrating on the test
        # set would be cheating.
        ("val_nat", cfg.n_val, False, cfg.seed + 3),
        ("test", cfg.n_test, False, cfg.seed + 2),
    ]
    out: Dict[str, TileSet] = {}
    for name, n, balanced, seed in specs:
        if os.path.exists(_cache_path(name)) and not force:
            ts = load_set(name)
            # Guard against a stale cache whose features were computed with a
            # different Stage-0 definition.
            if ts.feats.shape[1] != 32 or not np.isfinite(ts.feats).all():
                ts = refresh_features(ts)
                save_set(ts, name)
            out[name] = ts
            if verbose:
                print(f"  [{name}] loaded from cache: {len(out[name])} tiles")
            continue
        t0 = time.time()
        gen = SceneGenerator(seed=seed)
        tiles = (gen.sample_balanced_batch(n) if balanced else gen.sample_batch(n))
        ts = tiles_to_set(tiles)
        save_set(ts, name)
        out[name] = ts
        if verbose:
            ev = ts.true_event.mean()
            print(f"  [{name}] generated {n} tiles in {time.time()-t0:.1f}s "
                  f"| usable {ts.usable.mean():.1%} | observable events {ev:.1%}")
    return out


def refresh_features(ts: TileSet) -> TileSet:
    """Recompute Stage-0 features in place.

    Cheaper than regenerating the scenes when only the feature definition
    changes - which it does whenever the decimation factor is retuned.
    """
    ts.feats = np.stack([tile_features(c) for c in ts.cubes]).astype(np.float32)
    return ts


def summarise(ts: TileSet, name: str) -> str:
    counts = np.bincount(ts.event_lbl, minlength=len(C.EVENT_CLASSES))
    parts = [f"{c}={n}" for c, n in zip(C.EVENT_CLASSES, counts)]
    return (f"{name}: N={len(ts)} | {' '.join(parts)} | "
            f"mean cloud frac {ts.cloud_frac.mean():.2f} | "
            f"usable {ts.usable.mean():.1%} | observable events {ts.true_event.mean():.1%}")
