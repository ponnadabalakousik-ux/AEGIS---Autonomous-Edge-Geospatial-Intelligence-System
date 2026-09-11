"""
The onboard inference cascade.
==============================

Three stages, deliberately ordered cheapest-first. This is the core design
idea: on a power- and thermally-constrained payload you do not run a neural
network on every tile, you run the cheapest thing that can safely say "no".

    Stage 0  PhysicsScreener   ~18 kFLOP/tile     always runs
             32 spectral-statistic features on a 4x-decimated view -> logistic
             gate. Rejects thick cloud without waking the accelerator. Tuned
             for high recall on "possibly useful": it may pass junk upward but
             must almost never reject a usable tile.

             The decimation matters. Computed at full resolution this stage
             cost 0.9 mJ/tile against the CNN's 1.1 mJ - the gate was more
             expensive than the thing it gates, and the cascade lost energy
             overall. Decimating 4x drops it to 0.055 mJ and the cascade saves
             ~46%. The cost model caught this; intuition did not.

    Stage 1  TriageNet         ~6.3 MMAC/tile     runs on Stage-0 survivors
             Depthwise-separable CNN, 3 heads: cloud fraction (regression),
             event class (4-way), coarse event mask (segmentation).

    Stage 2  ROI extraction    negligible         runs on Stage-1 detections
             Turns the coarse mask into a bounding box so only the part of
             the scene that matters gets downlinked.

Everything is sized so the whole cascade fits comfortably in the ~2 W, few
hundred MB envelope of a flown VPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C


# ===========================================================================
# Stage 0 - physics screener (pure numpy, no framework, no accelerator)
# ===========================================================================
class PhysicsScreener:
    """Logistic gate on 32 spectral-statistic features.

    Trained with plain gradient descent in numpy so that it has no runtime
    dependency at all - on the real spacecraft this is ~200 lines of C on the
    housekeeping processor, which is exactly why it can afford to run on
    100% of tiles.
    """

    N_FEAT = 32

    def __init__(self) -> None:
        self.w = np.zeros(self.N_FEAT, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mu = np.zeros(self.N_FEAT, dtype=np.float32)
        self.sd = np.ones(self.N_FEAT, dtype=np.float32)
        self.threshold = 0.5

    # -- fit ---------------------------------------------------------------
    def fit(self, feats: np.ndarray, keep: np.ndarray, epochs: int = 900,
            lr: float = 0.35, l2: float = 1e-4) -> "PhysicsScreener":
        """`keep` is 1 where the tile is worth passing to Stage 1."""
        self.mu = feats.mean(0).astype(np.float32)
        self.sd = (feats.std(0) + 1e-6).astype(np.float32)
        x = (feats - self.mu) / self.sd
        y = keep.astype(np.float32)
        n = len(x)
        # Class weighting: rejecting a usable tile is far worse than passing
        # a cloudy one, because a wrongly rejected tile is gone forever.
        pos_w = 3.0
        w = np.zeros(self.N_FEAT, np.float32)
        b = np.float32(0.0)
        for _ in range(epochs):
            z = x @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            sample_w = np.where(y > 0.5, pos_w, 1.0)
            g = (p - y) * sample_w
            w -= lr * ((x.T @ g) / n + l2 * w)
            b -= lr * g.mean()
        self.w, self.b = w.astype(np.float32), np.float32(b)
        return self

    def calibrate(self, feats: np.ndarray, keep: np.ndarray,
                  min_recall: float = 0.995) -> float:
        """Pick the operating threshold that rejects the most tiles while
        still passing `min_recall` of the genuinely useful ones."""
        p = self.predict_proba(feats)
        pos = p[keep.astype(bool)]
        if len(pos) == 0:
            self.threshold = 0.5
            return self.threshold
        self.threshold = float(np.quantile(pos, 1.0 - min_recall))
        return self.threshold

    # -- inference ---------------------------------------------------------
    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        x = (feats - self.mu) / self.sd
        z = x @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def gate(self, feats: np.ndarray) -> np.ndarray:
        """True = wake the accelerator for this tile."""
        return self.predict_proba(feats) >= self.threshold

    # -- cost --------------------------------------------------------------
    @staticmethod
    def flops_per_tile(decimate: Optional[int] = None) -> float:
        """Feature extraction dominates: 8 indices over the decimated tile at
        ~5 ops each, plus 4 reductions per index, plus a 32-length dot product.

        The decimation factor is the difference between Stage 0 paying for
        itself and Stage 0 being more expensive than the network it gates.
        """
        from .scene import STAGE0_DECIMATION
        d = STAGE0_DECIMATION if decimate is None else decimate
        px = (C.TILE_PX // d) * (C.TILE_PX // d)
        index_ops = 8 * px * 5          # 8 indices, ~5 ops/pixel
        reduce_ops = 8 * px * 4         # mean/std/2 quantiles
        dot = 2 * PhysicsScreener.N_FEAT
        return float(index_ops + reduce_ops + dot)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"w": self.w, "b": np.array([self.b]), "mu": self.mu,
                "sd": self.sd, "threshold": np.array([self.threshold])}

    def load_state(self, d) -> "PhysicsScreener":
        self.w = d["w"]; self.b = float(d["b"][0]); self.mu = d["mu"]
        self.sd = d["sd"]; self.threshold = float(d["threshold"][0])
        return self


# ===========================================================================
# Stage 1 - TriageNet (PyTorch)
# ===========================================================================
def build_triagenet():
    """Constructed lazily so the package imports without torch installed."""
    import torch
    import torch.nn as nn

    class SeparableBlock(nn.Module):
        """Depthwise-separable conv: ~8-9x fewer MACs than a dense 3x3 at
        these widths, which is the difference between fitting in the power
        budget and not."""

        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1,
                                groups=cin, bias=False)
            self.bn1 = nn.BatchNorm2d(cin)
            self.pw = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.act(self.bn1(self.dw(x)))
            return self.act(self.bn2(self.pw(x)))

    class TriageNet(nn.Module):
        def __init__(self, in_ch: int = C.N_BANDS,
                     n_classes: int = len(C.EVENT_CLASSES)):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            )                                     # 24 x 32 x 32
            self.b1 = SeparableBlock(24, 48, stride=2)   # 48 x 16 x 16
            self.b1b = SeparableBlock(48, 48, stride=1)  # 48 x 16 x 16
            self.b2 = SeparableBlock(48, 72, stride=2)   # 72 x  8 x  8
            self.b3 = SeparableBlock(72, 96, stride=2)   # 96 x  4 x  4

            # Segmentation head taps the 16x16 features: coarse but enough to
            # place a bounding box, and 16x cheaper than a full-res decoder.
            self.seg = nn.Sequential(
                nn.Conv2d(48, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.cloud_head = nn.Linear(96, 1)
            self.event_head = nn.Sequential(
                nn.Linear(96, 64), nn.ReLU(inplace=True), nn.Linear(64, n_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            f1 = self.b1b(self.b1(x))
            seg = self.seg(f1)                       # (B,1,16,16) logits
            f = self.b3(self.b2(f1))
            v = self.pool(f).flatten(1)
            cloud = self.cloud_head(v).squeeze(1)    # logit
            event = self.event_head(v)               # logits
            return cloud, event, seg

    return TriageNet


def count_macs(model, input_shape=(1, C.N_BANDS, C.TILE_PX, C.TILE_PX)) -> float:
    """Multiply-accumulate count via forward hooks. INT8 hardware quotes
    throughput in ops/s where one MAC = 2 ops, handled in edge.py."""
    import torch
    import torch.nn as nn

    total = {"macs": 0.0}

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
            k = mod.kernel_size[0] * mod.kernel_size[1]
            cin_per_group = mod.in_channels // mod.groups
            total["macs"] += oc * oh * ow * k * cin_per_group
        elif isinstance(mod, nn.Linear):
            total["macs"] += mod.in_features * mod.out_features

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for h in handles:
        h.remove()
    return float(total["macs"])


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ===========================================================================
# Stage 2 - ROI extraction
# ===========================================================================
@dataclass
class ROI:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def area_px(self) -> int:
        return max(0, self.y1 - self.y0) * max(0, self.x1 - self.x0)

    @property
    def fraction_of_tile(self) -> float:
        return self.area_px / float(C.TILE_PX * C.TILE_PX)


def extract_roi(seg_logits: np.ndarray, threshold: float = 0.0,
                pad_px: int = 4, min_side: int = 8) -> Optional[ROI]:
    """Coarse 16x16 segmentation logits -> a padded bounding box at 64x64.

    Returns None when nothing fires, which is the signal to downlink a
    thumbnail (or nothing) rather than any imagery.
    """
    m = seg_logits.reshape(seg_logits.shape[-2], seg_logits.shape[-1]) > threshold
    if not m.any():
        return None
    scale = C.TILE_PX // m.shape[0]
    ys, xs = np.where(m)
    y0 = max(0, int(ys.min()) * scale - pad_px)
    y1 = min(C.TILE_PX, int(ys.max() + 1) * scale + pad_px)
    x0 = max(0, int(xs.min()) * scale - pad_px)
    x1 = min(C.TILE_PX, int(xs.max() + 1) * scale + pad_px)
    if y1 - y0 < min_side:
        c = (y0 + y1) // 2
        y0, y1 = max(0, c - min_side // 2), min(C.TILE_PX, c + min_side // 2)
    if x1 - x0 < min_side:
        c = (x0 + x1) // 2
        x0, x1 = max(0, c - min_side // 2), min(C.TILE_PX, c + min_side // 2)
    return ROI(y0, x0, y1, x1)


# ===========================================================================
# Product sizing - what actually goes down the pipe
# ===========================================================================
def product_bits(kind: str, roi: Optional[ROI] = None) -> float:
    """Downlink size of each product type, in bits.

    ALERT      metadata only: class, confidence, geolocation, timestamp,
               severity, plus a 32x32 single-band quick-look thumbnail.
    THUMBNAIL  heavily compressed 3-band preview for browse/audit.
    ROI        the bounding box around the detection, all 6 bands.
    FULL       the whole tile, all 6 bands - what a non-AI mission sends.
    """
    full = C.RAW_TILE_BITS / C.LOSSLESS_COMPRESSION_RATIO
    if kind == "full":
        return full
    if kind == "thumbnail":
        return (C.TILE_PX * C.TILE_PX * 3 * C.RAW_BITS_PER_SAMPLE
                / C.THUMBNAIL_COMPRESSION_RATIO)
    if kind == "alert":
        meta_bits = 96 * 8                          # 96-byte structured record
        thumb = 32 * 32 * 1 * 8 / 12.0              # tiny 8-bit quick-look
        return meta_bits + thumb
    if kind == "roi":
        if roi is None:
            return product_bits("thumbnail")
        px = roi.area_px
        return (px * C.N_BANDS * C.RAW_BITS_PER_SAMPLE / C.ROI_COMPRESSION_RATIO
                + 96 * 8)
    raise ValueError(f"unknown product kind: {kind}")
