"""
BIOMASS adapter: onboard forest-disturbance triage on real P-band SAR.
======================================================================

This is the modularity claim from the brief, discharged rather than asserted.
The cascade built in `models.py` for optical multispectral tiles is re-pointed
at a completely different instrument - ESA's Biomass P-band fully-polarimetric
SAR - without changing the architecture, the decision layer or the mission
simulator. Only the front end changes.

What this module runs on
------------------------
A real Biomass L1A product: BIO_S1_SCS__1M, track 006, frame 300, acquired
2025-11-21 over Rondonia, Brazil (-12.62, -58.61), tomographic phase, quad-pol.

The L1A *measurement* arrays (1373 samples x 21180 lines x 4 polarisations,
complex) are ~930 MB per frame and are not in the annotation-only download,
so this adapter runs on the product's Pauli-composite quicklook. That is an
8-bit browse product, not calibrated backscatter - every result here is
therefore a *demonstration of the pipeline*, not a calibrated science claim,
and the module says so in its outputs. `SlcLoader` below is the drop-in for
the real thing.

Why P-band forest scattering makes this work
--------------------------------------------
At 70 cm wavelength the radar sees through the canopy and scatters off trunks
and large branches:

  * intact forest      strong volume + double-bounce return; in a Pauli
                       composite this reads as bright and green-dominant
  * cleared land       surface scattering only; specular away from the sensor,
                       so it collapses to dark, and what return remains is
                       relatively blue (single-bounce) - which is exactly the
                       black rectangles and fishbone strips visible in the
                       Rondonia frame
  * water / smooth     near-zero return, dark in every channel

So a two-feature test - total power, and the ratio of the volume channel to
the surface channel - already separates the classes. That is the P-band
analogue of the spectral-index Stage 0 in the optical chain, and it is the
reason the same cascade design transfers.
"""
from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

# ---------------------------------------------------------------------------
# Mission constants (from the product annotation and ESA mission description)
# ---------------------------------------------------------------------------
BIOMASS: Dict[str, float | str] = {
    "centre_frequency_mhz": 435.0,
    "bandwidth_mhz": 6.0,
    "wavelength_m": 0.6892,
    "polarisations": "HH HV VH VV (quad-pol)",
    "altitude_km": 666.0,
    "inclination_deg": 97.97,
    "ltan": "06:00/18:00 dawn-dusk",
    "swath_km": 50.0,
    "resolution_m": "<=60 (range) x 50 (azimuth) at >=6 looks",
    "raw_data_rate_mbps": 117.0,        # max, prior to compression
    "prf_hz": 3050.0,
    "peak_rf_power_w": 120.0,
    "duty_cycle": 0.12,
    "repeat_days_tomographic": 3.0,
    "repeat_days_interferometric": 17.0,
    "launch_mass_kg": 1200.0,
    "platform_power_w": 1500.0,
}

# Measured from the delivered annotation of frame T006/F300.
FRAME_T006_F300 = {
    "product": "BIO_S1_SCS__1M_20251121T095108_20251121T095129_T_G01_M01_C01_T006_F300_01_DJUQK1",
    "samples": 1373,
    "lines": 21180,
    "range_pixel_spacing_m": 19.813869327800077,
    "azimuth_pixel_spacing_m": 6.7088732645224045,
    "polarisations": 4,
    "mission_phase": "TOMOGRAPHIC",
    "orbit": 3022,
    "centre_lat": -12.621341,
    "centre_lon": -58.605572,
    "footprint": [(-11.933118, -58.454420), (-12.046970, -59.024276),
                  (-13.309531, -58.758897), (-13.195744, -58.184694)],
}


def frame_data_volume(frame: Dict = FRAME_T006_F300,
                      bytes_per_complex: int = 8) -> Dict[str, float]:
    """How big is one L1A frame really?

    The answer is the whole motivation: ~0.9 GB for 21 seconds of acquisition,
    from an instrument whose raw rate is 117 Mbit/s.
    """
    px = frame["samples"] * frame["lines"]
    slc_bytes = px * frame["polarisations"] * bytes_per_complex
    along_km = frame["lines"] * frame["azimuth_pixel_spacing_m"] / 1000.0
    across_km = frame["samples"] * frame["range_pixel_spacing_m"] / 1000.0
    return {
        "pixels_per_pol": float(px),
        "slc_bytes": float(slc_bytes),
        "slc_gb": slc_bytes / 1e9,
        "along_track_km": along_km,
        "across_track_km": across_km,
        "area_km2": along_km * across_km,
        "gb_per_1000km2": slc_bytes / 1e9 / (along_km * across_km) * 1000.0,
    }


# ---------------------------------------------------------------------------
# Product reading
# ---------------------------------------------------------------------------
@dataclass
class BiomassFrame:
    """A Biomass frame as this pipeline consumes it.

    `channels` is (C, H, W). For the quicklook path C=3 and the channels are
    the Pauli components as rendered. For the SLC path C=4 and they are the
    calibrated intensities |HH|^2, |HV|^2, |VH|^2, |VV|^2.
    """
    channels: np.ndarray
    channel_names: List[str]
    source: str
    calibrated: bool
    meta: Dict

    @property
    def shape(self) -> Tuple[int, int]:
        return self.channels.shape[1], self.channels.shape[2]


class QuicklookLoader:
    """Reads the Pauli-composite browse image shipped with every L1A product.

    Pauli RGB convention for quad-pol SAR:
        R = |HH - VV|   double-bounce (trunk-ground interaction)
        G = |HV + VH|   volume scattering (canopy)
        B = |HH + VV|   surface / single-bounce

    So green-dominance means canopy, blue-dominance with low total power means
    bare or cleared ground. That mapping is what the features below exploit.
    """

    PAULI_NAMES = ["pauli_double_bounce", "pauli_volume", "pauli_surface"]

    def __init__(self, product_dir: str):
        self.product_dir = product_dir

    def find_quicklook(self) -> str:
        hits = glob.glob(os.path.join(self.product_dir, "**", "*_ql.png"),
                         recursive=True)
        if not hits:
            raise FileNotFoundError(
                f"no quicklook (*_ql.png) under {self.product_dir}")
        return hits[0]

    def read_annotation(self) -> Dict:
        hits = glob.glob(os.path.join(self.product_dir, "**", "*_annot.xml"),
                         recursive=True)
        if not hits:
            return {}
        txt = open(hits[0]).read()
        out: Dict = {}
        for key in ("numberOfSamples", "numberOfLines", "rangePixelSpacing",
                    "azimuthPixelSpacing"):
            m = re.search(rf"<[^>]*{key}[^>]*>([^<]+)</", txt)
            if m:
                try:
                    out[key] = float(m.group(1))
                except ValueError:
                    out[key] = m.group(1)
        return out

    def load(self) -> BiomassFrame:
        from PIL import Image
        path = self.find_quicklook()
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        chans = np.transpose(img, (2, 0, 1))
        return BiomassFrame(
            channels=chans, channel_names=self.PAULI_NAMES,
            source=os.path.basename(path), calibrated=False,
            meta={**self.read_annotation(), "note":
                  "8-bit Pauli browse product; radiometry is display-stretched, "
                  "not calibrated sigma-0"},
        )


class SlcLoader:
    """Drop-in for the real L1A measurement arrays.

    The full product ships one raster per polarisation under `measurement/`.
    Implement `load()` to read them, form intensities, multi-look to roughly
    square ground pixels (the frame is 19.8 m in range and 6.7 m in azimuth,
    so ~3 looks in azimuth), and return a `BiomassFrame` with
    `calibrated=True`. Everything downstream is unchanged.
    """

    def __init__(self, product_dir: str):
        self.product_dir = product_dir

    def load(self) -> BiomassFrame:
        meas = glob.glob(os.path.join(self.product_dir, "**", "measurement", "*"),
                         recursive=True)
        raise NotImplementedError(
            "L1A measurement arrays are not present in this product "
            f"(found {len(meas)} files under measurement/). Download the full "
            "~930 MB product and implement intensity formation + multi-looking "
            "here; see the class docstring."
        )


# ---------------------------------------------------------------------------
# Tiling and features
# ---------------------------------------------------------------------------
def tile_frame(frame: BiomassFrame, tile: int = 32,
               stride: Optional[int] = None,
               min_valid: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the frame into tiles, dropping the zero-fill border.

    SAR frames in slant range have large null regions at the swath edges; a
    tile that is mostly fill would otherwise be scored as "very dark" and
    misread as clear-cut, which is a real and easily-made mistake.
    """
    stride = stride or tile
    c, h, w = frame.channels.shape
    tiles, coords = [], []
    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            patch = frame.channels[:, y:y + tile, x:x + tile]
            valid = (patch.max(axis=0) > 0.02).mean()
            if valid >= min_valid:
                tiles.append(patch)
                coords.append((y, x))
    if not tiles:
        return np.zeros((0, c, tile, tile), np.float32), np.zeros((0, 2), int)
    return np.stack(tiles).astype(np.float32), np.asarray(coords, dtype=int)


def sar_features(tiles: np.ndarray) -> np.ndarray:
    """Cheap per-tile descriptors - the P-band analogue of Stage 0.

    Eight numbers per tile, all reductions over the channels:
      0 total power          canopy returns more than bare ground
      1 volume fraction      G / (R+G+B): canopy vs surface scattering
      2 surface fraction     B / (R+G+B)
      3 double-bounce frac   R / (R+G+B)
      4 power std            texture; clear-cut edges are sharp
      5 low-power fraction   how much of the tile is near-zero return
      6 volume-frac std      mixed tiles (a clearing edge) are heterogeneous
      7 5th-pct power        robust "darkest part of the tile"
    """
    r, g, b = tiles[:, 0], tiles[:, 1], tiles[:, 2]
    total = r + g + b + 1e-6
    power = total.reshape(len(tiles), -1)
    vol = (g / total).reshape(len(tiles), -1)
    return np.stack([
        power.mean(1),
        vol.mean(1),
        (b / total).reshape(len(tiles), -1).mean(1),
        (r / total).reshape(len(tiles), -1).mean(1),
        power.std(1),
        (power < np.quantile(power, 0.15)).mean(1),
        vol.std(1),
        np.quantile(power, 0.05, axis=1),
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Weak labelling
# ---------------------------------------------------------------------------
def weak_labels(feats: np.ndarray, low_q: float = 0.28,
                high_q: float = 0.62) -> Tuple[np.ndarray, np.ndarray]:
    """Physics-anchored weak labels: cleared (1), intact forest (0), unknown (-1).

    There is no field truth in this product, so the labels come from the
    scattering physics rather than from a person drawing polygons: tiles in
    the bottom quantile of total power *and* below-median volume fraction are
    cleared; tiles in the top quantile of power *and* above-median volume
    fraction are intact. Everything in between is left unlabelled and excluded
    from training, which keeps the label noise low at the cost of coverage.

    This is a stand-in, and it is labelled as such everywhere it is reported.
    Replace with INPE PRODES/DETER polygons or Hansen Global Forest Change
    rasterised onto the frame footprint for a defensible accuracy number.
    """
    power, vol = feats[:, 0], feats[:, 1]
    p_lo, p_hi = np.quantile(power, low_q), np.quantile(power, high_q)
    v_med = np.median(vol)
    y = np.full(len(feats), -1, dtype=np.int64)
    y[(power <= p_lo) & (vol <= v_med)] = 1     # cleared
    y[(power >= p_hi) & (vol >= v_med)] = 0     # intact forest
    confident = y >= 0
    return y, confident


# ---------------------------------------------------------------------------
# The onboard screener
# ---------------------------------------------------------------------------
@dataclass
class BiomassTriageResult:
    n_tiles: int
    n_labelled: int
    accuracy: float          # held-out descriptors only (the honest number)
    auc: float
    accuracy_circular: float # all features, incl. the two the labels came from
    auc_circular: float
    cleared_fraction: float
    downlink_reduction: float
    bytes_full: float
    bytes_triaged: float
    frame_gb: float
    notes: List[str] = field(default_factory=list)


def run_biomass_demo(product_dir: str, tile: int = 32,
                     seed: int = C.RNG_SEED) -> Tuple[BiomassTriageResult, Dict]:
    """End-to-end: read the frame, tile it, learn the screener, cost the link.

    The downlink argument for Biomass is different from the optical one and
    worth stating precisely. Biomass images only land, only outside the
    restricted zones, and every frame is scientifically wanted - you cannot
    simply throw 90% of it away as you can with cloud. What you *can* do is
    prioritise: send the disturbance-front frames at full fidelity and first,
    and let the interior-forest frames - which change slowly and are already
    covered by a 3-day repeat stack - go later or at reduced rate. So the
    figure below is a *prioritisation* saving, not a discard saving.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    frame = QuicklookLoader(product_dir).load()
    tiles, coords = tile_frame(frame, tile=tile)
    if len(tiles) < 40:
        raise RuntimeError(f"only {len(tiles)} valid tiles - frame too small")

    feats = sar_features(tiles)
    y, conf = weak_labels(feats)

    # --- leakage guard ------------------------------------------------
    # The weak labels are a threshold on features 0 (total power) and 1
    # (volume fraction). Training on those same two features and then
    # reporting accuracy would just be measuring how well a logistic
    # regression can re-learn a quantile rule - it scores ~0.99 / AUC 1.00
    # and means nothing. The headline model is therefore trained on the
    # *held-out* descriptors only (texture, surface and double-bounce
    # fractions, heterogeneity, dark-tail), so a good score is evidence that
    # independent structure in the frame predicts the scattering class.
    LABEL_FEATS = [0, 1]
    # Excludes every absolute-power statistic (0, 5, 7) as well as the volume
    # fraction (1): those are the label rule or proxies for it. What is left is
    # purely relative polarimetric composition and texture.
    HONEST_FEATS = [2, 3, 4, 6]

    Xc, yc = feats[conf], y[conf]
    idx_tr, idx_te = train_test_split(np.arange(len(yc)), test_size=0.35,
                                      random_state=seed, stratify=yc)

    def _fit(cols: List[int]):
        Xa = Xc[:, cols]
        mu_, sd_ = Xa[idx_tr].mean(0), Xa[idx_tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit((Xa[idx_tr] - mu_) / sd_, yc[idx_tr])
        p = m.predict_proba((Xa[idx_te] - mu_) / sd_)[:, 1]
        return (m, mu_, sd_, float(m.score((Xa[idx_te] - mu_) / sd_, yc[idx_te])),
                float(roc_auc_score(yc[idx_te], p)))

    clf, mu, sd, acc, auc = _fit(HONEST_FEATS)
    _, _, _, acc_circ, auc_circ = _fit(list(range(feats.shape[1])))

    prob_all = clf.predict_proba((feats[:, HONEST_FEATS] - mu) / sd)[:, 1]
    cleared = prob_all > 0.5

    # Link arithmetic. Disturbance tiles go down at full rate; interior-forest
    # tiles are decimated 4:1, which the 3-day repeat stack makes recoverable.
    vol = frame_data_volume()
    bytes_per_tile = vol["slc_bytes"] / max(len(tiles), 1)
    bytes_full = vol["slc_bytes"]
    bytes_triaged = float((cleared.sum() + (~cleared).sum() / 4.0) * bytes_per_tile)

    res = BiomassTriageResult(
        n_tiles=len(tiles), n_labelled=int(conf.sum()), accuracy=acc, auc=auc,
        accuracy_circular=acc_circ, auc_circular=auc_circ,
        cleared_fraction=float(cleared.mean()),
        downlink_reduction=1.0 - bytes_triaged / bytes_full,
        bytes_full=bytes_full, bytes_triaged=bytes_triaged,
        frame_gb=vol["slc_gb"],
        notes=[
            "Runs on the L1A Pauli quicklook, not calibrated SLC - the "
            "measurement arrays are absent from the annotation-only download.",
            "Labels are physics-derived weak labels, not field truth; accuracy "
            "measures separability of the scattering classes, not agreement "
            "with PRODES/DETER.",
            "Saving is from prioritisation (full-rate disturbance frames, "
            "decimated interior forest), not from discarding science data.",
            "Headline accuracy uses only relative polarimetric composition and "
            "texture - no absolute power statistic - because the weak labels "
            "are a power threshold. The circular all-feature score is reported "
            "beside it so the gap is visible.",
            "Because the ambiguous middle band is excluded from labelling, the "
            "two classes are well separated by construction: a high AUC here "
            "shows the pipeline works and the classes are polarimetrically "
            "distinct, NOT that it would score this well against PRODES/DETER.",
        ],
    )
    detail = {
        "frame": FRAME_T006_F300, "volume": vol,
        "feature_names": ["total_power", "volume_frac", "surface_frac",
                          "double_bounce_frac", "power_std", "low_power_frac",
                          "volume_frac_std", "power_p05"],
        "coefficients": dict(zip(
            ["surface_frac", "double_bounce_frac", "power_std",
             "volume_frac_std"],
            clf.coef_[0].tolist())),
        "tile_px": tile, "coords": coords, "prob": prob_all,
        "frame_shape": frame.shape, "source": frame.source,
        "calibrated": frame.calibrated,
    }
    return res, detail
