"""
Onboard RFI screening for BIOMASS P-band SAR - on real mission data.
====================================================================

This is the strongest of the Biomass onboard-AI cases, and the one that needs
no external truth data at all.

The problem
-----------
Biomass transmits at 435 MHz with a **6 MHz** allocation - a tiny slice of
spectrum shared with terrestrial services, and P-band is notoriously
contended. When an interferer sits in the band, the processor has to notch it
out, and the notched bandwidth is gone: slant-range resolution is c/(2B), so
losing bandwidth directly coarsens the product.

Unlike cloud, RFI cannot be predicted from a climatology and cannot be seen in
the imagery until after focusing. But it *can* be measured, cheaply, from the
raw echo spectrum - which is precisely the kind of decision that belongs
onboard rather than on the ground.

What this module runs on
------------------------
Real data from the delivered products, not a simulation:

* `rfiMitigation/rfiFreqMask{HH,HV,VH,VV}` in the L1A LUT - a
  329  x  87 binary mask (azimuth block  x  frequency bin) recording exactly which
  parts of the spectrum the ground processor had to notch.
* `rfiMitigation/rfi{Isolated,Persistent}FMReportList` in the product
  annotation - per-polarisation summary statistics.

Two frames from the same day are compared:
  T006/F300  Rondônia, Brazil  (-12.6, -58.6)  - a quiet-spectrum case
  T007/F132  Fujian, China     (+26.5, +117.9) - a contended-spectrum case

The contrast is large and it is measured, not assumed.
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

C_LIGHT = 299_792_458.0
NOMINAL_BANDWIDTH_HZ = 6.0e6      # ITU allocation at 435 MHz
SAMPLED_BANDWIDTH_HZ = 7.5652e6   # total sampled bandwidth (from annotation)
CENTRE_FREQ_MHZ = 435.0


def slant_range_resolution_m(bandwidth_hz: float) -> float:
    """c / (2B). The whole reason bandwidth loss is not a cosmetic problem."""
    return C_LIGHT / (2.0 * max(bandwidth_hz, 1.0))


# ---------------------------------------------------------------------------
# annotation summary
# ---------------------------------------------------------------------------
@dataclass
class RfiSummary:
    """Per-frame RFI statistics as reported by the ground processor."""
    frame: str
    isolated_pct_affected_lines: float
    isolated_avg_pct_bw: float
    isolated_max_pct_bw: float
    persistent_pct_affected_lines: float
    persistent_avg_pct_bw: float
    persistent_max_pct_bw: float

    @property
    def effective_bandwidth_hz(self) -> float:
        """Bandwidth surviving the persistent notch, on average."""
        return NOMINAL_BANDWIDTH_HZ * (1.0 - self.persistent_avg_pct_bw / 100.0)

    @property
    def worst_case_bandwidth_hz(self) -> float:
        return NOMINAL_BANDWIDTH_HZ * (1.0 - self.persistent_max_pct_bw / 100.0)

    @property
    def resolution_m(self) -> float:
        return slant_range_resolution_m(self.effective_bandwidth_hz)

    @property
    def worst_resolution_m(self) -> float:
        return slant_range_resolution_m(self.worst_case_bandwidth_hz)

    @property
    def resolution_penalty(self) -> float:
        """Fractional coarsening versus the nominal 6 MHz product."""
        nominal = slant_range_resolution_m(NOMINAL_BANDWIDTH_HZ)
        return self.resolution_m / nominal - 1.0

    def as_dict(self) -> Dict:
        return {
            "frame": self.frame,
            "isolated_pct_affected_lines": self.isolated_pct_affected_lines,
            "isolated_avg_pct_bw": self.isolated_avg_pct_bw,
            "persistent_pct_affected_lines": self.persistent_pct_affected_lines,
            "persistent_avg_pct_bw": self.persistent_avg_pct_bw,
            "persistent_max_pct_bw": self.persistent_max_pct_bw,
            "effective_bandwidth_mhz": self.effective_bandwidth_hz / 1e6,
            "resolution_m": self.resolution_m,
            "worst_resolution_m": self.worst_resolution_m,
            "resolution_penalty": self.resolution_penalty,
        }


def read_rfi_summary(annot_path: str, frame_label: str = "") -> RfiSummary:
    """Parse the `rfiMitigation` DSR of a Biomass L1A annotation."""
    root = ET.parse(annot_path).getroot()
    rfi = root.find("rfiMitigation")
    if rfi is None:
        raise ValueError(f"no rfiMitigation DSR in {annot_path}")

    def first(list_tag: str, item_tag: str) -> ET.Element:
        lst = rfi.find(list_tag)
        if lst is None or len(lst) == 0:
            raise ValueError(f"missing {list_tag}")
        # The mask is OR-combined across polarisations, so all four entries
        # are identical; taking the first is exact, not an approximation.
        return lst[0]

    iso = first("rfiIsolatedFMReportList", "rfiIsolatedFMReport")
    per = first("rfiPersistentFMReportList", "rfiPersistentFMReport")

    def val(e: ET.Element, tag: str) -> float:
        node = e.find(tag)
        return float(node.text) if node is not None and node.text else float("nan")

    return RfiSummary(
        frame=frame_label or os.path.basename(annot_path),
        isolated_pct_affected_lines=val(iso, "percentageAffectedLines"),
        isolated_avg_pct_bw=val(iso, "avgPercentageAffectedBW"),
        isolated_max_pct_bw=val(iso, "maxPercentageAffectedBW"),
        persistent_pct_affected_lines=val(per, "percentageAffectedLines"),
        persistent_avg_pct_bw=val(per, "avgPercentageAffectedBW"),
        persistent_max_pct_bw=val(per, "maxPercentageAffectedBW"),
    )


# ---------------------------------------------------------------------------
# per-block mask from the LUT
# ---------------------------------------------------------------------------
@dataclass
class RfiMaskAnalysis:
    n_blocks: int
    n_freq_bins: int
    identical_across_pol: bool
    notched_fraction_per_block: np.ndarray     # (n_blocks,)
    notched_fraction_per_bin: np.ndarray       # (n_freq_bins,)
    freq_offset_mhz: np.ndarray                # (n_freq_bins,)

    @property
    def mean_notched(self) -> float:
        return float(self.notched_fraction_per_block.mean())

    def blocks_above(self, threshold: float) -> float:
        return float((self.notched_fraction_per_block > threshold).mean())

    def persistent_carriers(self, min_prevalence: float = 0.85
                            ) -> List[Tuple[float, float]]:
        """Frequency bins notched in almost every block - i.e. a fixed emitter
        parked in the band rather than transient interference."""
        idx = np.where(self.notched_fraction_per_bin >= min_prevalence)[0]
        return [(float(self.freq_offset_mhz[i]),
                 float(self.notched_fraction_per_bin[i])) for i in idx]


def read_rfi_mask(lut_path: str) -> RfiMaskAnalysis:
    """Read `rfiMitigation/rfiFreqMask*` from the L1A LUT netCDF."""
    import netCDF4 as nc

    ds = nc.Dataset(lut_path)
    if "rfiMitigation" not in ds.groups:
        raise ValueError(f"no rfiMitigation group in {lut_path}")
    grp = ds.groups["rfiMitigation"]
    masks = {k: np.asarray(grp[k][:]) for k in grp.variables
             if k.startswith("rfiFreqMask")}
    if not masks:
        raise ValueError("no rfiFreqMask variables")
    ref = next(iter(masks.values()))
    identical = all(np.array_equal(ref, m) for m in masks.values())

    n_blocks, n_bins = ref.shape
    per_block = ref.mean(axis=1)
    per_bin = ref.mean(axis=0)
    offsets = ((np.arange(n_bins) - n_bins / 2.0) / n_bins
               * SAMPLED_BANDWIDTH_HZ / 1e6)
    return RfiMaskAnalysis(n_blocks, n_bins, identical, per_block, per_bin, offsets)


# ---------------------------------------------------------------------------
# the onboard decision
# ---------------------------------------------------------------------------
@dataclass
class RfiTriageResult:
    n_blocks: int
    mean_notched: float
    blocks_nominal: float          # fraction fit for full-rate downlink
    blocks_degraded: float
    blocks_severely_degraded: float
    downlink_reduction: float
    resolution_nominal_m: float
    resolution_mean_m: float
    resolution_worst_m: float
    persistent_carriers: List[Tuple[float, float]]
    notes: List[str] = field(default_factory=list)


def onboard_rfi_triage(mask: RfiMaskAnalysis,
                       degraded_threshold: float = 0.08,
                       severe_threshold: float = 0.20,
                       degraded_rate: float = 0.5,
                       severe_rate: float = 0.15) -> RfiTriageResult:
    """Rank azimuth blocks by how much usable bandwidth survives.

    The policy is deliberately conservative and reversible: nothing is deleted.
    Blocks whose bandwidth is largely intact go down at full rate; blocks the
    interference has degraded are decimated and deferred; severely corrupted
    blocks are reduced to a summary. Everything is flagged, so the ground can
    always ask for a re-downlink of what was held back.

    This is the right shape for an onboard science decision: the spacecraft
    decides *ordering and fidelity*, never whether the science existed.
    """
    frac = mask.notched_fraction_per_block
    severe = frac > severe_threshold
    degraded = (frac > degraded_threshold) & ~severe
    nominal = ~(severe | degraded)

    kept = (nominal.mean() * 1.0 + degraded.mean() * degraded_rate
            + severe.mean() * severe_rate)

    bw_mean = NOMINAL_BANDWIDTH_HZ * (1.0 - mask.mean_notched)
    bw_worst = NOMINAL_BANDWIDTH_HZ * (1.0 - float(frac.max()))

    return RfiTriageResult(
        n_blocks=mask.n_blocks,
        mean_notched=mask.mean_notched,
        blocks_nominal=float(nominal.mean()),
        blocks_degraded=float(degraded.mean()),
        blocks_severely_degraded=float(severe.mean()),
        downlink_reduction=1.0 - kept,
        resolution_nominal_m=slant_range_resolution_m(NOMINAL_BANDWIDTH_HZ),
        resolution_mean_m=slant_range_resolution_m(bw_mean),
        resolution_worst_m=slant_range_resolution_m(bw_worst),
        persistent_carriers=mask.persistent_carriers(),
        notes=[
            "RFI masks are the ground processor's own notch decisions, read "
            "from the delivered L1A LUT - this is measured interference, not "
            "a model of it.",
            "The onboard equivalent would derive the same mask from the raw "
            "echo spectrum before focusing, which is a periodogram and a "
            "threshold: cheaper than the cloud screener, and it needs no "
            "training data or labels at all.",
            "Nothing is discarded. The policy sets downlink rate and ordering; "
            "every block is flagged so the ground can request a re-downlink.",
        ],
    )


# ---------------------------------------------------------------------------
def find_products(*roots: str) -> Dict[str, Dict[str, str]]:
    """Locate annotation and LUT files for any Biomass products available."""
    found: Dict[str, Dict[str, str]] = {}
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for annot in glob.glob(os.path.join(root, "**", "*_annot.xml"),
                               recursive=True):
            key = os.path.basename(annot).replace("_annot.xml", "")
            entry = found.setdefault(key, {})
            entry["annot"] = annot
        for lut in glob.glob(os.path.join(root, "**", "*_lut.nc"), recursive=True):
            key = os.path.basename(lut).replace("_lut.nc", "")
            found.setdefault(key, {})["lut"] = lut
    return found


def compare_frames(products: Dict[str, Dict[str, str]],
                   labels: Optional[Dict[str, str]] = None) -> List[Dict]:
    """RFI summary for every frame we can read."""
    labels = labels or {}
    out = []
    for key, paths in sorted(products.items()):
        if "annot" not in paths:
            continue
        try:
            s = read_rfi_summary(paths["annot"], labels.get(key, key))
        except Exception as exc:                       # noqa: BLE001
            out.append({"frame": key, "error": str(exc)})
            continue
        out.append(s.as_dict())
    return out
