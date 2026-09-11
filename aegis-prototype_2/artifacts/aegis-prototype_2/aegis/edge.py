"""
Edge-SoC model: latency, energy, memory and radiation effects.
==============================================================

Two things are modelled here that a pure-software ML prototype usually
skips, and that decide whether onboard AI is actually flyable:

1. **Cost.** How long does an inference take on a real payload processor,
   and how many joules does it cost? The latency model is calibrated against
   a published in-orbit measurement (CloudScout on Phi-Sat-1: 325 ms for a
   512x512x3 input on a Myriad 2 at ~2 W) rather than a vendor peak-TOPS
   number, which no real network ever reaches.

2. **Radiation.** A neural network in orbit is a large block of memory being
   continuously corrupted by single-event upsets. INT8 weights are
   particularly nasty: a flip of bit 7 changes a weight by 128 quantisation
   steps. `RadiationModel` injects real bit flips and measures the accuracy
   cliff with and without mitigation, which is the argument for why you
   scrub rather than hope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# latency / energy
# ---------------------------------------------------------------------------
@dataclass
class InferenceCost:
    latency_ms: float
    energy_mj: float          # millijoules
    ops: float                # INT8 ops (1 MAC = 2 ops)

    def __add__(self, other: "InferenceCost") -> "InferenceCost":
        return InferenceCost(self.latency_ms + other.latency_ms,
                             self.energy_mj + other.energy_mj,
                             self.ops + other.ops)


def calibrate_utilisation(hw: C.HardwareProfile,
                          anchor: Dict = C.CLOUDSCOUT_ANCHOR) -> float:
    """Fraction of sustained throughput a real CNN achieves on this device.

    Solved from the Phi-Sat-1 CloudScout measurement. Comes out around 0.12
    for the Myriad 2, i.e. an unoptimised network reaches roughly an eighth
    of the device's sustained rate - which is exactly why "it's a 1 TOPS
    part" tells you almost nothing about whether your model closes.
    """
    ops = anchor["gmacs"] * 1e9 * 2.0
    achieved_gops = ops / (anchor["latency_ms"] / 1000.0) / 1e9
    return float(np.clip(achieved_gops / hw.int8_gops_effective, 0.02, 1.0))


class EdgeProcessor:
    """Cost model for one payload processor."""

    def __init__(self, key: str = C.DEFAULT_HARDWARE,
                 utilisation: Optional[float] = None):
        if key not in C.HARDWARE:
            raise KeyError(f"unknown hardware '{key}'; have {list(C.HARDWARE)}")
        self.key = key
        self.hw = C.HARDWARE[key]
        # The Myriad 2 gets the measured utilisation. Other devices are
        # scaled from it - documented as an assumption, not a measurement.
        base = calibrate_utilisation(C.HARDWARE["myriad2"])
        if utilisation is not None:
            self.utilisation = utilisation
        elif key == "myriad2":
            self.utilisation = base
        elif key == "leon3_baseline":
            self.utilisation = 0.55     # scalar CPU: no dark silicon to waste
        else:
            self.utilisation = base * 1.35   # newer toolchains, better mapping
        self.utilisation = float(np.clip(self.utilisation, 0.02, 1.0))

    @property
    def sustained_gops(self) -> float:
        return self.hw.int8_gops_effective * self.utilisation

    def cost_for_macs(self, macs: float, precision: str = "int8") -> InferenceCost:
        """`precision` fp32 costs ~4x the time of int8 on these devices."""
        ops = macs * 2.0
        penalty = {"int8": 1.0, "fp16": 2.0, "fp32": 4.0}[precision]
        latency_s = ops * penalty / (self.sustained_gops * 1e9)
        energy_j = latency_s * self.hw.power_active_w
        return InferenceCost(latency_s * 1000.0, energy_j * 1000.0, ops)

    def cost_for_flops_cpu(self, flops: float) -> InferenceCost:
        """Stage 0 runs on the housekeeping CPU, not the accelerator."""
        cpu_gflops = 0.30          # ASSUMPTION: rad-hard OBC, ~0.3 GFLOP/s
        cpu_w = 0.9
        latency_s = flops / (cpu_gflops * 1e9)
        return InferenceCost(latency_s * 1000.0, latency_s * cpu_w * 1000.0, flops)

    def idle_energy_mj(self, seconds: float) -> float:
        return self.hw.power_idle_w * seconds * 1000.0

    def summary(self) -> Dict:
        return {
            "key": self.key, "name": self.hw.name, "vendor": self.hw.vendor,
            "sustained_gops": self.sustained_gops,
            "utilisation": self.utilisation,
            "power_active_w": self.hw.power_active_w,
            "power_idle_w": self.hw.power_idle_w,
            "dram_mb": self.hw.dram_mb, "mass_g": self.hw.mass_g,
            "tid_krad": self.hw.tid_krad,
            "flight_heritage": self.hw.flight_heritage,
        }


def cascade_cost(proc: EdgeProcessor, stage0_flops: float, stage1_macs: float,
                 n_tiles: int, gate_pass_rate: float,
                 detect_rate: float) -> Dict[str, float]:
    """Total cost of running the three-stage cascade over `n_tiles`.

    The cascade is the whole economic argument: Stage 1 only ever sees the
    tiles Stage 0 could not confidently reject, so the average cost per tile
    is far below the cost of the CNN itself.
    """
    c0 = proc.cost_for_flops_cpu(stage0_flops)
    c1 = proc.cost_for_macs(stage1_macs, "int8")
    # Stage 2 (ROI extraction from a 16x16 mask) is a few thousand ops.
    c2 = proc.cost_for_flops_cpu(4096.0)

    n1 = n_tiles * gate_pass_rate
    n2 = n_tiles * detect_rate
    total_latency = n_tiles * c0.latency_ms + n1 * c1.latency_ms + n2 * c2.latency_ms
    total_energy = n_tiles * c0.energy_mj + n1 * c1.energy_mj + n2 * c2.energy_mj

    naive_latency = n_tiles * c1.latency_ms
    naive_energy = n_tiles * c1.energy_mj
    return {
        "stage0_ms": c0.latency_ms, "stage0_mj": c0.energy_mj,
        "stage1_ms": c1.latency_ms, "stage1_mj": c1.energy_mj,
        "tiles": float(n_tiles), "stage1_invocations": n1,
        "total_latency_s": total_latency / 1000.0,
        "total_energy_j": total_energy / 1000.0,
        "mean_latency_ms_per_tile": total_latency / max(n_tiles, 1),
        "mean_energy_mj_per_tile": total_energy / max(n_tiles, 1),
        "cnn_on_everything_energy_j": naive_energy / 1000.0,
        "cascade_energy_saving": 1.0 - (total_energy / max(naive_energy, 1e-9)),
        "cnn_on_everything_latency_s": naive_latency / 1000.0,
    }


# ---------------------------------------------------------------------------
# radiation
# ---------------------------------------------------------------------------
@dataclass
class RadiationResult:
    flips: int
    exposure_days: float
    mitigation: str
    event_macro_f1: float
    cloud_mae: float
    max_weight_error: float


class RadiationModel:
    """Single-event upsets in INT8 weight memory, with mitigation options.

    Mitigations modelled:
      none      - the weights rot and nobody notices until the science does
      ecc       - SECDED on weight memory: corrects single-bit, so only
                  multi-bit-per-word upsets survive (~4% of events here)
      scrub     - periodic refresh from a golden copy held in rad-hard
                  non-volatile memory; residual corruption is whatever
                  accumulates within one scrub interval
      ecc+scrub - both, which is what a real payload would fly
    """

    MITIGATIONS = ("none", "ecc", "scrub", "ecc+scrub")

    def __init__(self, hw: C.HardwareProfile, scrub_interval_min: float = 30.0,
                 seed: int = C.RNG_SEED):
        self.hw = hw
        self.scrub_interval_min = scrub_interval_min
        self.rng = np.random.default_rng(seed)

    def expected_flips(self, weight_bytes: int, days: float,
                       mitigation: str = "none") -> float:
        mbits = weight_bytes * 8 / 1e6
        raw = self.hw.seu_rate_per_mbit_day * mbits * days
        if mitigation == "none":
            return raw
        if mitigation == "ecc":
            return raw * 0.04            # only multi-bit upsets survive SECDED
        if mitigation == "scrub":
            # Only upsets inside the current scrub window are live.
            return raw * (self.scrub_interval_min / (days * 24 * 60))
        if mitigation == "ecc+scrub":
            return raw * 0.04 * (self.scrub_interval_min / (days * 24 * 60))
        raise ValueError(mitigation)

    # -- injection ---------------------------------------------------------
    @staticmethod
    def fake_quantise(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric per-output-channel INT8 quantisation.

        Per-channel, not per-tensor, because that is what the deployed ONNX
        model uses. Measuring radiation tolerance on a per-tensor model would
        be measuring a network that never flies - and it starts from a lower
        accuracy, which flatters the degradation curve.
        """
        flat = w.reshape(w.shape[0], -1)
        scale = np.abs(flat).max(axis=1) / 127.0
        scale = np.where(scale <= 0, 1e-8, scale).astype(np.float32)
        q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8)
        return q.reshape(w.shape), scale

    def inject(self, q: np.ndarray, n_flips: int) -> Tuple[np.ndarray, float]:
        """Flip `n_flips` uniformly-random bits in an INT8 tensor.

        Bit position matters enormously: flipping the sign/MSB of an INT8
        weight moves it by up to 128 steps, which is why unmitigated upsets
        in a quantised network are so much worse than in an FP32 one where
        the exponent is usually protected by ECC anyway.
        """
        flat = q.reshape(-1).astype(np.int16)
        if n_flips <= 0 or flat.size == 0:
            return q, 0.0
        idx = self.rng.integers(0, flat.size, size=n_flips)
        bit = self.rng.integers(0, 8, size=n_flips)
        before = flat[idx].copy()
        flat[idx] = flat[idx] ^ (1 << bit)
        flat = np.clip(flat, -128, 127)
        max_err = float(np.abs(flat[idx] - before).max()) if n_flips else 0.0
        return flat.astype(np.int8).reshape(q.shape), max_err

    def corrupt_model(self, model, n_flips: int):
        """Apply SEUs across a torch model's conv/linear weights, in place.

        Flips are distributed proportionally to tensor size, which is what
        physically happens - the biggest tensor takes the most hits.
        """
        import torch
        import torch.nn as nn
        targets = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        sizes = np.array([m.weight.numel() for m in targets], dtype=float)
        if sizes.sum() == 0 or n_flips <= 0:
            return 0.0
        share = self.rng.multinomial(int(n_flips), sizes / sizes.sum())
        max_err = 0.0
        with torch.no_grad():
            for mod, k in zip(targets, share):
                w = mod.weight.detach().numpy()
                q, scale = self.fake_quantise(w)
                q2, err = self.inject(q, int(k))
                max_err = max(max_err, float(err * scale.max()))
                deq = (q2.reshape(q2.shape[0], -1).astype(np.float32)
                       * scale[:, None]).reshape(w.shape)
                mod.weight.copy_(torch.from_numpy(deq))
        return max_err

    @staticmethod
    def quantise_model_inplace(model):
        """Apply INT8 fake-quantisation with no corruption - the control."""
        import torch
        import torch.nn as nn
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, (nn.Conv2d, nn.Linear)):
                    w = mod.weight.detach().numpy()
                    q, scale = RadiationModel.fake_quantise(w)
                    deq = (q.reshape(q.shape[0], -1).astype(np.float32)
                           * scale[:, None]).reshape(w.shape)
                    mod.weight.copy_(torch.from_numpy(deq))


def weight_bytes_int8(model) -> int:
    import torch.nn as nn
    return int(sum(m.weight.numel() for m in model.modules()
                   if isinstance(m, (nn.Conv2d, nn.Linear))))
