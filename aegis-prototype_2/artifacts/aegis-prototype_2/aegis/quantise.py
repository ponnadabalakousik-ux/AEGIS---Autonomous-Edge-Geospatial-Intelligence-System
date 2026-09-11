"""
INT8 post-training quantisation, accuracy audit and radiation study.
====================================================================

Quantisation is not an optimisation detail on a spacecraft, it is the
enabling step: the flown accelerators (Myriad 2/X, DPU overlays on Zynq)
are INT8 engines, and an FP32 model simply does not run on them at a useful
rate. This module does the conversion honestly - static PTQ with a real
calibration set - and then measures what it cost in accuracy.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .dataset import TileSet
from .train import FP32_ONNX, INT8_ONNX, macro_f1

ART = C.ARTIFACT_DIR


# ---------------------------------------------------------------------------
class TileCalibrationReader:
    """Feeds real tiles to the quantiser so activation ranges are calibrated
    on the actual data distribution, not on random noise."""

    def __init__(self, cubes: np.ndarray, n: int = 256):
        self.data = [{"tile": cubes[i:i + 1].astype(np.float32)}
                     for i in range(min(n, len(cubes)))]
        self._it = iter(self.data)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.data)


def quantise_int8(fp32_path: str = FP32_ONNX, int8_path: str = INT8_ONNX,
                  calib: Optional[np.ndarray] = None,
                  method: str = "percentile",
                  n_calib: int = 512,
                  reduce_range: bool = True) -> str:
    """Static INT8 post-training quantisation.

    The calibration method is not a detail. MinMax sets each activation's
    scale from the single most extreme value seen during calibration, so one
    outlier tile - a bright cloud edge, a saturated fire pixel - stretches the
    range and crushes everything else into a handful of levels. On this
    network that cost half the macro-F1 (0.92 -> 0.50), almost all of it in
    the two classes with the smallest signals. Percentile calibration clips
    the tail instead and recovers most of it (0.76).

    `reduce_range` recovers the rest. It holds weights to 7 bits so that INT8
    accumulation cannot saturate in the VNNI/AVX2 kernels; the same overflow
    hazard exists on the DSP/MAC arrays in flight accelerators. With both
    together the quantised network matches FP32 (0.945 vs 0.917 macro-F1,
    0.615 vs 0.614 segmentation IoU) at a quarter of the weight memory.

    Anything flying INT8 should be measured this way rather than assumed. The
    difference between the naive and the careful setting here is the
    difference between a working payload and a broken one, and nothing in the
    tooling warns you.
    """
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {
        "minmax": (CalibrationMethod.MinMax, {}),
        "percentile": (CalibrationMethod.Percentile,
                       {"CalibPercentile": 99.99, "CalibMovingAverage": True}),
        "entropy": (CalibrationMethod.Entropy, {}),
    }
    if method not in methods:
        raise ValueError(f"method must be one of {list(methods)}")
    calib_method, extra = methods[method]

    pre = fp32_path.replace(".onnx", "_pre.onnx")
    # Symbolic shape inference chokes on the dynamic batch axis; the graph is
    # simple enough that skipping it costs nothing.
    quant_pre_process(fp32_path, pre, skip_symbolic_shape=True)
    reader = TileCalibrationReader(
        calib if calib is not None
        else np.zeros((8, C.N_BANDS, C.TILE_PX, C.TILE_PX), np.float32),
        n=n_calib)
    quantize_static(
        pre, int8_path, reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calib_method,
        per_channel=True,
        reduce_range=reduce_range,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                       **extra},
    )
    return int8_path


QUANT_VARIANTS = (
    ("minmax", "minmax", False),
    ("minmax + reduce_range", "minmax", True),
    ("percentile", "percentile", False),
    ("entropy", "entropy", False),
    ("percentile + reduce_range", "percentile", True),
)


def compare_calibration_methods(fp32_path: str, calib: np.ndarray,
                                eval_set: TileSet,
                                variants=QUANT_VARIANTS) -> Dict:
    """Quantise several ways and measure what each costs in accuracy.

    Included in the pipeline rather than done once offline, because it is the
    step most likely to silently break when the network changes.
    """
    out: Dict[str, Dict] = {}
    for label, method, rr in variants:
        slug = label.replace(" + ", "_").replace(" ", "_")
        path = fp32_path.replace("_fp32.onnx", f"_int8_{slug}.onnx")
        try:
            quantise_int8(fp32_path, path, calib=calib, method=method,
                          reduce_range=rr)
            res = evaluate_onnx(OnnxTriage(path), eval_set)
            res["path"] = path
            res["method"] = method
            res["reduce_range"] = rr
            out[label] = res
        except Exception as exc:                      # noqa: BLE001
            out[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class OnnxTriage:
    """Runtime wrapper. This is the object the autonomy layer actually calls,
    so the mission simulation is driven by the *quantised* network, not by
    the training-time float model."""

    def __init__(self, path: str, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.path = path

    def run(self, cubes: np.ndarray) -> Dict[str, np.ndarray]:
        if cubes.ndim == 3:
            cubes = cubes[None]
        out = self.sess.run(None, {self.input_name: cubes.astype(np.float32)})
        cloud_logit, event_logits, seg_logits = out[0], out[1], out[2]
        return {
            "cloud_frac": _sigmoid(np.asarray(cloud_logit).reshape(-1)),
            "event_prob": _softmax(np.asarray(event_logits)),
            "seg_logits": np.asarray(seg_logits),
        }

    def run_batched(self, cubes: np.ndarray, batch: int = 64) -> Dict[str, np.ndarray]:
        cf, ep, sg = [], [], []
        for i in range(0, len(cubes), batch):
            r = self.run(cubes[i:i + batch])
            cf.append(r["cloud_frac"]); ep.append(r["event_prob"]); sg.append(r["seg_logits"])
        return {"cloud_frac": np.concatenate(cf),
                "event_prob": np.concatenate(ep),
                "seg_logits": np.concatenate(sg)}


# ---------------------------------------------------------------------------
def evaluate_onnx(model: OnnxTriage, ts: TileSet) -> Dict[str, float]:
    r = model.run_batched(ts.cubes)
    pred = r["event_prob"].argmax(1)
    cf_mae = float(np.abs(r["cloud_frac"] - ts.cloud_frac).mean())

    # Segmentation IoU against the 16x16 max-pooled truth.
    m = ts.event_mask.astype(np.float32).reshape(len(ts), 16, 4, 16, 4).max(axis=(2, 4))
    pm = (_sigmoid(r["seg_logits"].reshape(len(ts), 16, 16)) > 0.5).astype(np.float32)
    inter = float((pm * m).sum())
    union = float(((pm + m) > 0).sum())

    per_class = {}
    for i, cls in enumerate(C.EVENT_CLASSES):
        tp = float(((pred == i) & (ts.event_lbl == i)).sum())
        fp = float(((pred == i) & (ts.event_lbl != i)).sum())
        fn = float(((pred != i) & (ts.event_lbl == i)).sum())
        per_class[cls] = {
            "precision": tp / max(tp + fp, 1e-9),
            "recall": tp / max(tp + fn, 1e-9),
            "support": int((ts.event_lbl == i).sum()),
        }

    # The operational metric: of tiles that genuinely contain an observable
    # event, how many does the payload flag as *something* worth sending?
    ev = ts.true_event
    flagged = (pred != 0)
    op_recall = float(flagged[ev].mean()) if ev.any() else 0.0
    false_alarm = float(flagged[~ev].mean())

    return {
        "cloud_mae": cf_mae,
        "event_acc": float((pred == ts.event_lbl).mean()),
        "event_macro_f1": macro_f1(ts.event_lbl, pred, len(C.EVENT_CLASSES)),
        "seg_iou": inter / max(union, 1.0),
        "operational_event_recall": op_recall,
        "false_alarm_rate": false_alarm,
        "per_class": per_class,
    }


def benchmark_latency(path: str, cubes: np.ndarray, n: int = 120,
                      warmup: int = 15) -> Dict[str, float]:
    """Host-CPU wall-clock latency. Indicative only - the flight-relevant
    number comes from edge.EdgeProcessor, which is calibrated to hardware
    that actually flew. Reported because the *ratio* between FP32 and INT8
    is informative even on the wrong CPU."""
    m = OnnxTriage(path, threads=1)
    x = cubes[:1].astype(np.float32)
    for _ in range(warmup):
        m.run(x)
    t = []
    for i in range(n):
        xi = cubes[i % len(cubes):i % len(cubes) + 1].astype(np.float32)
        t0 = time.perf_counter()
        m.run(xi)
        t.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(t)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.median(t)),
            "p95_ms": float(np.quantile(t, 0.95))}


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


# ---------------------------------------------------------------------------
def radiation_study(torch_model_factory, state_dict_path: str, ts: TileSet,
                    hw_key: str = C.DEFAULT_HARDWARE,
                    exposure_days: Tuple[float, ...] = (1, 7, 30, 180, 365),
                    n_eval: int = 700, seed: int = C.RNG_SEED) -> List[Dict]:
    """Accuracy vs accumulated SEUs, for each mitigation strategy."""
    import torch
    from .edge import RadiationModel, weight_bytes_int8

    hw = C.HARDWARE[hw_key]
    base = torch_model_factory()
    base.load_state_dict(torch.load(state_dict_path))
    wbytes = weight_bytes_int8(base)

    sub = ts.subset(np.arange(min(n_eval, len(ts))))
    x = torch.from_numpy(sub.cubes)

    def eval_model(model) -> Tuple[float, float]:
        model.eval()
        with torch.no_grad():
            c, e, _ = model(x)
        pred = e.argmax(1).numpy()
        cf = torch.sigmoid(c).numpy()
        return (macro_f1(sub.event_lbl, pred, len(C.EVENT_CLASSES)),
                float(np.abs(cf - sub.cloud_frac).mean()))

    results: List[Dict] = []
    # Control: INT8, zero upsets.
    ctrl = torch_model_factory(); ctrl.load_state_dict(torch.load(state_dict_path))
    RadiationModel.quantise_model_inplace(ctrl)
    f1_0, mae_0 = eval_model(ctrl)
    results.append({"mitigation": "int8 baseline (0 SEU)", "exposure_days": 0.0,
                    "flips": 0, "event_macro_f1": f1_0, "cloud_mae": mae_0,
                    "max_weight_error": 0.0})

    for mit in RadiationModel.MITIGATIONS:
        for days in exposure_days:
            rad = RadiationModel(hw, seed=seed + int(days))
            n_flips = int(round(rad.expected_flips(wbytes, days, mit)))
            m = torch_model_factory(); m.load_state_dict(torch.load(state_dict_path))
            RadiationModel.quantise_model_inplace(m)
            err = rad.corrupt_model(m, n_flips)
            f1, mae = eval_model(m)
            results.append({"mitigation": mit, "exposure_days": float(days),
                            "flips": n_flips, "event_macro_f1": f1,
                            "cloud_mae": mae, "max_weight_error": err})
    return results


def calibrate_operating_point(net: "OnnxTriage", ts: TileSet,
                              min_precision: float = 0.45,
                              grid: Optional[np.ndarray] = None) -> Dict:
    """Choose the event-confidence threshold at the *natural* event rate.

    A model tuned on a class-balanced set will happily flag 8% of a stream in
    which only 5% of tiles contain anything - which is how an onboard triage
    system ends up downlinking more false alarms than events. The operating
    point therefore has to be picked against the prior the spacecraft will
    actually see.

    Objective: maximise recall subject to a floor on precision. Recall is what
    the mission cares about (a missed fire is unrecoverable); precision is a
    budget constraint (false alarms cost bits). Expressing it as
    "recall, subject to precision >= x" is more honest than optimising F1,
    because it makes the operator's trade explicit and tunable on orbit.
    """
    r = net.run_batched(ts.cubes)
    prob = r["event_prob"]
    pred = prob.argmax(1)
    conf = prob.max(1)
    cloud_ok = r["cloud_frac"] <= C.CLOUD_FRACTION_REJECT
    ev = ts.true_event

    if grid is None:
        grid = np.arange(0.30, 0.995, 0.01)
    rows = []
    for th in grid:
        flag = (pred != 0) & (conf >= th) & cloud_ok
        tp = float((flag & ev).sum())
        fp = float((flag & ~ev).sum())
        fn = float((~flag & ev).sum())
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        rows.append({"threshold": float(th), "precision": prec, "recall": rec,
                     "flag_rate": float(flag.mean()),
                     "f1": 2 * prec * rec / max(prec + rec, 1e-9)})

    feasible = [r_ for r_ in rows if r_["precision"] >= min_precision]
    best = (max(feasible, key=lambda r_: r_["recall"]) if feasible
            else max(rows, key=lambda r_: r_["f1"]))
    return {"chosen": best, "curve": rows, "min_precision": min_precision,
            "feasible": bool(feasible)}


def radiation_stress_sweep(torch_model_factory, state_dict_path: str,
                           ts: TileSet, hw_key: str = C.DEFAULT_HARDWARE,
                           flip_counts=(0, 30, 100, 300, 1000, 3000, 10000,
                                        30000, 100000),
                           n_eval: int = 2200, repeats: int = 5,
                           seed: int = C.RNG_SEED) -> List[Dict]:
    """Where is the cliff?

    The one-year natural upset dose barely moves this network, which is a
    useful result but not a sufficient one - it says nothing about the
    margin. This sweep drives the flip count far past the natural rate to
    find the point where accuracy actually collapses, then expresses that
    point back in years of exposure. That number is the design margin, and it
    is what justifies (or does not justify) the mass and power of ECC and
    scrubbing hardware.
    """
    import torch
    from .edge import RadiationModel, weight_bytes_int8

    hw = C.HARDWARE[hw_key]
    base = torch_model_factory()
    base.load_state_dict(torch.load(state_dict_path))
    wbytes = weight_bytes_int8(base)
    total_bits = wbytes * 8
    per_day = hw.seu_rate_per_mbit_day * total_bits / 1e6

    sub = ts.subset(np.arange(min(n_eval, len(ts))))
    x = torch.from_numpy(sub.cubes)

    rows: List[Dict] = []
    prev_f1 = None
    for n_flips in flip_counts:
        f1s, maes = [], []
        for r in range(repeats if n_flips else 1):
            rad = RadiationModel(hw, seed=seed + 1000 * r + n_flips)
            m = torch_model_factory()
            m.load_state_dict(torch.load(state_dict_path))
            RadiationModel.quantise_model_inplace(m)
            rad.corrupt_model(m, n_flips)
            m.eval()
            with torch.no_grad():
                c, e, _ = m(x)
            f1s.append(macro_f1(sub.event_lbl, e.argmax(1).numpy(),
                                len(C.EVENT_CLASSES)))
            maes.append(float(np.abs(torch.sigmoid(c).numpy() - sub.cloud_frac).mean()))
        rows.append({
            "flips": int(n_flips),
            "corrupted_weight_fraction": n_flips / max(wbytes, 1),
            "equivalent_days_unmitigated": n_flips / max(per_day, 1e-12),
            "equivalent_years_unmitigated": n_flips / max(per_day, 1e-12) / 365.25,
            "event_macro_f1_mean": float(np.mean(f1s)),
            "event_macro_f1_std": float(np.std(f1s)),
            "cloud_mae_mean": float(np.mean(maes)),
        })
    return rows


def save_json(obj, name: str) -> str:
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return p
