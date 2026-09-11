"""Training + ONNX export for the onboard cascade."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

import numpy as np

from . import config as C
from .dataset import TileSet, build_datasets, summarise
from .models import PhysicsScreener, build_triagenet, count_macs, count_params

ART = C.ARTIFACT_DIR
FP32_ONNX = os.path.join(ART, "triagenet_fp32.onnx")
INT8_ONNX = os.path.join(ART, "triagenet_int8.onnx")
SCREENER_NPZ = os.path.join(ART, "screener.npz")
TORCH_PT = os.path.join(ART, "triagenet.pt")


# ---------------------------------------------------------------------------
def train_screener(train: TileSet, val: TileSet, verbose: bool = True) -> Dict:
    """Stage 0: the cheap gate."""
    keep_tr = (train.usable | train.true_event).astype(np.float32)
    keep_va = (val.usable | val.true_event).astype(np.float32)

    scr = PhysicsScreener().fit(train.feats, keep_tr)
    scr.calibrate(val.feats, keep_va, min_recall=0.995)

    gate = scr.gate(val.feats)
    recall = float(gate[keep_va.astype(bool)].mean())
    reject_rate = float(1.0 - gate.mean())
    # How much of what it rejects was genuinely junk?
    rejected = ~gate
    precision_of_rejection = float((~keep_va.astype(bool))[rejected].mean()) if rejected.any() else 1.0

    os.makedirs(ART, exist_ok=True)
    np.savez(SCREENER_NPZ, **scr.state_dict())

    stats = {
        "threshold": scr.threshold,
        "recall_on_useful": recall,
        "reject_rate": reject_rate,
        "rejection_precision": precision_of_rejection,
        "flops_per_tile": PhysicsScreener.flops_per_tile(),
    }
    if verbose:
        print(f"  Stage 0 screener: rejects {reject_rate:.1%} of all tiles, "
              f"retains {recall:.2%} of useful ones "
              f"({precision_of_rejection:.1%} of rejections were true junk)")
    return stats


# ---------------------------------------------------------------------------
def _make_loaders(ts: TileSet, batch: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F

    x = torch.from_numpy(ts.cubes)
    cf = torch.from_numpy(ts.cloud_frac)
    lb = torch.from_numpy(ts.event_lbl)
    # Coarse 16x16 target: max-pool preserves small fire fronts that mean-
    # pooling would erase.
    m = torch.from_numpy(ts.event_mask.astype(np.float32)).unsqueeze(1)
    m16 = F.max_pool2d(m, kernel_size=4, stride=4)
    ds = TensorDataset(x, cf, lb, m16)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


def _dice_loss(logits, target, eps: float = 1.0):
    import torch
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()


def train_triagenet(train: TileSet, val: TileSet,
                    cfg: C.TrainConfig = C.DEFAULT_TRAIN,
                    verbose: bool = True) -> Tuple[object, Dict]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    Net = build_triagenet()
    model = Net()
    macs = count_macs(model)
    params = count_params(model)
    if verbose:
        print(f"  TriageNet: {params:,} parameters, {macs/1e6:.2f} MMAC/tile")

    tl = _make_loaders(train, cfg.batch_size, True)
    vl = _make_loaders(val, cfg.batch_size, False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(tl), pct_start=0.25)

    # Class weights counter the residual imbalance left after balanced
    # sampling, and encode that missing a fire costs more than a false alarm.
    # Mildly event-favouring, not aggressively so. An earlier 0.6/1.6 split
    # bought recall at the cost of a false-alarm rate that the downlink
    # budget then had to pay for; the operating point is a better place to
    # make that trade than the loss function, because it can be retuned on
    # orbit without retraining.
    cw = torch.tensor([1.0, 1.25, 1.15, 1.05], dtype=torch.float32)

    history = []
    best = {"score": -1.0, "state": None}
    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for x, cf, lb, m16 in tl:
            # Dihedral augmentation. Satellite imagery has no canonical
            # orientation - the spacecraft flies ascending and descending
            # passes - so the 8-element symmetry group is free extra data and
            # a real regulariser rather than a token one.
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, dims=(2, 3))
                m16 = torch.rot90(m16, k, dims=(2, 3))
            if bool(torch.randint(0, 2, (1,))):
                x = torch.flip(x, dims=(3,))
                m16 = torch.flip(m16, dims=(3,))
            opt.zero_grad(set_to_none=True)
            c_logit, e_logit, seg = model(x)
            loss_cloud = F.binary_cross_entropy_with_logits(c_logit, cf)
            loss_event = F.cross_entropy(e_logit, lb, weight=cw)
            loss_seg = (F.binary_cross_entropy_with_logits(seg, m16)
                        + 0.5 * _dice_loss(seg, m16))
            loss = 1.0 * loss_cloud + 1.4 * loss_event + 0.7 * loss_seg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(x)
        m = evaluate_torch(model, vl)
        # Selection metric weights event recall heavily - the operational
        # failure that matters is a missed fire, not a mis-set cloud number.
        score = 0.55 * m["event_macro_f1"] + 0.25 * (1 - m["cloud_mae"]) + 0.20 * m["seg_iou"]
        history.append({"epoch": ep, "train_loss": run / len(train), **m})
        if score > best["score"]:
            best = {"score": score,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if verbose:
            print(f"    ep {ep+1:02d}/{cfg.epochs} loss {run/len(train):.4f} "
                  f"| cloud MAE {m['cloud_mae']:.3f} "
                  f"| event F1 {m['event_macro_f1']:.3f} "
                  f"| seg IoU {m['seg_iou']:.3f} ({time.time()-t0:.0f}s)")
    model.load_state_dict(best["state"])
    os.makedirs(ART, exist_ok=True)
    torch.save(model.state_dict(), TORCH_PT)
    return model, {"params": params, "macs": macs, "history": history}


def evaluate_torch(model, loader) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F
    model.eval()
    cf_err, n = 0.0, 0
    preds, labels = [], []
    inter, union = 0.0, 0.0
    with torch.no_grad():
        for x, cf, lb, m16 in loader:
            c_logit, e_logit, seg = model(x)
            p_cf = torch.sigmoid(c_logit)
            cf_err += float((p_cf - cf).abs().sum())
            n += len(x)
            preds.append(e_logit.argmax(1).numpy())
            labels.append(lb.numpy())
            pm = (torch.sigmoid(seg) > 0.5).float()
            inter += float((pm * m16).sum())
            union += float(((pm + m16) > 0).float().sum())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "cloud_mae": cf_err / max(n, 1),
        "event_acc": float((preds == labels).mean()),
        "event_macro_f1": macro_f1(labels, preds, len(C.EVENT_CLASSES)),
        "seg_iou": inter / max(union, 1.0),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
def export_onnx(model, path: str = FP32_ONNX) -> str:
    """Export to ONNX - the practical handoff format for space AI toolchains
    (Ubotica's CVAI, Vitis-AI and OpenVINO all ingest ONNX)."""
    import torch
    os.makedirs(ART, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, C.N_BANDS, C.TILE_PX, C.TILE_PX)
    torch.onnx.export(
        model, dummy, path,
        input_names=["tile"],
        output_names=["cloud_logit", "event_logits", "seg_logits"],
        dynamic_axes={"tile": {0: "batch"}, "cloud_logit": {0: "batch"},
                      "event_logits": {0: "batch"}, "seg_logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    return path


def save_report(d: Dict, name: str) -> str:
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, name)
    with open(p, "w") as f:
        json.dump(d, f, indent=2, default=float)
    return p
