#!/usr/bin/env python3
"""
AEGIS end-to-end pipeline.
==========================

    python run_all.py                # full run (regenerates nothing cached)
    python run_all.py --force-data   # regenerate the datasets
    python run_all.py --skip-train   # reuse the trained network
    python run_all.py --days 14      # longer mission campaign

Stages
    1  datasets              synthetic Sentinel-2-like tiles with truth masks
    2  Stage 0 screener      cheap spectral gate
    3  Stage 1 TriageNet     multi-task CNN
    4  INT8 quantisation     ONNX static PTQ
    5  accuracy audit        FP32 vs INT8, natural-prior operating point
    6  edge hardware         latency/energy on four payload processors
    7  radiation             SEU injection with and without mitigation
    8  mission campaign      baseline vs AEGIS, plus ablations
    9  dashboard             self-contained HTML report
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from aegis import config as C
from aegis.dataset import build_datasets, summarise
from aegis.edge import EdgeProcessor, cascade_cost, weight_bytes_int8
from aegis.mission import MissionSimulator, build_tile_pool, compare
from aegis.models import PhysicsScreener, build_triagenet, count_macs, count_params
from aegis.quantise import (OnnxTriage, benchmark_latency,
                            calibrate_operating_point,
                            compare_calibration_methods, evaluate_onnx,
                            model_size_mb, quantise_int8, radiation_study,
                            radiation_stress_sweep, save_json)
from aegis.spacecraft import orbit_summary
from aegis.train import (FP32_ONNX, INT8_ONNX, SCREENER_NPZ, TORCH_PT,
                         export_onnx, train_screener, train_triagenet)

ART = C.ARTIFACT_DIR


def banner(n: int, title: str) -> None:
    print(f"\n{'='*72}\n  STAGE {n}  {title}\n{'='*72}")


def main() -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--days", type=float, default=C.SIM_DAYS)
    ap.add_argument("--hardware", default=C.DEFAULT_HARDWARE, choices=list(C.HARDWARE))
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    out: Dict = {"config": {
        "seed": C.RNG_SEED, "tile_px": C.TILE_PX, "bands": C.BANDS,
        "gsd_m": C.GSD_M, "sim_days": args.days, "hardware": args.hardware,
    }}
    t_start = time.time()

    # ---------------------------------------------------------------- 1
    banner(1, "DATASETS")
    data = build_datasets(force=args.force_data)
    for k in ("train", "val", "val_nat", "test"):
        print("  " + summarise(data[k], k))
    out["datasets"] = {k: {"n": len(v),
                           "usable_fraction": float(v.usable.mean()),
                           "observable_event_rate": float(v.true_event.mean())}
                       for k, v in data.items()}

    # ---------------------------------------------------------------- 2
    banner(2, "STAGE 0 - PHYSICS SCREENER")
    if args.skip_train and os.path.exists(SCREENER_NPZ):
        screener = PhysicsScreener().load_state(np.load(SCREENER_NPZ))
        # Re-measure rather than reload: the reported gate statistics must
        # describe this screener on this data, not whatever was cached.
        keep = (data["val"].usable | data["val"].true_event)
        gate = screener.gate(data["val"].feats)
        rejected = ~gate
        out["screener"] = {
            "threshold": screener.threshold,
            "flops_per_tile": PhysicsScreener.flops_per_tile(),
            "reject_rate": float(1.0 - gate.mean()),
            "recall_on_useful": float(gate[keep].mean()),
            "rejection_precision": float((~keep)[rejected].mean()) if rejected.any() else 1.0,
        }
        print(f"  reused cached screener: rejects "
              f"{out['screener']['reject_rate']:.1%}, retains "
              f"{out['screener']['recall_on_useful']:.2%} of useful tiles")
    else:
        out["screener"] = train_screener(data["train"], data["val"])
        screener = PhysicsScreener().load_state(np.load(SCREENER_NPZ))

    # ---------------------------------------------------------------- 3
    banner(3, "STAGE 1 - TRIAGENET")
    Net = build_triagenet()
    if args.skip_train and os.path.exists(TORCH_PT):
        import torch
        model = Net(); model.load_state_dict(torch.load(TORCH_PT)); model.eval()
        info = {"params": count_params(model), "macs": count_macs(model)}
        print(f"  reused cached network: {info['params']:,} params, "
              f"{info['macs']/1e6:.2f} MMAC")
    else:
        model, info = train_triagenet(data["train"], data["val"])
    out["triagenet"] = {"params": info["params"], "mmac": info["macs"] / 1e6}
    stage1_macs = info["macs"]

    # ---------------------------------------------------------------- 4
    banner(4, "INT8 QUANTISATION")
    export_onnx(model, FP32_ONNX)
    # The calibration study runs first: it is the step most likely to break
    # silently, and the chosen setting is the one the whole mission then uses.
    quant_study = compare_calibration_methods(FP32_ONNX, data["val"].cubes[:512],
                                              data["test"])
    fp32_eval = evaluate_onnx(OnnxTriage(FP32_ONNX), data["test"])
    print(f"  {'FP32 reference':28s} macro-F1 {fp32_eval['event_macro_f1']:.3f}  "
          f"seg IoU {fp32_eval['seg_iou']:.3f}")
    for label, v in quant_study.items():
        if "error" in v:
            print(f"  {label:28s} FAILED: {v['error']}")
            continue
        print(f"  {label:28s} macro-F1 {v['event_macro_f1']:.3f}  "
              f"seg IoU {v['seg_iou']:.3f}  "
              f"(vs FP32 {v['event_macro_f1']-fp32_eval['event_macro_f1']:+.3f})")
    out["quantisation_study"] = {
        k: ({kk: vv for kk, vv in v.items() if kk != "per_class"})
        for k, v in quant_study.items()}
    quantise_int8(FP32_ONNX, INT8_ONNX, calib=data["val"].cubes[:512],
                  method="percentile", reduce_range=False)
    print(f"  FP32 graph {model_size_mb(FP32_ONNX)*1000:.0f} kB  ->  "
          f"INT8 graph {model_size_mb(INT8_ONNX)*1000:.0f} kB")
    print(f"  weight payload: {weight_bytes_int8(model)*4/1024:.1f} kB FP32 "
          f"-> {weight_bytes_int8(model)/1024:.1f} kB INT8 (4.0x)")

    # ---------------------------------------------------------------- 5
    banner(5, "ACCURACY AUDIT (held-out test set, natural event prior)")
    acc: Dict = {}
    for path, tag in ((FP32_ONNX, "fp32"), (INT8_ONNX, "int8")):
        net = OnnxTriage(path)
        m = evaluate_onnx(net, data["test"])
        m["host_latency"] = benchmark_latency(path, data["test"].cubes, n=80)
        m["graph_kb"] = model_size_mb(path) * 1000
        acc[tag] = m
        print(f"  {tag.upper():5s} cloud MAE {m['cloud_mae']:.4f} | "
              f"macro-F1 {m['event_macro_f1']:.3f} | seg IoU {m['seg_iou']:.3f} | "
              f"event recall {m['operational_event_recall']:.3f}")
    print(f"  INT8 accuracy delta: macro-F1 "
          f"{acc['int8']['event_macro_f1']-acc['fp32']['event_macro_f1']:+.4f}, "
          f"cloud MAE {acc['int8']['cloud_mae']-acc['fp32']['cloud_mae']:+.4f}")

    net8 = OnnxTriage(INT8_ONNX)
    cal = calibrate_operating_point(net8, data["val_nat"], min_precision=0.45)
    ch = cal["chosen"]
    print(f"  operating point (chosen on val_nat, never on test): "
          f"conf >= {ch['threshold']:.2f} -> precision {ch['precision']:.3f}, "
          f"recall {ch['recall']:.3f}")
    out["accuracy"] = acc
    out["operating_point"] = cal

    # ---------------------------------------------------------------- 6
    banner(6, "EDGE HARDWARE")
    stage0_flops = PhysicsScreener.flops_per_tile()
    gate_rate = float(screener.gate(data["test"].feats).mean())
    detect_rate = float(((net8.run_batched(data["test"].cubes)["event_prob"]
                          .argmax(1)) != 0).mean())
    hw_rows = []
    for key in C.HARDWARE:
        proc = EdgeProcessor(key)
        cc = cascade_cost(proc, stage0_flops, stage1_macs, 1000, gate_rate, detect_rate)
        row = {**proc.summary(), **cc}
        # Does the cascade keep up with the instrument in real time?
        tiles_per_s = 0.0
        row["required_tiles_per_s"] = 0.0
        hw_rows.append(row)
        print(f"  {proc.hw.name:44s} {proc.sustained_gops:7.1f} GOP/s  "
              f"CNN {cc['stage1_ms']:8.2f} ms  cascade {cc['mean_energy_mj_per_tile']:7.3f} mJ/tile")
    out["hardware"] = hw_rows
    out["cascade"] = {"gate_pass_rate": gate_rate, "detect_rate": detect_rate,
                      "stage0_flops": stage0_flops, "stage1_macs": stage1_macs}

    # ---------------------------------------------------------------- 7
    banner(7, "RADIATION (single-event upsets in INT8 weights)")
    rad = radiation_study(Net, TORCH_PT, data["test"], hw_key=args.hardware)
    for r in rad:
        if r["exposure_days"] in (0.0, 365.0):
            print(f"  {r['mitigation']:22s} {r['exposure_days']:5.0f} d  "
                  f"{r['flips']:7d} flips  macro-F1 {r['event_macro_f1']:.3f}")
    out["radiation"] = rad

    sweep = radiation_stress_sweep(Net, TORCH_PT, data["test"], hw_key=args.hardware)
    intact = sweep[0]["event_macro_f1_mean"]
    knee = next((r for r in sweep if r["event_macro_f1_mean"] < 0.8 * intact), None)
    print("  stress sweep (unmitigated-equivalent exposure to reach each level):")
    for r in sweep:
        print(f"    {r['flips']:7d} flips = {r['equivalent_years_unmitigated']:8.1f} yr  "
              f"macro-F1 {r['event_macro_f1_mean']:.3f} +/- {r['event_macro_f1_std']:.3f}")
    if knee:
        print(f"  20% degradation at {knee['flips']} flips = "
              f"{knee['equivalent_years_unmitigated']:.0f} years unmitigated "
              f"({knee['corrupted_weight_fraction']:.1%} of weights corrupted)")
    out["radiation_sweep"] = sweep

    # ---------------------------------------------------------------- 8
    banner(8, "MISSION CAMPAIGN")
    policy = C.AutonomyPolicy(
        event_confidence_threshold=float(ch["threshold"]),
        alert_confidence_threshold=max(float(ch["threshold"]), 0.90),
    )
    pool = build_tile_pool(data["test"], screener, net8)
    sim = MissionSimulator(pool, policy=policy, hardware=args.hardware,
                           stage1_macs=stage1_macs, days=args.days)
    print(f"  orbit: {sim.orbit.period_s/60:.1f} min period, "
          f"{len(sim.passes)} geometric passes, "
          f"{len(sim.xband_passes)} X-band contacts granted")
    print(f"  instrument: {sim.acq['tiles_per_day']/1e6:.2f} M tiles/day "
          f"= {sim.acq['compressed_gb_per_day']:.1f} GB/day compressed")
    print(f"  ground segment: {sim.dl['capacity_gb_per_day']:.1f} GB/day usable "
          f"-> {sim.acq['compressed_gb_per_day']/sim.dl['capacity_gb_per_day']:.1f}x oversubscribed")
    print(f"  Monte-Carlo weight: 1 simulated tile = {sim.weight:.0f} real tiles")

    base = sim.run("baseline")
    aeg = sim.run("aegis")
    cmp_ = compare(base, aeg)
    for k, v in cmp_["benefits"].items():
        print(f"  {k}: {json.dumps({kk: (round(vv,3) if isinstance(vv,(int,float)) else vv) for kk,vv in v.items()})}")
    resid = abs(aeg.event_accounting_residual) / max(aeg.events_observable, 1.0)
    print(f"  event accounting residual: {resid:.2e} (must be ~0)")
    assert resid < 1e-6, "event accounting does not close - a bucket is leaking"

    out["orbit"] = orbit_summary(sim.orbit, sim.passes, sim.duration_s)
    out["acquisition"] = sim.acq
    out["downlink"] = sim.dl
    out["mission_weight"] = sim.weight
    out["campaign"] = cmp_
    out["traces"] = {
        "baseline_memory": base.memory_trace, "aegis_memory": aeg.memory_trace,
        "baseline_soc": base.soc_trace, "aegis_soc": aeg.soc_trace,
    }

    # ablations: which part of the design is doing the work?
    banner(8.5, "ABLATIONS")
    abl = {}
    variants = {
        "no_cloud_rejection": (C.AutonomyPolicy(**{**policy.__dict__,
                                                   "cloud_reject_threshold": 1.01}), True),
        "no_novelty": (C.AutonomyPolicy(**{**policy.__dict__,
                                           "novelty_decay_hours": 1e-6}), True),
        "no_voi_priority": (policy, False),   # AI on, but FIFO downlink
    }
    for name, (pol, prio) in variants.items():
        s2 = MissionSimulator(pool, policy=pol, hardware=args.hardware,
                              stage1_macs=stage1_macs, days=args.days)
        r = s2.run("aegis", prioritise=prio).summary()
        abl[name] = {"gb_per_day": r["gb_downlinked_per_day"],
                     "event_delivery_rate": r["event_delivery_rate"],
                     "median_latency_min": r["median_latency_min"]}
        print(f"  {name:18s} {r['gb_downlinked_per_day']:7.2f} GB/day  "
              f"delivery {r['event_delivery_rate']:.3f}  "
              f"median latency {r['median_latency_min']:.0f} min")
    # When the link is comfortable, value-ordering has nothing to do: every
    # queued product fits, so the order it goes in is irrelevant. Squeezing
    # the ground segment puts the constraint back. Sweeping it finds the
    # regime where the scheduler earns its place - which is a more useful
    # answer than a single ablation number either way.
    stress = []
    for util in (0.60, 0.10, 0.04, 0.02, 0.01):
        prof = C.MissionProfile(pass_utilisation=util)
        s3 = MissionSimulator(pool, profile=prof, policy=policy,
                              hardware=args.hardware, stage1_macs=stage1_macs,
                              days=args.days)
        voi = s3.run("aegis", prioritise=True).summary()
        fifo = s3.run("aegis", prioritise=False).summary()
        stress.append({
            "pass_utilisation": util,
            "capacity_gb_per_day": s3.dl["capacity_gb_per_day"],
            "demand_gb_per_day": voi["gb_downlinked_per_day"],
            "voi_delivery": voi["event_delivery_rate"],
            "fifo_delivery": fifo["event_delivery_rate"],
            "voi_latency_min": voi["median_latency_min"],
            "fifo_latency_min": fifo["median_latency_min"],
        })
        print(f"  link stress util={util:.2f} cap {s3.dl['capacity_gb_per_day']:6.2f} GB/day  "
              f"VoI delivery {voi['event_delivery_rate']:.3f} @ {voi['median_latency_min']:4.0f} min  |  "
              f"FIFO {fifo['event_delivery_rate']:.3f} @ {fifo['median_latency_min']:4.0f} min")
    out["link_stress"] = stress

    abl["full_aegis"] = {
        "gb_per_day": cmp_["aegis"]["gb_downlinked_per_day"],
        "event_delivery_rate": cmp_["aegis"]["event_delivery_rate"],
        "median_latency_min": cmp_["aegis"]["median_latency_min"]}
    out["ablations"] = abl

    # ---------------------------------------------------------------- 8.6
    banner(8.6, "MISSION ADAPTERS (real spacecraft)")
    from aegis.missions import analyse_all
    cases = analyse_all(hardware=args.hardware, macs_per_tile=stage1_macs)
    for m in cases:
        print(f"  {m['mission']:32s} {m['daily_gbit_before']:7.0f} -> "
              f"{m['daily_gbit_after']:7.0f} Gbit/day "
              f"({m['downlink_reduction']*100:3.0f}%)  compute "
              f"{m['compute_duty_cycle']*100:4.1f}% duty, "
              f"{m['compute_power_fraction_of_platform']*100:.2f}% of platform")
    out["mission_cases"] = cases

    out["runtime_s"] = time.time() - t_start
    save_json(out, "results.json")
    print(f"\n  results -> {os.path.join(ART, 'results.json')}  "
          f"({out['runtime_s']:.0f}s total)")

    # ---------------------------------------------------------------- 8.7
    bio_roots = [os.environ.get("AEGIS_BIOMASS_DIR", ""),
                 os.environ.get("AEGIS_BIOMASS_DIR2", "")]
    bio_roots = [r for r in bio_roots if r]
    if bio_roots:
        banner(8.7, "BIOMASS RFI SCREENING (real mission data)")
        try:
            from aegis.rfi import (find_products, onboard_rfi_triage,
                                   read_rfi_mask, read_rfi_summary)
            prods = find_products(*bio_roots)
            rfi_rows, triage = [], None
            for key, paths in sorted(prods.items()):
                if "annot" not in paths:
                    continue
                label = ("F300 Rondonia, Brazil" if "f300" in key
                         else "F132 Fujian, China" if "f132" in key else key)
                d = read_rfi_summary(paths["annot"], label).as_dict()
                if "lut" in paths:
                    mask = read_rfi_mask(paths["lut"])
                    t = onboard_rfi_triage(mask)
                    d["triage"] = {
                        "n_blocks": t.n_blocks,
                        "blocks_nominal": t.blocks_nominal,
                        "blocks_degraded": t.blocks_degraded,
                        "blocks_severely_degraded": t.blocks_severely_degraded,
                        "downlink_reduction": t.downlink_reduction,
                        "persistent_carriers": t.persistent_carriers,
                        "spectrum": {
                            "freq_offset_mhz": mask.freq_offset_mhz.tolist(),
                            "notched_fraction": mask.notched_fraction_per_bin.tolist(),
                        },
                        "notes": t.notes,
                    }
                    triage = t
                rfi_rows.append(d)
                print(f"  {d['frame']:26s} persistent RFI notches "
                      f"{d['persistent_avg_pct_bw']:5.2f}% of the 6 MHz band "
                      f"(max {d['persistent_max_pct_bw']:5.2f}%) -> "
                      f"{d['resolution_m']:.2f} m range resolution "
                      f"(+{d['resolution_penalty']*100:.1f}%)")
            if triage:
                print(f"  onboard triage on the frame with a LUT: "
                      f"{triage.blocks_nominal*100:.1f}% nominal / "
                      f"{triage.blocks_degraded*100:.1f}% degraded / "
                      f"{triage.blocks_severely_degraded*100:.1f}% severe "
                      f"-> {triage.downlink_reduction*100:.1f}% link saving")
                print("  persistent carriers: " + ", ".join(
                    f"{f:+.2f} MHz ({p*100:.0f}% of blocks)"
                    for f, p in triage.persistent_carriers))
            out["rfi"] = rfi_rows
        except Exception as exc:                        # noqa: BLE001
            print(f"  RFI analysis skipped: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 9
    if not args.no_report:
        banner(9, "DASHBOARD")
        from aegis.report import build_dashboard
        bio = None
        bio_dir = os.environ.get("AEGIS_BIOMASS_DIR", "")
        if bio_dir and os.path.isdir(bio_dir):
            try:
                from aegis.biomass import run_biomass_demo
                bio = run_biomass_demo(bio_dir)
                print(f"  Biomass adapter: {bio[0].n_tiles} tiles, "
                      f"held-out accuracy {bio[0].accuracy:.3f} "
                      f"(circular control {bio[0].accuracy_circular:.3f})")
                out["biomass"] = {
                    "n_tiles": bio[0].n_tiles, "accuracy": bio[0].accuracy,
                    "auc": bio[0].auc, "accuracy_circular": bio[0].accuracy_circular,
                    "cleared_fraction": bio[0].cleared_fraction,
                    "downlink_reduction": bio[0].downlink_reduction,
                    "frame_gb": bio[0].frame_gb, "notes": bio[0].notes}
                save_json(out, "results.json")
            except Exception as exc:                       # noqa: BLE001
                print(f"  Biomass adapter skipped: {type(exc).__name__}: {exc}")
        p = build_dashboard(out, data, screener, net8, biomass=bio)
        print(f"  dashboard -> {p}")
    return out


if __name__ == "__main__":
    main()
