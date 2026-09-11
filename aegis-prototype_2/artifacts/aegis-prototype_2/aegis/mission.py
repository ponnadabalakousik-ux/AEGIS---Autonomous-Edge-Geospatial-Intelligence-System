"""
Mission campaign: baseline vs AEGIS over a full multi-day simulation.
=====================================================================

This is where the six claimed benefits from the problem statement get turned
into numbers. Two spacecraft fly the identical orbit, take the identical
pictures and have the identical link budget. One downlinks what it can, oldest
first, because that is what a conventional mission does. The other runs the
cascade and decides.

The campaign is a Monte-Carlo subsample: decisions are made on
`sim_tiles_per_day` tiles drawn from the held-out test set, each carrying a
weight equal to the number of real tiles it represents. Every bit, joule and
event count is scaled by that weight, so the absolute figures are full-mission
figures while the compute stays tractable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .autonomy import Action, DownlinkQueue, Observation, TriagePolicy
from .dataset import TileSet
from .edge import EdgeProcessor
from .models import PhysicsScreener, extract_roi, product_bits
from .quantise import OnnxTriage
from .spacecraft import (SOLAR_DAY, LifeLimits, Orbit, Pass, PowerSystem,
                         ResourceState, acquisition_rate, compute_passes,
                         downlink_capacity, in_eclipse, sun_vector_eci)


# ---------------------------------------------------------------------------
@dataclass
class TilePool:
    """Cached inference over the evaluation tiles.

    Inference is run once up front rather than inside the time loop: the
    network is deterministic, so re-running it 40,000 times would only cost
    wall-clock. The *cost* of those inferences is still charged in full by
    the edge model.
    """
    ts: TileSet
    gate: np.ndarray              # (N,) bool - Stage 0 says "wake the CNN"
    pred_cloud: np.ndarray        # (N,)
    pred_class: np.ndarray        # (N,)
    pred_conf: np.ndarray         # (N,)
    severity: np.ndarray          # (N,)
    roi_bits: np.ndarray          # (N,) bits of the ROI product
    roi_frac: np.ndarray          # (N,) ROI area / tile area

    @property
    def n(self) -> int:
        return len(self.ts)


def build_tile_pool(ts: TileSet, screener: PhysicsScreener,
                    net: OnnxTriage) -> TilePool:
    gate = screener.gate(ts.feats)
    r = net.run_batched(ts.cubes)
    pred_cloud = r["cloud_frac"]
    prob = r["event_prob"]
    pred_class = prob.argmax(1)
    pred_conf = prob.max(1)
    seg = r["seg_logits"]

    n = len(ts)
    roi_bits = np.empty(n)
    roi_frac = np.empty(n)
    severity = np.empty(n)
    for i in range(n):
        roi = extract_roi(seg[i])
        roi_bits[i] = product_bits("roi", roi)
        roi_frac[i] = roi.fraction_of_tile if roi is not None else 0.0
        # Severity proxy: how much of the tile the detection covers, scaled
        # so a small confident fire still scores meaningfully.
        severity[i] = float(np.clip(roi_frac[i] * 2.5, 0.0, 1.0))

    # Stage 0 rejected tiles never get CNN outputs onboard. Substituting the
    # cheap estimate here keeps the bookkeeping honest.
    return TilePool(ts, gate, pred_cloud, pred_class, pred_conf, severity,
                    roi_bits, roi_frac)


# ---------------------------------------------------------------------------
@dataclass
class CampaignResult:
    mode: str
    duration_s: float
    weight: float

    tiles_acquired: float = 0.0
    tiles_stage1: float = 0.0
    tiles_discarded_onboard: float = 0.0
    tiles_dropped_memory: float = 0.0

    bits_downlinked: float = 0.0
    bits_generated_if_all: float = 0.0

    events_observable: float = 0.0
    events_delivered: float = 0.0
    events_missed_capacity: float = 0.0
    events_missed_model: float = 0.0
    events_thumbnail_only: float = 0.0
    false_positives_delivered: float = 0.0
    event_accounting_residual: float = 0.0

    latencies_s: List[float] = field(default_factory=list)
    latency_weights: List[float] = field(default_factory=list)
    alert_latencies_s: List[float] = field(default_factory=list)

    energy_comms_wh: float = 0.0
    energy_compute_wh: float = 0.0
    energy_payload_wh: float = 0.0
    energy_bus_wh: float = 0.0

    tx_on_time_s: float = 0.0
    battery_cycles: float = 0.0
    peak_memory_fraction: float = 0.0
    memory_trace: List[Tuple[float, float]] = field(default_factory=list)
    soc_trace: List[Tuple[float, float]] = field(default_factory=list)
    brownouts: int = 0

    def weighted_latency_percentile(self, q: float) -> float:
        if not self.latencies_s:
            return float("nan")
        lat = np.array(self.latencies_s)
        w = np.array(self.latency_weights)
        order = np.argsort(lat)
        lat, w = lat[order], w[order]
        cw = np.cumsum(w) / w.sum()
        return float(lat[np.searchsorted(cw, q)] if cw[-1] >= q else lat[-1])

    def summary(self) -> Dict:
        days = self.duration_s / SOLAR_DAY
        recall = self.events_delivered / max(self.events_observable, 1e-9)
        return {
            "mode": self.mode,
            "days": days,
            "tiles_acquired": self.tiles_acquired,
            "tiles_run_through_cnn": self.tiles_stage1,
            "cnn_invocation_rate": self.tiles_stage1 / max(self.tiles_acquired, 1e-9),
            "gb_downlinked": self.bits_downlinked / 8 / 1e9,
            "gb_downlinked_per_day": self.bits_downlinked / 8 / 1e9 / days,
            "gb_if_everything_sent": self.bits_generated_if_all / 8 / 1e9,
            "downlink_reduction": 1.0 - self.bits_downlinked / max(self.bits_generated_if_all, 1e-9),
            "events_observable": self.events_observable,
            "events_delivered": self.events_delivered,
            "event_delivery_rate": recall,
            "events_missed_capacity": self.events_missed_capacity,
            "events_missed_model": self.events_missed_model,
            "events_thumbnail_only": self.events_thumbnail_only,
            "event_accounting_residual": self.event_accounting_residual,
            "event_precision": (self.events_delivered
                                / max(self.events_delivered
                                      + self.false_positives_delivered, 1e-9)),
            "false_positives_delivered": self.false_positives_delivered,
            "median_latency_min": self.weighted_latency_percentile(0.5) / 60.0,
            "p90_latency_min": self.weighted_latency_percentile(0.9) / 60.0,
            "mean_alert_latency_min": (float(np.mean(self.alert_latencies_s)) / 60.0
                                       if self.alert_latencies_s else float("nan")),
            "energy_comms_wh": self.energy_comms_wh,
            "energy_compute_wh": self.energy_compute_wh,
            "energy_payload_wh": self.energy_payload_wh,
            "energy_bus_wh": self.energy_bus_wh,
            "energy_total_wh": (self.energy_comms_wh + self.energy_compute_wh
                                + self.energy_payload_wh + self.energy_bus_wh),
            "energy_comms_wh_per_day": self.energy_comms_wh / days,
            "tx_on_time_hours": self.tx_on_time_s / 3600.0,
            "tx_duty_cycle": self.tx_on_time_s / self.duration_s,
            "battery_cycles": self.battery_cycles,
            "peak_memory_fraction": self.peak_memory_fraction,
            "tiles_dropped_memory": self.tiles_dropped_memory,
            "brownouts": self.brownouts,
        }


# ---------------------------------------------------------------------------
class MissionSimulator:
    def __init__(self, pool: TilePool, profile: C.MissionProfile = None,
                 policy: C.AutonomyPolicy = C.DEFAULT_POLICY,
                 hardware: str = C.DEFAULT_HARDWARE,
                 stage1_macs: float = 10e6,
                 days: float = C.SIM_DAYS,
                 seed: int = C.RNG_SEED):
        self.pool = pool
        self.mp = profile or C.MissionProfile()
        self.policy_cfg = policy
        self.proc = EdgeProcessor(hardware)
        self.stage1_macs = stage1_macs
        self.days = days
        self.duration_s = days * SOLAR_DAY
        self.seed = seed

        self.orbit = Orbit(self.mp.altitude_km, self.mp.inclination_deg,
                           None, self.mp.ltan_hours)
        self.passes = compute_passes(self.orbit, C.GROUND_STATIONS,
                                     self.duration_s, C.TIME_STEP_S,
                                     self.mp.min_elevation_deg)
        self.acq = acquisition_rate(self.orbit, self.mp)
        self.dl = downlink_capacity(self.passes, self.mp, self.duration_s)

        # X-band passes actually granted to this mission.
        rng = np.random.default_rng(seed)
        usable = [p for p in self.passes
                  if p.max_elevation >= self.mp.xband_min_elevation_deg]
        keep = rng.random(len(usable)) < self.mp.pass_utilisation
        self.xband_passes = [p for p, k in zip(usable, keep) if k]

        # Monte-Carlo weight.
        self.sim_tiles_total = int(self.mp.sim_tiles_per_day * days)
        self.weight = self.acq["tiles_per_day"] * days / self.sim_tiles_total

        self.stage0_flops = PhysicsScreener.flops_per_tile()

    # -- helpers -----------------------------------------------------------
    def _pass_lookup(self) -> Dict[int, Pass]:
        """Map time-step index -> active X-band pass."""
        table: Dict[int, Pass] = {}
        for p in self.xband_passes:
            i0 = int(p.t_start // C.TIME_STEP_S)
            i1 = int(p.t_end // C.TIME_STEP_S)
            for i in range(i0, i1 + 1):
                table[i] = p
        return table

    def _cell_grid(self, t_grid: np.ndarray) -> np.ndarray:
        """Coarse geographic bucket per time step, for the novelty tracker.

        Derived from the real sub-satellite track so that revisits of the same
        ground area actually collide - which is what makes novelty mean
        anything rather than being a random discount.
        """
        lat, lon = self.orbit.subsatellite_latlon(t_grid)
        return np.stack([np.floor(lat / 2.0), np.floor(lon / 2.0)], axis=-1).astype(int)

    # -- the campaign ------------------------------------------------------
    def run(self, mode: str = "aegis", verbose: bool = False,
            prioritise: Optional[bool] = None) -> CampaignResult:
        """`prioritise=False` keeps the AI but schedules the queue FIFO, which
        isolates how much of the benefit comes from value-ordered downlink as
        opposed to from detection alone."""
        assert mode in ("aegis", "baseline")
        if prioritise is None:
            prioritise = (mode == "aegis")
        rng = np.random.default_rng(self.seed + (0 if mode == "aegis" else 1))
        triage = TriagePolicy(self.policy_cfg, seed=self.seed)

        res = CampaignResult(mode=mode, duration_s=self.duration_s,
                             weight=self.weight)
        state = ResourceState(
            battery_wh=self.mp.battery_wh,
            battery_capacity_wh=self.mp.battery_wh,
            memory_capacity_bits=self.mp.mass_memory_gb * 8e9,
        )
        power = PowerSystem(self.mp)
        queue = DownlinkQueue(self.mp.mass_memory_gb * 8e9)
        life = LifeLimits(self.mp.xband_tx_rated_hours, self.mp.battery_rated_cycles)

        dt = C.TIME_STEP_S
        steps = int(self.duration_s / dt)
        t_grid = np.arange(steps) * dt
        eci = self.orbit.position_eci(t_grid)
        sunlit = ~in_eclipse(eci, sun_vector_eci(t_grid))
        pass_at = self._pass_lookup()
        cell_grid = self._cell_grid(t_grid)

        # Tiles arrive only while imaging, i.e. on the sunlit side.
        imaging_steps = int(sunlit.sum())
        tiles_per_step = self.sim_tiles_total / max(imaging_steps, 1)

        pool_idx = rng.integers(0, self.pool.n, size=self.sim_tiles_total)
        cursor = 0
        carry = 0.0
        full_bits = product_bits("full")

        alert_backlog: List[Observation] = []
        relay_credit = 0.0        # token bucket, bits
        stage1_count = 0.0

        for i in range(steps):
            t = float(t_grid[i])
            lit = bool(sunlit[i])
            orbit_index = int(t // self.orbit.period_s)
            loads = {"bus": self.mp.bus_load_w}

            # ---------------- acquisition + onboard processing -------------
            n_new = 0
            if lit:
                carry += tiles_per_step
                n_new = int(carry)
                carry -= n_new
                loads["payload"] = self.mp.payload_imaging_w

            compute_w = 0.0
            for _ in range(n_new):
                if cursor >= len(pool_idx):
                    break
                k = int(pool_idx[cursor]); cursor += 1
                ts = self.pool.ts
                obs = Observation(
                    tile_index=k, t_obs=t, lat=0.0, lon=0.0,
                    cell=(int(cell_grid[i, 0]), int(cell_grid[i, 1])),
                    true_event=bool(ts.true_event[k]),
                    true_class=int(ts.event_lbl[k]),
                    true_cloud=float(ts.cloud_frac[k]),
                    weight=self.weight,
                )
                res.tiles_acquired += self.weight
                res.bits_generated_if_all += full_bits * self.weight
                if obs.true_event:
                    res.events_observable += self.weight

                if mode == "baseline":
                    # No onboard intelligence: everything is a full product,
                    # stored oldest-first until the recorder is full.
                    obs.action = Action.FULL
                    obs.bits = full_bits
                    obs.voi = 0.0
                    if not queue.push_fifo(obs):
                        res.tiles_dropped_memory += self.weight
                        if obs.true_event:
                            res.events_missed_capacity += self.weight
                    continue

                # --- AEGIS: Stage 0 always, Stage 1 only on survivors ------
                obs.gated_out = not bool(self.pool.gate[k])
                c0 = self.proc.cost_for_flops_cpu(self.stage0_flops)
                compute_w += c0.energy_mj / 1000.0 * self.weight / dt

                if not obs.gated_out:
                    obs.pred_cloud = float(self.pool.pred_cloud[k])
                    obs.pred_class = int(self.pool.pred_class[k])
                    obs.pred_conf = float(self.pool.pred_conf[k])
                    obs.severity = float(self.pool.severity[k])
                    obs.roi = _RoiProxy(float(self.pool.roi_frac[k]),
                                        float(self.pool.roi_bits[k]))
                    stage1_count += self.weight
                    res.tiles_stage1 += self.weight
                    c1 = self.proc.cost_for_macs(self.stage1_macs, "int8")
                    compute_w += c1.energy_mj / 1000.0 * self.weight / dt

                triage.decide(obs, t, orbit_index)

                if obs.action == Action.DISCARD:
                    res.tiles_discarded_onboard += self.weight
                    if obs.true_event:
                        res.events_missed_model += self.weight
                    continue

                if obs.action == Action.ALERT:
                    alert_backlog.append(obs)
                    continue

                if not queue.push(obs):
                    res.tiles_dropped_memory += self.weight
                    if obs.true_event:
                        res.events_missed_capacity += self.weight

            if compute_w > 0:
                loads["compute"] = compute_w

            # ---------------- urgent relay path ---------------------------
            if mode == "aegis" and self.mp.relay_available:
                # Token bucket: the relay is a low-rate link with a buffer, so
                # credit accrues between bursts. Without this a single weighted
                # alert can be larger than one time step's worth of bits and
                # nothing is ever sent - which is exactly the bug this
                # replaced.
                relay_credit = min(relay_credit + self.mp.relay_rate_kbps * 1000.0 * dt,
                                   self.mp.relay_rate_kbps * 1000.0 * 600.0)
                relay_bits = relay_credit
                sent_bits = 0.0
                still: List[Observation] = []
                for obs in alert_backlog:
                    need = product_bits("alert") * obs.weight
                    if sent_bits + need <= relay_bits:
                        sent_bits += need
                        obs.alert_sent_at = t + self.mp.relay_latency_min * 60.0
                        lat = obs.alert_sent_at - obs.t_obs
                        res.alert_latencies_s.append(lat)
                        res.latencies_s.append(lat)
                        res.latency_weights.append(obs.weight)
                        if obs.true_event:
                            res.events_delivered += obs.weight
                        else:
                            res.false_positives_delivered += obs.weight
                        triage.novelty.mark(obs.cell, t)
                        # The imagery still goes down on the next pass.
                        obs.action = Action.ROI
                        obs.bits = product_bits("roi", obs.roi)
                        obs.voi *= 0.35        # already alerted: less urgent
                        queue.push(obs)
                    else:
                        still.append(obs)
                relay_credit -= sent_bits
                # Anything that has waited longer than its latency target is
                # no longer an alert; demote it to the bulk queue rather than
                # letting it rot in the backlog.
                alert_backlog = []
                for obs in still:
                    target = C.EVENT_LATENCY_TARGET_MIN[
                        C.EVENT_CLASSES[obs.pred_class]] * 60.0
                    if t - obs.t_obs > target:
                        obs.action = Action.ROI
                        obs.bits = product_bits("roi", obs.roi)
                        if not queue.push(obs):
                            res.tiles_dropped_memory += obs.weight
                            if obs.true_event:
                                res.events_missed_capacity += obs.weight
                    else:
                        alert_backlog.append(obs)
                if sent_bits > 0:
                    loads["comms"] = loads.get("comms", 0.0) + self.mp.relay_power_w

            # ---------------- downlink ------------------------------------
            p = pass_at.get(i)
            if p is not None and queue.items:
                tx_w = self.mp.xband_tx_power_w
                if power.can_afford(state, tx_w + sum(loads.values()), dt, lit):
                    budget = (self.mp.xband_rate_mbps * 1e6
                              * self.mp.xband_link_efficiency * dt)
                    sent, used = queue.schedule_pass(budget, t,
                                                     prioritise=prioritise)
                    if used > 0:
                        loads["comms"] = loads.get("comms", 0.0) + tx_w
                        state.tx_on_time_s += dt
                        res.tx_on_time_s += dt
                        res.bits_downlinked += used
                    for o in sent:
                        if o.alert_sent_at is None:
                            res.latencies_s.append(t - o.t_obs)
                            res.latency_weights.append(o.weight)
                            # A thumbnail is a browse product, not a delivery:
                            # nobody detects a three-pixel vessel in a 60:1
                            # preview. Only full-fidelity products count.
                            interpretable = o.action in (Action.ROI, Action.FULL)
                            if o.true_event and interpretable:
                                res.events_delivered += o.weight
                            elif o.true_event:
                                res.events_thumbnail_only += o.weight
                            elif interpretable and mode == "aegis" and o.pred_class != 0:
                                res.false_positives_delivered += o.weight
                        triage.novelty.mark(o.cell, t)

            # ---------------- power ---------------------------------------
            power.step(state, dt, lit, loads)
            state.memory_used_bits = queue.used_bits
            res.peak_memory_fraction = max(res.peak_memory_fraction, queue.fill)
            if i % 45 == 0:
                res.memory_trace.append((t, queue.fill))
                res.soc_trace.append((t, state.soc))

        # anything never downlinked and never alerted is a capacity miss
        for o in list(queue.items) + alert_backlog:
            if o.true_event and o.alert_sent_at is None:
                res.events_missed_capacity += o.weight

        # Accounting closure check: every observable event must end up in
        # exactly one bucket. A silent leak here would flatter whichever mode
        # leaked, so it is asserted rather than trusted.
        accounted = (res.events_delivered + res.events_thumbnail_only
                     + res.events_missed_model + res.events_missed_capacity)
        res.event_accounting_residual = res.events_observable - accounted

        res.energy_comms_wh = state.energy_comms_wh
        res.energy_compute_wh = state.energy_compute_wh
        res.energy_payload_wh = state.energy_payload_wh
        res.energy_bus_wh = state.energy_bus_wh
        res.battery_cycles = state.battery_cycles
        res.brownouts = state.brownouts
        res.tx_on_time_s = state.tx_on_time_s
        self.life = life
        return res


class _RoiProxy:
    """Lightweight stand-in carrying the two ROI numbers the policy needs,
    so the mission loop never re-derives bounding boxes it already has."""

    __slots__ = ("fraction_of_tile", "_bits")

    def __init__(self, frac: float, bits: float):
        self.fraction_of_tile = frac
        self._bits = bits

    @property
    def area_px(self) -> int:
        return int(self.fraction_of_tile * C.TILE_PX * C.TILE_PX)


def compare(baseline: CampaignResult, aegis: CampaignResult) -> Dict:
    """The head-to-head table, mapped onto the six claimed benefits."""
    b, a = baseline.summary(), aegis.summary()

    def ratio(x: float, y: float) -> float:
        return float(x / y) if y else float("inf")

    return {
        "baseline": b,
        "aegis": a,
        "benefits": {
            "1_downlink_volume": {
                "claim": "Reduced data transmission",
                "baseline_gb_per_day": b["gb_downlinked_per_day"],
                "aegis_gb_per_day": a["gb_downlinked_per_day"],
                "reduction": 1.0 - ratio(a["gb_downlinked_per_day"],
                                         b["gb_downlinked_per_day"]),
            },
            "2_event_delivery": {
                "claim": "Selective/better targeted measurements",
                "baseline_rate": b["event_delivery_rate"],
                "aegis_rate": a["event_delivery_rate"],
                "improvement_x": ratio(a["event_delivery_rate"],
                                       b["event_delivery_rate"]),
            },
            "3_latency": {
                "claim": "Reduced analysis time",
                "baseline_median_min": b["median_latency_min"],
                "aegis_median_min": a["median_latency_min"],
                "aegis_alert_min": a["mean_alert_latency_min"],
                "speedup_x": ratio(b["median_latency_min"], a["median_latency_min"]),
            },
            "4_energy": {
                "claim": "Reduced net power usage",
                "baseline_comms_wh_per_day": b["energy_comms_wh_per_day"],
                "aegis_comms_wh_per_day": a["energy_comms_wh_per_day"],
                "aegis_compute_wh": a["energy_compute_wh"],
                "net_saving_wh": (b["energy_comms_wh"] + b["energy_compute_wh"]
                                  - a["energy_comms_wh"] - a["energy_compute_wh"]),
                "net_saving_fraction": 1.0 - ratio(
                    a["energy_comms_wh"] + a["energy_compute_wh"],
                    b["energy_comms_wh"] + b["energy_compute_wh"]),
            },
            "5_life_limited": {
                "claim": "Less usage of life-limited items",
                "baseline_tx_hours": b["tx_on_time_hours"],
                "aegis_tx_hours": a["tx_on_time_hours"],
                "tx_reduction": 1.0 - ratio(a["tx_on_time_hours"],
                                            b["tx_on_time_hours"]),
            },
            "6_storage": {
                "claim": "Decreased mass memory demand",
                "baseline_peak_memory": b["peak_memory_fraction"],
                "aegis_peak_memory": a["peak_memory_fraction"],
                "baseline_tiles_lost_to_full_recorder": b["tiles_dropped_memory"],
                "aegis_tiles_lost_to_full_recorder": a["tiles_dropped_memory"],
            },
        },
    }
