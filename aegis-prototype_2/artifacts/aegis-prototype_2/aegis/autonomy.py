"""
The autonomy layer: deciding what is worth keeping, sending and interrupting for.
================================================================================

Detection is the easy half. The half that actually changes a mission is the
policy that turns a detection into a decision under hard constraints - bits,
joules, memory, and a ground station that is only overhead for six minutes.

Value of Information
--------------------
Every observation gets a scalar VoI:

    VoI = w_class . confidence . (1 - cloud) . severity . novelty . urgency

  w_class     operator-set worth of that event type (uplinkable)
  confidence  the network's own posterior - so the policy is calibrated,
              not binary
  (1 - cloud) an image nobody can interpret has no value however exciting
              the classifier finds it
  severity    physical intensity, from the segmentation mask extent
  novelty     decays exponentially if the same ground cell was downlinked
              recently. Stops the spacecraft spending its whole budget
              re-sending one large fire for three days.
  urgency     rises as the event's latency target approaches

Routing then depends on VoI *and* on how the value decays with time:
high-confidence urgent events take the low-rate relay immediately rather
than waiting for a station, because a wildfire alert delivered six hours
late is worth almost nothing.

Downlink scheduling
-------------------
Each pass is a knapsack: maximise total VoI subject to the bits the pass can
actually carry and the energy the battery can actually spare. Solved greedily
by VoI-per-bit, which is the standard 1/2-approximation and is the right call
onboard - it is O(n log n), needs no solver, and is trivially explainable to
an operations team, which matters more than the last few percent of optimality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .models import ROI, extract_roi, product_bits


class Action(str, Enum):
    DISCARD = "discard"          # never stored - the tile ceases to exist
    THUMBNAIL = "thumbnail"      # browse/audit product only
    ROI = "roi"                  # the interesting box, all bands
    FULL = "full"                # whole tile - reserved for the ambiguous
    ALERT = "alert"              # immediate low-rate relay + ROI queued


@dataclass
class Observation:
    """One tile as seen by the payload, after inference."""
    tile_index: int              # index into the evaluation tile pool
    t_obs: float                 # seconds since epoch
    lat: float
    lon: float
    cell: Tuple[int, int]        # coarse geographic bucket, for novelty

    # ground truth (never visible to the onboard logic - audit only)
    true_event: bool
    true_class: int
    true_cloud: float

    # inference outputs
    pred_cloud: float = 0.0
    pred_class: int = 0
    pred_conf: float = 0.0
    severity: float = 0.0
    roi: Optional[ROI] = None
    gated_out: bool = False      # rejected by Stage 0, CNN never ran

    # decisions
    action: Action = Action.DISCARD
    voi: float = 0.0
    bits: float = 0.0
    weight: float = 1.0          # real tiles represented by this observation

    # outcomes
    t_downlinked: Optional[float] = None
    alert_sent_at: Optional[float] = None

    @property
    def latency_s(self) -> Optional[float]:
        best = None
        if self.alert_sent_at is not None:
            best = self.alert_sent_at - self.t_obs
        if self.t_downlinked is not None:
            d = self.t_downlinked - self.t_obs
            best = d if best is None else min(best, d)
        return best


# ---------------------------------------------------------------------------
class NoveltyTracker:
    """Exponentially-decaying memory of what has already been sent, per cell."""

    def __init__(self, decay_hours: float):
        self.tau = decay_hours * 3600.0
        self._last: Dict[Tuple[int, int], float] = {}

    def novelty(self, cell: Tuple[int, int], t: float) -> float:
        last = self._last.get(cell)
        if last is None:
            return 1.0
        return float(1.0 - math.exp(-(t - last) / self.tau))

    def mark(self, cell: Tuple[int, int], t: float) -> None:
        self._last[cell] = t


# ---------------------------------------------------------------------------
class TriagePolicy:
    """Turns inference outputs into an Action and a VoI score."""

    def __init__(self, policy: C.AutonomyPolicy = C.DEFAULT_POLICY,
                 seed: int = C.RNG_SEED):
        self.p = policy
        self.novelty = NoveltyTracker(policy.novelty_decay_hours)
        self.rng = np.random.default_rng(seed)
        self._alerts_this_orbit = 0
        self._orbit_index = -1

    # -- scoring -----------------------------------------------------------
    def urgency(self, cls_idx: int, age_s: float) -> float:
        """Ramps from 1.0 to 2.0 as the event approaches its latency target,
        so a stale high-value detection outranks a fresh mediocre one."""
        target = C.EVENT_LATENCY_TARGET_MIN[C.EVENT_CLASSES[cls_idx]] * 60.0
        if target > 1e8:
            return 1.0
        return float(1.0 + min(age_s / target, 1.0))

    def score(self, obs: Observation, t: float) -> float:
        cls = C.EVENT_CLASSES[obs.pred_class]
        w = C.EVENT_VALUE_WEIGHT[cls]
        # Value degrades in proportion to how much of the scene is hidden.
        # (An earlier version scaled by the reject threshold, which quietly
        # halved the value of a perfectly usable 20%-cloud tile.)
        clarity = float(np.clip(1.0 - obs.pred_cloud, 0.0, 1.0))
        nov = self.novelty.novelty(obs.cell, t)
        # A small fire is still a fire: the floor stops footprint size from
        # dominating the score.
        sev = 0.55 + 0.45 * obs.severity
        return float(w * obs.pred_conf * clarity * sev * nov
                     * self.urgency(obs.pred_class, 0.0))

    # -- decision ----------------------------------------------------------
    def decide(self, obs: Observation, t: float, orbit_index: int) -> Observation:
        if orbit_index != self._orbit_index:
            self._orbit_index = orbit_index
            self._alerts_this_orbit = 0

        # Stage 0 rejected it: the CNN never ran, so there is nothing to
        # score. This is where the bulk of the savings come from.
        if obs.gated_out:
            obs.action = Action.DISCARD
            obs.voi = 0.0
            obs.bits = 0.0
            return obs

        # Too cloudy to be worth anything, regardless of what was detected.
        if obs.pred_cloud > self.p.cloud_reject_threshold:
            # Keep a small audit sample so the ground can check the screener
            # is not silently eating good data. Never let a model be the only
            # witness to its own errors.
            if self.rng.random() < self.p.audit_sample_rate:
                obs.action = Action.THUMBNAIL
                obs.bits = product_bits("thumbnail")
                obs.voi = 0.01
            else:
                obs.action = Action.DISCARD
                obs.bits = 0.0
            return obs

        obs.voi = self.score(obs, t)
        is_event = (obs.pred_class != 0
                    and obs.pred_conf >= self.p.event_confidence_threshold)

        if is_event and obs.pred_conf >= self.p.alert_confidence_threshold and \
                C.EVENT_LATENCY_TARGET_MIN[C.EVENT_CLASSES[obs.pred_class]] <= 120.0 and \
                self._alerts_this_orbit + obs.weight <= self.p.max_alerts_per_orbit:
            obs.action = Action.ALERT
            obs.bits = product_bits("roi", obs.roi)
            self._alerts_this_orbit += obs.weight
            return obs

        if is_event and obs.voi >= self.p.voi_downlink_threshold:
            # Ambiguous but valuable detections get the full tile: if the
            # model is unsure, do not let it also decide what to crop.
            if obs.pred_conf < 0.72 or obs.roi is None or obs.roi.fraction_of_tile > 0.45:
                obs.action = Action.FULL
                obs.bits = product_bits("full")
            else:
                obs.action = Action.ROI
                obs.bits = product_bits("roi", obs.roi)
            return obs

        if obs.voi >= self.p.voi_thumbnail_threshold or \
                self.rng.random() < self.p.audit_sample_rate:
            obs.action = Action.THUMBNAIL
            obs.bits = product_bits("thumbnail")
            return obs

        obs.action = Action.DISCARD
        obs.bits = 0.0
        return obs


# ---------------------------------------------------------------------------
class DownlinkQueue:
    """Prioritised store of products waiting for a pass."""

    def __init__(self, capacity_bits: float):
        self.capacity_bits = capacity_bits
        self.items: List[Observation] = []

    @property
    def used_bits(self) -> float:
        return float(sum(o.bits * o.weight for o in self.items))

    @property
    def fill(self) -> float:
        return self.used_bits / max(self.capacity_bits, 1.0)

    def push(self, obs: Observation) -> bool:
        """Returns False if the product had to be dropped for lack of memory.

        When memory is full the *lowest*-value item is evicted rather than
        the oldest. A FIFO mass memory throws away the fire to keep the
        cloud; that is the failure mode onboard prioritisation removes.
        """
        need = obs.bits * obs.weight
        if need <= 0:
            return True
        if self.used_bits + need <= self.capacity_bits:
            self.items.append(obs)
            return True
        self.items.sort(key=lambda o: o.voi)
        freed = 0.0
        evicted = 0
        while self.items and self.used_bits + need - freed > self.capacity_bits:
            if self.items[0].voi >= obs.voi:
                break
            freed += self.items[0].bits * self.items[0].weight
            self.items.pop(0)
            evicted += 1
        if self.used_bits + need <= self.capacity_bits:
            self.items.append(obs)
            return True
        return False

    def push_fifo(self, obs: Observation) -> bool:
        """Baseline behaviour: no notion of value, so no informed eviction.
        Full memory simply means the next observation is lost."""
        need = obs.bits * obs.weight
        if self.used_bits + need <= self.capacity_bits:
            self.items.append(obs)
            return True
        return False

    # -- scheduling --------------------------------------------------------
    def schedule_pass(self, budget_bits: float, t_now: float,
                      prioritise: bool = True) -> Tuple[List[Observation], float]:
        """Greedy VoI-per-bit knapsack over one contact window."""
        if prioritise:
            def density(o: Observation) -> float:
                age = max(t_now - o.t_obs, 0.0)
                target = C.EVENT_LATENCY_TARGET_MIN[C.EVENT_CLASSES[o.pred_class]] * 60.0
                urg = 1.0 + min(age / target, 1.0) if target < 1e8 else 1.0
                return (o.voi * urg) / max(o.bits, 1.0)
            order = sorted(self.items, key=density, reverse=True)
        else:
            order = sorted(self.items, key=lambda o: o.t_obs)   # FIFO

        sent: List[Observation] = []
        used = 0.0
        for o in order:
            need = o.bits * o.weight
            if used + need <= budget_bits:
                used += need
                sent.append(o)
        keep = set(id(o) for o in sent)
        self.items = [o for o in self.items if id(o) not in keep]
        for o in sent:
            o.t_downlinked = t_now
        return sent, used
