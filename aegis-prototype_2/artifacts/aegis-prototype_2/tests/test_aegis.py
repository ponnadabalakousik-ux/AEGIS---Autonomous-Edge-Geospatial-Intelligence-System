"""
Invariant tests.
================
These are the checks that catch a silently wrong result rather than a crash.
The mission-accounting closure test in particular exists because an early
version of the campaign leaked observations into an alert backlog that could
never be transmitted, which made the AI look *worse* than it was - a bug that
produced entirely plausible numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aegis import config as C
from aegis.autonomy import Action, DownlinkQueue, NoveltyTracker, Observation, TriagePolicy
from aegis.edge import EdgeProcessor, RadiationModel, calibrate_utilisation
from aegis.models import PhysicsScreener, ROI, extract_roi, product_bits
from aegis.scene import SceneGenerator, spectral_indices, tile_features
from aegis.spacecraft import (SOLAR_DAY, Orbit, acquisition_rate, compute_passes,
                              elevation_deg, in_eclipse, raan_from_ltan,
                              station_ecef, sun_vector_eci)


# ===========================================================================
# scene physics
# ===========================================================================
def test_snow_and_cloud_separate_in_swir():
    """The discriminating case for the whole 'why not a threshold' argument.

    Snow and cloud are both blinding in the visible. Only SWIR tells them
    apart, which is what NDSI encodes. If this ever fails, the simulator has
    stopped being physically meaningful.
    """
    g = SceneGenerator(seed=7)
    snow, cloud = [], []
    for _ in range(6):
        t = g.sample_tile(force_land="snow", force_event="nominal",
                          force_cloud_fraction=0.0)
        snow.append(spectral_indices(t.cube)["ndsi"].mean())
        t = g.sample_tile(force_event="nominal", force_cloud_fraction=0.95)
        cloud.append(spectral_indices(t.cube)["ndsi"].mean())
    assert np.mean(snow) > 0.4, f"snow NDSI too low: {np.mean(snow):.3f}"
    assert np.mean(cloud) < 0.25, f"cloud NDSI too high: {np.mean(cloud):.3f}"
    assert np.mean(snow) - np.mean(cloud) > 0.3


def test_fire_raises_swir2_more_than_swir1():
    """Sub-pixel high-temperature emission must follow the Planck ordering
    B12 > B11 >> NIR. Get this backwards and every fire index inverts."""
    g = SceneGenerator(seed=11)
    ratios = []
    for _ in range(8):
        t = g.sample_tile(force_event="wildfire", force_cloud_fraction=0.0)
        m = t.event_mask
        if m.sum() < 4:
            continue
        b11, b12 = t.cube[4][m].mean(), t.cube[5][m].mean()
        ratios.append(b12 / max(b11, 1e-6))
    assert ratios, "no fire pixels generated"
    assert np.mean(ratios) > 1.2, f"B12/B11 in fire = {np.mean(ratios):.2f}"


def test_cloud_fraction_matches_request():
    g = SceneGenerator(seed=3)
    for target in (0.0, 0.25, 0.6, 0.9):
        errs = [abs(g.sample_tile(force_cloud_fraction=target,
                                  force_event="nominal").cloud_fraction - target)
                for _ in range(4)]
        assert np.mean(errs) < 0.16, f"target {target}: mean error {np.mean(errs):.3f}"


def test_reflectance_stays_physical():
    g = SceneGenerator(seed=5)
    for _ in range(12):
        c = g.sample_tile().cube
        assert c.min() >= -1e-6 and c.max() <= 1.36
        assert np.isfinite(c).all()


def test_true_event_requires_visibility():
    """An event buried under cloud is not an observable event. If this is
    wrong, the baseline gets credit for delivering things nobody can see."""
    g = SceneGenerator(seed=13)
    for _ in range(10):
        t = g.sample_tile(force_event="wildfire", force_cloud_fraction=0.99)
        assert not t.is_true_event
    n_ok = sum(g.sample_tile(force_event="wildfire",
                             force_cloud_fraction=0.0).is_true_event
               for _ in range(10))
    assert n_ok >= 8


def test_features_are_finite_and_sized():
    g = SceneGenerator(seed=2)
    f = tile_features(g.sample_tile().cube)
    assert f.shape == (PhysicsScreener.N_FEAT,)
    assert np.isfinite(f).all()


# ===========================================================================
# orbital mechanics
# ===========================================================================
def test_sso_precession_matches_solar_rate():
    """A sun-synchronous orbit is *defined* by its node precessing at the
    Earth's mean motion about the Sun, 0.9856 deg/day. Reproducing that from
    J2 alone is a strong check on the whole orbit model."""
    o = Orbit(500.0, 97.4)
    drift = math.degrees(o.raan_dot) * SOLAR_DAY
    assert abs(drift - 0.9856) < 0.01, f"RAAN drift {drift:.4f} deg/day"


def test_orbital_period_matches_kepler():
    o = Orbit(500.0, 97.4)
    assert 94.0 < o.period_s / 60.0 < 95.5


def test_ltan_produces_realistic_eclipse():
    """A mid-morning SSO must have eclipse; if the RAAN is left at an
    arbitrary value the orbit can end up in permanent sunlight, which would
    silently delete the eclipse from the power budget."""
    o = Orbit(500.0, 97.4, None, ltan_hours=10.5)
    t = np.arange(0, 3 * SOLAR_DAY, 30.0)
    frac = in_eclipse(o.position_eci(t), sun_vector_eci(t)).mean()
    assert 0.25 < frac < 0.42, f"eclipse fraction {frac:.3f}"


def test_dawn_dusk_orbit_has_less_eclipse_than_mid_morning():
    t = np.arange(0, 3 * SOLAR_DAY, 30.0)
    dd = Orbit(500.0, 97.4, None, ltan_hours=6.0)
    mm = Orbit(500.0, 97.4, None, ltan_hours=10.5)
    f_dd = in_eclipse(dd.position_eci(t), sun_vector_eci(t)).mean()
    f_mm = in_eclipse(mm.position_eci(t), sun_vector_eci(t)).mean()
    assert f_dd < f_mm


def test_polar_station_sees_more_passes_than_mid_latitude():
    """Svalbard exists for a reason. If a mid-latitude station beats it, the
    access geometry is wrong."""
    o = Orbit(500.0, 97.4, None, 10.5)
    dur = 3 * SOLAR_DAY
    svalbard = compute_passes(o, [("Svalbard", 78.23, 15.41)], dur, 20.0, 10.0)
    harwell = compute_passes(o, [("Harwell", 51.57, -1.31)], dur, 20.0, 10.0)
    assert len(svalbard) > len(harwell)


def test_elevation_is_90_at_zenith():
    stn = station_ecef(0.0, 0.0)
    overhead = stn * (1.0 + 500e3 / np.linalg.norm(stn))
    assert abs(float(elevation_deg(overhead[None], stn)[0]) - 90.0) < 0.5


def test_acquisition_rate_is_self_consistent():
    o = Orbit(500.0, 97.4, None, 10.5)
    acq = acquisition_rate(o, C.MissionProfile())
    # Ground speed for a 500 km circular orbit is ~7.0 km/s.
    assert 6.8 < acq["ground_speed_kms"] < 7.2
    area_per_tile = C.TILE_GROUND_KM ** 2
    assert abs(acq["tiles_per_day"] * area_per_tile - acq["area_km2_per_day"]) < 1.0


# ===========================================================================
# edge model
# ===========================================================================
def test_latency_model_reproduces_the_flown_anchor():
    """The calibration must round-trip: feeding CloudScout's op count back
    through the model has to give back CloudScout's measured 325 ms."""
    proc = EdgeProcessor("myriad2")
    cost = proc.cost_for_macs(C.CLOUDSCOUT_ANCHOR["gmacs"] * 1e9, "int8")
    assert abs(cost.latency_ms - C.CLOUDSCOUT_ANCHOR["latency_ms"]) < 1.0


def test_accelerator_beats_scalar_obc_by_orders_of_magnitude():
    fast = EdgeProcessor("myriad2").cost_for_macs(6.3e6)
    slow = EdgeProcessor("leon3_baseline").cost_for_macs(6.3e6)
    assert slow.latency_ms > 50 * fast.latency_ms


def test_int8_is_cheaper_than_fp32():
    p = EdgeProcessor("myriad2")
    assert p.cost_for_macs(1e6, "int8").latency_ms < p.cost_for_macs(1e6, "fp32").latency_ms


def test_cascade_costs_less_than_cnn_on_everything():
    from aegis.edge import cascade_cost
    p = EdgeProcessor("myriad2")
    cc = cascade_cost(p, PhysicsScreener.flops_per_tile(), 6.3e6, 1000, 0.5, 0.2)
    assert cc["cascade_energy_saving"] > 0.3
    assert cc["total_energy_j"] < cc["cnn_on_everything_energy_j"]


def test_mitigation_monotonically_reduces_upsets():
    hw = C.HARDWARE["myriad2"]
    rad = RadiationModel(hw)
    n = [rad.expected_flips(40000, 365.0, m) for m in RadiationModel.MITIGATIONS]
    assert n[0] > n[1] > n[3]        # none > ecc > ecc+scrub
    assert n[0] > n[2] > n[3]        # none > scrub > ecc+scrub


def test_bitflip_injection_actually_flips_bits():
    rad = RadiationModel(C.HARDWARE["myriad2"], seed=1)
    q = np.zeros((16, 64), dtype=np.int8)
    q2, err = rad.inject(q, 200)
    assert (q2 != q).sum() > 0
    assert err > 0
    assert q2.dtype == np.int8


def test_per_channel_quantisation_round_trips():
    w = np.random.default_rng(0).standard_normal((8, 4, 3, 3)).astype(np.float32)
    q, scale = RadiationModel.fake_quantise(w)
    deq = (q.reshape(8, -1).astype(np.float32) * scale[:, None]).reshape(w.shape)
    assert np.abs(deq - w).max() < np.abs(w).max() / 100.0


# ===========================================================================
# products and policy
# ===========================================================================
def test_product_sizes_are_ordered():
    roi = ROI(0, 0, 16, 16)
    assert (product_bits("alert") < product_bits("thumbnail")
            < product_bits("roi", roi) < product_bits("full"))


def test_roi_extraction_finds_the_blob():
    seg = np.full((1, 16, 16), -6.0, dtype=np.float32)
    seg[0, 6:9, 5:8] = 6.0
    roi = extract_roi(seg)
    assert roi is not None
    assert roi.y0 <= 24 and roi.y1 >= 36
    assert 0 < roi.fraction_of_tile < 1.0
    assert extract_roi(np.full((1, 16, 16), -6.0, dtype=np.float32)) is None


def test_novelty_decays_then_recovers():
    nt = NoveltyTracker(decay_hours=10.0)
    assert nt.novelty((0, 0), 0.0) == 1.0
    nt.mark((0, 0), 0.0)
    assert nt.novelty((0, 0), 60.0) < 0.05
    assert nt.novelty((0, 0), 40 * 3600.0) > 0.9


def _obs(**kw):
    base = dict(tile_index=0, t_obs=0.0, lat=0.0, lon=0.0, cell=(0, 0),
                true_event=False, true_class=0, true_cloud=0.0)
    base.update(kw)
    return Observation(**base)


def test_gated_tiles_are_discarded_without_scoring():
    pol = TriagePolicy()
    o = pol.decide(_obs(gated_out=True), 0.0, 0)
    assert o.action == Action.DISCARD and o.bits == 0.0


def test_cloudy_tiles_are_rejected_regardless_of_detection():
    pol = TriagePolicy(C.AutonomyPolicy(audit_sample_rate=0.0))
    o = pol.decide(_obs(pred_cloud=0.9, pred_class=1, pred_conf=0.99), 0.0, 0)
    assert o.action == Action.DISCARD


def test_confident_urgent_detection_raises_an_alert():
    pol = TriagePolicy(C.AutonomyPolicy(alert_confidence_threshold=0.8))
    o = pol.decide(_obs(pred_cloud=0.02, pred_class=1, pred_conf=0.97,
                        severity=0.8, roi=ROI(10, 10, 30, 30), weight=1.0), 0.0, 0)
    assert o.action == Action.ALERT


def test_low_confidence_detection_sends_the_full_tile():
    """If the model is unsure, it must not also be trusted to crop."""
    pol = TriagePolicy(C.AutonomyPolicy(event_confidence_threshold=0.5,
                                        alert_confidence_threshold=0.95))
    o = pol.decide(_obs(pred_cloud=0.02, pred_class=1, pred_conf=0.60,
                        severity=0.9, roi=ROI(20, 20, 30, 30), weight=1.0), 0.0, 0)
    assert o.action == Action.FULL


def test_queue_evicts_lowest_value_not_oldest():
    q = DownlinkQueue(capacity_bits=3 * product_bits("full"))
    for i in range(3):
        o = _obs(voi=0.1 * (i + 1)); o.bits = product_bits("full")
        assert q.push(o)
    high = _obs(voi=5.0); high.bits = product_bits("full")
    assert q.push(high)
    assert high in q.items
    assert min(o.voi for o in q.items) > 0.1     # the 0.1 item was evicted


def test_fifo_queue_simply_drops_when_full():
    q = DownlinkQueue(capacity_bits=2 * product_bits("full"))
    for i in range(2):
        o = _obs(voi=0.0); o.bits = product_bits("full")
        assert q.push_fifo(o)
    extra = _obs(voi=9.9); extra.bits = product_bits("full")
    assert not q.push_fifo(extra)                # no notion of value: lost


def test_scheduler_prefers_value_per_bit():
    q = DownlinkQueue(capacity_bits=1e12)
    cheap_good = _obs(voi=1.0); cheap_good.bits = 1000.0
    dear_bad = _obs(voi=1.1); dear_bad.bits = 1e6
    q.items = [dear_bad, cheap_good]
    sent, _ = q.schedule_pass(budget_bits=2000.0, t_now=100.0, prioritise=True)
    assert sent == [cheap_good]


# ===========================================================================
# screener
# ===========================================================================
def test_screener_learns_and_prefers_recall():
    rng = np.random.default_rng(0)
    n = 1500
    feats = rng.standard_normal((n, PhysicsScreener.N_FEAT)).astype(np.float32)
    keep = (feats[:, 0] + 0.5 * feats[:, 3] > 0).astype(np.float32)
    scr = PhysicsScreener().fit(feats, keep, epochs=400)
    scr.calibrate(feats, keep, min_recall=0.99)
    gate = scr.gate(feats)
    assert gate[keep.astype(bool)].mean() >= 0.97
    assert gate.mean() < 0.95            # it must actually reject something


def test_screener_state_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((300, PhysicsScreener.N_FEAT)).astype(np.float32)
    keep = (feats[:, 1] > 0).astype(np.float32)
    a = PhysicsScreener().fit(feats, keep, epochs=100)
    a.calibrate(feats, keep)
    p = tmp_path / "s.npz"
    np.savez(p, **a.state_dict())
    b = PhysicsScreener().load_state(np.load(p))
    assert np.allclose(a.predict_proba(feats), b.predict_proba(feats))
    assert a.threshold == pytest.approx(b.threshold)


# ===========================================================================
# mission accounting - the one that matters most
# ===========================================================================
def _tiny_pool(n=180, seed=4):
    from aegis.dataset import tiles_to_set
    from aegis.mission import TilePool
    g = SceneGenerator(seed=seed)
    ts = tiles_to_set(g.sample_batch(n))
    rng = np.random.default_rng(seed)
    prob = rng.dirichlet(np.ones(4), size=n)
    return TilePool(
        ts=ts,
        gate=rng.random(n) > 0.35,
        pred_cloud=np.clip(ts.cloud_frac + rng.normal(0, 0.05, n), 0, 1),
        pred_class=prob.argmax(1),
        pred_conf=prob.max(1),
        severity=rng.random(n),
        roi_bits=np.full(n, product_bits("roi", ROI(0, 0, 24, 24))),
        roi_frac=np.full(n, 0.14),
    )


@pytest.mark.parametrize("mode", ["aegis", "baseline"])
def test_event_accounting_closes(mode):
    """Every observable event must land in exactly one bucket: delivered,
    thumbnail-only, missed by the model, or missed for capacity. A leak here
    silently changes the headline benefit, so it is asserted, not eyeballed."""
    from aegis.mission import MissionSimulator
    sim = MissionSimulator(_tiny_pool(), days=1.0)
    r = sim.run(mode)
    accounted = (r.events_delivered + r.events_thumbnail_only
                 + r.events_missed_model + r.events_missed_capacity)
    assert abs(r.events_observable - accounted) < 1e-6 * max(r.events_observable, 1.0)
    assert abs(r.event_accounting_residual) < 1e-6 * max(r.events_observable, 1.0)


def test_ai_reduces_downlink_and_memory_versus_baseline():
    from aegis.mission import MissionSimulator
    sim = MissionSimulator(_tiny_pool(), days=1.0)
    base = sim.run("baseline").summary()
    aeg = sim.run("aegis").summary()
    assert aeg["gb_downlinked_per_day"] < base["gb_downlinked_per_day"]
    assert aeg["peak_memory_fraction"] <= base["peak_memory_fraction"]
    assert aeg["tx_on_time_hours"] <= base["tx_on_time_hours"]


def test_both_modes_acquire_identical_data():
    """The comparison is only fair if both spacecraft take the same pictures."""
    from aegis.mission import MissionSimulator
    sim = MissionSimulator(_tiny_pool(), days=1.0)
    base, aeg = sim.run("baseline"), sim.run("aegis")
    assert abs(base.tiles_acquired - aeg.tiles_acquired) < 1e-6
    assert abs(base.bits_generated_if_all - aeg.bits_generated_if_all) < 1e-3


def test_battery_never_violates_its_floor_or_capacity():
    from aegis.mission import MissionSimulator
    sim = MissionSimulator(_tiny_pool(), days=1.0)
    r = sim.run("aegis")
    socs = [s for _, s in r.soc_trace]
    assert socs and min(socs) >= 0.0 and max(socs) <= 1.0 + 1e-9
    assert r.brownouts == 0


def test_memory_never_exceeds_capacity():
    from aegis.mission import MissionSimulator
    for mode in ("aegis", "baseline"):
        r = MissionSimulator(_tiny_pool(), days=1.0).run(mode)
        assert r.peak_memory_fraction <= 1.0 + 1e-9


def test_simulation_is_deterministic():
    from aegis.mission import MissionSimulator
    a = MissionSimulator(_tiny_pool(), days=1.0).run("aegis").summary()
    b = MissionSimulator(_tiny_pool(), days=1.0).run("aegis").summary()
    assert a["gb_downlinked"] == pytest.approx(b["gb_downlinked"])
    assert a["event_delivery_rate"] == pytest.approx(b["event_delivery_rate"])


# ===========================================================================
# mission adapters
# ===========================================================================
def test_biomass_frame_volume_is_about_one_gigabyte():
    from aegis.biomass import frame_data_volume
    v = frame_data_volume()
    assert 0.8 < v["slc_gb"] < 1.1
    assert 120 < v["along_track_km"] < 160


def test_mission_adapters_are_real_time_feasible():
    from aegis.missions import analyse_all
    for m in analyse_all():
        assert m["real_time_feasible"], f"{m['mission']} cannot keep up"
        assert 0.0 <= m["downlink_reduction"] <= 1.0
        # The AI must not cost a meaningful slice of platform power.
        assert m["compute_power_fraction_of_platform"] < 0.05


# ===========================================================================
# RFI (real Biomass annotation data)
# ===========================================================================
def test_range_resolution_follows_bandwidth():
    from aegis.rfi import NOMINAL_BANDWIDTH_HZ, slant_range_resolution_m
    # c/(2B): 6 MHz -> ~25 m, and halving the bandwidth must double the number.
    assert abs(slant_range_resolution_m(NOMINAL_BANDWIDTH_HZ) - 24.98) < 0.1
    assert abs(slant_range_resolution_m(3e6)
               / slant_range_resolution_m(6e6) - 2.0) < 1e-6


def test_rfi_triage_policy_is_monotonic_and_bounded():
    import numpy as np
    from aegis.rfi import RfiMaskAnalysis, onboard_rfi_triage
    bins = 87
    for level in (0.01, 0.10, 0.40):
        frac = np.full(64, level)
        m = RfiMaskAnalysis(64, bins, True, frac, np.full(bins, level),
                            np.linspace(-3.8, 3.8, bins))
        r = onboard_rfi_triage(m)
        assert 0.0 <= r.downlink_reduction <= 1.0
        shares = (r.blocks_nominal + r.blocks_degraded
                  + r.blocks_severely_degraded)
        assert abs(shares - 1.0) < 1e-9
    # More interference must never mean less saving.
    def saving(level):
        frac = np.full(64, level)
        m = RfiMaskAnalysis(64, bins, True, frac, np.full(bins, level),
                            np.linspace(-3.8, 3.8, bins))
        return onboard_rfi_triage(m).downlink_reduction
    assert saving(0.01) <= saving(0.10) <= saving(0.40)
