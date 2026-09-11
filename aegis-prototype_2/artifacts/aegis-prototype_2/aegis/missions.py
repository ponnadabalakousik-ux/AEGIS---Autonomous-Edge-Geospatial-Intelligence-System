"""
Mission adapters: applying the AEGIS cascade to real spacecraft.
===============================================================

The brief asks for a system that can either augment an existing satellite
through modular integration, or be built bespoke. This module is the
"augment an existing satellite" half: it takes the published characteristics
of three real missions and works out what an onboard triage payload would
actually buy on each, using the same cost model as the simulator.

The three cases were chosen because they fail differently:

  MICROCARB   The purest case. A CO2 spectrometer whose soundings are useless
              when the footprint is cloudy, and cloud is discarded on the
              ground. Screening in orbit costs almost nothing and removes most
              of the downlink. Nothing is lost that was not going to be thrown
              away anyway.

  BIOMASS     The hardest case, and the more interesting one to a systems
              engineer. Every frame is scientifically wanted - there is no
              cloud to throw away at P-band - so the saving cannot come from
              discarding. It comes from *prioritising*: disturbance-front
              frames at full rate and first, slow-changing interior forest
              later or decimated, backed by the repeat stack.

  6U OPTICAL  The bespoke case: the simulated smallsat in `mission.py`, where
              the payload is designed around the AI from the start.

Every figure below is either a published mission number (marked SOURCE) or an
explicit assumption (marked ASSUMPTION). Nothing is smuggled in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config as C
from .edge import EdgeProcessor
from .models import PhysicsScreener


@dataclass
class MissionCase:
    key: str
    name: str
    operator: str
    instrument: str
    # --- published characteristics ---
    daily_data_gbit: float
    downlink_rate_mbps: float
    orbit_km: float
    platform_power_w: float
    # --- what the AI would do ---
    ai_app: str
    rejectable_fraction: float      # data that is worthless and identifiable
    rejectable_basis: str
    residual_rate_on_kept: float    # 1.0 = kept data sent at full fidelity
    sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # -- derived ----------------------------------------------------------
    def contact_seconds_needed(self, gbit: float) -> float:
        return gbit * 1e9 / (self.downlink_rate_mbps * 1e6)

    def analyse(self, proc: EdgeProcessor, macs_per_tile: float,
                tile_bits: float) -> Dict:
        """Cost the triage payload against the mission's own data budget."""
        kept = (1.0 - self.rejectable_fraction) * self.residual_rate_on_kept
        gbit_after = self.daily_data_gbit * kept

        # How many inference invocations does a day of data imply?
        tiles_per_day = self.daily_data_gbit * 1e9 / max(tile_bits, 1.0)
        c0 = proc.cost_for_flops_cpu(PhysicsScreener.flops_per_tile())
        c1 = proc.cost_for_macs(macs_per_tile, "int8")
        # Stage 1 only runs on what Stage 0 could not reject outright.
        stage1_rate = max(1.0 - self.rejectable_fraction * 0.85, 0.05)
        energy_j = tiles_per_day * (c0.energy_mj + stage1_rate * c1.energy_mj) / 1000.0
        compute_s = tiles_per_day * (c0.latency_ms + stage1_rate * c1.latency_ms) / 1000.0

        contact_before = self.contact_seconds_needed(self.daily_data_gbit)
        contact_after = self.contact_seconds_needed(gbit_after)

        return {
            "mission": self.name,
            "instrument": self.instrument,
            "ai_app": self.ai_app,
            "daily_gbit_before": self.daily_data_gbit,
            "daily_gbit_after": gbit_after,
            "downlink_reduction": 1.0 - kept,
            "contact_s_per_day_before": contact_before,
            "contact_s_per_day_after": contact_after,
            "contact_s_saved_per_day": contact_before - contact_after,
            "tiles_per_day": tiles_per_day,
            "compute_s_per_day": compute_s,
            "compute_duty_cycle": compute_s / 86400.0,
            "compute_energy_wh_per_day": energy_j / 3600.0,
            "compute_power_fraction_of_platform":
                (energy_j / 86400.0) / max(self.platform_power_w, 1e-9),
            "real_time_feasible": compute_s < 86400.0,
            "rejectable_basis": self.rejectable_basis,
            "sources": self.sources,
            "assumptions": self.assumptions,
        }


# ---------------------------------------------------------------------------
MICROCARB = MissionCase(
    key="microcarb",
    name="MicroCarb",
    operator="CNES / UK Space Agency (Airbus-built payload heritage)",
    instrument="4-band SWIR grating spectrometer (764, 1273, 1608, 2037 nm)",
    daily_data_gbit=500.0,          # SOURCE: published daily data volume
    downlink_rate_mbps=150.0,       # SOURCE: X-band downlink rate
    orbit_km=650.0,
    platform_power_w=110.0,         # SOURCE: average platform power
    ai_app=("Onboard cloud/aerosol screening of CO2 soundings using the O2 "
            "A-band radiance and the co-registered cloud imager, discarding "
            "contaminated soundings before they are stored"),
    rejectable_fraction=0.75,
    rejectable_basis=(
        "Soundings whose 4.5 x 9 km footprint is cloud- or aerosol-contaminated "
        "cannot yield a 1 ppm XCO2 retrieval and are rejected during ground "
        "processing. Screening them in orbit removes the same data earlier."),
    residual_rate_on_kept=1.0,
    sources=[
        "eoPortal MicroCarb: 500 Gbit/day, 150 Mbit/s X-band, 110 W average "
        "platform power, 4.5 x 9 km IFOV, 13.5 km swath, bands at 764/1273/"
        "1608/2037 nm",
    ],
    assumptions=[
        "ASSUMPTION: 75% sounding rejection. Cloud/aerosol screening yields for "
        "nadir CO2 spectrometers of this class are commonly quoted in the "
        "70-90% range; 75% is the conservative end. THIS IS THE SINGLE NUMBER "
        "TO REPLACE with the mission's own figure - every MicroCarb result "
        "scales linearly with it.",
        "ASSUMPTION: onboard screening reproduces the ground screen closely "
        "enough that the retained set is the same. In practice you would fly a "
        "deliberately conservative screen and keep an audit sample.",
    ],
)

BIOMASS_CASE = MissionCase(
    key="biomass",
    name="Biomass (Earth Explorer 7)",
    operator="ESA (Airbus prime)",
    instrument="P-band (435 MHz) fully-polarimetric SAR, 6 MHz bandwidth",
    # 117 Mbit/s raw x 12% radar duty cycle, over a day, is the honest way to
    # get to a daily volume from the published instantaneous rate.
    daily_data_gbit=117.0 * 0.12 * 86400.0 / 1000.0,
    downlink_rate_mbps=310.0,       # ASSUMPTION, see below
    orbit_km=666.0,
    platform_power_w=1500.0,        # SOURCE: platform electrical power
    ai_app=("Onboard RFI screening plus forest-disturbance detection, used to "
            "rank frames for downlink: disturbance fronts at full rate and "
            "first, slow-changing interior forest decimated and deferred"),
    rejectable_fraction=0.10,
    rejectable_basis=(
        "Only RFI-corrupted and no-data segments are genuinely discardable - "
        "P-band shares spectrum with terrestrial services and the 435 MHz band "
        "is heavily contended. Everything else is wanted science, so the "
        "benefit is ordering, not deletion."),
    residual_rate_on_kept=0.60,
    sources=[
        "eoPortal Biomass: 435 MHz, 6 MHz bandwidth, quad-pol, ~50 km swath, "
        "117 Mbit/s max raw rate, 12% radar duty cycle, 666 km dawn-dusk SSO, "
        "3-day tomographic / 17-day interferometric repeat, 1500 W platform",
        "Product annotation, BIO_S1_SCS__1M T006/F300 (2025-11-21): 1373 x "
        "21180 samples, 4 polarisations, 19.81 m range / 6.71 m azimuth pixel "
        "spacing -> 0.93 GB per frame as complex float32",
    ],
    assumptions=[
        "ASSUMPTION: 310 Mbit/s X-band. Not published in the sources consulted; "
        "chosen as representative for a 1200 kg Earth Explorer. Contact-time "
        "figures scale inversely with it.",
        "ASSUMPTION: 10% RFI/no-data rejection and 40% average rate reduction "
        "on deferred interior-forest frames. Both are policy choices an "
        "operator sets, not physical constants.",
        "ASSUMPTION: daily volume derived as raw rate x radar duty cycle, "
        "ignoring onboard compression and the fact that Biomass does not image "
        "continuously (land only, outside restricted zones). The true figure "
        "is lower; this is an upper bound.",
    ],
)

AEGIS_6U = MissionCase(
    key="aegis6u",
    name="AEGIS 6U demonstrator (bespoke)",
    operator="this prototype",
    instrument="6-band multispectral pushbroom, 20 m GSD, 68 km swath",
    daily_data_gbit=155.5 * 8,      # from the simulator's acquisition model
    downlink_rate_mbps=100.0,
    orbit_km=500.0,
    platform_power_w=65.0,
    ai_app=("Full three-stage cascade: spectral gate, multi-task CNN, ROI "
            "extraction, with value-of-information downlink scheduling"),
    rejectable_fraction=0.90,
    rejectable_basis=(
        "About two thirds of the Earth is cloudy at any moment and cloudy "
        "optical tiles carry no information; a further slice is nominal scenes "
        "with no event of interest, kept only as an audit sample."),
    residual_rate_on_kept=0.55,
    sources=["Simulated in mission.py; see the campaign results."],
    assumptions=["Design point of this prototype rather than a flown system."],
)

CASES: Dict[str, MissionCase] = {
    m.key: m for m in (MICROCARB, BIOMASS_CASE, AEGIS_6U)
}


def analyse_all(hardware: str = C.DEFAULT_HARDWARE,
                macs_per_tile: float = 6.3e6,
                tile_bits: float = C.RAW_TILE_BITS) -> List[Dict]:
    proc = EdgeProcessor(hardware)
    return [c.analyse(proc, macs_per_tile, tile_bits) for c in CASES.values()]
