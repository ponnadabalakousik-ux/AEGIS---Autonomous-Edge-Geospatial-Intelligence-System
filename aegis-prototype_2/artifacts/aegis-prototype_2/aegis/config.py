"""
AEGIS - Autonomous Edge Geospatial Intelligence System
======================================================
Central configuration: sensor model, hardware profiles, mission profiles.

Every number in this file is either (a) taken from published in-orbit AI
missions / hardware datasheets, or (b) an explicit engineering assumption
flagged with ASSUMPTION. Nothing is hidden inside the algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

RNG_SEED = 20260810

# ---------------------------------------------------------------------------
# 1. SENSOR MODEL
# ---------------------------------------------------------------------------
# Six-band subset of the Sentinel-2 MSI band set. These six carry almost all
# of the information needed for cloud screening and thermal-anomaly / water
# discrimination, which is why onboard demonstrators tend to use a reduced
# band set rather than the full 13.
BANDS: List[str] = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAND_WAVELENGTH_NM: Dict[str, int] = {
    "B02": 490,    # blue
    "B03": 560,    # green
    "B04": 665,    # red
    "B08": 842,    # NIR
    "B11": 1610,   # SWIR-1
    "B12": 2190,   # SWIR-2
}
N_BANDS = len(BANDS)

TILE_PX = 64                 # tile edge in pixels
GSD_M = 20.0                 # ground sample distance (m) - S2 20 m bands
TILE_GROUND_KM = TILE_PX * GSD_M / 1000.0   # 1.28 km edge

RAW_BITS_PER_SAMPLE = 12     # detector quantisation
# Raw tile size on the bus before any compression, in bits.
RAW_TILE_BITS = TILE_PX * TILE_PX * N_BANDS * RAW_BITS_PER_SAMPLE

# Lossless-ish onboard compression (CCSDS 123.0-B-2 typical ratio on
# multispectral EO data is ~2.5-3.5x). ASSUMPTION: 3.0x, applied in both the
# baseline and the AEGIS pipeline so it never flatters the AI result.
LOSSLESS_COMPRESSION_RATIO = 3.0
# Lossy ROI/preview compression used only for thumbnails and alert packets.
THUMBNAIL_COMPRESSION_RATIO = 60.0
ROI_COMPRESSION_RATIO = 8.0

# ---------------------------------------------------------------------------
# 2. SURFACE + ATMOSPHERE SPECTRAL LIBRARY
# ---------------------------------------------------------------------------
# Top-of-atmosphere reflectance (0-1) per band, ordered as BANDS.
# Values are representative literature means for Sentinel-2 TOA reflectance.
# The snow entry matters: snow is as bright as cloud in the visible but
# collapses in SWIR. That is exactly the case a brightness threshold gets
# wrong and a learned model gets right, so it is the discriminating test
# case for the whole "why AI, not a threshold" argument.
SURFACE_SPECTRA: Dict[str, List[float]] = {
    "water":      [0.055, 0.045, 0.030, 0.018, 0.010, 0.008],
    "vegetation": [0.030, 0.058, 0.032, 0.400, 0.200, 0.085],
    "soil":       [0.120, 0.160, 0.220, 0.300, 0.350, 0.300],
    "urban":      [0.150, 0.160, 0.180, 0.220, 0.250, 0.220],
    "snow":       [0.850, 0.880, 0.870, 0.750, 0.090, 0.045],
    "cloud":      [0.750, 0.780, 0.800, 0.780, 0.620, 0.500],
    "cirrus":     [0.220, 0.225, 0.230, 0.240, 0.180, 0.140],
    "smoke":      [0.320, 0.290, 0.265, 0.230, 0.120, 0.090],
}
SURFACE_TEXTURE_SIGMA: Dict[str, float] = {
    "water": 0.010, "vegetation": 0.055, "soil": 0.060, "urban": 0.075,
    "snow": 0.045, "cloud": 0.070, "cirrus": 0.040, "smoke": 0.050,
}

# Relative frequency of each land background in the simulated world.
LAND_CLASS_WEIGHTS: Dict[str, float] = {
    "vegetation": 0.34, "water": 0.30, "soil": 0.18,
    "urban": 0.10, "snow": 0.08,
}

# Global cloud cover statistics. ~67% of the Earth is cloudy at any time
# (MODIS climatology), which is the single biggest source of worthless
# downlink in an optical EO mission and the reason cloud screening is the
# first app flown on every onboard-AI demonstrator.
GLOBAL_CLOUD_PROB = 0.67
CLOUD_FRACTION_REJECT = 0.35    # tile is scientifically useless above this

# ---------------------------------------------------------------------------
# 3. EVENT MODEL
# ---------------------------------------------------------------------------
EVENT_CLASSES: List[str] = ["nominal", "wildfire", "flood", "vessel"]
EVENT_INDEX: Dict[str, int] = {c: i for i, c in enumerate(EVENT_CLASSES)}

# Prior probability that a *cloud-free* tile contains each event type.
# Real events are rare - that rarity is the whole point. If events were
# common there would be nothing to triage.
EVENT_PRIOR: Dict[str, float] = {
    "nominal": 0.880, "wildfire": 0.040, "flood": 0.030, "vessel": 0.050,
}

# Operational value weight of each class (used by the autonomy scorer).
# Wildfire scores highest because time-to-alert has direct life-safety value.
EVENT_VALUE_WEIGHT: Dict[str, float] = {
    "nominal": 0.05, "wildfire": 1.00, "flood": 0.80, "vessel": 0.45,
}
# Latency requirement in minutes; drives the alert-vs-bulk routing decision.
EVENT_LATENCY_TARGET_MIN: Dict[str, float] = {
    "nominal": 1e9, "wildfire": 30.0, "flood": 120.0, "vessel": 360.0,
}

# ---------------------------------------------------------------------------
# 4. EDGE COMPUTE PROFILES
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareProfile:
    """A payload processor the AI could run on.

    `int8_gops_effective` is *sustained* INT8 throughput, not the datasheet
    peak. The model is calibrated against a published in-orbit data point
    (see edge.calibrate_against_cloudscout) so latency predictions are
    anchored to something that actually flew.
    """
    name: str
    vendor: str
    int8_gops_effective: float   # sustained INT8 GOP/s
    power_active_w: float        # inference-time power draw
    power_idle_w: float
    dram_mb: float
    mass_g: float
    tid_krad: float              # total ionising dose tolerance
    seu_rate_per_mbit_day: float # upsets per Mbit of SRAM per day @ 500 km SSO
    flight_heritage: str

# Sustained-throughput figures are conservative fractions of vendor peak,
# consistent with the published Myriad/FPGA benchmarks for real CNNs.
HARDWARE: Dict[str, HardwareProfile] = {
    "myriad2": HardwareProfile(
        name="Intel Movidius Myriad 2 VPU", vendor="Intel/Ubotica",
        int8_gops_effective=180.0, power_active_w=2.0, power_idle_w=0.35,
        dram_mb=512, mass_g=45, tid_krad=10.0, seu_rate_per_mbit_day=0.9,
        flight_heritage="Phi-Sat-1 (2020) - first DNN inference in orbit",
    ),
    "zynq_us_dpu": HardwareProfile(
        name="Zynq UltraScale+ MPSoC (DPU overlay)", vendor="AMD/Xilinx",
        int8_gops_effective=600.0, power_active_w=2.5, power_idle_w=0.60,
        dram_mb=2048, mass_g=120, tid_krad=100.0, seu_rate_per_mbit_day=2.4,
        flight_heritage="Euclid, multiple CubeSat payloads",
    ),
    "myriad_x": HardwareProfile(
        name="Intel Movidius Myriad X VPU", vendor="Intel/Ubotica",
        int8_gops_effective=420.0, power_active_w=2.8, power_idle_w=0.40,
        dram_mb=512, mass_g=48, tid_krad=15.0, seu_rate_per_mbit_day=0.8,
        flight_heritage="CogniSat-6 / in-orbit demos",
    ),
    "leon3_baseline": HardwareProfile(
        # The honest control case: what a conventional rad-hard OBC can do.
        name="GR712RC LEON3-FT (no accelerator)", vendor="Frontgrade Gaisler",
        int8_gops_effective=0.35, power_active_w=1.5, power_idle_w=0.8,
        dram_mb=256, mass_g=80, tid_krad=300.0, seu_rate_per_mbit_day=0.02,
        flight_heritage="Extensive - classic rad-hard OBC",
    ),
}
DEFAULT_HARDWARE = "myriad2"

# Published anchor point used to calibrate the latency model:
# CloudScout on Phi-Sat-1 / Myriad 2, 512x512x3 input, 325 ms, ~2 W, 92% acc.
CLOUDSCOUT_ANCHOR = {
    "input_hw": 512, "input_c": 3, "latency_ms": 325.0, "power_w": 2.0,
    "gmacs": 3.6,   # ASSUMPTION: CloudScout fwd pass ~3.6 GMAC at 512x512x3
}

# ---------------------------------------------------------------------------
# 5. SPACECRAFT / MISSION PROFILE
# ---------------------------------------------------------------------------
@dataclass
class MissionProfile:
    """A 6U-class optical EO smallsat in a sun-synchronous orbit."""
    altitude_km: float = 500.0
    inclination_deg: float = 97.4          # SSO at 500 km
    raan_deg: float = 0.0
    ltan_hours: float = 10.5               # local time of ascending node

    # --- Power ---
    solar_array_w: float = 65.0            # 6U deployable, EOL, sun-pointing
    bus_load_w: float = 12.0               # OBC, ADCS, thermal, housekeeping
    payload_imaging_w: float = 9.0         # imager during acquisition
    battery_wh: float = 80.0
    battery_min_soc: float = 0.35          # never discharge below this

    # --- Imager ---
    # A 20 m GSD, 6-band pushbroom with a 68 km swath imaging on the daylit
    # half of each orbit. This is what makes the mission downlink-limited:
    # the instrument comfortably out-produces the ground segment, which is
    # precisely the situation the problem statement describes.
    swath_km: float = 68.0
    daylight_duty: float = 0.50            # fraction of the orbit spent imaging
    # The mission simulation is a Monte-Carlo subsample: it makes decisions on
    # `sim_tiles_per_day` representative tiles and scales every bit, joule and
    # byte by the resulting weight. Keeps the campaign runnable without
    # pretending the spacecraft only takes 6000 pictures a day.
    sim_tiles_per_day: int = 6000

    # --- Comms ---
    xband_rate_mbps: float = 100.0         # high-rate payload downlink
    xband_tx_power_w: float = 28.0         # DC draw of the transmit chain
    xband_link_efficiency: float = 0.80    # framing, coding, acquisition time
    sband_rate_kbps: float = 256.0         # TT&C + urgent alert path
    sband_tx_power_w: float = 6.0
    min_elevation_deg: float = 10.0
    xband_min_elevation_deg: float = 20.0  # high-rate link needs a good pass
    # Ground stations are shared assets; a smallsat mission does not get every
    # geometric pass it could theoretically use.
    pass_utilisation: float = 0.60
    # Optional inter-satellite relay for time-critical alerts. Modelled as an
    # always-available low-rate path with a fixed latency, i.e. a commercial
    # data-relay service rather than a dedicated constellation.
    relay_available: bool = True
    relay_rate_kbps: float = 100.0   # token-bucketed: credit accrues between bursts
    relay_latency_min: float = 4.0
    relay_power_w: float = 9.0

    # --- Storage ---
    mass_memory_gb: float = 32.0

    # --- Life-limited items ---
    xband_tx_rated_hours: float = 4000.0
    battery_rated_cycles: float = 30000.0

# Ground stations: (name, lat_deg, lon_deg). A modest commercial network -
# deliberately not a huge one, because pass scarcity is what makes onboard
# triage valuable.
GROUND_STATIONS: List[Tuple[str, float, float]] = [
    ("Svalbard",     78.23,   15.41),
    ("Harwell (UK)", 51.57,   -1.31),
    ("Punta Arenas", -53.16, -70.91),
    ("Perth",       -31.95,  115.86),
]

SIM_DAYS = 7.0
TIME_STEP_S = 20.0

# ---------------------------------------------------------------------------
# 6. AUTONOMY POLICY
# ---------------------------------------------------------------------------
@dataclass
class AutonomyPolicy:
    """Thresholds for the onboard decision tree. Uplinkable as a small
    parameter block - the whole point of a modular AI payload is that the
    operator retunes policy without re-flashing the network."""
    cloud_reject_threshold: float = 0.35
    event_confidence_threshold: float = 0.55
    alert_confidence_threshold: float = 0.80
    voi_downlink_threshold: float = 0.18
    voi_thumbnail_threshold: float = 0.06
    novelty_decay_hours: float = 36.0
    # Counted in *real* alerts per orbit, not simulated ones. Acts as a
    # ground-segment sanity valve so a model failure cannot flood the relay.
    max_alerts_per_orbit: float = 25000.0
    # Safety net: fraction of nominal-looking tiles kept anyway, so the
    # ground can audit what the model threw away. Non-negotiable in a real
    # mission - never let a model be the only witness to its own errors.
    audit_sample_rate: float = 0.02

DEFAULT_POLICY = AutonomyPolicy()

# ---------------------------------------------------------------------------
# 7. TRAINING
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    n_train: int = 14000
    n_val: int = 1600
    n_test: int = 2200
    batch_size: int = 64
    epochs: int = 40
    lr: float = 2.5e-3
    weight_decay: float = 1e-4
    seed: int = RNG_SEED

DEFAULT_TRAIN = TrainConfig()

ARTIFACT_DIR = "artifacts"
