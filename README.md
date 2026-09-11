# AEGIS — Autonomous Edge Geospatial Intelligence System

A working prototype of an AI-driven payload that decides, **in orbit**, what is worth
computing, keeping and downlinking.

Built against the problem statement: *"Using AI to autonomously decipher and process
data in situ through edge-processing would mean that only the useful measurements and
data are performed and sent back to Earth for analysis."*

Every number in this repository is produced by `python run_all.py` from a single seed.
Nothing is typed in by hand.

---

## Headline result

Two spacecraft. Identical orbit, identical imagery, identical link budget. One
downlinks oldest-first, because that is what a conventional mission does. The other
runs the onboard cascade and decides. The only difference is payload software.

| Claimed benefit | Baseline (no AI) | AEGIS | Change |
|---|---|---|---|
| Data transmission | 39.0 GB/day | 2.9 GB/day | **−92.5 %** |
| Real events delivered to ground | 25.2 % | 91.9 % | **3.6× more** |
| Median observation → ground | 1095 min | 12 min | **89× faster** |
| Urgent alerts (relay path) | — | 4 min | new capability |
| Net energy (comms + compute) | 234 Wh | 139 Wh | **−40.6 %** |
| Transmitter on-time (life-limited) | 8.4 h | 1.7 h | **−79.6 %** |
| Peak mass-memory fill | 100 % | 2.7 % | 63.8 M tiles lost → 0 |

The mission is deliberately downlink-limited, as real optical EO missions are: the
instrument produces 155 GB/day compressed against a ground segment worth 41 GB/day —
**3.8× oversubscribed**.

---

## What is actually in here

### 1. A physically-motivated scene simulator (`aegis/scene.py`)

Sentinel-2-like 6-band tiles (B02/B03/B04/B08/B11/B12 at 20 m) with pixel-accurate
truth for both cloud and event. Not arbitrary noise:

- literature TOA reflectance spectra per surface class
- spatially correlated 1/f^β fractal fields, so texture is scale-free like real land
  cover and cloud
- **fire modelled the way an imager actually sees it** — sub-pixel ~800 K emission
  lifts SWIR-2 far more than SWIR-1, which is the basis of every operational
  Sentinel-2 fire index
- smoke plumes that scatter in the blue and vanish in SWIR — the confuser that makes
  naive brightness-based cloud screening throw the fire away
- **snow**, which is as bright as cloud in the visible and collapses in SWIR. This is
  the case a threshold gets wrong and a learned model gets right, and it is asserted
  in the test suite
- shot noise, read noise, 12-bit quantisation

`scene.RealSceneAdapter` documents the single interface to implement to swap in real
Sentinel-2 L1C.

### 2. A three-stage onboard cascade (`aegis/models.py`)

Cheapest-first, because on a power-constrained payload you do not run a neural network
on every tile — you run the cheapest thing that can safely say *no*.

| Stage | What | Cost/tile | Runs on | Effect |
|---|---|---|---|---|
| 0 — spectral gate | 32 index statistics → logistic gate | 18 kFLOP | 100 % of tiles, housekeeping CPU | rejects 51 % while keeping 99.4 % of useful tiles |
| 1 — TriageNet | depthwise-separable CNN; cloud fraction + 4-class event + coarse mask | 6.3 MMAC, 38 k params | ~49 % of tiles, INT8 accelerator | macro-F1 0.945 |
| 2 — ROI extraction | 16×16 mask → padded bounding box | ~4 kFLOP | detections only | sends the box, not the scene |

### 3. An edge-hardware model calibrated to something that flew (`aegis/edge.py`)

Latency is not derived from vendor peak-TOPS. It is solved from the published Φ-Sat-1
measurement — CloudScout, 512×512×3, **325 ms on a Myriad 2 at ~2 W** — which implies a
sustained utilisation of about 12 %. The test suite asserts the model round-trips that
anchor to within 1 ms.

Four payload processors are compared, including the honest control case: a
conventional rad-hard LEON3 OBC with no accelerator, where the same network takes
65 ms/tile and burns 35× the energy. Onboard AI of this kind is not a software upgrade
to existing avionics.

### 4. A real spacecraft model (`aegis/spacecraft.py`)

Because the whole benefit case is a *resource* argument, and a resource argument is
only credible if the resources are modelled:

- circular SSO with J2 nodal precession — the model reproduces the sun-synchronous
  drift rate of **0.9854 °/day** against the defining 0.9856 from first principles
- rotating Earth, geometric elevation-angle access to four real ground stations
- cylindrical-shadow eclipse driving a battery with a depth-of-discharge floor
- life-limited-item accounting: transmitter on-time and battery cycles

### 5. A value-of-information autonomy layer (`aegis/autonomy.py`)

```
VoI = w_class · confidence · (1 − cloud) · severity · novelty · urgency
```

Novelty decays exponentially per ground cell, so the spacecraft does not spend three
days re-sending the same large fire. Downlink scheduling is a greedy VoI-per-bit
knapsack — O(n log n), no solver, and explainable to an operations team, which matters
more onboard than the last few percent of optimality.

Mass memory evicts the **lowest-value** item, not the oldest. A FIFO recorder throws
away the fire to keep the cloud; that is the failure mode onboard prioritisation
removes.

### 6. Modularity, demonstrated on real ESA Biomass data (`aegis/biomass.py`)

The brief asks for a system that can augment an existing satellite through modular
integration. The front end is swapped from a 6-band optical imager to **Biomass**,
ESA's P-band fully-polarimetric SAR (Airbus prime) — completely different physics —
and the decision layer, cost model and mission simulator are unchanged.

Real frame: `BIO_S1_SCS__1M … T006_F300`, 2025-11-21, Rondônia, Brazil, tomographic
phase, quad-pol. 1373 × 21180 × 4 pol complex ≈ **0.93 GB for 21 seconds of
acquisition**.

Forest/clearing separation at P-band works because a 70 cm wavelength penetrates the
canopy and scatters off trunks: intact forest returns strong volume and double-bounce,
cleared land scatters specularly away and collapses to dark.

**Caveats, stated up front:** it runs on the product's Pauli-composite quicklook, not
calibrated SLC (the measurement arrays are absent from the annotation-only download);
labels are physics-derived weak labels, not PRODES/DETER truth; and the headline
accuracy uses only descriptors *not* used to build those labels, with the circular
all-feature score reported beside it so the gap is visible.

### 7. Onboard RFI screening on real Biomass interference data (`aegis/rfi.py`)

The strongest Biomass case, and the only one in this repository that needs no
labels, no training data and no network at all.

Biomass transmits at 435 MHz into a **6 MHz** allocation — a sliver of heavily
contended spectrum. When an interferer sits in the band the processor must notch
it out, and notched bandwidth is gone: slant-range resolution is c/2B, so
interference directly coarsens the science product.

The figures are read from the delivered products — the ground processor's own
notch decisions in the L1A LUT (`rfiMitigation/rfiFreqMask*`, 329 azimuth blocks
× 87 frequency bins) and the RFI report in the annotation. Two frames from the
same day, 2025-11-21:

| Frame | Persistent RFI (avg / max % of band) | Isolated RFI (% of lines) | Effective bandwidth | Range resolution |
|---|---|---|---|---|
| T006/F300 Rondônia, Brazil | 4.13 % / 6.19 % | 45.2 % | 5.752 MHz | 26.06 m (+4.3 %) |
| T007/F132 Fujian, China | **15.13 % / 31.31 %** | **99.8 %** | 5.092 MHz | 29.44 m (**+17.8 %**), worst 36.37 m |

The mask is identical across all four polarisations, and it shows a **fixed
emitter parked almost exactly at band centre**: bins at −0.22 and +0.13 MHz are
notched in 100 % of blocks, ±0.5 MHz in over 90 %. That is not transient
interference — it is a permanent tax on the mission's bandwidth over that region.

Applied as an onboard policy on the Rondônia frame: 75.7 % of blocks intact and
sent at full rate, 24.0 % degraded and deferred, 0.3 % severely corrupted and
reduced to a summary — a 12.3 % link saving on the *quiet* frame, and far more
over a contended region. Nothing is deleted; the policy sets rate and ordering,
and every block is flagged so the ground can request a re-downlink.

The onboard implementation is a periodogram of the raw echo and a threshold,
before focusing. Cheaper than the cloud screener.

### 8. Mission adapters for real spacecraft (`aegis/missions.py`)

| Mission | Onboard app | Reduction | Why it's interesting |
|---|---|---|---|
| **MicroCarb** | cloud/aerosol screening of CO₂ soundings | ~75 % | the purest case — contaminated soundings are already discarded on the ground, so screening in orbit loses nothing |
| **Biomass** | RFI screening + disturbance-led frame ranking | ~46 % | the hard case — no cloud to discard, every frame is wanted science, so the saving must come from *ordering*, not deletion |
| **AEGIS 6U** | full cascade | ~94 % | the bespoke case |

---

## Four findings worth reading

**INT8 quantisation nearly shipped a broken payload.** Default MinMax calibration sets
each activation's scale from the single most extreme value seen, so one bright cloud
edge stretches the range and crushes everything else. It cost **half the macro-F1
(0.92 → 0.50)**. Percentile calibration recovered most of it (0.76); adding
`reduce_range` — holding weights to 7 bits so INT8 accumulation cannot saturate —
recovered the rest and matched FP32 at a quarter of the weight memory. Nothing in the
tooling warns you. The comparison now runs inside the pipeline on every execution.

**The cheap gate was more expensive than the network it gates.** Computed at full
resolution, Stage 0 cost 0.9 mJ/tile against the CNN's 1.1 mJ — the cascade *lost*
energy overall. Decimating the feature computation 4× (spectral statistics do not need
full resolution) dropped it to 0.055 mJ and the cascade now saves 46 %. A unit test
caught this, not intuition.

**Interference at P-band is worse than the mission's own averages suggest.** Two
frames on the same day differ by nearly 4× in bandwidth lost, and over China the
notch costs 15 % of a 6 MHz allocation — an 18 % coarsening of range resolution,
peaking at 46 %. That is measured from ESA's own products, not modelled.

**Value-ordered scheduling does nothing at the nominal ground segment — and that is
the correct result.** Once triage removes 92 % of the volume, everything queued fits
in the available contacts, so the order is irrelevant. The link-stress sweep takes
contacts away until the constraint returns: at the crossover, value-ordering delivers
the same events roughly **2× faster**, and under severe constraint it changes *what*
arrives, not just when. The scheduler is insurance against a degraded ground segment,
not the main mechanism — worth knowing which of the two it is.

---

## Running it

```bash
pip install numpy scipy scikit-learn pillow matplotlib torch onnx onnxruntime onnxscript pytest

python run_all.py                 # full pipeline, ~12 min on 2 cores
python run_all.py --skip-train    # reuse the trained network, ~3 min
python run_all.py --days 14       # longer campaign
python run_all.py --hardware zynq_us_dpu

# Include the Biomass adapter and RFI analysis (one or two products):
AEGIS_BIOMASS_DIR=/path/to/BIO_S1_SCS__1M_...F300 \
AEGIS_BIOMASS_DIR2=/path/to/BIO_S1_SCS__1M_...F132 python run_all.py

pytest tests/ -q                  # 43 invariant tests
```

Outputs land in `artifacts/`: `results.json` (everything) and
`aegis_dashboard.html` (self-contained report, no CDN, works offline).

---

## What this prototype does **not** show

- **The imagery is simulated.** Physically motivated, but simulated. Real Sentinel-2
  L1C will be harder, particularly at cloud edges and over bright desert.
- **Event priors are inflated** — wildfire at 4 % of cloud-free tiles is orders of
  magnitude above reality, set that way so the campaign has statistical power. Rates
  and ratios hold; absolute event counts do not.
- **Vessel detection is weak** and should be — a ship is a few pixels at 20 m GSD.
- **The Biomass demo runs on browse imagery**, not calibrated SLC.
- **No ADCS, thermal or slew modelling.** Life-limited-item accounting covers
  transmitter on-time and battery cycles only.
- **The radiation model is statistical**, not a beam-test result. It gives the shape of
  the degradation curve, not a qualification number.

---

## Layout

```
aegis/
  config.py       all tunable constants; every figure sourced or marked ASSUMPTION
  scene.py        multispectral scene simulator + spectral indices
  dataset.py      dataset construction and caching
  models.py       the three-stage cascade
  train.py        training + ONNX export
  quantise.py     INT8 PTQ, calibration study, operating point, radiation study
  edge.py         payload-processor cost model + SEU injection
  spacecraft.py   orbit, ground-station access, power, storage, wear
  autonomy.py     value-of-information scoring + downlink scheduling
  mission.py      baseline-vs-AEGIS campaign
  missions.py     MicroCarb / Biomass / bespoke adapters
  biomass.py      real ESA Biomass P-band SAR front end
  rfi.py          onboard RFI screening on real Biomass interference data
  report.py       self-contained HTML dashboard
run_all.py        the pipeline
tests/            43 invariant tests
```

---

## Sources

- [Φ-Sat-1 / Φ-Sat-2 mission characteristics — eoPortal](https://www.eoportal.org/satellite-missions/phisat-1)
- [CloudScout figures: 92 % accuracy, 1 % FPR, 325 ms, 2 W on Myriad 2](https://arxiv.org/html/2504.03891v1)
- [FPGA-based neural network accelerators for space: a survey](https://arxiv.org/html/2504.16173v1)
- [Biomass mission characteristics — eoPortal](https://www.eoportal.org/satellite-missions/biomass)
- [Biomass Level 1A product definition — ESA Earth Online](https://earth.esa.int/eogateway/catalog/biomass-level-1a)
- [MicroCarb mission characteristics — eoPortal](https://www.eoportal.org/satellite-missions/microcarb)
- Product annotation of `BIO_S1_SCS__1M_20251121T095108_…_T006_F300_01_DJUQK1`
