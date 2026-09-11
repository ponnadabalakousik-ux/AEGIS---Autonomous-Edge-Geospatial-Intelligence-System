"""
Multispectral scene simulator.
==============================
Generates Sentinel-2-like 6-band tiles with pixel-accurate ground truth for
cloud fraction and event masks.

Why synthetic: the prototype needs *labelled* data where the truth is known
exactly at pixel level for both cloud and event, over tens of thousands of
tiles, with controllable event rarity. That does not exist as a single public
dataset. The generator is physically motivated rather than arbitrary:

  * per-class TOA reflectance spectra from published Sentinel-2 statistics
  * spatially correlated 1/f^beta fractal fields, so texture has the
    scale-free structure real land cover and cloud fields have
  * fire modelled the way it actually appears to a multispectral imager:
    sub-pixel high-temperature emission raises SWIR-2 far more than SWIR-1
    (Planck at ~800 K), which is the basis of every operational S2 fire index
  * shot + read noise then 12-bit quantisation

`RealSceneAdapter` documents the single interface a real Sentinel-2 L1C
loader must satisfy to replace this module wholesale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# fractal field generation
# ---------------------------------------------------------------------------
def fractal_field(size: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially correlated random field with a 1/f^beta power spectrum.

    beta ~ 1.6-2.2 gives cloud-like / terrain-like structure. Returned field
    is normalised to zero mean, unit std.
    """
    white = rng.standard_normal((size, size))
    fx = np.fft.fftfreq(size)[:, None]
    fy = np.fft.fftfreq(size)[None, :]
    radial = np.sqrt(fx ** 2 + fy ** 2)
    radial[0, 0] = 1.0 / size          # avoid divide-by-zero at DC
    spectrum = np.fft.fft2(white) / (radial ** (beta / 2.0))
    spectrum[0, 0] = 0.0
    out = np.real(np.fft.ifft2(spectrum))
    std = out.std()
    return out / std if std > 1e-9 else out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _gauss_blob(size: int, cy: float, cx: float, sy: float, sx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2)))


# ---------------------------------------------------------------------------
# tile container
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    """One observation. `cube` is (n_bands, H, W) TOA reflectance in [0, ~1.3]."""
    cube: np.ndarray
    cloud_mask: np.ndarray            # bool (H, W): opaque cloud or cirrus
    event_mask: np.ndarray            # bool (H, W): event footprint
    event_class: str
    land_class: str
    severity: float                   # 0-1, physical intensity of the event
    lat: float = 0.0
    lon: float = 0.0
    t_obs_s: float = 0.0
    tile_id: int = -1

    @property
    def cloud_fraction(self) -> float:
        return float(self.cloud_mask.mean())

    @property
    def event_label(self) -> int:
        return C.EVENT_INDEX[self.event_class]

    @property
    def is_usable(self) -> bool:
        """Ground truth: would an analyst get anything out of this tile?"""
        return self.cloud_fraction <= C.CLOUD_FRACTION_REJECT

    @property
    def is_true_event(self) -> bool:
        """A real, *observable* event: present AND not hidden by cloud."""
        if self.event_class == "nominal":
            return False
        if not self.is_usable:
            return False
        # Event must not itself be buried under the cloud that is present.
        visible = np.logical_and(self.event_mask, ~self.cloud_mask).sum()
        total = max(int(self.event_mask.sum()), 1)
        return (visible / total) > 0.5


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------
class SceneGenerator:
    def __init__(self, seed: int = C.RNG_SEED, size: int = C.TILE_PX):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self._land_names = list(C.LAND_CLASS_WEIGHTS.keys())
        w = np.array([C.LAND_CLASS_WEIGHTS[k] for k in self._land_names])
        self._land_p = w / w.sum()
        self._event_names = list(C.EVENT_PRIOR.keys())
        ew = np.array([C.EVENT_PRIOR[k] for k in self._event_names])
        self._event_p = ew / ew.sum()

    # -- building blocks ----------------------------------------------------
    def _spectrum(self, name: str) -> np.ndarray:
        return np.asarray(C.SURFACE_SPECTRA[name], dtype=np.float32)

    def _base_surface(self, land: str) -> Tuple[np.ndarray, np.ndarray]:
        """Background reflectance cube plus a secondary-class mixing map."""
        n = self.size
        spec = self._spectrum(land)
        sigma = C.SURFACE_TEXTURE_SIGMA[land]

        # Primary texture: one shared spatial field modulates all bands
        # coherently (albedo variation), plus small per-band decorrelated
        # texture. Real surfaces are strongly band-correlated; pure per-band
        # noise would make the problem artificially easy.
        shared = fractal_field(n, beta=2.0, rng=self.rng)
        cube = np.empty((C.N_BANDS, n, n), dtype=np.float32)
        for b in range(C.N_BANDS):
            per_band = fractal_field(n, beta=1.7, rng=self.rng)
            modulation = 1.0 + sigma * (0.8 * shared + 0.45 * per_band)
            cube[b] = spec[b] * modulation

        # Mix in a secondary land class over part of the tile (coastlines,
        # field boundaries, urban edges) so tiles are not single-class.
        mix_map = np.zeros((n, n), dtype=np.float32)
        if self.rng.random() < 0.55:
            other = self.rng.choice(self._land_names, p=self._land_p)
            if other != land:
                edge = fractal_field(n, beta=2.4, rng=self.rng)
                thresh = self.rng.uniform(-0.6, 0.6)
                mix_map = _smoothstep((edge - thresh) * 1.6 + 0.5)
                ospec = self._spectrum(other)
                osig = C.SURFACE_TEXTURE_SIGMA[other]
                otex = fractal_field(n, beta=2.0, rng=self.rng)
                for b in range(C.N_BANDS):
                    other_band = ospec[b] * (1.0 + osig * otex)
                    cube[b] = cube[b] * (1 - mix_map) + other_band * mix_map
        return cube, mix_map

    def _add_clouds(self, cube: np.ndarray, target_cf: float) -> np.ndarray:
        """Composite an optically-thick cloud field plus cirrus and shadow."""
        n = self.size
        if target_cf <= 0.001:
            return np.zeros((n, n), dtype=bool)

        field_ = fractal_field(n, beta=self.rng.uniform(1.7, 2.3), rng=self.rng)
        # Pick the threshold that yields the requested cloud fraction exactly.
        thresh = np.quantile(field_, 1.0 - target_cf)
        # Optical thickness ramps up over the cloud edge rather than being
        # binary - thin edges are what makes cloud screening genuinely hard.
        alpha = _smoothstep((field_ - thresh) / max(field_.std() * 0.55, 1e-6))
        cloud_mask = alpha > 0.5

        cspec = self._spectrum("cloud")
        ctex = fractal_field(n, beta=2.0, rng=self.rng)
        for b in range(C.N_BANDS):
            cloud_band = cspec[b] * (1.0 + C.SURFACE_TEXTURE_SIGMA["cloud"] * ctex)
            cube[b] = cube[b] * (1 - alpha) + cloud_band * alpha

        # Cloud shadow, offset by the solar geometry.
        dy, dx = self.rng.integers(-10, 11), self.rng.integers(-10, 11)
        shadow = np.roll(np.roll(alpha, dy, axis=0), dx, axis=1)
        shadow = np.clip(shadow - alpha, 0.0, 1.0) * 0.55
        cube *= (1.0 - shadow)[None, :, :]

        # Semi-transparent cirrus veil over the whole tile, sometimes.
        if self.rng.random() < 0.30:
            veil = _smoothstep(fractal_field(n, beta=2.6, rng=self.rng) * 0.5 + 0.5)
            veil *= self.rng.uniform(0.10, 0.40)
            cspec = self._spectrum("cirrus")
            for b in range(C.N_BANDS):
                cube[b] = cube[b] * (1 - veil) + cspec[b] * veil
            cloud_mask |= veil > 0.28
        return cloud_mask

    # -- events -------------------------------------------------------------
    def _add_wildfire(self, cube: np.ndarray, land: str) -> Tuple[np.ndarray, float]:
        """Sub-pixel hot-spot emission + smoke plume.

        A fire front at ~800 K occupying a small fraction of a 20 m pixel
        contributes negligible signal at 490-842 nm but a large one at
        1610 nm and a larger one still at 2190 nm. Modelling that ordering
        (B12 > B11 >> NIR) is what makes the simulated fire detectable by the
        same spectral logic operational algorithms use.
        """
        n = self.size
        severity = float(self.rng.uniform(0.25, 1.0))
        n_fronts = int(self.rng.integers(1, 4))
        hot = np.zeros((n, n), dtype=np.float32)
        for _ in range(n_fronts):
            cy, cx = self.rng.uniform(8, n - 8, size=2)
            sy = self.rng.uniform(1.2, 3.4)
            sx = self.rng.uniform(1.2, 3.4)
            blob = _gauss_blob(n, cy, cx, sy, sx)
            # Fire fronts are filamentary, not circular.
            filament = _smoothstep(fractal_field(n, beta=2.0, rng=self.rng) * 0.7 + 0.5)
            hot = np.maximum(hot, blob * (0.55 + 0.45 * filament))
        hot /= max(hot.max(), 1e-6)

        # Spectral gain of the thermal component, normalised to B12.
        # (Planck ratio at 800 K between 2190 nm and 1610 nm is ~2.3x.)
        gain = np.array([0.0, 0.0, 0.02, 0.06, 0.42, 1.00], dtype=np.float32)
        amplitude = 0.85 * severity
        for b in range(C.N_BANDS):
            cube[b] += amplitude * gain[b] * hot

        event_mask = hot > 0.22

        # Smoke plume drifting downwind: scatters strongly in the blue,
        # nearly transparent in SWIR. This is the confuser that makes naive
        # brightness-based cloud screening throw away the fire.
        if self.rng.random() < 0.8:
            drift_y = self.rng.integers(-16, 17)
            drift_x = self.rng.integers(-16, 17)
            plume = np.zeros((n, n), dtype=np.float32)
            for k in range(1, 7):
                shifted = np.roll(np.roll(hot, drift_y * k // 3, 0), drift_x * k // 3, 1)
                plume = np.maximum(plume, shifted * (1.0 - k / 8.0))
            plume = _smoothstep(plume * 1.4) * self.rng.uniform(0.25, 0.6)
            sspec = self._spectrum("smoke")
            for b in range(C.N_BANDS):
                cube[b] = cube[b] * (1 - plume) + sspec[b] * plume
        return event_mask, severity

    def _add_flood(self, cube: np.ndarray, land: str) -> Tuple[np.ndarray, float]:
        """Turbid standing water over what was dry land."""
        n = self.size
        severity = float(self.rng.uniform(0.3, 1.0))
        terrain = fractal_field(n, beta=2.6, rng=self.rng)
        # Water fills the low ground: threshold the terrain field.
        frac = 0.06 + 0.34 * severity
        thresh = np.quantile(terrain, frac)
        alpha = _smoothstep((thresh - terrain) / max(terrain.std() * 0.35, 1e-6))
        mask = alpha > 0.5

        wspec = self._spectrum("water").copy()
        # Flood water is sediment-laden: brighter in red/green than clear water.
        sediment = self.rng.uniform(0.3, 1.0)
        wspec = wspec + np.array([0.01, 0.03, 0.05, 0.02, 0.005, 0.003]) * sediment
        for b in range(C.N_BANDS):
            cube[b] = cube[b] * (1 - alpha) + wspec[b] * alpha
        return mask, severity

    def _add_vessel(self, cube: np.ndarray, land: str) -> Tuple[np.ndarray, float]:
        """Bright compact targets with a wake, only meaningful over water."""
        n = self.size
        severity = float(self.rng.uniform(0.2, 0.9))
        n_ships = int(self.rng.integers(1, 5))
        mask = np.zeros((n, n), dtype=bool)
        for _ in range(n_ships):
            cy, cx = self.rng.uniform(6, n - 6, size=2)
            length = self.rng.uniform(1.4, 3.2)
            ang = self.rng.uniform(0, np.pi)
            sy = length * abs(np.sin(ang)) + 0.6
            sx = length * abs(np.cos(ang)) + 0.6
            blob = _gauss_blob(n, cy, cx, sy, sx)
            hull = blob > 0.45
            # Metal hull: bright and fairly flat across VNIR, low SWIR.
            hull_spec = np.array([0.28, 0.30, 0.31, 0.26, 0.14, 0.10], np.float32)
            for b in range(C.N_BANDS):
                cube[b] = np.where(hull, hull_spec[b] * (0.7 + 0.6 * severity), cube[b])
            # Turbulent wake: slightly brighter than surrounding water.
            wake_dir = (-np.sin(ang), -np.cos(ang))
            wake = np.zeros((n, n), np.float32)
            for k in range(1, 12):
                yy = int(round(cy + wake_dir[0] * k * 1.6))
                xx = int(round(cx + wake_dir[1] * k * 1.6))
                if 1 <= yy < n - 1 and 1 <= xx < n - 1:
                    wake[yy - 1:yy + 2, xx - 1:xx + 2] = max(0.0, 1.0 - k / 12.0)
            for b in range(C.N_BANDS):
                cube[b] += wake * 0.035 * (1.0 if b < 4 else 0.2)
            mask |= hull
        return mask, severity

    # -- sensor -------------------------------------------------------------
    def _apply_sensor(self, cube: np.ndarray) -> np.ndarray:
        """Shot noise, read noise, then 12-bit quantisation."""
        cube = np.clip(cube, 0.0, 1.35)
        # Photon shot noise: sigma proportional to sqrt(signal).
        full_well_e = 30000.0
        electrons = cube * full_well_e
        electrons = electrons + self.rng.standard_normal(cube.shape) * np.sqrt(
            np.maximum(electrons, 1.0)
        )
        electrons += self.rng.standard_normal(cube.shape) * 22.0   # read noise
        cube = np.clip(electrons / full_well_e, 0.0, 1.35)
        levels = 2 ** C.RAW_BITS_PER_SAMPLE - 1
        cube = np.round(cube / 1.35 * levels) / levels * 1.35
        return cube.astype(np.float32)

    # -- public API ---------------------------------------------------------
    def sample_tile(
        self,
        force_event: Optional[str] = None,
        force_cloud_fraction: Optional[float] = None,
        force_land: Optional[str] = None,
        tile_id: int = -1,
    ) -> Tile:
        land = force_land or str(self.rng.choice(self._land_names, p=self._land_p))

        # Cloud fraction: bimodal in reality - mostly clear or mostly covered.
        if force_cloud_fraction is not None:
            cf_target = float(force_cloud_fraction)
        elif self.rng.random() < C.GLOBAL_CLOUD_PROB:
            cf_target = float(np.clip(self.rng.beta(2.2, 1.3), 0.0, 0.99))
        else:
            cf_target = float(np.clip(self.rng.beta(0.6, 9.0), 0.0, 0.99))

        # Choose event. Vessels only occur over water; fire/flood only on land.
        if force_event is not None:
            event = force_event
        else:
            event = str(self.rng.choice(self._event_names, p=self._event_p))
            if event == "vessel" and land != "water":
                land = "water"
            elif event in ("wildfire", "flood") and land == "water":
                land = "vegetation" if event == "wildfire" else "soil"
        if event == "vessel" and land != "water":
            land = "water"

        cube, _ = self._base_surface(land)

        event_mask = np.zeros((self.size, self.size), dtype=bool)
        severity = 0.0
        if event == "wildfire":
            event_mask, severity = self._add_wildfire(cube, land)
        elif event == "flood":
            event_mask, severity = self._add_flood(cube, land)
        elif event == "vessel":
            event_mask, severity = self._add_vessel(cube, land)

        cloud_mask = self._add_clouds(cube, cf_target)
        cube = self._apply_sensor(cube)

        return Tile(
            cube=cube, cloud_mask=cloud_mask, event_mask=event_mask,
            event_class=event, land_class=land, severity=severity,
            tile_id=tile_id,
        )

    def sample_batch(self, n: int, start_id: int = 0) -> List[Tile]:
        return [self.sample_tile(tile_id=start_id + i) for i in range(n)]

    def sample_balanced_batch(self, n: int, start_id: int = 0) -> List[Tile]:
        """Class-balanced sampling for *training only*.

        Training on the natural 88/4/3/5 prior starves the rare classes. We
        oversample events during training and then evaluate on the natural
        prior, which is the honest way round.
        """
        tiles: List[Tile] = []
        classes = C.EVENT_CLASSES
        # 40% nominal, 20% each event class.
        weights = np.array([0.40, 0.20, 0.20, 0.20])
        counts = (weights * n).astype(int)
        counts[0] += n - counts.sum()
        tid = start_id
        for cls, k in zip(classes, counts):
            for _ in range(int(k)):
                tiles.append(self.sample_tile(force_event=cls, tile_id=tid))
                tid += 1
        self.rng.shuffle(tiles)
        return tiles


# ---------------------------------------------------------------------------
# spectral indices - the cheap physics stage
# ---------------------------------------------------------------------------
def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, 1e-6)


def spectral_indices(cube: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-pixel indices computable in a handful of ops per pixel.

    These are the operational workhorses; the CNN's job is to beat them, and
    the Stage-0 screener's job is to use them to avoid ever waking the CNN.
    """
    b2, b3, b4, b8, b11, b12 = (cube[i] for i in range(6))
    return {
        "ndvi": _nd(b8, b4),                     # vegetation
        "ndwi": _nd(b3, b8),                     # open water
        "ndsi": _nd(b3, b11),                    # snow vs cloud (the key one)
        "nbr": _nd(b8, b12),                     # burn / active fire
        "fire_ratio": b12 / np.maximum(b11, 1e-6),
        "brightness": (b2 + b3 + b4) / 3.0,
        "swir_bright": (b11 + b12) / 2.0,
        "whiteness": 1.0 - (np.std(cube[:3], axis=0) / np.maximum(np.mean(cube[:3], axis=0), 1e-6)),
    }


# Stage 0 works on a decimated copy of the tile. This is not a shortcut, it
# is the point: the screener needs *statistics* of spectral indices, and the
# mean and spread of an index are estimated perfectly well from 256 samples.
# Computing them at full resolution made Stage 0 more expensive than the CNN
# it exists to avoid running - a real design error that the cost model caught.
STAGE0_DECIMATION = 4


def tile_features(cube: np.ndarray, decimate: int = STAGE0_DECIMATION) -> np.ndarray:
    """Compact feature vector for the Stage-0 screener (32 floats).

    Deliberately tiny, and computed on a `decimate`x decimated view so it runs
    on the housekeeping processor for a fraction of the accelerator's cost.
    """
    if decimate > 1:
        cube = cube[:, ::decimate, ::decimate]
    idx = spectral_indices(cube)
    feats: List[float] = []
    for key in ["ndvi", "ndwi", "ndsi", "nbr", "fire_ratio",
                "brightness", "swir_bright", "whiteness"]:
        v = idx[key]
        feats.extend([
            float(np.mean(v)), float(np.std(v)),
            float(np.quantile(v, 0.95)), float(np.quantile(v, 0.05)),
        ])
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# real-data hook
# ---------------------------------------------------------------------------
class RealSceneAdapter:
    """Drop-in replacement contract for real Sentinel-2 L1C data.

    To swap the simulator for flight-representative data, implement:

        sample_tile(...) -> Tile
        sample_batch(n)  -> List[Tile]

    reading 64x64 chips of bands B02,B03,B04,B08,B11,B12 resampled to 20 m,
    scaled to TOA reflectance, with `cloud_mask` from the L1C/L2A cloud
    product (or Fmask/s2cloudless) and `event_mask` from the corresponding
    hazard product (FIRMS active fire, Copernicus EMS flood delineation, AIS
    for vessels). Nothing downstream of this module needs to change - the
    models, quantisation, edge model and autonomy layer all consume `Tile`.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "RealSceneAdapter is an interface stub. Implement sample_tile() "
            "against your Sentinel-2 archive; see the class docstring."
        )


def to_rgb(cube: np.ndarray, gain: float = 2.6) -> np.ndarray:
    """Quick-look true-colour render for previews and the dashboard."""
    rgb = np.stack([cube[2], cube[1], cube[0]], axis=-1)   # B04,B03,B02
    rgb = np.clip(rgb * gain, 0, 1) ** (1 / 1.8)
    return (rgb * 255).astype(np.uint8)


def to_swir_composite(cube: np.ndarray, gain: float = 1.8) -> np.ndarray:
    """B12/B11/B04 false colour - the composite fire shows up in."""
    rgb = np.stack([cube[5], cube[4], cube[2]], axis=-1)
    rgb = np.clip(rgb * gain, 0, 1) ** (1 / 1.6)
    return (rgb * 255).astype(np.uint8)
