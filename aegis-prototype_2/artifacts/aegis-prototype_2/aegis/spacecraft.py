"""
Spacecraft model: orbit, ground-station access, power, storage, wear.
====================================================================

Deliberately physical rather than hand-waved, because the entire benefit
case for onboard AI is a *resource* argument, and a resource argument is
only credible if the resources are modelled. Specifically:

  * a real sun-synchronous orbit with J2 nodal precession, a rotating Earth
    and geometric elevation-angle access to real ground-station locations -
    so downlink opportunities are scarce and unevenly spaced the way they
    actually are, not a smooth average
  * a cylindrical-shadow eclipse model driving a battery with a floor, so
    "just downlink more" runs into energy, not just bits
  * life-limited-item accounting (transmitter on-time, battery cycles),
    which is the benefit everyone forgets to quantify
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

MU_EARTH = 3.986004418e14        # m^3/s^2
R_EARTH = 6_378_137.0            # m
J2 = 1.08262668e-3
OMEGA_EARTH = 7.2921159e-5       # rad/s
SOLAR_DAY = 86400.0


# ---------------------------------------------------------------------------
# orbit
# ---------------------------------------------------------------------------
def raan_from_ltan(ltan_hours: float, day_of_year: float = 172.0) -> float:
    """RAAN (deg) that puts the ascending node at the requested local time.

    Without this the orbit plane ends up wherever the seed put it, and an SSO
    can land in permanent full sun - which would quietly delete the eclipse
    from the power budget and make every energy claim optimistic.
    """
    lam = math.radians(280.46 + 0.9856474 * day_of_year)
    eps = math.radians(23.439)
    ra_sun = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam)))
    return (ra_sun + (ltan_hours - 12.0) * 15.0) % 360.0


class Orbit:
    """Circular orbit with J2 secular RAAN drift."""

    def __init__(self, altitude_km: float, inclination_deg: float,
                 raan_deg: Optional[float] = None, ltan_hours: float = 10.5,
                 day_of_year: float = 172.0):
        if raan_deg is None:
            raan_deg = raan_from_ltan(ltan_hours, day_of_year)
        self.a = R_EARTH + altitude_km * 1000.0
        self.i = math.radians(inclination_deg)
        self.raan0 = math.radians(raan_deg)
        self.ltan_hours = ltan_hours
        self.n = math.sqrt(MU_EARTH / self.a ** 3)          # rad/s
        self.period_s = 2 * math.pi / self.n
        # Secular nodal regression from J2. For a 500 km SSO this comes out
        # at ~+0.986 deg/day, matching the Earth's mean motion about the Sun,
        # which is the definition of sun-synchronous - a useful self-check.
        self.raan_dot = (-1.5 * J2 * (R_EARTH / self.a) ** 2
                         * self.n * math.cos(self.i))

    def raan(self, t: float) -> float:
        return self.raan0 + self.raan_dot * t

    def position_eci(self, t: np.ndarray) -> np.ndarray:
        """(N,3) ECI position in metres."""
        u = self.n * t
        om = self.raan(t)
        cu, su = np.cos(u), np.sin(u)
        co, so = np.cos(om), np.sin(om)
        ci, si = math.cos(self.i), math.sin(self.i)
        x = self.a * (cu * co - su * ci * so)
        y = self.a * (cu * so + su * ci * co)
        z = self.a * (su * si)
        return np.stack([x, y, z], axis=-1)

    def position_ecef(self, t: np.ndarray) -> np.ndarray:
        r = self.position_eci(t)
        th = OMEGA_EARTH * t
        c, s = np.cos(th), np.sin(th)
        x = c * r[..., 0] + s * r[..., 1]
        y = -s * r[..., 0] + c * r[..., 1]
        return np.stack([x, y, r[..., 2]], axis=-1)

    def subsatellite_latlon(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.position_ecef(t)
        norm = np.linalg.norm(r, axis=-1)
        lat = np.degrees(np.arcsin(r[..., 2] / norm))
        lon = np.degrees(np.arctan2(r[..., 1], r[..., 0]))
        return lat, lon


def station_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return R_EARTH * np.array([math.cos(la) * math.cos(lo),
                               math.cos(la) * math.sin(lo),
                               math.sin(la)])


def elevation_deg(sat_ecef: np.ndarray, stn: np.ndarray) -> np.ndarray:
    """Elevation of the satellite above the station's local horizon."""
    rel = sat_ecef - stn
    rng = np.linalg.norm(rel, axis=-1)
    zenith = stn / np.linalg.norm(stn)
    sin_el = np.einsum("...j,j->...", rel, zenith) / np.maximum(rng, 1.0)
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def sun_vector_eci(t: np.ndarray, day_of_year: float = 172.0) -> np.ndarray:
    """Unit sun vector. Solstice default (day 172) is the worst case for
    eclipse fraction asymmetry in an SSO, so the power budget is not being
    quietly evaluated at its easiest point in the year."""
    doy = day_of_year + t / SOLAR_DAY
    lam = np.radians(280.46 + 0.9856474 * doy)          # ecliptic longitude
    eps = math.radians(23.439)
    return np.stack([np.cos(lam),
                     np.sin(lam) * math.cos(eps),
                     np.sin(lam) * math.sin(eps)], axis=-1)


def in_eclipse(sat_eci: np.ndarray, sun_hat: np.ndarray) -> np.ndarray:
    """Cylindrical shadow: behind the Earth and within one Earth radius of
    the anti-sun axis."""
    proj = np.einsum("...j,...j->...", sat_eci, sun_hat)
    perp = np.linalg.norm(sat_eci - proj[..., None] * sun_hat, axis=-1)
    return (proj < 0) & (perp < R_EARTH)


# ---------------------------------------------------------------------------
# access windows
# ---------------------------------------------------------------------------
@dataclass
class Pass:
    station: str
    t_start: float
    t_end: float
    max_elevation: float

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


def compute_passes(orbit: Orbit, stations, duration_s: float, dt: float,
                   min_el: float) -> List[Pass]:
    t = np.arange(0.0, duration_s, dt)
    sat = orbit.position_ecef(t)
    passes: List[Pass] = []
    for name, lat, lon in stations:
        el = elevation_deg(sat, station_ecef(lat, lon))
        vis = el >= min_el
        if not vis.any():
            continue
        edges = np.diff(vis.astype(np.int8))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if vis[0]:
            starts.insert(0, 0)
        if vis[-1]:
            ends.append(len(t) - 1)
        for s, e in zip(starts, ends):
            if e > s:
                passes.append(Pass(name, float(t[s]), float(t[e]),
                                   float(el[s:e].max())))
    passes.sort(key=lambda p: p.t_start)
    return passes


# ---------------------------------------------------------------------------
# resource state
# ---------------------------------------------------------------------------
@dataclass
class ResourceState:
    """Everything the autonomy layer is allowed to spend."""
    battery_wh: float
    battery_capacity_wh: float
    memory_used_bits: float = 0.0
    memory_capacity_bits: float = 0.0

    # cumulative tallies
    energy_comms_wh: float = 0.0
    energy_compute_wh: float = 0.0
    energy_payload_wh: float = 0.0
    energy_bus_wh: float = 0.0
    tx_on_time_s: float = 0.0
    bits_downlinked: float = 0.0
    battery_cycles: float = 0.0
    tiles_dropped_memory_full: int = 0
    brownouts: int = 0

    @property
    def soc(self) -> float:
        return self.battery_wh / self.battery_capacity_wh

    @property
    def memory_fraction(self) -> float:
        return self.memory_used_bits / max(self.memory_capacity_bits, 1.0)


class PowerSystem:
    def __init__(self, profile: C.MissionProfile):
        self.p = profile

    def step(self, state: ResourceState, dt: float, sunlit: bool,
             loads_w: Dict[str, float]) -> None:
        gen = self.p.solar_array_w if sunlit else 0.0
        draw = sum(loads_w.values())
        net_wh = (gen - draw) * dt / 3600.0
        before = state.battery_wh
        state.battery_wh = float(np.clip(state.battery_wh + net_wh,
                                         0.0, self.p.battery_wh))
        # Depth-of-discharge accounting for battery life.
        if state.battery_wh < before:
            state.battery_cycles += (before - state.battery_wh) / self.p.battery_wh
        if state.battery_wh <= 1e-6 and draw > gen:
            state.brownouts += 1
        for k, v in loads_w.items():
            wh = v * dt / 3600.0
            if k == "comms":
                state.energy_comms_wh += wh
            elif k == "compute":
                state.energy_compute_wh += wh
            elif k == "payload":
                state.energy_payload_wh += wh
            else:
                state.energy_bus_wh += wh

    def can_afford(self, state: ResourceState, watts: float, seconds: float,
                   sunlit: bool) -> bool:
        gen = self.p.solar_array_w if sunlit else 0.0
        net_wh = (watts - gen) * seconds / 3600.0
        floor = self.p.battery_min_soc * self.p.battery_wh
        return (state.battery_wh - net_wh) >= floor


# ---------------------------------------------------------------------------
@dataclass
class LifeLimits:
    """Life-limited items. The headline claim 'AI extends mission life' is
    only meaningful if you say which item and by how much."""
    tx_rated_hours: float
    battery_rated_cycles: float

    def tx_consumed_fraction(self, on_time_s: float) -> float:
        return on_time_s / 3600.0 / self.tx_rated_hours

    def battery_consumed_fraction(self, cycles: float) -> float:
        return cycles / self.battery_rated_cycles

    def projected_life_years(self, on_time_s: float, cycles: float,
                             elapsed_s: float) -> Dict[str, float]:
        """Linear extrapolation of each consumable to its rated limit."""
        years = elapsed_s / (365.25 * SOLAR_DAY)
        if years <= 0:
            return {"tx_years": float("inf"), "battery_years": float("inf")}
        tx_rate = self.tx_consumed_fraction(on_time_s) / years
        bat_rate = self.battery_consumed_fraction(cycles) / years
        return {
            "tx_years": (1.0 / tx_rate) if tx_rate > 0 else float("inf"),
            "battery_years": (1.0 / bat_rate) if bat_rate > 0 else float("inf"),
        }


def acquisition_rate(orbit: Orbit, profile: C.MissionProfile) -> Dict[str, float]:
    """How many 64x64 tiles the instrument actually produces per day.

    Ground-track speed for a circular orbit is v_sat scaled by R_E/a; the
    imager sweeps `swath_km` across it whenever it is on the daylit side.
    """
    v_orbit = math.sqrt(MU_EARTH / orbit.a)
    v_ground = v_orbit * R_EARTH / orbit.a                    # m/s
    imaging_s_per_day = SOLAR_DAY * profile.daylight_duty
    area_km2 = (v_ground / 1000.0) * imaging_s_per_day * profile.swath_km
    tile_area_km2 = C.TILE_GROUND_KM ** 2
    tiles_per_day = area_km2 / tile_area_km2
    return {
        "ground_speed_kms": v_ground / 1000.0,
        "area_km2_per_day": area_km2,
        "tiles_per_day": tiles_per_day,
        "raw_gb_per_day": tiles_per_day * C.RAW_TILE_BITS / 8 / 1e9,
        "compressed_gb_per_day": (tiles_per_day * C.RAW_TILE_BITS
                                  / C.LOSSLESS_COMPRESSION_RATIO / 8 / 1e9),
    }


def downlink_capacity(passes: List[Pass], profile: C.MissionProfile,
                      duration_s: float) -> Dict[str, float]:
    """Usable X-band capacity, after elevation masking and station sharing."""
    usable = [p for p in passes if p.max_elevation >= profile.xband_min_elevation_deg]
    contact_s = sum(p.duration_s for p in usable) * profile.pass_utilisation
    bits = contact_s * profile.xband_rate_mbps * 1e6 * profile.xband_link_efficiency
    days = duration_s / SOLAR_DAY
    return {
        "usable_passes": len(usable),
        "usable_passes_per_day": len(usable) / days,
        "effective_contact_s_per_day": contact_s / days,
        "capacity_gb_per_day": bits / 8 / 1e9 / days,
    }


def orbit_summary(orbit: Orbit, passes: List[Pass], duration_s: float) -> Dict:
    days = duration_s / SOLAR_DAY
    total = sum(p.duration_s for p in passes)
    by_station: Dict[str, int] = {}
    for p in passes:
        by_station[p.station] = by_station.get(p.station, 0) + 1
    return {
        "period_min": orbit.period_s / 60.0,
        "orbits_per_day": SOLAR_DAY / orbit.period_s,
        "raan_drift_deg_per_day": math.degrees(orbit.raan_dot) * SOLAR_DAY,
        "n_passes": len(passes),
        "passes_per_day": len(passes) / days,
        "mean_pass_s": total / max(len(passes), 1),
        "total_contact_min_per_day": total / 60.0 / days,
        "contact_duty_cycle": total / duration_s,
        "passes_by_station": by_station,
    }
