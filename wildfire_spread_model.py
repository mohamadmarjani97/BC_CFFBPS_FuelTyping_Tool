"""Wildfire spread simulation using fuel, topography, and canopy data.

This module provides a lightweight, explainable spread model that can be used
for scenario testing. It is not intended to replace operational fire behavior
systems, but gives a practical baseline for spatial spread prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import cos, exp, radians, sqrt
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# Approximate baseline ROS values (m/min) by CFFDRS-style fuel type.
BASE_RATE_OF_SPREAD = {
    "C-1": 14.0,
    "C-2": 10.0,
    "C-3": 8.0,
    "C-4": 12.0,
    "C-5": 7.0,
    "C-6": 6.0,
    "C-7": 9.0,
    "D-1": 3.0,
    "D-2": 2.0,
    "M-1": 7.0,
    "M-2": 8.0,
    "M-3": 4.0,
    "M-4": 5.0,
    "O-1A": 9.0,
    "O-1B": 11.0,
    "S-1": 1.5,
    "S-2": 1.0,
    "S-3": 0.8,
    "N": 0.0,
}


NEIGHBOR_DIRECTIONS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 0.0),
    (-1, 1, 45.0),
    (0, 1, 90.0),
    (1, 1, 135.0),
    (1, 0, 180.0),
    (1, -1, 225.0),
    (0, -1, 270.0),
    (-1, -1, 315.0),
)


def _normalize_fuel_type(value: object) -> str:
    if value is None:
        return "N"
    text = str(value).strip().upper()
    if text in {"", "NONE", "NAN"}:
        return "N"
    return text


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed angle difference in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


@dataclass(frozen=True)
class WildfireInputs:
    """Inputs used by the spread simulator.

    All grids must have the same shape.
    - fuel_type: fuel code per cell (e.g., C-2, M-1, O-1a, N)
    - slope_degrees: terrain slope in degrees
    - aspect_degrees: downslope azimuth (0=N, 90=E, 180=S, 270=W)
    - canopy_cover: canopy cover percentage in [0, 100]
    - fuel_modifier_pct: optional modifier percentage (e.g., mixedwood conifer%)
    """

    fuel_type: np.ndarray
    slope_degrees: np.ndarray
    aspect_degrees: np.ndarray
    canopy_cover: np.ndarray
    fuel_modifier_pct: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        shape = self.fuel_type.shape
        for name, arr in (
            ("slope_degrees", self.slope_degrees),
            ("aspect_degrees", self.aspect_degrees),
            ("canopy_cover", self.canopy_cover),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} does not match fuel grid shape {shape}")
        if self.fuel_modifier_pct is not None and self.fuel_modifier_pct.shape != shape:
            raise ValueError(
                f"fuel_modifier_pct shape {self.fuel_modifier_pct.shape} "
                f"does not match fuel grid shape {shape}"
            )


class WildfireSpreadModel:
    """Grid-based wildfire spread model.

    The model computes directional rates of spread (ROS) from:
      1) baseline fuel ROS,
      2) terrain effects (slope and aspect),
      3) wind alignment,
      4) canopy cover.
    """

    def __init__(
        self,
        cell_size_m: float = 30.0,
        wind_speed_kmh: float = 10.0,
        wind_direction_deg: float = 90.0,
        slope_factor_per_degree: float = 0.045,
        wind_speed_factor: float = 0.035,
        headwind_factor: float = 0.018,
    ) -> None:
        self.cell_size_m = float(cell_size_m)
        self.wind_speed_kmh = float(wind_speed_kmh)
        self.wind_direction_deg = float(wind_direction_deg) % 360.0
        self.slope_factor_per_degree = float(slope_factor_per_degree)
        self.wind_speed_factor = float(wind_speed_factor)
        self.headwind_factor = float(headwind_factor)

    def baseline_rate_of_spread(self, fuel_type: object) -> float:
        key = _normalize_fuel_type(fuel_type)
        if key in BASE_RATE_OF_SPREAD:
            return BASE_RATE_OF_SPREAD[key]
        # Accept lower-case variants of O-1 fuel types.
        key = key.replace("A", "a").replace("B", "b")
        key = key.upper()
        return BASE_RATE_OF_SPREAD.get(key, BASE_RATE_OF_SPREAD["N"])

    def _fuel_modifier_multiplier(self, fuel_type: str, modifier_pct: Optional[float]) -> float:
        if modifier_pct is None:
            return 1.0
        if fuel_type.startswith("M-"):
            p = min(max(float(modifier_pct), 0.0), 100.0) / 100.0
            # Mixedwood spread rises as conifer fraction increases.
            return 0.4 + 0.6 * p
        return 1.0

    def directional_rate_of_spread(
        self,
        fuel_type: object,
        slope_deg: float,
        aspect_deg: float,
        canopy_cover_pct: float,
        direction_deg: float,
        fuel_modifier_pct: Optional[float] = None,
    ) -> float:
        base_ros = self.baseline_rate_of_spread(fuel_type)
        if base_ros <= 0.0:
            return 0.0

        fuel_code = _normalize_fuel_type(fuel_type)
        fuel_mult = self._fuel_modifier_multiplier(fuel_code, fuel_modifier_pct)

        # Aspect is downslope direction; uphill is opposite.
        uphill_deg = (float(aspect_deg) + 180.0) % 360.0
        slope_alignment = cos(radians(_angle_diff_deg(direction_deg, uphill_deg)))
        effective_slope = float(slope_deg) * slope_alignment
        slope_factor = exp(self.slope_factor_per_degree * effective_slope)
        slope_factor = min(max(slope_factor, 0.25), 3.5)

        wind_alignment = cos(radians(_angle_diff_deg(direction_deg, self.wind_direction_deg)))
        if wind_alignment >= 0.0:
            wind_factor = 1.0 + self.wind_speed_factor * self.wind_speed_kmh * wind_alignment
        else:
            wind_factor = 1.0 + self.headwind_factor * self.wind_speed_kmh * wind_alignment
        wind_factor = min(max(wind_factor, 0.20), 3.5)

        canopy = min(max(float(canopy_cover_pct), 0.0), 100.0)
        canopy_factor = 0.8 + 0.7 * (canopy / 100.0)

        return base_ros * fuel_mult * slope_factor * wind_factor * canopy_factor

    def simulate_arrival_time(
        self,
        inputs: WildfireInputs,
        ignition_cells: Sequence[Tuple[int, int]],
        max_duration_minutes: float = 24.0 * 60.0,
    ) -> np.ndarray:
        """Return a grid of ignition arrival times in minutes."""
        rows, cols = inputs.fuel_type.shape
        arrival = np.full((rows, cols), np.inf, dtype=float)
        queue: list[Tuple[float, int, int]] = []

        for r, c in ignition_cells:
            if r < 0 or r >= rows or c < 0 or c >= cols:
                raise ValueError(f"Ignition cell {(r, c)} is outside grid bounds {rows}x{cols}")
            arrival[r, c] = 0.0
            heappush(queue, (0.0, r, c))

        while queue:
            current_time, r, c = heappop(queue)
            if current_time > arrival[r, c]:
                continue
            if current_time > max_duration_minutes:
                continue

            for dr, dc, direction_deg in NEIGHBOR_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                ros = self.directional_rate_of_spread(
                    fuel_type=inputs.fuel_type[r, c],
                    slope_deg=float(inputs.slope_degrees[r, c]),
                    aspect_deg=float(inputs.aspect_degrees[r, c]),
                    canopy_cover_pct=float(inputs.canopy_cover[r, c]),
                    direction_deg=direction_deg,
                    fuel_modifier_pct=(
                        None
                        if inputs.fuel_modifier_pct is None
                        else float(inputs.fuel_modifier_pct[r, c])
                    ),
                )
                if ros <= 0.0:
                    continue

                distance_m = self.cell_size_m * (sqrt(2.0) if dr != 0 and dc != 0 else 1.0)
                travel_minutes = distance_m / ros
                new_time = current_time + travel_minutes
                if new_time < arrival[nr, nc] and new_time <= max_duration_minutes:
                    arrival[nr, nc] = new_time
                    heappush(queue, (new_time, nr, nc))

        return arrival

    def simulate_burn_mask(
        self,
        inputs: WildfireInputs,
        ignition_cells: Sequence[Tuple[int, int]],
        duration_minutes: float,
    ) -> np.ndarray:
        arrival = self.simulate_arrival_time(
            inputs=inputs,
            ignition_cells=ignition_cells,
            max_duration_minutes=duration_minutes,
        )
        return np.isfinite(arrival) & (arrival <= duration_minutes)


def _example() -> None:
    fuel = np.array(
        [
            ["C-3", "C-3", "C-4", "M-1", "N"],
            ["C-3", "C-4", "M-1", "M-1", "S-2"],
            ["C-2", "C-2", "C-3", "O-1b", "S-2"],
            ["D-1", "C-2", "C-3", "O-1b", "S-3"],
            ["N", "D-1", "C-2", "S-1", "S-3"],
        ],
        dtype=object,
    )
    slope = np.array(
        [
            [10, 12, 16, 20, 5],
            [8, 11, 14, 18, 5],
            [6, 8, 10, 14, 4],
            [5, 6, 8, 12, 3],
            [2, 4, 6, 8, 2],
        ],
        dtype=float,
    )
    aspect = np.full_like(slope, 270.0)  # Downslope west; uphill east.
    canopy = np.array(
        [
            [70, 75, 80, 65, 0],
            [68, 72, 70, 60, 15],
            [62, 65, 70, 35, 15],
            [25, 45, 65, 30, 10],
            [0, 20, 40, 25, 10],
        ],
        dtype=float,
    )
    modifier = np.array(
        [
            [0, 0, 0, 65, 0],
            [0, 0, 70, 70, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    inputs = WildfireInputs(
        fuel_type=fuel,
        slope_degrees=slope,
        aspect_degrees=aspect,
        canopy_cover=canopy,
        fuel_modifier_pct=modifier,
    )
    model = WildfireSpreadModel(
        cell_size_m=30.0,
        wind_speed_kmh=18.0,
        wind_direction_deg=90.0,  # Wind blowing toward east.
    )
    arrival = model.simulate_arrival_time(
        inputs=inputs,
        ignition_cells=[(2, 1)],
        max_duration_minutes=180.0,
    )

    np.set_printoptions(precision=1, suppress=True)
    print("Arrival time (minutes):")
    print(arrival)
    print("\nBurned within 3 hours:")
    print(np.isfinite(arrival))


if __name__ == "__main__":
    _example()
