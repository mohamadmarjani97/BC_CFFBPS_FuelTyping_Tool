import unittest

import numpy as np

from wildfire_spread_model import WildfireInputs, WildfireSpreadModel


class WildfireSpreadModelTests(unittest.TestCase):
    def test_uphill_spread_is_faster_than_downhill(self) -> None:
        # Upslope is east (aspect=270 means downslope west).
        fuel = np.array([["C-3", "C-3", "C-3"]], dtype=object)
        slope = np.array([[25.0, 25.0, 25.0]])
        aspect = np.array([[270.0, 270.0, 270.0]])
        canopy = np.array([[60.0, 60.0, 60.0]])
        inputs = WildfireInputs(
            fuel_type=fuel,
            slope_degrees=slope,
            aspect_degrees=aspect,
            canopy_cover=canopy,
        )
        model = WildfireSpreadModel(cell_size_m=30.0, wind_speed_kmh=0.0)
        arrival = model.simulate_arrival_time(inputs, ignition_cells=[(0, 1)], max_duration_minutes=120.0)

        self.assertLess(arrival[0, 2], arrival[0, 0])

    def test_higher_canopy_increases_directional_ros(self) -> None:
        model = WildfireSpreadModel(cell_size_m=30.0, wind_speed_kmh=0.0)

        low_canopy = model.directional_rate_of_spread(
            fuel_type="C-4",
            slope_deg=0.0,
            aspect_deg=0.0,
            canopy_cover_pct=10.0,
            direction_deg=90.0,
        )
        high_canopy = model.directional_rate_of_spread(
            fuel_type="C-4",
            slope_deg=0.0,
            aspect_deg=0.0,
            canopy_cover_pct=90.0,
            direction_deg=90.0,
        )
        self.assertGreater(high_canopy, low_canopy)

    def test_non_burnable_barrier_blocks_spread(self) -> None:
        fuel = np.array([["C-3", "N", "C-3"]], dtype=object)
        slope = np.zeros((1, 3), dtype=float)
        aspect = np.zeros((1, 3), dtype=float)
        canopy = np.zeros((1, 3), dtype=float)
        inputs = WildfireInputs(
            fuel_type=fuel,
            slope_degrees=slope,
            aspect_degrees=aspect,
            canopy_cover=canopy,
        )
        model = WildfireSpreadModel(cell_size_m=30.0, wind_speed_kmh=0.0)
        arrival = model.simulate_arrival_time(inputs, ignition_cells=[(0, 0)], max_duration_minutes=300.0)

        self.assertTrue(np.isinf(arrival[0, 2]))


if __name__ == "__main__":
    unittest.main()
