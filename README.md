# Wildfire Spread Model

This repository now includes a grid-based wildfire spread model in
`wildfire_spread_model.py`.

## What it uses

The model estimates spread using:

- **Fuel data** (CFFDRS-style fuel type per cell)
- **Topography**:
  - slope (degrees)
  - aspect (downslope azimuth)
- **Canopy cover** (%)
- Optional mixedwood fuel modifier (e.g., conifer percent for M-1/M-2 cells)
- Wind speed and wind direction

## Quick start

Run the built-in example:

```bash
python wildfire_spread_model.py
```

Or import it:

```python
import numpy as np
from wildfire_spread_model import WildfireInputs, WildfireSpreadModel

fuel = np.array([["C-3", "C-3"], ["C-4", "M-1"]], dtype=object)
slope = np.array([[10, 12], [14, 18]], dtype=float)
aspect = np.array([[270, 270], [270, 270]], dtype=float)
canopy = np.array([[70, 75], [80, 60]], dtype=float)
modifier = np.array([[0, 0], [0, 65]], dtype=float)

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
    wind_direction_deg=90.0,
)

arrival = model.simulate_arrival_time(
    inputs=inputs,
    ignition_cells=[(0, 0)],
    max_duration_minutes=180.0,
)
```

## Tests

```bash
python -m unittest -v
```

## Notes

This is an explainable baseline simulator for planning and experimentation. It
is not a replacement for full operational fire behavior systems.
