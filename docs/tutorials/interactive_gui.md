# Tutorial: Interactive Web GUI Dashboard

Dyno includes a modern web-based graphical user interface (GUI) powered by [Solara](https://solara.dev) and [ipyvuetify]. The GUI enables researchers, students, and analysts to explore models interactively without writing code.

---

## Launching the Web Dashboard

To start the interactive interface, use the predefined Pixi task:

```bash
pixi run -e dev solara
```

This starts a local development server (typically at `http://localhost:8765`) and opens the interactive dashboard in your browser.

---

## Features of the Interactive GUI

### 1. Real-Time Parameter Sliders
- Adjust key parameters such as capital share $\alpha$, subjective discount factor $\beta$, depreciation $\delta$, and shock persistence $\rho$ using interactive sliders.
- The steady-state solver runs reactively upon every parameter change, updating equilibrium values instantly.

### 2. Live Impulse Response Plots
- Observe how shifts in calibration alter impulse response functions in real time.
- Compare multiple calibration scenarios visually.

### 3. Model Introspection Tabs
- **Equations View**: Formatted mathematical rendering of model equations.
- **Steady-State Table**: Real-time display of all endogenous variables, residuals, and parameter calibrations.
- **Jacobian Matrices**: Inspect transition matrix $X$ and shock impact matrix $Y$.

---

## Running Custom Models in the GUI

You can launch any `.dyno` or `.mod` model in the GUI by pointing Solara to your script:

```python
# app.py
from dyno.gui.components import ModelDashboard
from dyno import DynoModel
import solara

model = DynoModel("my_model.dyno")

@solara.component
def Page():
    ModelDashboard(model)
```

Run it via:

```bash
pixi run -e dev solara run app.py
```
