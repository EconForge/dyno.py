# Tutorial: Interactive Modeling with Dyno Lab

This tutorial walks through using **Dyno Lab** (`jupyterlab_dyno`), the interactive JupyterLab extension for Dyno, to explore, calibrate, and simulate a Real Business Cycle (RBC) model in real time.

---

## Prerequisites & Setup

Ensure you have `jupyterlab_dyno` installed in your environment:

```bash
pixi add --channel https://repo.prefix.dev/econforge jupyterlab_dyno
```

Launch JupyterLab:

```bash
pixi run -e dev jupyter lab
```

JupyterLab will start and open your browser to the workspace.

---

## Step 1: Open an RBC Model

In the JupyterLab file browser on the left, create or open a model file named `rbc.dyno`:

```text
# 1. Calibration
alpha <- 0.33
beta  <- 0.99
delta <- 0.025
rho   <- 0.95
eta   <- 1.0

# 2. Steady State
k[~] <- ((1/beta - (1-delta)) / alpha)**(1 / (alpha - 1))
y[~] <- k[~]^alpha
i[~] <- delta * k[~]
c[~] <- y[~] - i[~]
z[~] <- 0.0

# 3. Dynamic Equations
z[t] = rho * z[t-1] + e_z[t]
y[t] = exp(z[t]) * k[t-1]^alpha
k[t] = (1-delta)*k[t-1] + i[t]
y[t] = c[t] + i[t]
beta * (c[t+1]/c[t])^(-1) * (alpha * y[t+1]/k[t] + 1 - delta) = 1

# 4. Stochastic Shocks
e_z[t] <- N(0, 0.01^2)
```

Upon opening `rbc.dyno`, Dyno Lab immediately configures a split-pane layout:

- The **Code Editor** remains on the left with full syntax highlighting.
- The **Dyno Report** opens on the right, connected to a background Python kernel.

Within seconds, the background solver evaluates steady states, solves the first-order perturbation decision rule, and generates interactive Plotly impulse response charts.

---

## Step 2: Live Reactive Exploration

Dyno Lab eliminates the need to manually re-run scripts or switch between windows.

### Experiment 1: Altering Technology Shock Persistence

1. In the left code editor, change the persistence parameter `rho` from `0.95` to `0.70`:
   ```text
   rho <- 0.70
   ```
2. Stop typing for half a second.
3. Observe how the impulse response plots on the right instantly re-render! The response of output $y_t$ and consumption $c_t$ now decays much faster back to the steady-state baseline.

### Experiment 2: Modifying Capital Depreciation

1. Change `delta` from `0.025` to `0.050`:
   ```text
   delta <- 0.050
   ```
2. The report updates immediately:
   - Steady-state capital $k^*$ adjusts downwards to reflect higher depreciation costs.
   - Investment $i^*$ increases proportionally to replenish capital.
   - The scroll position of your report view remains fixed right where you were looking.

---

## Step 3: Using the Dyno Options Sidebar

To adjust solution parameters without editing model code:

1. Click the **Dyno Options** tab in the left JupyterLab sidebar (or click the **Options** button in the report toolbar).
2. Switch **Simulation Type** from `Log-Deviation` to `Level`. The IRF charts now display exact physical units rather than percentage deviations.
3. Change **Horizon** from `40` to `80` periods to observe long-term capital accumulation trajectories.
4. Check **Steady state only** if you are calibrating deep structural parameters and want instantaneous algebraic feedback without running stochastic simulations.

---

## Step 4: Interactive Diagnostic Highlighting

Dyno Lab guides you directly to syntax and mathematical mistakes:

1. In the code editor, introduce a deliberate typo in an equation (e.g. `c[t] = y[t] - i[t] + unknown_var[t]`).
2. Notice that the background solver catches the missing symbol and immediately highlights the erroneous line in red inside the code editor.
3. Correct the line back to `c[t] = y[t] - i[t]`. The error highlight clears and the full report reappears instantly.

---

## Next Steps

- For an in-depth reference of all editor controls and options, read the [Dyno Lab Interface Guide](../dyno_lab/interface.md).
- To configure preprocessors and default settings, see [Options & Settings](../dyno_lab/options_and_settings.md).
