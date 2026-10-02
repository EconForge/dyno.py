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
e_z[t] <- N(0.01)
```

Upon opening `rbc.dyno`, Dyno Lab configures a split-pane layout:

- **Left Pane — Code Editor**: Code editor with syntax highlighting for `.dyno` and `.mod` files.
- **Right Pane — Dyno Report**: Interactive report viewer connected to a background Python kernel, displaying steady-state values, eigenvalues, and IRFs.

---

## Step 2: Live Model Exploration

Dyno Lab updates the report automatically as you modify model specifications.

### Experiment 1: Modifying Shock Persistence

1. In the editor, change the persistence parameter `rho` from `0.95` to `0.70`:
   ```text
   rho <- 0.70
   ```
2. After a brief pause, the background solver recomputes the decision rule. The impulse responses on the right update to reflect the faster decay back to steady state.

### Experiment 2: Modifying Capital Depreciation

1. Change `delta` from `0.025` to `0.050`:
   ```text
   delta <- 0.050
   ```
2. The report updates:
   - Steady-state capital $k^*$ falls to reflect higher depreciation.
   - Investment $i^*$ adjusts to maintain the stationary capital stock.
   - The report view maintains its scroll position during re-computation.

---

## Step 3: Using the Dyno Options Sidebar

Solution parameters can be adjusted via the sidebar interface:

1. Click the **Dyno Options** tab in the JupyterLab sidebar (or the **Options** button in the report toolbar).
2. Change **Simulation Type** between `Deviation`, `Log-Deviation`, or `Level`.
3. Adjust the **Horizon** (e.g. from `40` to `80` periods) to examine longer-term dynamics.
4. Enable **Steady state only** when calibrating parameters to inspect steady-state values without running perturbation solvers.

---

## Step 4: Diagnostic Highlighting

Dyno Lab flags syntax and mathematical errors directly in the editor:

1. In the code editor, introduce an undefined variable (e.g. `c[t] = y[t] - i[t] + unknown_var[t]`).
2. The background solver identifies the undefined symbol and highlights the offending line in the editor gutter.
3. Correcting the line clears the diagnostic marker and restores the report.

---

## Next Steps

- For an overview of Dyno Lab features and installation, see the [Dyno Lab Guide](../dyno_lab/index.md).
