# The Dyno Lab Interface & Reactive Workspace

Dyno Lab provides a synchronized, side-by-side workspace designed for fluid exploratory modeling and rapid feedback.

---

## Workspace Layout

When you open any supported model file (`.dyno`, `.dyno.yaml`, or `.mod`), Dyno Lab splits the main area into coordinated panes:

```text
┌─────────────────────────────────┬──────────────────────────────────┐
│ Left Pane: Code Editor          │ Right Pane: Dyno Report          │
├─────────────────────────────────┼──────────────────────────────────┤
│                                 │ [Re-run] [Clear] [Options]       │
│  # Parameters                   ├──────────────────────────────────┤
│  alpha <- 0.33                  │ Model Summary & Steady State     │
│  beta  <- 0.99                  │                                  │
│  ...                            │ Eigenvalues & Stability Check    │
│                                 │                                  │
│  # Equations                    │ Interactive IRF Plots (Plotly)   │
│  y[t] = a[t]*k[t-1]^alpha       │                                  │
│  ...                            │                                  │
└─────────────────────────────────┴──────────────────────────────────┘
```

- **Left Pane — Code Editor**: Built on CodeMirror with custom syntax highlighting tailored to Dyno expressions and Dynare grammar.
- **Right Pane — Dyno Report Viewer**: A rich interactive viewer powered by a background Python kernel (`ipykernel` or `xpython`) executing `dyno.report.dsge_report()`.
- **Multi-Document Grouping**: Opening several model files organizes all code editors on the left and docks their respective report viewers on the right. Switching between model tabs automatically synchronizes the active report view.

---

## Interactive Capabilities

### 1. Live Reactive Re-Rendering

- As you edit parameters, equations, or shock distributions in the editor, Dyno Lab monitors modifications.
- After a brief typing pause (debounced), the model is sent to the background kernel, solved, and the report updates reactively.
- No terminal commands, external scripts, or manual recompilations are needed.

### 2. Scroll Position Preservation

- When a model re-solves, you often want to watch the effect of a parameter adjustment on a specific IRF plot or steady-state variable.
- Dyno Lab tracks your scroll position in the report panel and restores it across re-renders, preventing jarring jumps to the top of the document.

### 3. Inline Editor Diagnostics & Error Highlighting

When a model fails verification or contains syntax errors:

- Dyno Lab sends diagnostic positions via the custom MIME channel `application/vnd.jupyterlab-dyno.highlighting+json`.
- Problematic lines (e.g., misspelled variable names, non-zero steady-state residuals, or syntax mistakes) are highlighted with error markers directly in the left editor pane.
- The right panel displays the formatted error traceback and diagnostic hints.

---

## Report Viewer Toolbar

At the top of every Dyno Report view, a convenient action bar allows direct control:

| Button | Action |
|---|---|
| **Re-run** | Forces an immediate re-evaluation and fresh render of the model. |
| **Clear** | Clears the current report output from the panel. |
| **Options** | Opens or focuses the Dyno Options sidebar panel for per-file settings. |

---

## What the Dyno Report Displays

A fully solved Dyno model presents four comprehensive analytical sections:

### 1. Model Summary & Symbols

- High-level overview of symbols: endogenous variables, exogenous shocks, and calibrated constants.
- Clear breakdown of declared equations.

### 2. Steady-State Equilibrium

- Displays steady-state values for all endogenous and exogenous variables.
- Lists equation-level residuals to verify that $|LHS - RHS| < 10^{-6}$.

### 3. Dynamic Stability & Blanchard-Kahn Diagnostics

- Generalized eigenvalues from the matrix pencil $(A, B, C)$.
- Counts eigenvalues outside the unit circle ($|\lambda_i| > 1$) relative to the number of forward-looking variables.
- Displays whether Blanchard-Kahn rank and order conditions are satisfied for existence and uniqueness.

### 4. Interactive Impulse Response Functions (IRFs)

- High-resolution interactive charts powered by **Plotly**.
- Zoom into specific horizons, hover over time points to read exact percentage values, toggle individual variables on and off in the legend, and download publication-quality SVG/PNG plots.
