# Dyno Lab Options & Settings

Dyno Lab provides granular configuration controls at both the individual file level and across your entire JupyterLab environment.

---

## The Dyno Options Sidebar

Clicking the **Dyno Options** tab in the left JupyterLab sidebar (or clicking **Options** in the report toolbar) reveals a dedicated parameter panel:

```text
┌──────────────────────────────────────────────┐
│ Dyno Options                                 │
├──────────────────────────────────────────────┤
│ Approximation Order:                         │
│ [ 1                                        ] │
│                                              │
│ Simulation Type:                             │
│ ( ) Level                                    │
│ ( ) Deviation                                │
│ (•) Log-Deviation                            │
│                                              │
│ Horizon (Periods):                           │
│ [ 40                                       ] │
│                                              │
│ [ ] Steady state only                        │
│ [x] Preserve scroll position                 │
└──────────────────────────────────────────────┘
```

### File-Specific State Memory

- Options set in this panel are **tracked per open model file**.
- For example, you can analyze an RBC model at a 40-quarter horizon with log-deviation IRFs, while inspecting a New Keynesian model with level trajectories over 100 periods.
- Switching active tabs in JupyterLab automatically updates the sidebar to reflect the selected file's settings.

### Available Per-File Settings

- **Approximation Order**: Numerical perturbation order (currently defaults to `1` for first-order linear approximation).
- **Simulation Type**:
  - `Level`: Raw variable trajectories in model units ($y_t$).
  - `Deviation`: Arithmetic deviations from steady state ($y_t - \bar{y}$).
  - `Log-Deviation`: Proportional percentage deviations ($\frac{y_t - \bar{y}}{\bar{y}}$ or $\log(y_t / \bar{y})$).
- **Horizon**: Number of forward time steps to simulate for impulse response functions (default: `40`).
- **Steady State Only**: When enabled, the background solver evaluates only the deterministic steady-state and skips computing decision rules and IRFs. Useful for fast calibration iteration on large models.
- **Preserve Scroll Position**: Overrides global behavior to determine whether the report preserves scroll coordinates upon re-evaluation.

---

## Global Extension Preferences

Default behavior across all files can be customized in JupyterLab's settings system.

To edit global settings:

1. Open **Settings** from the JupyterLab top menu.
2. Select **Advanced Settings Editor**.
3. Choose **Dyno Lab** from the sidebar list.

### Configuration Properties

```json
{
  "preserve-scroll-position": true,
  "modfile-preprocessor": "lark",
  "kernel-restart": false,
  "output_type": "markdown",
  "display_graph": true,
  "check_output": false
}
```

### Property Reference

| Property | Type | Default | Description |
|---|---|---|---|
| `preserve-scroll-position` | boolean | `true` | Retain report scroll position when models are re-rendered reactively. |
| `modfile-preprocessor` | string | `"lark"` | Parser used for Dynare `.mod` files: `"lark"` (Dyno native) or `"dynare"` (`dynare-preprocessor-pylib`). |
| `kernel-restart` | boolean | `false` | Automatically restart the background Python kernel prior to each solve (useful during development). |
| `output_type` | string | `"markdown"` | Primary format for rendered reports: `"markdown"`, `"html"`, or `"text"`. |
| `display_graph` | boolean | `true` | Enable or disable interactive Plotly charts in reports. |
| `check_output` | boolean | `false` | Diagnostic mode: prints raw kernel execution streams for debugging. |

---

## Modfile Preprocessor Selection

Dyno Lab offers two parsing pipelines for Dynare `.mod` files:

1. **`lark` (Default)**: Uses Dyno's built-in Lark EBNF parser. Provides instant parsing without C++ compilation requirements and translates expressions directly into Dyno AST nodes.
2. **`dynare`**: Uses the official Dynare C++ preprocessor wrapped via `dynare-preprocessor-pylib`. Recommended when importing complex `.mod` files utilizing advanced Dynare macro-processing commands (`@#include`, `@#for`).
