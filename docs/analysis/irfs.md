# Impulse Response Functions (IRFs)

Impulse Response Functions (IRFs) measure the dynamic response of endogenous variables over time following a one-standard-deviation innovation to an exogenous shock.

---

## Computing IRFs

Once a model is solved via perturbation, call `solution.irfs()`:

```python
from dyno import DynoModel

model = DynoModel("examples/neo.dyno")
solution = model.solve()

# Compute IRFs over 40 periods
irfs = solution.irfs(type="log-deviation", T=40)
```

The returned `irfs` object is a dictionary mapping each shock name to a `pandas.DataFrame`:

```python
for shock_name, df_irf in irfs.items():
    print(f"=== Shock: {shock_name} ===")
    print(df_irf.head())
```

---

## IRF Types

Dyno supports three scaling transformations via the `type` parameter:

| `type` | Mathematical Definition | Interpretation | Typical Use Case |
|---|---|---|---|
| `"log-deviation"` *(default)* | $\frac{y_t - \bar{y}}{\bar{y}}$ | Percentage deviation from steady state ($100 \times \Delta\%$) | Growth rates, macro aggregates |
| `"deviation"` | $y_t - \bar{y}$ | Level difference from steady state | Rates, ratios (inflation, interest rates) |
| `"level"` | $y_t = \bar{y} + \hat{y}_t$ | Actual level values in original units | Forecasting, levels analysis |

### Examples

=== "Log-Deviation (% dev)"
    ```python
    irfs_pct = solution.irfs(type="log-deviation", T=40)
    ```

=== "Level Deviation"
    ```python
    irfs_diff = solution.irfs(type="deviation", T=40)
    ```

=== "Raw Levels"
    ```python
    irfs_levels = solution.irfs(type="level", T=40)
    ```

---

## Interactive Visualization with Plotly

Dyno integrates seamlessly with Plotly to generate interactive charts:

```python
fig = solution.plot(type="log-deviation")

# Display in Jupyter notebook or browser
fig.show()

# Customize layout
fig.update_layout(
    title="Impulse Response Analysis",
    template="plotly_white",
    height=600,
)
fig.show()
```

### Exporting Static Figures

Export publication-ready vector images (requires `kaleido`):

```python
fig.write_image("irf_plots.pdf")
fig.write_image("irf_plots.png", scale=3)
```

---

## Plotting with Matplotlib & Pandas

You can plot specific variables directly using Pandas and Matplotlib:

```python
import matplotlib.pyplot as plt

df = irfs["e_z"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
variables = ["y", "c", "i", "k"]

for ax, var in zip(axes.flatten(), variables):
    ax.plot(df.index, df[var], lw=2, color="navy")
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.set_title(f"Response of {var}")
    ax.set_xlabel("Periods")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
