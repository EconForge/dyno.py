# Tutorial: Deterministic Transition Dynamics & Policy Experiments

Many economic research and policy questions involve non-linear transitions: What happens when a government permanently reduces corporate taxes? How does the economy adjust when capital is destroyed following a natural disaster?

Dyno's stacked-time deterministic solver handles these questions with full non-linear fidelity.

---

## The Scenario: Post-Crisis Capital Rebuilding

Suppose an economy at steady state suffers a sudden disaster that destroys 15% of its capital stock at date $t=0$. Households and firms possess perfect foresight regarding the future economic environment and rebuild capital over time.

---

## Model Specification (`transition.dyno`)

```text
# Parameters
alph <- 0.50
gam  <- 0.50
delt <- 0.02
bet  <- 0.051
aa   <- 0.511
T    <- 50

# Steady-state values
x[~] <- 1.0
k[~] <- ((delt + bet) / (1.0 * aa * alph))^(1 / (alph - 1))
c[~] <- aa * k[~]^alph - delt * k[~]

# Dynamic equations
0 = c[t] + k[t] - aa*x[t]*k[t-1]^alph - (1 - delt)*k[t-1]
0 = c[t]^(-gam) - (1 + bet)^(-1) * (aa*alph*x[t+1]*k[t]^(alph-1) + 1 - delt) * c[t+1]^(-gam)

# Constant productivity path
x[1] <- 1.0
forall t, 2 <= t < T : x[t] <- 1.0

# Initial condition override: capital destroyed by 40%
k[0] <- 0.60 * k[~]
```

---

## Solving the Transition Path

```python
from dyno import DynoModel

model = DynoModel(txt="""
alph <- 0.50
gam  <- 0.50
delt <- 0.02
bet  <- 0.051
aa   <- 0.511
T    <- 50

x[~] <- 1.0
k[~] <- ((delt + bet) / (1.0 * aa * alph))^(1 / (alph - 1))
c[~] <- aa * k[~]^alph - delt * k[~]

0 = c[t] + k[t] - aa*x[t]*k[t-1]^alph - (1 - delt)*k[t-1]
0 = c[t]^(-gam) - (1 + bet)^(-1) * (aa*alph*x[t+1]*k[t]^(alph-1) + 1 - delt) * c[t+1]^(-gam)

x[1] <- 1.0
forall t, 2 <= t < T : x[t] <- 1.0

k[0] <- 0.60 * k[~]
""")

# Simulate non-linear stacked system
trajectory = model.simulate()
df = trajectory.to_df()

print("Initial period t=0:")
print(df.loc[0, ["k", "c"]])

print("\nFinal period t=50:")
print(df.loc[50, ["k", "c"]])
```

---

## Visualizing the Transition

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 1. Capital
axes[0].plot(df["t"], df["k"], lw=2, color="crimson")
axes[0].axhline(model.steady_state["k"], color="black", linestyle="--", alpha=0.7)
axes[0].set_title("Capital Stock $k_t$")
axes[0].grid(True, alpha=0.3)

# 2. Consumption
axes[1].plot(df["t"], df["c"], lw=2, color="navy")
axes[1].axhline(model.steady_state["c"], color="black", linestyle="--", alpha=0.7)
axes[1].set_title("Consumption $c_t$")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

### Transition Dynamics:

- **Capital Accumulation**: Capital starts depressed at $0.85 \times \bar{k}$ and monotonically converges back to steady state.
- **Consumption Cut**: Households suppress consumption immediately to finance high investment.
- **High Return on Capital**: The marginal product of capital $\alpha y_{t+1}/k_t$ is elevated, incentivizing strong initial capital accumulation.
