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
alpha <- 0.33
beta  <- 0.985
delta <- 0.025
gamma <- 2.0
rho   <- 0.95

# Time horizon
T <- 80

# Steady State
r_ss <- 1/beta - 1 + delta
k[~] <- (alpha / r_ss)^(1/(1-alpha))
y[~] <- k[~]^alpha
i[~] <- delta * k[~]
c[~] <- y[~] - i[~]
a[~] <- 0.0

# Dynamic Equations
y[t] = exp(a[t]) * k[t-1]^alpha
k[t] = (1-delta)*k[t-1] + i[t]
c[t] + i[t] = y[t]
beta * (c[t+1]/c[t])^(-gamma) * (alpha*y[t+1]/k[t] + 1 - delta) = 1
a[t] = rho*a[t-1] + e[t]

# 15% Destruction of Initial Capital
k[0] <- k[~] * 0.85

# Exogenous Shocks remain zero
e[0] <- 0.0
forall t, 1 <= t < 80 : e[t] <- 0.0
```

---

## Solving the Transition Path

```python
from dyno import DynoModel

model = DynoModel("transition.dyno")

# Solve non-linear stacked system
trajectory = model.solve()

print("Initial period t=0:")
print(trajectory.loc[0, ["k", "c", "i", "y"]])

print("\nFinal period t=80:")
print(trajectory.loc[80, ["k", "c", "i", "y"]])
```

---

## Visualizing the Transition

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# 1. Capital
axes[0, 0].plot(trajectory.index, trajectory["k"], lw=2, color="crimson")
axes[0, 0].axhline(model.steady_state["k"], color="black", linestyle="--", alpha=0.7)
axes[0, 0].set_title("Capital Stock $k_t$")
axes[0, 0].grid(True, alpha=0.3)

# 2. Consumption
axes[0, 1].plot(trajectory.index, trajectory["c"], lw=2, color="navy")
axes[0, 1].axhline(model.steady_state["c"], color="black", linestyle="--", alpha=0.7)
axes[0, 1].set_title("Consumption $c_t$")
axes[0, 1].grid(True, alpha=0.3)

# 3. Investment
axes[1, 0].plot(trajectory.index, trajectory["i"], lw=2, color="forestgreen")
axes[1, 0].axhline(model.steady_state["i"], color="black", linestyle="--", alpha=0.7)
axes[1, 0].set_title("Investment $i_t$")
axes[1, 0].grid(True, alpha=0.3)

# 4. Output
axes[1, 1].plot(trajectory.index, trajectory["y"], lw=2, color="darkorange")
axes[1, 1].axhline(model.steady_state["y"], color="black", linestyle="--", alpha=0.7)
axes[1, 1].set_title("Output $y_t$")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Transition Dynamics:
- **Capital Accumulation**: Capital starts depressed at $0.85 \times \bar{k}$ and monotonically converges back to steady state.
- **Consumption Cut**: Households suppress consumption immediately to finance high investment.
- **High Return on Capital**: The marginal product of capital $\alpha y_{t+1}/k_t$ is elevated, incentivizing strong initial capital accumulation.
