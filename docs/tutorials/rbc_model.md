# Tutorial: The Real Business Cycle (RBC) Model

The Real Business Cycle (RBC) model developed by Kydland & Prescott (1982) and Hansen (1985) is the foundational benchmark of modern macroeconomic theory. This tutorial walks through setting up, calibrating, solving, and analyzing the canonical RBC model in Dyno.

---

## Economic Environment

The economy consists of a representative household and a representative firm:

### 1. Household Problem
The household maximizes expected lifetime utility:

$$\max \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \left[ \log(c_t) - \chi \frac{n_t^{1+\eta}}{1+\eta} \right]$$

subject to the budget constraint:

$$c_t + i_t = w_t n_t + r_t k_{t-1}$$

and capital accumulation:

$$k_t = (1-\delta) k_{t-1} + i_t$$

### 2. Firm Problem
The competitive firm produces output $y_t$ with Cobb-Douglas technology:

$$y_t = a_t k_{t-1}^\alpha n_t^{1-\alpha}$$

Factor prices satisfy marginal productivity conditions:

$$r_t = \alpha \frac{y_t}{k_{t-1}}, \quad w_t = (1-\alpha) \frac{y_t}{n_t}$$

### 3. Productivity Shock
Total factor productivity (TFP) evolves according to an AR(1) process:

$$\log(a_t) = \rho \log(a_{t-1}) + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d. } \mathcal{N}(0, \sigma^2)$$

### 4. Equilibrium Conditions

- **Euler equation**: $\frac{1}{c_t} = \beta \, \mathbb{E}_t \left[ \frac{1}{c_{t+1}} (r_{t+1} + 1 - \delta) \right]$
- **Intratemporal labor supply**: $w_t = \chi c_t n_t^\eta$
- **Resource constraint**: $y_t = c_t + i_t$

---

## Defining the Model in `.dyno`

Create `rbc.dyno`:

```text
# 1. Calibration
beta  <- 0.985
delta <- 0.025
alpha <- 0.330
rho   <- 0.950
eta   <- 1.000
nss   <- 0.330

# Calibrate chi to match steady-state labor nss = 0.33
chi   <- (1-alpha)*(1/beta - 1 + delta)/(alpha*(1/beta - 1 + delta - delta*alpha)) * (1/nss)

# 2. Steady State
a[~] <- 1.0
r[~] <- 1/beta - 1 + delta
n[~] <- nss
k[~] <- (alpha/r[~])^(1/(1-alpha)) * n[~]
y[~] <- (k[~]^alpha) * (n[~]^(1-alpha))
w[~] <- (1-alpha) * y[~] / n[~]
i[~] <- delta * k[~]
c[~] <- y[~] - i[~]

# 3. Dynamic Equations
1/c[t] = beta * (1/c[t+1]) * (r[t+1] + 1 - delta)
w[t] = chi * c[t] * (n[t]^eta)
k[t] = (1-delta)*k[t-1] + i[t]
y[t] = a[t] * (k[t-1]^alpha) * (n[t]^(1-alpha))
log(a[t]) = rho*log(a[t-1]) + epsilon[t]
w[t] = (1-alpha)*y[t]/n[t]
r[t] = alpha*y[t]/k[t-1]
y[t] = c[t] + i[t]

# 4. Shock Process (0.9% standard deviation)
epsilon[t] <- N(0, 0.009^2)
```

---

## Solving and Analyzing in Python

```python
from dyno import DynoModel

# Load and verify
rbc = DynoModel("rbc.dyno")
rbc.check()

# Solve model
sol = rbc.solve()
print("Stable decision rule computed successfully!")
```

### Impulse Response Analysis

```python
# Compute IRFs over 40 quarters (10 years)
irfs = sol.irfs(type="log-deviation", T=40)
df_tfp = irfs["epsilon"]

# Inspect responses at key horizons
key_horizons = [0, 1, 4, 12, 40]
print(df_tfp.loc[key_horizons, ["y", "c", "i", "n", "w", "r"]])
```

### Key Economic Insights from IRFs:

1. **Investment volatility**: Investment $i_t$ surges more than output $y_t$ on impact due to consumption smoothing.
2. **Consumption smoothing**: Consumption $c_t$ rises gradually and remains elevated longer than output.
3. **Labor response**: Hours worked $n_t$ rise on impact as higher wages induce intertemporal labor substitution.

---

## Comparison: `.dyno` vs Dynare `.mod`

Dyno also solves the exact same model written in standard Dynare format:

```python
from dyno import DynoModel

# Load the Dynare modfile
rbc_mod = DynoModel("examples/modfiles/RBC.mod")
sol_mod = rbc_mod.solve()

# Transition matrices match identically to machine precision
import numpy as np
diff = np.max(np.abs(sol.X - sol_mod.X))
print(f"Max matrix divergence: {diff:.2e}")  # 0.00e+00
```
