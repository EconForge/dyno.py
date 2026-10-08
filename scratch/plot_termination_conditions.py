from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dyno import DynoModel, deterministic_solve

RAMSEY_MODEL_TXT = """
# Neoclassical Ramsey Model with Perfect Foresight / Deterministic Transition
# Parameters
alph <- 0.50
gam  <- 0.50
delt <- 0.02
bet  <- 0.051
aa   <- 0.511
T    <- 50

# Steady State
x[~] <- 1.0
k[~] <- ((delt + bet) / (1.0 * aa * alph))^(1 / (alph - 1))
c[~] <- aa * k[~]^alph - delt * k[~]

# Dynamic Equations
0 = c[t] + k[t] - aa*x[t]*k[t-1]^alph - (1 - delt)*k[t-1]
0 = c[t]^(-gam) - (1 + bet)^(-1) * (aa*alph*x[t+1]*k[t]^(alph-1) + 1 - delt) * c[t+1]^(-gam)

# Anticipated Exogenous Trajectory
x[1] <- 1.10
x[2] <- 1.30
forall t, 3 <= t < T : x[t] <- 1.0 + (1.30 - 1.0) * exp(-(t - 1))
"""


def resource_constraint_residual(
    k_prev: float,
    k_curr: float,
    c_curr: float,
    x_curr: float,
    alph: float = 0.50,
    delt: float = 0.02,
    aa: float = 0.511,
) -> float:
    return c_curr + k_curr - (aa * x_curr * (k_prev**alph) + (1.0 - delt) * k_prev)


def build_plot(output_path: Path) -> Path:
    model = DynoModel(txt=RAMSEY_MODEL_TXT)
    horizons = [50, 100]
    modes = [
        ("static", {"continuation": "static"}),
        ("stationary", {"continuation": "stationary"}),
        ("steady_state", {"continuation": "steady_state"}),
        (
            "constant_growth",
            {"continuation": "constant_growth", "growth_type": "geometric"},
        ),
    ]

    trajectories = {
        (name, horizon): deterministic_solve(model, T=horizon, **options)
        for horizon in horizons
        for name, options in modes
    }

    colors = {
        "static": "#c0392b",
        "stationary": "#1f77b4",
        "steady_state": "#2ca02c",
        "constant_growth": "#9467bd",
    }
    linestyles = {50: "-", 100: "--"}
    markers = {50: None, 100: "o"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.axvspan(50, 100, color="#f4f4f4", alpha=0.8, zorder=0)
    ax.axvline(50, color="#999999", linestyle=":", linewidth=1)
    for horizon in horizons:
        for name, _ in modes:
            traj = trajectories[(name, horizon)]
            ax.plot(
                traj["t"],
                traj["k"],
                label=f"{name}, T={horizon}",
                color=colors[name],
                linestyle=linestyles[horizon],
                linewidth=2.2 if horizon == 100 else 2,
                marker=markers[horizon],
                markevery=5 if horizon == 100 else None,
                markersize=3.5 if horizon == 100 else None,
                zorder=3 if horizon == 100 else 2,
            )
    ax.axhline(model.steady_state["k"], color="#555555", linestyle=":", linewidth=1.2)
    ax.set_title("Capital Path")
    ax.set_xlabel("t")
    ax.set_ylabel("k")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    for horizon in horizons:
        for name, _ in modes:
            traj = trajectories[(name, horizon)]
            tail = traj.iloc[-8:].copy()
            tail["tau"] = tail["t"] - horizon
            ax.plot(
                tail["tau"],
                tail["k"],
                marker="o",
                color=colors[name],
                linestyle=linestyles[horizon],
                linewidth=2,
            )
    ax.axhline(model.steady_state["k"], color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Capital Near Terminal Date")
    ax.set_xlabel("t - T")
    ax.set_ylabel("k")

    ax = axes[1, 0]
    ax.axvspan(50, 100, color="#f4f4f4", alpha=0.8, zorder=0)
    ax.axvline(50, color="#999999", linestyle=":", linewidth=1)
    for horizon in horizons:
        for name, _ in modes:
            traj = trajectories[(name, horizon)]
            ax.plot(
                traj["t"],
                traj["c"],
                label=f"{name}, T={horizon}",
                color=colors[name],
                linestyle=linestyles[horizon],
                linewidth=2.2 if horizon == 100 else 2,
                marker=markers[horizon],
                markevery=5 if horizon == 100 else None,
                markersize=3.5 if horizon == 100 else None,
                zorder=3 if horizon == 100 else 2,
            )
    ax.axhline(model.steady_state["c"], color="#555555", linestyle=":", linewidth=1.2)
    ax.set_title("Consumption Path")
    ax.set_xlabel("t")
    ax.set_ylabel("c")
    ax.set_xlim(0, 100)

    ax = axes[1, 1]
    labels = [name for name, _ in modes]
    positions = np.arange(len(labels))
    width = 0.18
    residual_colors = {50: "#7f8c8d", 100: "#bdc3c7"}
    step_colors = {50: "#f39c12", 100: "#f7c66b"}
    for offset_i, horizon in enumerate(horizons):
        residuals = []
        terminal_steps = []
        for name in labels:
            traj = trajectories[(name, horizon)]
            k = traj["k"].to_numpy()
            c = traj["c"].to_numpy()
            x = traj["x"].to_numpy()
            residuals.append(
                resource_constraint_residual(
                    k[horizon - 1],
                    k[horizon],
                    c[horizon],
                    x[horizon],
                )
            )
            terminal_steps.append(k[horizon] - k[horizon - 1])

        shift = (offset_i - 0.5) * 2 * width
        ax.bar(
            positions + shift - width / 2,
            residuals,
            width=width,
            color=residual_colors[horizon],
            label=f"resource residual at T, T={horizon}",
        )
        ax.bar(
            positions + shift + width / 2,
            terminal_steps,
            width=width,
            color=step_colors[horizon],
            label=f"k[T] - k[T-1], T={horizon}",
        )

    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Terminal Effects")
    ax.legend(frameon=False)

    fig.suptitle(
        "Deterministic Solver Continuation Conditions on the Ramsey Example",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    output = build_plot(Path("scratch/termination_condition_comparison.png"))
    print(output)
