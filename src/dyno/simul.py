from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import numpy as np

from .typedefs import IRFType, SimulateMode, TTensor, TVector, UnitsType

if TYPE_CHECKING:
    import pandas as pd

    from .model import AbstractModel
    from .solver import PerturbationSolution, RecursiveDecisionRule

__all__ = [
    "SimulationResult",
    "IRFSimulation",
    "RandomSimulation",
    "TransitionSimulation",
    "irf",
    "irfs",
    "simulate",
    "sim_to_nsim",
]


class SimulationResult:
    """Base container for model simulations backed by a 3D (N, T+1, V) array.

    Attributes
    ----------
    data : np.ndarray
        3D array of shape ``(N, T + 1, V)`` storing trajectories in canonical units.
    steady_state : np.ndarray
        1D array of shape ``(V,)`` with steady-state values for each variable.
    variables : list[str]
        Variable names of length ``V``.
    index : list[Any]
        Experiment/trajectory identifiers of length ``N`` (shock names, draw indices,
        or transition label).
    units : UnitsType
        Default output units (``"deviation"``, ``"level"``, ``"percent"``, or ``"log-deviation"``).
    attrs : dict[str, Any]
        Optional solver metadata (e.g. ``converged``, ``iterations``, ``residual``).
    """

    data: TTensor
    steady_state: TVector
    variables: list[str]
    index: list[Any]
    units: UnitsType
    _canonical_units: Literal["deviation", "level"]
    attrs: dict[str, Any]

    def __init__(
        self,
        data: np.ndarray,
        variables: list[str],
        index: list[Any] | None = None,
        steady_state: np.ndarray | list[float] | None = None,
        units: UnitsType = "deviation",
        canonical_units: Literal["deviation", "level"] = "deviation",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(
                f"SimulationResult data must have 2 or 3 dimensions, got shape {arr.shape}"
            )

        self.data = arr
        self.variables = list(variables)
        n_exp, _, n_vars = arr.shape
        if len(self.variables) != n_vars:
            raise ValueError(
                f"Length of variables ({len(self.variables)}) does not match data V dimension ({n_vars})"
            )

        if index is None:
            self.index = list(range(1, n_exp + 1))
        else:
            self.index = list(index)
            if len(self.index) != n_exp:
                raise ValueError(
                    f"Length of index ({len(self.index)}) does not match data N dimension ({n_exp})"
                )

        if steady_state is None:
            self.steady_state = np.zeros(n_vars, dtype=float)
        else:
            ss = np.asarray(steady_state, dtype=float).reshape(-1)
            if len(ss) != n_vars:
                raise ValueError(
                    f"Length of steady_state ({len(ss)}) does not match data V dimension ({n_vars})"
                )
            self.steady_state = ss

        self.units = units
        self._canonical_units = canonical_units
        self.attrs = dict(attrs) if attrs is not None else {}

    @property
    def N(self) -> int:
        """Number of experiments / draws / shocks."""
        return int(self.data.shape[0])

    @property
    def T(self) -> int:
        """Simulation horizon (number of periods after t=0)."""
        return int(self.data.shape[1] - 1)

    @property
    def V(self) -> int:
        """Number of variables."""
        return int(self.data.shape[2])

    def in_units(self, units: UnitsType | None = None) -> np.ndarray:
        """Return a 3D ``(N, T+1, V)`` array converted to the requested units."""
        target: UnitsType = self.units if units is None else units
        ss = self.steady_state[None, None, :]

        if self._canonical_units == "deviation":
            dev = self.data
        else:
            dev = np.where(np.isnan(ss), self.data, self.data - ss)

        if target == "deviation":
            return dev.copy()
        elif target == "level":
            if self._canonical_units == "level":
                return self.data.copy()
            return np.where(np.isnan(ss), dev, dev + ss)
        elif target in ("percent", "log-deviation"):
            with np.errstate(divide="ignore", invalid="ignore"):
                return (dev / ss) * 100.0
        else:
            raise ValueError(
                f"Unsupported units '{target}'. Choose from 'level', 'deviation', 'percent', or 'log-deviation'."
            )

    def to_dict(self, units: UnitsType | None = None) -> dict[Any, pd.DataFrame]:
        """Return a dictionary mapping each experiment key in ``index`` to its ``(T+1, V)`` DataFrame."""
        import pandas as pd

        arr = self.in_units(units)
        res: dict[Any, pd.DataFrame] = {}
        for i, key in enumerate(self.index):
            df = pd.DataFrame(arr[i], columns=self.variables)
            df.attrs.update(self.attrs)
            res[key] = df
        return res

    def to_df(self, units: UnitsType | None = None) -> pd.DataFrame:
        """Convert the simulation to a Pandas DataFrame in the specified ``units``.

        When ``N == 1`` (and not an ``IRFSimulation``), returns a 2D ``(T+1, V)`` DataFrame.
        When ``N > 1``, returns a MultiIndex DataFrame indexed by ``(experiment, t)``.
        """
        import pandas as pd

        arr = self.in_units(units)
        if self.N == 1 and not isinstance(self, IRFSimulation):
            df = pd.DataFrame(arr[0], columns=self.variables)
            df.attrs.update(self.attrs)
            return df

        index_name = "shock" if isinstance(self, IRFSimulation) else "n"
        frames: dict[Any, pd.DataFrame] = {}
        for i, key in enumerate(self.index):
            sub_df = pd.DataFrame(arr[i], columns=self.variables)
            sub_df.index.name = "t"
            frames[key] = sub_df
        out = pd.concat(frames, names=[index_name, "t"])
        out.attrs.update(self.attrs)
        return out

    @property
    def df(self) -> pd.DataFrame:
        """Shorthand property for ``self.to_df()``."""
        return self.to_df()

    def to_xarray(self, units: UnitsType | None = None) -> Any:
        """Convert the simulation to an ``xarray.DataArray`` with dimensions ``('N', 'T', 'V')``."""
        import xarray as xr

        arr = self.in_units(units)
        return xr.DataArray(
            arr,
            dims=("N", "T", "V"),
            coords={
                "N": self.index,
                "T": np.arange(self.T + 1),
                "V": self.variables,
            },
            attrs=dict(self.attrs, units=self.units if units is None else units),
        )

    def plot(
        self,
        variables: list[str] | None = None,
        T: int | None = None,
        units: UnitsType | None = None,
        engine: str = "altair",
        **kwargs: Any,
    ) -> Any:
        """Plot the simulation trajectories."""
        from .plots import plot_simulation

        return plot_simulation(
            self,
            variables=variables,
            T=T,
            units=units,
            engine=engine,
            **kwargs,
        )

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.in_units(self.units), dtype=dtype)


class IRFSimulation(SimulationResult, dict):
    """Impulse response function simulation indexed by exogenous shock name.

    Subclasses both ``SimulationResult`` and ``dict`` so ``isinstance(sim, dict)``
    and ``sim[shock_name]`` work alongside ``sim.to_df(units=...)`` and ``sim.plot()``.
    """

    def __init__(
        self,
        data: np.ndarray,
        variables: list[str],
        shocks: list[str],
        steady_state: np.ndarray | list[float] | None = None,
        units: UnitsType = "deviation",
        canonical_units: Literal["deviation", "level"] = "deviation",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        SimulationResult.__init__(
            self,
            data=data,
            variables=variables,
            index=shocks,
            steady_state=steady_state,
            units=units,
            canonical_units=canonical_units,
            attrs=attrs,
        )
        dict.__init__(self, self.to_dict(units=self.units))

    def __getitem__(self, key: Any) -> pd.DataFrame:
        import pandas as pd

        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if isinstance(key, str) and key in self.variables:
            j = self.variables.index(key)
            arr = self.in_units(self.units)[:, :, j].T  # (T+1, N)
            return pd.DataFrame(arr, columns=self.index)
        raise KeyError(key)


class RandomSimulation(SimulationResult):
    """Stochastic time-series simulation across ``N`` Monte Carlo draws."""

    def __init__(
        self,
        data: np.ndarray,
        variables: list[str],
        steady_state: np.ndarray | list[float] | None = None,
        units: UnitsType = "deviation",
        canonical_units: Literal["deviation", "level"] = "deviation",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        n_draws = arr.shape[0]
        super().__init__(
            data=arr,
            variables=variables,
            index=list(range(1, n_draws + 1)),
            steady_state=steady_state,
            units=units,
            canonical_units=canonical_units,
            attrs=attrs,
        )

    def __len__(self) -> int:
        if self.N == 1:
            return self.T + 1
        return self.N

    def __getitem__(self, key: Any) -> Any:
        import pandas as pd

        if isinstance(key, str) and key in self.variables:
            j = self.variables.index(key)
            arr = self.in_units(self.units)[:, :, j]
            if self.N == 1:
                return pd.Series(arr[0], name=key)
            return pd.DataFrame(arr.T, columns=self.index)
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if 0 <= idx < self.N:
                arr = self.in_units(self.units)[idx]
                return pd.DataFrame(arr, columns=self.variables)
            if idx == self.N and idx >= 1:
                arr = self.in_units(self.units)[idx - 1]
                return pd.DataFrame(arr, columns=self.variables)
            raise IndexError(f"Draw index {idx} out of range for N={self.N}")
        return self.to_df()[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.to_df(), name)


class TransitionSimulation(SimulationResult):
    """Deterministic / perfect-foresight transition trajectory over ``t = 0..T``."""

    def __init__(
        self,
        data: np.ndarray,
        variables: list[str],
        steady_state: np.ndarray | list[float] | None = None,
        units: UnitsType = "level",
        canonical_units: Literal["deviation", "level"] = "level",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        super().__init__(
            data=arr,
            variables=variables,
            index=["Transition"],
            steady_state=steady_state,
            units=units,
            canonical_units=canonical_units,
            attrs=attrs,
        )

    def to_df(self, units: UnitsType | None = None) -> pd.DataFrame:
        import pandas as pd

        arr = self.in_units(units)[0]
        df = pd.DataFrame(arr, columns=self.variables)
        df.index = pd.RangeIndex(self.T + 1, name="t")
        df.reset_index(inplace=True)
        df.attrs.update(self.attrs)
        return df

    def __len__(self) -> int:
        return self.T + 1

    def __getitem__(self, key: Any) -> Any:
        if key in ("Perfect Foresight", "Transition", "transition"):
            return self.to_df()
        return self.to_df()[key]

    def __contains__(self, key: Any) -> bool:
        if key in ("Perfect Foresight", "Transition", "transition"):
            return True
        return key in self.variables or key == "t"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.to_df(), name)


def _compute_irf_array(
    dr: RecursiveDecisionRule | PerturbationSolution,
    i: int,
    T: int = 40,
) -> np.ndarray:
    """Compute a single shock's IRF in deviation units, shape ``(T+1, V)``."""
    X = dr.X
    Y = dr.Y
    Σ = dr.Σ

    assert X.shape is not None
    v0 = np.zeros(X.shape[1])

    assert Y.shape is not None
    m0 = np.zeros(Y.shape[1])

    ss = [v0]
    m0[i] = np.sqrt(Σ[i, i])
    ss.append(X @ ss[-1] + Y @ m0)  # type: ignore

    for _ in range(T - 1):
        ss.append(X @ ss[-1])  # type: ignore

    return np.concatenate([e[None, :] for e in ss], axis=0)


def irf(
    dr: RecursiveDecisionRule | PerturbationSolution,
    i: int,
    T: int = 40,
    type: IRFType = "level",
) -> pd.DataFrame:
    """Impulse response function simulation in response to a shock on a specific exogenous variable."""
    import pandas as pd

    res = _compute_irf_array(dr, i, T=T)
    assert dr.x0 is not None
    sim = TransitionSimulation(
        data=res,
        variables=list(dr.symbols["endogenous"]),
        steady_state=dr.x0,
        units=type,
        canonical_units="deviation",
    )
    arr = sim.in_units(type)[0]
    return pd.DataFrame(arr, columns=dr.symbols["endogenous"])


def irfs(
    model: AbstractModel,
    dr: RecursiveDecisionRule | PerturbationSolution,
    type: IRFType = "log-deviation",
    T: int = 40,
) -> IRFSimulation:
    """Impulse response function simulation in response to shocks on each exogenous variable."""
    exo_names = list(model.symbols["exogenous"])
    endo_names = list(dr.symbols["endogenous"])
    n_endo = len(endo_names)

    if len(exo_names) == 0:
        data = np.zeros((0, T + 1, n_endo), dtype=float)
    else:
        arrays = [_compute_irf_array(dr, i, T=T) for i in range(len(exo_names))]
        data = np.stack(arrays, axis=0)

    x0 = dr.x0 if dr.x0 is not None else np.zeros(n_endo, dtype=float)
    return IRFSimulation(
        data=data,
        variables=endo_names,
        shocks=exo_names,
        steady_state=x0,
        units=type,
        canonical_units="deviation",
    )


def simulate(
    dr: RecursiveDecisionRule | PerturbationSolution,
    T: int = 40,
    shocks: dict[str, dict[int, float]] | np.ndarray | None = None,
    initial_states: dict[str, float] | np.ndarray | None = None,
    N: int = 1,
    mode: SimulateMode = "random",
    units: UnitsType = "deviation",
) -> SimulationResult:
    """Simulate the evolution of endogenous variables from a solved decision rule.

    Parameters
    ----------
    dr : RecursiveDecisionRule | PerturbationSolution
        Solved linearized model.
    T : int, default 40
        Time horizon over which the simulation is performed.
    shocks : dict[str, dict[int, float]] | np.ndarray | None, optional
        Preset shocks to force at specific dates.
    initial_states : dict[str, float] | np.ndarray | None, optional
        Preset initial state at date 0.
    N : int, default 1
        Number of random trajectories to simulate when ``mode='random'``.
    mode : {'auto', 'irf', 'random', 'transition', 'deterministic'}, default 'random'
        Simulation mode.
    units : {'deviation', 'level', 'percent', 'log-deviation'}, default 'deviation'
        Default output units for the returned ``SimulationResult``.

    Returns
    -------
    SimulationResult
        ``IRFSimulation`` when ``mode='irf'``, or ``RandomSimulation`` of shape ``(N, T+1, V)``.
    """
    if mode == "irf":
        model = getattr(dr, "_model", None) or getattr(
            getattr(dr, "decision_rule", None), "_model", None
        )
        if model is None:
            raise ValueError("Cannot compute IRFs without an attached model.")
        return irfs(model, dr, type=units, T=T)

    X = dr.X
    Y = dr.Y
    Σ = dr.Σ
    assert X.shape is not None
    assert Y.shape is not None

    n_endo = X.shape[1]
    n_exo = Y.shape[1]
    m0 = np.zeros(n_exo)

    endo_names = list(dr.symbols["endogenous"])
    exo_names = list(dr.symbols["exogenous"])

    model = getattr(dr, "_model", None) or getattr(
        getattr(dr, "decision_rule", None), "_model", None
    )
    context = getattr(model, "context", {}) if model is not None else {}
    model_values = context.get("values", {})
    steady_states = context.get("steady_states", {})

    # 1. Initial shock at t=0
    eps_0 = np.zeros(n_exo)
    for j, name in enumerate(exo_names):
        if (
            shocks is not None
            and isinstance(shocks, dict)
            and name in shocks
            and 0 in shocks[name]
        ):
            val = shocks[name][0]
            ss_val = steady_states.get(name, 0.0)
            eps_0[j] = val - ss_val if not np.isnan(ss_val) else val
        elif name in model_values and 0 in model_values[name]:
            val = model_values[name][0]
            ss_val = steady_states.get(name, 0.0)
            eps_0[j] = val - ss_val if not np.isnan(ss_val) else val

    # 2. Initial state at t=0
    v0 = Y @ eps_0

    if initial_states is not None:
        if isinstance(initial_states, dict):
            for i, name in enumerate(endo_names):
                if name in initial_states:
                    val = initial_states[name]
                    ss_val = steady_states.get(name, 0.0)
                    v0[i] = val - ss_val if not np.isnan(ss_val) else val
        elif isinstance(initial_states, (np.ndarray, list)):
            v0 = np.array(initial_states, dtype=float).copy()
    else:
        for i, name in enumerate(endo_names):
            if name in model_values and 0 in model_values[name]:
                val = model_values[name][0]
                ss_val = steady_states.get(name, 0.0)
                v0[i] = val - ss_val if not np.isnan(ss_val) else val

    n_draws = max(1, int(N))
    trajectories = np.zeros((n_draws, T + 1, n_endo), dtype=float)
    trajectories[:, 0, :] = v0[None, :]

    # 3. Dynamic simulation for t = 1 to T across N draws
    for draw_idx in range(n_draws):
        curr = v0.copy()
        for t in range(1, T + 1):
            if shocks is not None and isinstance(shocks, np.ndarray):
                if shocks.ndim == 3:
                    e = shocks[draw_idx, t - 1, :].copy()
                else:
                    e = shocks[t - 1, :].copy()
            else:
                if n_exo > 0:
                    e = np.random.multivariate_normal(m0, Σ)
                else:
                    e = np.zeros(0, dtype=float)
                for j, name in enumerate(exo_names):
                    if (
                        shocks is not None
                        and isinstance(shocks, dict)
                        and name in shocks
                        and t in shocks[name]
                    ):
                        val = shocks[name][t]
                        ss_val = steady_states.get(name, 0.0)
                        e[j] = val - ss_val if not np.isnan(ss_val) else val
                    elif name in model_values and t in model_values[name]:
                        val = model_values[name][t]
                        ss_val = steady_states.get(name, 0.0)
                        e[j] = val - ss_val if not np.isnan(ss_val) else val

            curr = X @ curr + Y @ e
            trajectories[draw_idx, t, :] = curr

    x0 = dr.x0 if dr.x0 is not None else np.zeros(n_endo, dtype=float)
    return RandomSimulation(
        data=trajectories,
        variables=endo_names,
        steady_state=x0,
        units=units,
        canonical_units="deviation",
    )


def sim_to_nsim(sim: Any, units: UnitsType | None = None) -> pd.DataFrame:
    """Convert any simulation object, dict of DataFrames, or DataFrame to tidy long format
    with columns ``['shock', 't', 'variable', 'value']``.
    """
    import pandas as pd

    if isinstance(sim, SimulationResult):
        sim_dict = sim.to_dict(units=units)
    elif isinstance(sim, dict):
        sim_dict = sim
    elif isinstance(sim, pd.DataFrame):
        sim_dict = {"simulation": sim}
    else:
        raise TypeError(f"Unsupported simulation type for sim_to_nsim: {type(sim)}")

    cleaned: dict[Any, pd.DataFrame] = {}
    for k, df in sim_dict.items():
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if "t" in df.columns:
            cleaned[str(k)] = df.drop(columns=["t"])
        else:
            cleaned[str(k)] = df

    if not cleaned:
        return pd.DataFrame(columns=["shock", "t", "variable", "value"])

    pdf = pd.concat(cleaned).reset_index()
    ppdf = pdf.rename(columns={"level_0": "shock", "level_1": "t"})
    ppdf = ppdf.melt(id_vars=["shock", "t"])
    return ppdf
