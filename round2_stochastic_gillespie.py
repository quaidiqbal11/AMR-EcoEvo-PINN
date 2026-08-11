#!/usr/bin/env python
# coding: utf-8

# Round 2

# In[1]:


# =============================================================================
# CELL 1 — Environment, folders, reproducibility, and simulation budgets
# =============================================================================
# PURPOSE
# -------
# This cell:
# 1. imports the packages needed for Round 2;
# 2. fixes all random seeds;
# 3. creates folders for stochastic data, figures, tables, and reports;
# 4. reports the available GPU;
# 5. selects development or manuscript-scale simulation budgets.
#
# IMPORTANT
# ---------
# The Gillespie direct algorithm is event-driven and sequential. It mainly
# uses the CPU. Your 8 GB GPU will become useful during PINN training.

from pathlib import Path
from dataclasses import dataclass, asdict, replace
from typing import Dict, Tuple, Optional

import math
import json
import random
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from IPython.display import display

warnings.filterwarnings("ignore", category=UserWarning)

# Reproducibility seed
SEED = 20260804

random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------------------------
# Simulation mode
# -------------------------------------------------------------------------
# First run the complete notebook using QUICK_MODE=True.
#
# After every cell runs successfully:
# 1. change QUICK_MODE to False;
# 2. restart the Jupyter kernel;
# 3. run all cells again;
# 4. use only the final-mode results in the manuscript.
# -------------------------------------------------------------------------
QUICK_MODE = False

SHORT_REPLICATES = 30 if QUICK_MODE else 500
OUTCOME_REPLICATES = 60 if QUICK_MODE else 1000
RESCUE_GRID_REPLICATES = 10 if QUICK_MODE else 300

# -------------------------------------------------------------------------
# Project directory
# -------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

ENSEMBLE_DIR = (
    PROJECT_ROOT
    / "data"
    / "stochastic"
    / "ensembles"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "stochastic"
    / "summaries"
)

PINN_EXPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "stochastic"
    / "pinn_exports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

for directory in [
    ENSEMBLE_DIR,
    SUMMARY_DIR,
    PINN_EXPORT_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# -------------------------------------------------------------------------
# Environment report
# -------------------------------------------------------------------------
print("Project root:", PROJECT_ROOT)
print("Python:", sys.version.split()[0])
print("Operating system:", platform.platform())
print("QUICK_MODE:", QUICK_MODE)
print("Short replicates per schedule:", SHORT_REPLICATES)
print("Outcome replicates per schedule:", OUTCOME_REPLICATES)
print("Rescue-grid replicates per case:", RESCUE_GRID_REPLICATES)

# GPU reporting
try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        memory_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"GPU memory: {memory_gb:.2f} GB")

except Exception as exc:
    print("GPU information unavailable:", exc)

# In[2]:


# =============================================================================
# CELL 2 — Publication figure settings and reusable helpers
# =============================================================================
# PURPOSE
# -------
# This cell:
# 1. applies consistent publication-oriented figure settings;
# 2. defines a function that saves every figure as PNG, PDF, and SVG.
#
# No scientific result is produced here.

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (8.6, 5.4),
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
    }
)


def save_figure(
    fig: plt.Figure,
    stem: str,
    close_after_saving: bool = False,
) -> None:
    """
    Save one figure in publication and editable formats.

    Parameters
    ----------
    fig:
        Matplotlib figure object.

    stem:
        Output filename without extension.

    close_after_saving:
        Close the figure after saving when True.

    Files produced
    --------------
    PNG:
        300-dpi raster image.

    PDF:
        Vector figure suitable for LaTeX.

    SVG:
        Editable vector figure.
    """

    safe_stem = stem.strip().replace(" ", "_")

    fig.savefig(
        FIGURE_DIR / f"{safe_stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIGURE_DIR / f"{safe_stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIGURE_DIR / f"{safe_stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )

    if close_after_saving:
        plt.close(fig)


print("Cell 2 completed successfully.")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Figure directory:", FIGURE_DIR.resolve())

# In[3]:


# =============================================================================
# CELL 3 — Biological parameters, stochastic configuration, and treatments
# =============================================================================
# PURPOSE
# -------
# This cell defines:
# 1. biological parameters;
# 2. stochastic simulation settings;
# 3. untreated, early, late, and pulsed antibiotic schedules;
# 4. treatment-switch and output-grid utilities.
#
# A piecewise treatment schedule has one more exposure level than switch times.
#
# Example
# -------
# switch_times = (5, 8)
# levels       = (0, 1, 0)
#
# This means:
# A(t)=0 before t=5;
# A(t)=1 for 5<=t<8;
# A(t)=0 after t=8.

@dataclass(frozen=True)
class AMRParameters:
    """Biological parameters of the sensitive-resistant model."""

    carrying_capacity: int = 1000

    sensitive_birth_rate: float = 1.0
    resistance_fitness_cost: float = 0.10

    sensitive_death_rate: float = 0.10
    resistant_death_rate: float = 0.10

    mutation_probability: float = 1.0e-4

    sensitive_drug_effect: float = 1.0
    resistant_drug_effect: float = 0.10

    @property
    def resistant_birth_rate(self) -> float:
        """
        Resistant birth rate after accounting for fitness cost.

        r_R = r_S(1-c_R)
        """

        return (
            self.sensitive_birth_rate
            * (
                1.0
                - self.resistance_fitness_cost
            )
        )

    def validate(self) -> None:
        """Check whether all biological parameters are admissible."""

        if self.carrying_capacity <= 0:
            raise ValueError(
                "carrying_capacity must be positive."
            )

        if self.sensitive_birth_rate <= 0:
            raise ValueError(
                "sensitive_birth_rate must be positive."
            )

        if self.sensitive_death_rate < 0:
            raise ValueError(
                "sensitive_death_rate cannot be negative."
            )

        if self.resistant_death_rate < 0:
            raise ValueError(
                "resistant_death_rate cannot be negative."
            )

        bounded_parameters = {
            "resistance_fitness_cost":
                self.resistance_fitness_cost,

            "mutation_probability":
                self.mutation_probability,

            "sensitive_drug_effect":
                self.sensitive_drug_effect,

            "resistant_drug_effect":
                self.resistant_drug_effect,
        }

        for name, value in bounded_parameters.items():

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0,1]."
                )


@dataclass(frozen=True)
class StochasticConfig:
    """Simulation and outcome settings."""

    initial_sensitive: int = 100
    initial_resistant: int = 0

    # Short trajectory comparison with Round 1
    short_final_time: float = 20.0
    short_output_points: int = 401

    # Longer horizon for extinction and rescue
    outcome_final_time: float = 120.0
    outcome_output_points: int = 601

    # Operational outcome definitions
    establishment_threshold: int = 10
    dominance_fraction: float = 0.50

    # Protection against accidental infinite runs
    maximum_events_per_replicate: int = 2_000_000


@dataclass(frozen=True)
class PiecewiseTreatment:
    """Piecewise-constant antibiotic-exposure schedule."""

    name: str
    switch_times: Tuple[float, ...]
    levels: Tuple[float, ...]
    description: str

    def __post_init__(self) -> None:

        if (
            len(self.levels)
            != len(self.switch_times) + 1
        ):
            raise ValueError(
                "levels must contain one more "
                "value than switch_times."
            )

        if (
            tuple(sorted(self.switch_times))
            != self.switch_times
        ):
            raise ValueError(
                "switch_times must be ordered."
            )

        if any(
            level < 0.0 or level > 1.0
            for level in self.levels
        ):
            raise ValueError(
                "Treatment levels must lie in [0,1]."
            )

    def exposure(
        self,
        t: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate A(t)."""

        values = np.asarray(
            t,
            dtype=float,
        )

        indexes = np.searchsorted(
            np.asarray(self.switch_times),
            values,
            side="right",
        )

        output = np.asarray(
            self.levels,
            dtype=float,
        )[indexes]

        if np.isscalar(t):
            return float(output)

        return output

    def next_switch_after(
        self,
        t: float,
        tolerance: float = 1.0e-12,
    ) -> float:
        """Return the next deterministic switch after time t."""

        for switch_time in self.switch_times:

            if switch_time > t + tolerance:
                return float(switch_time)

        return math.inf

    @property
    def first_on_time(self) -> float:
        """Return the first time at which treatment becomes active."""

        if self.levels[0] > 0:
            return 0.0

        for switch_time, level in zip(
            self.switch_times,
            self.levels[1:],
        ):

            if level > 0:
                return float(switch_time)

        return math.inf

    @property
    def permanent_at_end(self) -> bool:
        """Return True when treatment remains active at the final horizon."""

        return bool(self.levels[-1] > 0)

    def active_intervals(
        self,
        final_time: float,
    ) -> list[tuple[float, float]]:
        """Return treatment-active intervals for plotting."""

        boundaries = [0.0]

        boundaries.extend(
            switch_time
            for switch_time in self.switch_times
            if switch_time < final_time
        )

        boundaries.append(final_time)

        intervals = []

        for left, right in zip(
            boundaries[:-1],
            boundaries[1:],
        ):

            midpoint = 0.5 * (left + right)

            if self.exposure(midpoint) > 0:
                intervals.append(
                    (left, right)
                )

        return intervals


PARAMETERS = AMRParameters()
PARAMETERS.validate()

CONFIG = StochasticConfig()

initial_total = (
    CONFIG.initial_sensitive
    + CONFIG.initial_resistant
)

if initial_total > PARAMETERS.carrying_capacity:
    raise ValueError(
        "Initial population exceeds carrying capacity."
    )


SCHEDULES = {
    "untreated":
        PiecewiseTreatment(
            name="untreated",
            switch_times=(),
            levels=(0.0,),
            description="No treatment.",
        ),

    "early_continuous":
        PiecewiseTreatment(
            name="early_continuous",
            switch_times=(5.0,),
            levels=(0.0, 1.0),
            description="Permanent treatment from t=5.",
        ),

    "late_continuous":
        PiecewiseTreatment(
            name="late_continuous",
            switch_times=(10.0,),
            levels=(0.0, 1.0),
            description="Permanent treatment from t=10.",
        ),

    "pulsed":
        PiecewiseTreatment(
            name="pulsed",
            switch_times=(
                5.0,
                8.0,
                12.0,
                15.0,
            ),
            levels=(
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
            ),
            description=(
                "Treatment on [5,8) and [12,15)."
            ),
        ),
}


def make_output_grid(
    final_time: float,
    points: int,
    schedule: PiecewiseTreatment,
) -> np.ndarray:
    """
    Create a regular time grid containing every treatment switch exactly.
    """

    regular_grid = np.linspace(
        0.0,
        final_time,
        points,
    )

    switch_grid = np.asarray(
        [
            switch_time
            for switch_time in schedule.switch_times
            if 0.0 <= switch_time <= final_time
        ],
        dtype=float,
    )

    return np.unique(
        np.concatenate(
            [
                regular_grid,
                switch_grid,
            ]
        )
    )


def shade_treatment(
    ax: plt.Axes,
    schedule: PiecewiseTreatment,
    final_time: float,
) -> None:
    """Shade all intervals during which treatment is active."""

    for left, right in schedule.active_intervals(
        final_time
    ):

        ax.axvspan(
            left,
            right,
            alpha=0.08,
        )


schedule_table = pd.DataFrame(
    {
        "schedule": list(SCHEDULES.keys()),

        "description": [
            schedule.description
            for schedule in SCHEDULES.values()
        ],

        "switch_times": [
            schedule.switch_times
            for schedule in SCHEDULES.values()
        ],

        "levels": [
            schedule.levels
            for schedule in SCHEDULES.values()
        ],
    }
)

display(schedule_table)

# In[5]:


# =============================================================================
# CELL 4 — Five stochastic events, propensities, mean drift, and verification
# =============================================================================
# PURPOSE
# -------
# Implement and verify the five stochastic events.
#
# EVENT ORDER
# -----------
# 0. Sensitive birth without mutation
# 1. Resistant mutant offspring from sensitive division
# 2. Sensitive death
# 3. Resistant birth
# 4. Resistant death
#
# The order matches the well-mixed DrugRes_WMP.py implementation.
#
# CORRECTION
# ----------
# All birth propensities use:
#
#     max(0, 1-(S+R)/K)
#
# This prevents invalid negative event rates.

STOICHIOMETRIC_MATRIX = np.array(
    [
        [1, 0, -1, 0, 0],
        [0, 1,  0, 1, -1],
    ],
    dtype=np.int64,
)

EVENT_NAMES = (
    "sensitive_birth",
    "mutant_resistant_birth",
    "sensitive_death",
    "resistant_birth",
    "resistant_death",
)


def calculate_propensities(
    state: np.ndarray,
    t: float,
    parameters: AMRParameters,
    schedule: PiecewiseTreatment,
) -> np.ndarray:
    """
    Calculate five non-negative Gillespie event propensities.
    """

    sensitive = float(state[0])
    resistant = float(state[1])

    if sensitive < 0 or resistant < 0:
        raise ValueError(
            "Stochastic population counts cannot be negative."
        )

    exposure = float(
        schedule.exposure(t)
    )

    total = sensitive + resistant

    crowding = max(
        0.0,
        1.0
        - total
        / parameters.carrying_capacity,
    )

    sensitive_birth_rate = max(
        0.0,
        parameters.sensitive_birth_rate
        * (
            1.0
            - parameters.sensitive_drug_effect
            * exposure
        ),
    )

    resistant_birth_rate = max(
        0.0,
        parameters.resistant_birth_rate
        * (
            1.0
            - parameters.resistant_drug_effect
            * exposure
        ),
    )

    propensities = np.array(
        [
            # Sensitive birth without mutation
            sensitive_birth_rate
            * (
                1.0
                - parameters.mutation_probability
            )
            * crowding
            * sensitive,

            # Resistant mutant offspring
            sensitive_birth_rate
            * parameters.mutation_probability
            * crowding
            * sensitive,

            # Sensitive death
            parameters.sensitive_death_rate
            * sensitive,

            # Resistant birth
            resistant_birth_rate
            * crowding
            * resistant,

            # Resistant death
            parameters.resistant_death_rate
            * resistant,
        ],
        dtype=float,
    )

    if np.any(
        ~np.isfinite(propensities)
    ):
        raise FloatingPointError(
            "Non-finite propensity detected."
        )

    if np.any(
        propensities < -1.0e-14
    ):
        raise FloatingPointError(
            "Negative propensity detected."
        )

    return np.maximum(
        propensities,
        0.0,
    )


def deterministic_rhs(
    t: float,
    state: np.ndarray,
    parameters: AMRParameters,
    schedule: PiecewiseTreatment,
) -> np.ndarray:
    """
    Calculate the deterministic mean-field equations independently.

    The adaptive ODE solver uses floating-point intermediate states.
    Consequently, this function does not call the integer-state stochastic
    validator.
    """

    sensitive, resistant = np.asarray(
        state,
        dtype=float,
    )

    exposure = float(
        schedule.exposure(t)
    )

    total = sensitive + resistant

    crowding = (
        1.0
        - total
        / parameters.carrying_capacity
    )

    sensitive_birth_rate = (
        parameters.sensitive_birth_rate
        * (
            1.0
            - parameters.sensitive_drug_effect
            * exposure
        )
    )

    resistant_birth_rate = (
        parameters.resistant_birth_rate
        * (
            1.0
            - parameters.resistant_drug_effect
            * exposure
        )
    )

    d_sensitive = (
        sensitive_birth_rate
        * (
            1.0
            - parameters.mutation_probability
        )
        * crowding
        * sensitive
        - parameters.sensitive_death_rate
        * sensitive
    )

    d_resistant = (
        sensitive_birth_rate
        * parameters.mutation_probability
        * crowding
        * sensitive
        + resistant_birth_rate
        * crowding
        * resistant
        - parameters.resistant_death_rate
        * resistant
    )

    return np.array(
        [
            d_sensitive,
            d_resistant,
        ],
        dtype=float,
    )


def solve_deterministic_reference(
    parameters: AMRParameters,
    schedule: PiecewiseTreatment,
    output_grid: np.ndarray,
) -> pd.DataFrame:
    """Solve the deterministic mean-field equations."""

    solution = solve_ivp(
        fun=lambda t, state: deterministic_rhs(
            t,
            state,
            parameters,
            schedule,
        ),

        t_span=(
            float(output_grid[0]),
            float(output_grid[-1]),
        ),

        y0=np.array(
            [
                CONFIG.initial_sensitive,
                CONFIG.initial_resistant,
            ],
            dtype=float,
        ),

        t_eval=output_grid,

        method="DOP853",

        rtol=1.0e-10,
        atol=1.0e-12,
    )

    if not solution.success:
        raise RuntimeError(
            solution.message
        )

    sensitive = solution.y[0]
    resistant = solution.y[1]

    total = sensitive + resistant

    resistant_fraction = np.divide(
        resistant,
        total,
        out=np.zeros_like(resistant),
        where=total > 0,
    )

    return pd.DataFrame(
        {
            "time": solution.t,
            "sensitive": sensitive,
            "resistant": resistant,
            "total": total,
            "resistant_fraction":
                resistant_fraction,
            "antibiotic":
                schedule.exposure(solution.t),
        }
    )


# -------------------------------------------------------------------------
# Mathematical consistency test
# -------------------------------------------------------------------------
# Compare:
#
#     V @ a(X,t)
#
# with independently written mean-field equations at random feasible states.
# -------------------------------------------------------------------------
test_rng = np.random.default_rng(
    SEED
)

maximum_drift_difference = 0.0
minimum_propensity = math.inf

for _ in range(500):

    sensitive = int(
        test_rng.integers(
            0,
            PARAMETERS.carrying_capacity + 1,
        )
    )

    resistant = int(
        test_rng.integers(
            0,
            PARAMETERS.carrying_capacity
            - sensitive
            + 1,
        )
    )

    state = np.array(
        [
            sensitive,
            resistant,
        ],
        dtype=np.int64,
    )

    schedule = SCHEDULES[
        test_rng.choice(
            list(SCHEDULES.keys())
        )
    ]

    test_time = float(
        test_rng.uniform(
            0.0,
            CONFIG.short_final_time,
        )
    )

    propensities = calculate_propensities(
        state,
        test_time,
        PARAMETERS,
        schedule,
    )

    matrix_drift = (
        STOICHIOMETRIC_MATRIX
        @ propensities
    )

    direct_drift = deterministic_rhs(
        test_time,
        state,
        PARAMETERS,
        schedule,
    )

    maximum_drift_difference = max(
        maximum_drift_difference,
        float(
            np.max(
                np.abs(
                    matrix_drift
                    - direct_drift
                )
            )
        ),
    )

    minimum_propensity = min(
        minimum_propensity,
        float(
            propensities.min()
        ),
    )


# Births must stop at carrying capacity.
capacity_state = np.array(
    [
        900,
        100,
    ],
    dtype=np.int64,
)

capacity_propensities = (
    calculate_propensities(
        capacity_state,
        0.0,
        PARAMETERS,
        SCHEDULES["untreated"],
    )
)

assert maximum_drift_difference < 1.0e-12
assert minimum_propensity >= 0.0

assert np.all(
    capacity_propensities[
        [0, 1, 3]
    ] == 0.0
)

print("All propensity and drift tests passed.")
print(
    "Maximum drift difference:",
    maximum_drift_difference,
)
print(
    "Birth propensities at capacity:",
    capacity_propensities[
        [0, 1, 3]
    ],
)

# In[6]:


# =============================================================================
# CELL 5 — Corrected single-replicate Gillespie direct simulator
# =============================================================================
# Corrections relative to the audited repository:
# - accepts a seed;
# - records complete regularly sampled trajectories;
# - interrupts waiting times at every treatment switch and final horizon;
# - handles zero total propensity;
# - never loses a final-time replicate;
# - separately records emergence, establishment, dominance, and extinction.
# Establishment is operationally defined as first R>=10, rather than requiring
# R to equal an exact equilibrium value.

def simulate_gillespie_replicate(parameters,schedule,output_grid,seed,
                                  establishment_threshold=CONFIG.establishment_threshold,
                                  dominance_fraction=CONFIG.dominance_fraction,
                                  maximum_events=CONFIG.maximum_events_per_replicate,
                                  keep_event_log=False):
    grid=np.asarray(output_grid,dtype=float)
    if grid.ndim!=1 or len(grid)<2 or abs(grid[0])>1e-14 or np.any(np.diff(grid)<=0):
        raise ValueError('output_grid must be strictly increasing and start at zero.')
    rng=np.random.default_rng(seed)
    state=np.array([CONFIG.initial_sensitive,CONFIG.initial_resistant],dtype=np.int64)
    t=float(grid[0]); final=float(grid[-1]); index=0
    sampled_S=np.empty(len(grid),dtype=np.int32); sampled_R=np.empty(len(grid),dtype=np.int32)
    while index<len(grid) and grid[index]<=t+1e-14:
        sampled_S[index],sampled_R[index]=state; index+=1

    emergence=0.0 if state[1]>0 else math.nan
    establishment=0.0 if state[1]>=establishment_threshold else math.nan
    dominance=0.0 if state.sum()>0 and state[1]/state.sum()>=dominance_fraction else math.nan
    S_extinction=0.0 if state[0]==0 else math.nan
    total_extinction=0.0 if state.sum()==0 else math.nan
    first_on=schedule.first_on_time
    state_at_treatment=(int(state[0]),int(state[1])) if first_on==0 else None
    event_count=0

    if keep_event_log:
        log_t=[t]; log_S=[int(state[0])]; log_R=[int(state[1])]; log_event=[-1]

    while t<final-1e-14:
        if event_count>=maximum_events:
            raise RuntimeError('maximum_events reached.')
        boundary=min(schedule.next_switch_after(t),final)
        prop=calculate_propensities(state,t,parameters,schedule)
        a0=float(prop.sum())

        if a0<=0:
            while index<len(grid) and grid[index]<=boundary+1e-14:
                sampled_S[index],sampled_R[index]=state; index+=1
            t=boundary
            if state_at_treatment is None and math.isfinite(first_on) and t>=first_on-1e-12:
                state_at_treatment=(int(state[0]),int(state[1]))
            if keep_event_log:
                log_t.append(t); log_S.append(int(state[0])); log_R.append(int(state[1])); log_event.append(-2)
            if t>=final-1e-14: break
            continue

        proposed=t+float(rng.exponential(1.0/a0))
        if proposed>=boundary-1e-14:
            while index<len(grid) and grid[index]<=boundary+1e-14:
                sampled_S[index],sampled_R[index]=state; index+=1
            t=boundary
            if state_at_treatment is None and math.isfinite(first_on) and t>=first_on-1e-12:
                state_at_treatment=(int(state[0]),int(state[1]))
            if keep_event_log:
                log_t.append(t); log_S.append(int(state[0])); log_R.append(int(state[1])); log_event.append(-2)
            if t>=final-1e-14: break
            continue

        while index<len(grid) and grid[index]<proposed-1e-14:
            sampled_S[index],sampled_R[index]=state; index+=1

        draw=float(rng.random()*a0)
        event_index=min(int(np.searchsorted(np.cumsum(prop),draw,side='right')),4)
        state=state+STOICHIOMETRIC_MATRIX[:,event_index]
        t=proposed; event_count+=1
        if np.any(state<0) or state.sum()>parameters.carrying_capacity:
            raise RuntimeError('Invalid state generated.')

        if math.isnan(emergence) and state[1]>0: emergence=t
        if math.isnan(establishment) and state[1]>=establishment_threshold: establishment=t
        if math.isnan(dominance) and state.sum()>0 and state[1]/state.sum()>=dominance_fraction: dominance=t
        if math.isnan(S_extinction) and state[0]==0: S_extinction=t
        if math.isnan(total_extinction) and state.sum()==0: total_extinction=t
        if keep_event_log:
            log_t.append(t); log_S.append(int(state[0])); log_R.append(int(state[1])); log_event.append(event_index)
        if state.sum()==0:
            while index<len(grid):
                sampled_S[index]=0; sampled_R[index]=0; index+=1
            break

    while index<len(grid):
        sampled_S[index],sampled_R[index]=state; index+=1
    if state_at_treatment is None:
        state_at_treatment=(-1,-1)

    final_S=int(sampled_S[-1]); final_R=int(sampled_R[-1])
    established=bool(np.isfinite(establishment)); survived=bool(final_S+final_R>0)
    rescue=bool(schedule.permanent_at_end and final_S==0 and final_R>0 and established)
    output={
        'time':grid.copy(),'sensitive':sampled_S,'resistant':sampled_R,'event_count':event_count,
        'emergence_time':emergence,'establishment_time':establishment,'dominance_time':dominance,
        'sensitive_extinction_time':S_extinction,'total_extinction_time':total_extinction,
        'sensitive_at_treatment':state_at_treatment[0],'resistant_at_treatment':state_at_treatment[1],
        'final_sensitive':final_S,'final_resistant':final_R,
        'mutant_present_at_treatment':bool(state_at_treatment[1]>0),'established':established,
        'resistant_dominance_reached':bool(np.isfinite(dominance)),'survived':survived,
        'extinct':not survived,'evolutionary_rescue':rescue,
    }
    if keep_event_log:
        output['event_log']=pd.DataFrame({'time':log_t,'sensitive':log_S,'resistant':log_R,'event_id':log_event})
    return output

# In[7]:


# =============================================================================
# OPTIONAL TEST — Run one Gillespie replicate
# =============================================================================
# This cell checks that the function defined in Cell 5 can generate one complete
# stochastic AMR trajectory.

test_schedule = SCHEDULES["early_continuous"]

test_output_grid = make_output_grid(
    final_time=CONFIG.short_final_time,
    points=CONFIG.short_output_points,
    schedule=test_schedule,
)

test_result = simulate_gillespie_replicate(
    parameters=PARAMETERS,
    schedule=test_schedule,
    output_grid=test_output_grid,
    seed=SEED + 1,
    keep_event_log=True,
)

print("Single Gillespie replicate completed successfully.")
print("-" * 60)
print("Number of recorded output times:", len(test_result["time"]))
print("Number of stochastic events:", test_result["event_count"])
print("Final sensitive population:", test_result["final_sensitive"])
print("Final resistant population:", test_result["final_resistant"])
print("Resistance emergence time:", test_result["emergence_time"])
print("Resistance establishment time:", test_result["establishment_time"])
print("Sensitive extinction time:", test_result["sensitive_extinction_time"])
print("Total extinction time:", test_result["total_extinction_time"])
print("Resistant population established:", test_result["established"])
print("Population survived:", test_result["survived"])
print("Population became extinct:", test_result["extinct"])
print("Evolutionary rescue occurred:", test_result["evolutionary_rescue"])

display(test_result["event_log"].head(10))

# In[8]:


# =============================================================================
# CELL 6 — Unit tests and event-level publication figure
# =============================================================================
# The same seed must reproduce the same path. Treatment switches must be present
# exactly in the output grid, and every population must remain integer,
# non-negative, and below capacity.

test_schedule=SCHEDULES['pulsed']
test_grid=make_output_grid(CONFIG.short_final_time,CONFIG.short_output_points,test_schedule)
a=simulate_gillespie_replicate(PARAMETERS,test_schedule,test_grid,12345)
b=simulate_gillespie_replicate(PARAMETERS,test_schedule,test_grid,12345)
c=simulate_gillespie_replicate(PARAMETERS,test_schedule,test_grid,54321)
assert np.array_equal(a['sensitive'],b['sensitive']) and np.array_equal(a['resistant'],b['resistant'])
assert not (np.array_equal(a['sensitive'],c['sensitive']) and np.array_equal(a['resistant'],c['resistant']))
assert np.all(a['sensitive']>=0) and np.all(a['resistant']>=0)
assert np.all(a['sensitive']+a['resistant']<=PARAMETERS.carrying_capacity)
for switch in test_schedule.switch_times: assert np.any(np.isclose(test_grid,switch))
print('All single-replicate tests passed.')

example_schedule=SCHEDULES['early_continuous']
example_grid=make_output_grid(CONFIG.short_final_time,CONFIG.short_output_points,example_schedule)
example=simulate_gillespie_replicate(PARAMETERS,example_schedule,example_grid,SEED+777,keep_event_log=True)
log=example['event_log'].copy(); log['total']=log['sensitive']+log['resistant']
fig,ax=plt.subplots(figsize=(8.8,5.5))
ax.step(log['time'],log['sensitive'],where='post',label=r'Sensitive, $S(t)$')
ax.step(log['time'],log['resistant'],where='post',label=r'Resistant, $R(t)$')
ax.step(log['time'],log['total'],where='post',label=r'Total, $S(t)+R(t)$')
shade_treatment(ax,example_schedule,CONFIG.short_final_time)
ax.set(xlabel='Time',ylabel='Integer bacterial count',title='Exact event-level Gillespie trajectory',xlim=(0,CONFIG.short_final_time),ylim=(0,None))
ax.grid(alpha=0.25); ax.legend(loc='best'); fig.tight_layout(); save_figure(fig,'round2_event_level_example'); plt.show()
log.to_csv(SUMMARY_DIR/'round2_event_level_example.csv',index=False)
print('Example event count:',example['event_count'])
print('Emergence time:',example['emergence_time'])
print('Establishment time:',example['establishment_time'])

# In[9]:


# =============================================================================
# CELL 7 — Ensemble runner and ensemble-statistics functions
# =============================================================================
# Independent child seeds are generated using SeedSequence. Full event logs are
# not retained for all replicates; regularly sampled arrays and replicate-level
# outcome summaries are saved.

def run_gillespie_ensemble(parameters,schedule,output_grid,number_of_replicates,base_seed,label,show_progress=True):
    children=np.random.SeedSequence(base_seed).spawn(number_of_replicates)
    seeds=[int(child.generate_state(1,dtype=np.uint32)[0]) for child in children]
    S=np.empty((number_of_replicates,len(output_grid)),dtype=np.int32)
    R=np.empty_like(S); rows=[]; start=time.perf_counter(); step=max(1,number_of_replicates//10)
    for i,seed in enumerate(seeds):
        result=simulate_gillespie_replicate(parameters,schedule,output_grid,seed)
        S[i]=result['sensitive']; R[i]=result['resistant']
        rows.append({'replicate':i,'seed':seed,'schedule':schedule.name,
                     **{key:result[key] for key in [
                         'event_count','emergence_time','establishment_time','dominance_time',
                         'sensitive_extinction_time','total_extinction_time','sensitive_at_treatment',
                         'resistant_at_treatment','final_sensitive','final_resistant',
                         'mutant_present_at_treatment','established','resistant_dominance_reached',
                         'survived','extinct','evolutionary_rescue']}})
        if show_progress and ((i+1)%step==0 or i+1==number_of_replicates):
            print(f'{label}: {i+1}/{number_of_replicates}')
    summary=pd.DataFrame(rows); elapsed=time.perf_counter()-start
    np.savez_compressed(ENSEMBLE_DIR/f'{label}.npz',time=output_grid,sensitive=S,resistant=R,seeds=np.asarray(seeds,dtype=np.uint32))
    summary.to_csv(SUMMARY_DIR/f'{label}_replicate_summary.csv',index=False)
    return {'label':label,'schedule':schedule,'time':output_grid,'sensitive':S,'resistant':R,'summary':summary,'elapsed_seconds':elapsed,'parameters':parameters}

def calculate_ensemble_statistics(ensemble):
    S=ensemble['sensitive'].astype(float); R=ensemble['resistant'].astype(float); N=S+R
    output={'time':ensemble['time']}
    for prefix,values in [('sensitive',S),('resistant',R),('total',N)]:
        mean=values.mean(axis=0); variance=values.var(axis=0,ddof=1)
        output.update({
            f'{prefix}_mean':mean,f'{prefix}_median':np.median(values,axis=0),
            f'{prefix}_q05':np.quantile(values,0.05,axis=0),f'{prefix}_q25':np.quantile(values,0.25,axis=0),
            f'{prefix}_q75':np.quantile(values,0.75,axis=0),f'{prefix}_q95':np.quantile(values,0.95,axis=0),
            f'{prefix}_variance':variance,
            f'{prefix}_fano':np.divide(variance,mean,out=np.zeros_like(variance),where=mean>0),
        })
    return pd.DataFrame(output)

# In[10]:


# =============================================================================
# OPTIONAL TEST — Verify the Cell 7 ensemble functions
# =============================================================================
# PURPOSE
# -------
# Run a very small three-replicate ensemble to confirm that:
# - independent seeds are generated;
# - every trajectory is completed;
# - replicate summaries are created;
# - ensemble statistics are calculated;
# - NPZ and CSV files are saved correctly.
#
# This is only a function test, not a scientific experiment.

test_schedule = SCHEDULES["early_continuous"]

test_grid = make_output_grid(
    final_time=CONFIG.short_final_time,
    points=101,
    schedule=test_schedule,
)

test_ensemble = run_gillespie_ensemble(
    parameters=PARAMETERS,
    schedule=test_schedule,
    output_grid=test_grid,
    number_of_replicates=3,
    base_seed=SEED + 700,
    label="test_cell7_ensemble",
    show_progress=True,
)

test_statistics = calculate_ensemble_statistics(
    test_ensemble
)

print("\nCell 7 ensemble test completed successfully.")
print("-" * 65)

print(
    "Sensitive-array shape:",
    test_ensemble["sensitive"].shape,
)

print(
    "Resistant-array shape:",
    test_ensemble["resistant"].shape,
)

print(
    "Number of summary rows:",
    len(test_ensemble["summary"]),
)

print(
    "Elapsed simulation time:",
    f"{test_ensemble['elapsed_seconds']:.3f} seconds",
)

print("\nReplicate-level summary:")
display(
    test_ensemble["summary"][
        [
            "replicate",
            "seed",
            "event_count",
            "final_sensitive",
            "final_resistant",
            "established",
            "survived",
            "extinct",
            "evolutionary_rescue",
        ]
    ]
)

print("\nFirst five ensemble-statistics rows:")
display(
    test_statistics.head()
)

# In[11]:


# =============================================================================
# CELL 8 — Short-horizon ensembles for untreated, early, late, and pulsed cases
# =============================================================================
# These t<=20 ensembles match Round 1 and support deterministic-versus-stochastic
# comparisons and later stochastic-data robustness tests.

SHORT_ENSEMBLES={}; SHORT_STATISTICS={}
for schedule_index,(name,schedule) in enumerate(SCHEDULES.items()):
    grid=make_output_grid(CONFIG.short_final_time,CONFIG.short_output_points,schedule)
    ensemble=run_gillespie_ensemble(PARAMETERS,schedule,grid,SHORT_REPLICATES,SEED+10_000*schedule_index,f'short_{name}')
    statistics=calculate_ensemble_statistics(ensemble)
    statistics.to_csv(SUMMARY_DIR/f'short_{name}_time_statistics.csv',index=False)
    SHORT_ENSEMBLES[name]=ensemble; SHORT_STATISTICS[name]=statistics

display(pd.DataFrame({
    'schedule':list(SHORT_ENSEMBLES),
    'replicates':[x['sensitive'].shape[0] for x in SHORT_ENSEMBLES.values()],
    'time_points':[x['sensitive'].shape[1] for x in SHORT_ENSEMBLES.values()],
    'elapsed_seconds':[x['elapsed_seconds'] for x in SHORT_ENSEMBLES.values()],
}))

# In[12]:


# =============================================================================
# CELL 9 — Ensemble audit, statistics, and deterministic comparison figures
# =============================================================================
# All stochastic states are audited separately from prediction error. For each
# schedule, the ODE solution is compared with the stochastic mean and 5–95%
# interval. Nonlinear dynamics mean exact equality is not expected.

audit_rows=[]; metric_rows=[]
for name,ensemble in SHORT_ENSEMBLES.items():
    S,R=ensemble['sensitive'],ensemble['resistant']; summary=ensemble['summary']
    checks={
        'nonnegative':bool(np.all(S>=0) and np.all(R>=0)),
        'capacity_respected':bool(np.all(S+R<=PARAMETERS.carrying_capacity)),
        'complete_accounting':bool(np.all(summary['survived'].astype(int)+summary['extinct'].astype(int)==1)),
        'complete_rows':bool(len(summary)==SHORT_REPLICATES),
    }
    assert all(checks.values())
    audit_rows.append({'schedule':name,'replicates':len(summary),'minimum_sensitive':int(S.min()),'minimum_resistant':int(R.min()),'maximum_total':int((S+R).max()),**checks,'all_checks_passed':all(checks.values())})

    stats=SHORT_STATISTICS[name]; deterministic=solve_deterministic_reference(PARAMETERS,ensemble['schedule'],ensemble['time'])
    for variable in ['sensitive','resistant','total']:
        mean=stats[f'{variable}_mean'].to_numpy(); exact=deterministic[variable].to_numpy(); error=mean-exact
        metric_rows.append({'schedule':name,'variable':variable,
                            'relative_l2_mean_vs_ode':np.linalg.norm(error)/(np.linalg.norm(exact)+1e-12),
                            'rmse_mean_vs_ode':float(np.sqrt(np.mean(error**2))),
                            'fraction_ode_inside_5_95_interval':float(np.mean((exact>=stats[f'{variable}_q05'])&(exact<=stats[f'{variable}_q95'])))})

    for variable,label in [('total','Total bacterial population'),('resistant','Resistant bacterial population')]:
        fig,ax=plt.subplots(figsize=(8.8,5.5))
        ax.fill_between(stats['time'],stats[f'{variable}_q05'],stats[f'{variable}_q95'],alpha=0.18,label='Stochastic 5–95% interval')
        ax.plot(stats['time'],stats[f'{variable}_mean'],label='Stochastic ensemble mean')
        ax.plot(deterministic['time'],deterministic[variable],linestyle='--',label='Deterministic mean-field solution')
        shade_treatment(ax,ensemble['schedule'],CONFIG.short_final_time)
        ax.set(xlabel='Time',ylabel=label,title=f'Deterministic and stochastic {variable}: {name.replace("_"," ")}',xlim=(0,CONFIG.short_final_time),ylim=(0,None))
        ax.grid(alpha=0.25); ax.legend(loc='best'); fig.tight_layout(); save_figure(fig,f'round2_{variable}_comparison_{name}'); plt.show()

SHORT_AUDIT=pd.DataFrame(audit_rows); SHORT_AUDIT.to_csv(TABLE_DIR/'round2_short_ensemble_audit.csv',index=False)
COMPARISON_METRICS=pd.DataFrame(metric_rows); COMPARISON_METRICS.to_csv(TABLE_DIR/'round2_deterministic_stochastic_metrics.csv',index=False)
display(SHORT_AUDIT); display(COMPARISON_METRICS)

# In[13]:


# =============================================================================
# CELL 10 — Demographic variability and Fano-factor figures
# =============================================================================
# Fano factor = variance/mean. Values above one indicate overdispersion relative
# to a Poisson-like count process.

for variable,label in [('sensitive','Sensitive-population Fano factor'),('resistant','Resistant-population Fano factor')]:
    fig,ax=plt.subplots(figsize=(8.8,5.5))
    for name,stats in SHORT_STATISTICS.items():
        ax.plot(stats['time'],stats[f'{variable}_fano'],label=name.replace('_',' '))
    ax.set(xlabel='Time',ylabel=label,title=f'Demographic variability of the {variable} population',xlim=(0,CONFIG.short_final_time),ylim=(0,None))
    ax.grid(alpha=0.25); ax.legend(loc='best'); fig.tight_layout(); save_figure(fig,f'round2_{variable}_fano_comparison'); plt.show()

# In[14]:


# =============================================================================
# CELL 11 — Extended-horizon ensembles for extinction and rescue
# =============================================================================
# t=20 is often too short for complete extinction. Permanent early and late
# treatment schedules are therefore simulated to t=120. Rescue is defined only
# for permanent treatment: final S=0, final R>0, and establishment was reached.

OUTCOME_ENSEMBLES={}
for schedule_index,name in enumerate(['early_continuous','late_continuous']):
    schedule=SCHEDULES[name]
    grid=make_output_grid(CONFIG.outcome_final_time,CONFIG.outcome_output_points,schedule)
    OUTCOME_ENSEMBLES[name]=run_gillespie_ensemble(PARAMETERS,schedule,grid,OUTCOME_REPLICATES,SEED+200_000+10_000*schedule_index,f'outcome_{name}')
print('Extended-outcome ensembles completed.')

# In[15]:


# =============================================================================
# CELL 12 — Outcome probabilities, Wilson intervals, and probability figures
# =============================================================================
# Wilson intervals remain well behaved when probabilities are close to zero or
# one. Separate figures are generated for each outcome.

def wilson_interval(successes,trials,z=1.959963984540054):
    if trials<=0: return math.nan,math.nan,math.nan
    p=successes/trials; denominator=1+z*z/trials
    center=(p+z*z/(2*trials))/denominator
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return p,max(0,center-half),min(1,center+half)

OUTCOME_COLUMNS={
    'mutant_present_at_treatment':'Resistance present at treatment',
    'established':'Resistant establishment',
    'resistant_dominance_reached':'Resistant dominance',
    'extinct':'Population extinction','survived':'Population survival',
    'evolutionary_rescue':'Evolutionary rescue',
}
rows=[]
for name,ensemble in OUTCOME_ENSEMBLES.items():
    summary=ensemble['summary']; n=len(summary)
    for column,label in OUTCOME_COLUMNS.items():
        successes=int(summary[column].sum()); p,lower,upper=wilson_interval(successes,n)
        rows.append({'schedule':name,'outcome':label,'successes':successes,'replicates':n,'probability':p,'ci95_lower':lower,'ci95_upper':upper})
OUTCOME_PROBABILITIES=pd.DataFrame(rows); OUTCOME_PROBABILITIES.to_csv(TABLE_DIR/'round2_outcome_probabilities.csv',index=False)
display(OUTCOME_PROBABILITIES)

for label in OUTCOME_COLUMNS.values():
    subset=OUTCOME_PROBABILITIES[OUTCOME_PROBABILITIES['outcome']==label]
    x=np.arange(len(subset)); lower=subset['probability']-subset['ci95_lower']; upper=subset['ci95_upper']-subset['probability']
    fig,ax=plt.subplots(figsize=(7.4,5.2))
    ax.errorbar(x,subset['probability'],yerr=np.vstack([lower,upper]),fmt='o',capsize=5,markersize=7)
    ax.set_xticks(x); ax.set_xticklabels(subset['schedule'].str.replace('_',' '))
    ax.set(ylabel='Estimated probability',title=f'{label} under continuous treatment',ylim=(0,1))
    ax.grid(axis='y',alpha=0.25); fig.tight_layout(); save_figure(fig,'round2_probability_'+label.lower().replace(' ','_')); plt.show()

# In[16]:


# =============================================================================
# CELL 13 — Cumulative-incidence figures for event times
# =============================================================================
# Replicates without an event are treated as right-censored at the final time.
# The plotted quantity is the fraction of all replicates that experienced the
# event by each time.

def cumulative_incidence(event_times,evaluation_times):
    event_times=np.asarray(event_times,dtype=float); finite=np.isfinite(event_times)
    return np.array([np.mean(finite&(event_times<=t)) for t in evaluation_times])

EVENT_COLUMNS={
    'emergence_time':'Resistance emergence',
    'establishment_time':'Resistant establishment',
    'sensitive_extinction_time':'Sensitive-lineage extinction',
    'total_extinction_time':'Total-population extinction',
}
evaluation_times=np.linspace(0,CONFIG.outcome_final_time,500)
for column,title in EVENT_COLUMNS.items():
    fig,ax=plt.subplots(figsize=(8.6,5.4))
    for name,ensemble in OUTCOME_ENSEMBLES.items():
        ax.step(evaluation_times,cumulative_incidence(ensemble['summary'][column],evaluation_times),where='post',label=name.replace('_',' '))
    ax.set(xlabel='Time',ylabel='Cumulative incidence',title=title,xlim=(0,CONFIG.outcome_final_time),ylim=(0,1))
    ax.grid(alpha=0.25); ax.legend(loc='best'); fig.tight_layout(); save_figure(fig,'round2_cumulative_'+column.replace('_time','')); plt.show()

# In[17]:


# =============================================================================
# CELL 14 — Terminal-distribution publication figures
# =============================================================================
# Violin plots show the distribution shape, mean, and median instead of only a
# point estimate.

names=list(OUTCOME_ENSEMBLES)
final_R=[OUTCOME_ENSEMBLES[name]['summary']['final_resistant'].to_numpy() for name in names]
fig,ax=plt.subplots(figsize=(7.8,5.5))
ax.violinplot(final_R,positions=np.arange(1,len(names)+1),showmeans=True,showmedians=True)
ax.set_xticks(np.arange(1,len(names)+1)); ax.set_xticklabels([x.replace('_',' ') for x in names])
ax.set(ylabel='Final resistant population',title='Distribution of terminal resistant population',ylim=(0,None))
ax.grid(axis='y',alpha=0.25); fig.tight_layout(); save_figure(fig,'round2_terminal_resistant_violin'); plt.show()

fractions=[]
for name in names:
    summary=OUTCOME_ENSEMBLES[name]['summary']; total=(summary['sensitive_at_treatment']+summary['resistant_at_treatment']).to_numpy(float)
    R=summary['resistant_at_treatment'].to_numpy(float)
    fractions.append(np.divide(R,total,out=np.zeros_like(R),where=total>0))
fig,ax=plt.subplots(figsize=(7.8,5.5))
ax.violinplot(fractions,positions=np.arange(1,len(names)+1),showmeans=True,showmedians=True)
ax.set_xticks(np.arange(1,len(names)+1)); ax.set_xticklabels([x.replace('_',' ') for x in names])
ax.set(ylabel='Resistant fraction at treatment',title='Pre-treatment resistance composition',ylim=(0,1))
ax.grid(axis='y',alpha=0.25); fig.tight_layout(); save_figure(fig,'round2_resistant_fraction_at_treatment_violin'); plt.show()

# In[18]:


# =============================================================================
# CELL 15 — Mutation-by-treatment-time rescue experiment and heat maps
# =============================================================================
# This experiment tests how rescue depends jointly on mutation probability and
# permanent-treatment timing. Only t=0, treatment time, and t=120 are sampled to
# reduce memory; first-passage outcomes are still checked after every event.

MUTATION_GRID=np.array([1e-5,3e-5,1e-4,3e-4,1e-3])
TREATMENT_TIME_GRID=np.array([2.5,5.0,7.5,10.0,12.5])
rows=[]; case=0; total_cases=len(MUTATION_GRID)*len(TREATMENT_TIME_GRID)
for mu in MUTATION_GRID:
    parameters=replace(PARAMETERS,mutation_probability=float(mu))
    for treatment_time in TREATMENT_TIME_GRID:
        case+=1
        schedule=PiecewiseTreatment(f'continuous_T{treatment_time:g}',(float(treatment_time),),(0.0,1.0),f'Permanent treatment from t={treatment_time}.')
        grid=np.array([0.0,float(treatment_time),CONFIG.outcome_final_time])
        children=np.random.SeedSequence(SEED+1_000_000+int(round(-math.log10(mu)*1000))+int(round(treatment_time*100))).spawn(RESCUE_GRID_REPLICATES)
        rescue=extinction=established=present=0
        for child in children:
            result=simulate_gillespie_replicate(parameters,schedule,grid,int(child.generate_state(1,dtype=np.uint32)[0]))
            rescue+=int(result['evolutionary_rescue']); extinction+=int(result['extinct']); established+=int(result['established']); present+=int(result['mutant_present_at_treatment'])
        p,lower,upper=wilson_interval(rescue,RESCUE_GRID_REPLICATES)
        rows.append({'mutation_probability':mu,'treatment_time':treatment_time,'replicates':RESCUE_GRID_REPLICATES,
                     'rescue_probability':p,'rescue_ci95_lower':lower,'rescue_ci95_upper':upper,
                     'extinction_probability':extinction/RESCUE_GRID_REPLICATES,
                     'establishment_probability':established/RESCUE_GRID_REPLICATES,
                     'mutant_at_treatment_probability':present/RESCUE_GRID_REPLICATES})
        print(f'Grid {case}/{total_cases}: mu={mu:.1e}, T={treatment_time:g}, P_rescue={p:.3f}')
RESCUE_GRID_RESULTS=pd.DataFrame(rows); RESCUE_GRID_RESULTS.to_csv(TABLE_DIR/'round2_rescue_parameter_grid.csv',index=False)

def probability_heatmap(column,title,label,stem):
    pivot=RESCUE_GRID_RESULTS.pivot(index='mutation_probability',columns='treatment_time',values=column)
    fig,ax=plt.subplots(figsize=(8.0,5.8))
    image=ax.imshow(pivot.to_numpy(),origin='lower',aspect='auto',extent=[pivot.columns.min(),pivot.columns.max(),np.log10(pivot.index.min()),np.log10(pivot.index.max())],vmin=0,vmax=1)
    ax.set(xlabel='Permanent-treatment introduction time',ylabel=r'$\log_{10}(\mu)$',title=title)
    colorbar=fig.colorbar(image,ax=ax); colorbar.set_label(label)
    fig.tight_layout(); save_figure(fig,stem); plt.show()

probability_heatmap('rescue_probability','Evolutionary rescue across mutation and treatment timing','Rescue probability','round2_rescue_probability_heatmap')
probability_heatmap('extinction_probability','Population extinction across mutation and treatment timing','Extinction probability','round2_extinction_probability_heatmap')
probability_heatmap('mutant_at_treatment_probability','Resistance present when treatment begins','Probability','round2_mutant_at_treatment_heatmap')

# In[19]:


# =============================================================================
# CELL 16 — Monte Carlo convergence of rescue probability
# =============================================================================
# The cumulative estimate and Wilson interval are plotted as replicate count
# grows. Stable estimates and narrowing intervals support the selected budget.

indicators=OUTCOME_ENSEMBLES['early_continuous']['summary']['evolutionary_rescue'].astype(int).to_numpy()
sample_sizes=np.unique(np.linspace(10,len(indicators),min(60,max(2,len(indicators)-9)),dtype=int))
rows=[]
for n in sample_sizes:
    p,lower,upper=wilson_interval(int(indicators[:n].sum()),int(n))
    rows.append({'sample_size':n,'probability':p,'ci95_lower':lower,'ci95_upper':upper})
MONTE_CARLO_CONVERGENCE=pd.DataFrame(rows); MONTE_CARLO_CONVERGENCE.to_csv(TABLE_DIR/'round2_monte_carlo_convergence.csv',index=False)
fig,ax=plt.subplots(figsize=(8.4,5.4))
ax.fill_between(MONTE_CARLO_CONVERGENCE['sample_size'],MONTE_CARLO_CONVERGENCE['ci95_lower'],MONTE_CARLO_CONVERGENCE['ci95_upper'],alpha=0.18,label='95% Wilson interval')
ax.plot(MONTE_CARLO_CONVERGENCE['sample_size'],MONTE_CARLO_CONVERGENCE['probability'],label='Cumulative rescue estimate')
ax.set(xlabel='Number of stochastic replicates',ylabel='Estimated rescue probability',title='Monte Carlo convergence of evolutionary-rescue probability',ylim=(0,1))
ax.grid(alpha=0.25); ax.legend(loc='best'); fig.tight_layout(); save_figure(fig,'round2_rescue_monte_carlo_convergence'); plt.show()

# In[21]:


# =============================================================================
# CELL 17 — Export a stochastic sparse-data benchmark for later PINN tests
# =============================================================================
# The main PINN experiments remain based on deterministic ground truth. This
# export is a separate robustness benchmark containing 20 stochastic cultures,
# 21 total-load measurements, and phenotype measurements at t={7,12,18}.

number_to_export=min(20,SHORT_REPLICATES)
total_times=np.linspace(0,CONFIG.short_final_time,21)
phenotype_times=np.array([7.0,12.0,18.0])
rows=[]
for name,ensemble in SHORT_ENSEMBLES.items():
    grid=ensemble['time']; schedule=ensemble['schedule']
    total_indexes=[int(np.argmin(np.abs(grid-t))) for t in total_times]
    phenotype_indexes={int(np.argmin(np.abs(grid-t))) for t in phenotype_times}
    for replicate in range(number_to_export):
        S=ensemble['sensitive'][replicate]; R=ensemble['resistant'][replicate]
        for index in total_indexes:
            total=int(S[index]+R[index]); fraction=float(R[index]/total) if total>0 else 0.0
            rows.append({'schedule':name,'replicate':replicate,'time':float(grid[index]),
                         'sensitive_hidden':int(S[index]),'resistant_hidden':int(R[index]),
                         'total_observed':total,
                         'resistant_fraction_observed':fraction if index in phenotype_indexes else math.nan,
                         'phenotype_observed':bool(index in phenotype_indexes),
                         'antibiotic':float(schedule.exposure(grid[index]))})
STOCHASTIC_PINN_EXPORT=pd.DataFrame(rows)
STOCHASTIC_PINN_EXPORT.to_csv(PINN_EXPORT_DIR/'round2_stochastic_sparse_pinn_benchmark.csv',index=False)
display(STOCHASTIC_PINN_EXPORT.head(25))

# In[23]:


# =============================================================================
# CELL 18 — LaTeX export, manifest, file inventory, and final audit
# =============================================================================
# QUICK_MODE results are for code verification. Set QUICK_MODE=False and rerun
# before inserting probabilities into the manuscript.

latex=OUTCOME_PROBABILITIES.copy(); latex['schedule']=latex['schedule'].str.replace('_',' ')
latex['Probability [95\\% CI]']=latex.apply(lambda row:f"{row['probability']:.3f} [{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}]",axis=1)
latex_code=latex[['schedule','outcome','Probability [95\\% CI]']].rename(columns={'schedule':'Treatment','outcome':'Outcome'}).to_latex(index=False,escape=False,caption='Stochastic AMR outcomes under continuous treatment with 95\\% Wilson intervals.',label='tab:stochastic_outcomes')
(TABLE_DIR/'round2_stochastic_outcomes.tex').write_text(latex_code,encoding='utf-8')

manifest={
    'seed':SEED,'quick_mode':QUICK_MODE,'short_replicates':SHORT_REPLICATES,
    'outcome_replicates':OUTCOME_REPLICATES,'rescue_grid_replicates':RESCUE_GRID_REPLICATES,
    'parameters':asdict(PARAMETERS),'resistant_birth_rate':PARAMETERS.resistant_birth_rate,
    'config':asdict(CONFIG),
    'schedules':{name:{'switch_times':s.switch_times,'levels':s.levels,'description':s.description} for name,s in SCHEDULES.items()},
    'outcome_definitions':{
        'emergence':'first R>=1','establishment':f'first R>={CONFIG.establishment_threshold}',
        'dominance':f'first R/(S+R)>={CONFIG.dominance_fraction}','extinction':'S=R=0',
        'evolutionary_rescue':'permanent treatment, final S=0, final R>0, establishment reached',
    },
}
with open(REPORT_DIR/'round2_manifest.json','w') as file: json.dump(manifest,file,indent=2)

required=[TABLE_DIR/'round2_short_ensemble_audit.csv',TABLE_DIR/'round2_deterministic_stochastic_metrics.csv',TABLE_DIR/'round2_outcome_probabilities.csv',TABLE_DIR/'round2_rescue_parameter_grid.csv',TABLE_DIR/'round2_monte_carlo_convergence.csv',TABLE_DIR/'round2_stochastic_outcomes.tex',REPORT_DIR/'round2_manifest.json',PINN_EXPORT_DIR/'round2_stochastic_sparse_pinn_benchmark.csv']
missing=[str(path) for path in required if not path.exists()]
if missing: raise FileNotFoundError(missing)
assert SHORT_AUDIT['all_checks_passed'].all()
assert len(RESCUE_GRID_RESULTS)==len(MUTATION_GRID)*len(TREATMENT_TIME_GRID)

inventory=sorted(str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.rglob('*') if path.is_file())
(REPORT_DIR/'round2_file_inventory.txt').write_text('\n'.join(inventory),encoding='utf-8')
print('Round 2 completed successfully.')
print('QUICK_MODE:',QUICK_MODE)
print('Ensemble files:',len(list(ENSEMBLE_DIR.glob('*.npz'))))
print('Summary CSV files:',len(list(SUMMARY_DIR.glob('*.csv'))))
print('PINN benchmark files:',len(list(PINN_EXPORT_DIR.glob('*.csv'))))
print('Figure files:',len(list(FIGURE_DIR.glob('*'))))
print('Table files:',len(list(TABLE_DIR.glob('*'))))
print('Report files:',len(list(REPORT_DIR.glob('*'))))
if QUICK_MODE:
    print("\nPreliminary run completed. Set QUICK_MODE=False for manuscript-scale results.")
else:
    print("\nManuscript-scale final run completed successfully.")
