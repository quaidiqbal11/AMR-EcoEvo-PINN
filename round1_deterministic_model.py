#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# CELL 1 — Environment, reproducibility, project folders, and GPU information
# =============================================================================
# PURPOSE
# -------
# 1. Fix random seeds so that generated noise and later experiments are
#    reproducible.
# 2. Detect the available hardware, including the GPU.
# 3. Create an organized project structure for datasets, figures, tables,
#    and validation reports.
#
# IMPORTANT
# ---------
# Round 1 uses SciPy's solve_ivp function, which runs mainly on the CPU.
# Your 8 GB GPU will become important when the PINNs are trained in later
# experimental rounds.

from pathlib import Path
import os
import sys
import platform
import random
import json
import warnings

import numpy as np

# Reproducibility seed
SEED = 20260803

random.seed(SEED)
np.random.seed(SEED)

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# Determine the project root.
# If this notebook is inside a folder called "notebooks", the project root
# is assumed to be its parent folder.
# -------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

# -------------------------------------------------------------------------
# Create the project folders.
# -------------------------------------------------------------------------
DATA_DENSE = PROJECT_ROOT / "data" / "dense"
DATA_SPARSE = PROJECT_ROOT / "data" / "sparse"
DATA_NOISY = PROJECT_ROOT / "data" / "noisy"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

for directory in [
    DATA_DENSE,
    DATA_SPARSE,
    DATA_NOISY,
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Display the computational environment.
# -------------------------------------------------------------------------
print("Project root:", PROJECT_ROOT)
print("Python version:", sys.version.split()[0])
print("Operating system:", platform.platform())
print("Random seed:", SEED)

# -------------------------------------------------------------------------
# Detect PyTorch and GPU information.
# Round 1 does not require the GPU, but this confirms that the system is
# ready for the later PINN experiments.
# -------------------------------------------------------------------------
try:
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        total_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )

        print(f"GPU memory: {total_memory_gb:.2f} GB")

except Exception as exc:
    print("PyTorch/GPU information could not be obtained:", exc)

# In[3]:


# =============================================================================
# CELL 2 — Scientific imports and publication-output settings
# =============================================================================
# PURPOSE
# -------
# This cell imports all scientific libraries needed for:
#
# 1. Solving the deterministic AMR differential equations;
# 2. Storing and processing simulation results;
# 3. Interpolating numerical trajectories;
# 4. Conducting parameter-sensitivity analysis;
# 5. Generating high-resolution publication figures;
# 6. Saving figures in PNG, PDF, and SVG formats.
#
# IMPORTANT
# ---------
# This cell does not solve the AMR model yet.
# It only prepares the scientific environment and defines a reusable
# figure-saving function.
#
# Therefore, a cell containing only imports and function definitions normally
# produces no output. Confirmation messages are included at the end of this
# version so that successful execution is visible.

# -------------------------------------------------------------------------
# Python utilities
# -------------------------------------------------------------------------
from dataclasses import dataclass, asdict, replace
from typing import Callable, Dict, Tuple, Iterable
from pathlib import Path

# -------------------------------------------------------------------------
# Numerical and tabular computing
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Scientific plotting
# -------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Differential-equation solver and numerical interpolation
# -------------------------------------------------------------------------
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import linregress


# =============================================================================
# CHECK THAT CELL 1 HAS ALREADY BEEN EXECUTED
# =============================================================================
# Cell 1 creates FIGURE_DIR. This test prevents later errors caused by running
# the notebook cells in the wrong order.

if "FIGURE_DIR" not in globals():
    raise RuntimeError(
        "FIGURE_DIR has not been defined. "
        "Please run Cell 1 before running Cell 2."
    )

# Ensure that the figure folder exists.
FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# PUBLICATION-QUALITY MATPLOTLIB SETTINGS
# =============================================================================
# These settings are applied to all figures generated later in the notebook.
#
# The figures will use:
# - readable font sizes;
# - clear axis lines;
# - high-resolution output;
# - sufficiently thick curves;
# - framed legends;
# - editable vector-format export.
#
# Matplotlib's default color sequence is retained. Individual publication
# figures can later be refined according to the target journal's requirements.

plt.rcParams.update(
    {
        # Display resolution inside Jupyter Notebook
        "figure.dpi": 120,

        # Resolution used when raster figures are saved
        "savefig.dpi": 300,

        # Default figure size
        "figure.figsize": (8.6, 5.4),

        # Font settings
        "font.family": "sans-serif",
        "font.size": 11,

        # Axis-label and title sizes
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",

        # Tick-label sizes
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Axis-line properties
        "axes.linewidth": 1.0,

        # Curve properties
        "lines.linewidth": 2.0,

        # Legend properties
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 1.0,

        # Saved-figure settings
        "savefig.bbox": "tight",

        # Prevent minus signs from being shown incorrectly
        "axes.unicode_minus": False,
    }
)


# =============================================================================
# REUSABLE FIGURE-SAVING FUNCTION
# =============================================================================

def save_figure(
    fig: plt.Figure,
    stem: str,
    close_after_saving: bool = False,
) -> None:
    """
    Save one publication figure in PNG, PDF, and SVG formats.

    Parameters
    ----------
    fig:
        Matplotlib figure object to be saved.

    stem:
        Output filename without a file extension.

        Example
        -------
        stem = "figure_total_population"

        This produces:
        - figure_total_population.png
        - figure_total_population.pdf
        - figure_total_population.svg

    close_after_saving:
        When True, close the figure after saving it. This is useful when many
        figures are generated inside a loop because it prevents unnecessary
        memory use.

        The default is False so that figures remain visible in Jupyter.

    Output formats
    --------------
    PNG:
        High-resolution 300-dpi raster image suitable for previews,
        presentations, and many manuscript-submission systems.

    PDF:
        Vector format suitable for LaTeX manuscripts and publication.

    SVG:
        Editable vector format suitable for Adobe Illustrator, Inkscape,
        PowerPoint, and further graphical refinement.
    """

    # ---------------------------------------------------------------------
    # Validate the inputs.
    # ---------------------------------------------------------------------
    if not isinstance(fig, plt.Figure):
        raise TypeError(
            "fig must be a Matplotlib Figure object."
        )

    if not isinstance(stem, str):
        raise TypeError(
            "stem must be a string."
        )

    if not stem.strip():
        raise ValueError(
            "stem cannot be empty."
        )

    # Remove spaces at the beginning and end.
    stem = stem.strip()

    # Replace internal spaces with underscores to create safe filenames.
    safe_stem = stem.replace(" ", "_")

    # ---------------------------------------------------------------------
    # Define the three output paths.
    # ---------------------------------------------------------------------
    png_path = FIGURE_DIR / f"{safe_stem}.png"
    pdf_path = FIGURE_DIR / f"{safe_stem}.pdf"
    svg_path = FIGURE_DIR / f"{safe_stem}.svg"

    # ---------------------------------------------------------------------
    # Save high-resolution PNG.
    # ---------------------------------------------------------------------
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    # ---------------------------------------------------------------------
    # Save vector PDF.
    # ---------------------------------------------------------------------
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    # ---------------------------------------------------------------------
    # Save editable vector SVG.
    # ---------------------------------------------------------------------
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    # Optionally close the figure to release memory.
    if close_after_saving:
        plt.close(fig)


# =============================================================================
# EXECUTION CONFIRMATION
# =============================================================================
# These print statements confirm that:
# - all imports worked;
# - Matplotlib was configured;
# - the saving function was defined;
# - the output folder is available.

print("Cell 2 completed successfully.")
print("-" * 60)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("SciPy version:", scipy.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Publication figure settings have been applied.")
print("The save_figure() function has been defined.")
print("Figure output folder:", FIGURE_DIR.resolve())

# In[4]:


# =============================================================================
# CELL 3 — Biological parameters and numerical configuration
# =============================================================================
# PURPOSE
# -------
# Store the biological and numerical parameters in validated dataclasses.
#
# WHY THIS IS IMPORTANT
# ---------------------
# Using validated parameter containers prevents accidental use of impossible
# values such as:
# - negative carrying capacity;
# - mutation probability greater than one;
# - negative death rates;
# - initial population above carrying capacity.
#
# The baseline mutation probability is initially set to 1e-4 because it
# produces a visible resistant trajectory during debugging. Later experiments
# will also examine 1e-5 and 1e-3.

@dataclass(frozen=True)
class AMRParameters:
    """Biological parameters of the sensitive-resistant AMR model."""

    carrying_capacity: float = 1000.0
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
        Calculate the resistant birth rate after accounting for fitness cost.

        r_R = r_S(1-c_R)
        """

        return self.sensitive_birth_rate * (
            1.0 - self.resistance_fitness_cost
        )

    def validate(self) -> None:
        """Check that all biological parameters are admissible."""

        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive.")

        if self.sensitive_birth_rate <= 0:
            raise ValueError("sensitive_birth_rate must be positive.")

        if self.sensitive_death_rate < 0:
            raise ValueError("sensitive_death_rate cannot be negative.")

        if self.resistant_death_rate < 0:
            raise ValueError("resistant_death_rate cannot be negative.")

        bounded_parameters = {
            "resistance_fitness_cost": self.resistance_fitness_cost,
            "mutation_probability": self.mutation_probability,
            "sensitive_drug_effect": self.sensitive_drug_effect,
            "resistant_drug_effect": self.resistant_drug_effect,
        }

        for name, value in bounded_parameters.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie between zero and one."
                )

        if self.resistant_birth_rate < 0:
            raise ValueError(
                "The calculated resistant birth rate cannot be negative."
            )

    def as_dictionary(self) -> Dict[str, float]:
        """Return all parameters as a dictionary."""

        result = asdict(self)
        result["resistant_birth_rate"] = self.resistant_birth_rate

        return result


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical configuration for deterministic data generation."""

    initial_sensitive: float = 100.0
    initial_resistant: float = 0.0

    start_time: float = 0.0
    final_time: float = 20.0

    dense_points: int = 1001
    sparse_total_points: int = 21

    relative_tolerance: float = 1.0e-9
    absolute_tolerance: float = 1.0e-11

    solver_method: str = "DOP853"

    def validate(self, carrying_capacity: float) -> None:
        """Check that the numerical configuration is admissible."""

        if self.initial_sensitive < 0:
            raise ValueError(
                "Initial sensitive population cannot be negative."
            )

        if self.initial_resistant < 0:
            raise ValueError(
                "Initial resistant population cannot be negative."
            )

        initial_total = (
            self.initial_sensitive + self.initial_resistant
        )

        if initial_total > carrying_capacity:
            raise ValueError(
                "Initial total population exceeds carrying capacity."
            )

        if self.final_time <= self.start_time:
            raise ValueError(
                "final_time must be greater than start_time."
            )

        if self.dense_points < 2:
            raise ValueError(
                "dense_points must contain at least two points."
            )

        if self.sparse_total_points < 2:
            raise ValueError(
                "sparse_total_points must contain at least two points."
            )


# -------------------------------------------------------------------------
# Instantiate and validate the baseline configurations.
# -------------------------------------------------------------------------
PARAMETERS = AMRParameters()
CONFIG = SimulationConfig()

PARAMETERS.validate()
CONFIG.validate(PARAMETERS.carrying_capacity)

# -------------------------------------------------------------------------
# Display a parameter table.
# -------------------------------------------------------------------------
parameter_dictionary = PARAMETERS.as_dictionary()

parameter_table = pd.DataFrame(
    {
        "Parameter": list(parameter_dictionary.keys()),
        "Value": list(parameter_dictionary.values()),
    }
)

parameter_table

# In[5]:


# =============================================================================
# CELL 4 — Prescribed antibiotic-treatment schedules
# =============================================================================
# PURPOSE
# -------
# Define the external antibiotic input A(t) for four experiments:
#
# 1. Untreated control;
# 2. Early continuous treatment;
# 3. Late continuous treatment;
# 4. Pulsed treatment.
#
# These schedules are prescribed rather than optimized. This is deliberate:
# Control-PINNs and reinforcement-learning antibiotic policies already exist.
# The first paper focuses on state and parameter inference.

TreatmentFunction = Callable[
    [float | np.ndarray],
    float | np.ndarray,
]


@dataclass(frozen=True)
class TreatmentSchedule:
    """Named normalized antibiotic-exposure function."""

    name: str
    function: TreatmentFunction
    description: str

    def __call__(
        self,
        t: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Evaluate the treatment schedule and ensure 0 <= A(t) <= 1.
        """

        values = np.asarray(
            self.function(t),
            dtype=float,
        )

        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"{self.name} produced non-finite values."
            )

        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(
                f"{self.name} must remain inside [0,1]."
            )

        if np.isscalar(t):
            return float(values)

        return values


def untreated_schedule() -> TreatmentSchedule:
    """No antibiotic exposure."""

    return TreatmentSchedule(
        name="untreated",
        function=lambda t: np.zeros_like(
            np.asarray(t, dtype=float)
        ),
        description="No antibiotic exposure.",
    )


def early_continuous_schedule(
    start: float = 5.0,
) -> TreatmentSchedule:
    """Continuous antibiotic exposure beginning at t=start."""

    return TreatmentSchedule(
        name="early_continuous",
        function=lambda t: (
            np.asarray(t, dtype=float) >= start
        ).astype(float),
        description=(
            f"Continuous antibiotic exposure beginning at t={start}."
        ),
    )


def late_continuous_schedule(
    start: float = 10.0,
) -> TreatmentSchedule:
    """Continuous antibiotic exposure beginning later."""

    return TreatmentSchedule(
        name="late_continuous",
        function=lambda t: (
            np.asarray(t, dtype=float) >= start
        ).astype(float),
        description=(
            f"Continuous antibiotic exposure beginning at t={start}."
        ),
    )


def pulsed_schedule(
    first_on: float = 5.0,
    first_off: float = 8.0,
    second_on: float = 12.0,
    second_off: float = 15.0,
) -> TreatmentSchedule:
    """Two finite antibiotic pulses."""

    def exposure(
        t: float | np.ndarray,
    ) -> np.ndarray:

        t_array = np.asarray(t, dtype=float)

        first_pulse = (
            (t_array >= first_on)
            & (t_array < first_off)
        )

        second_pulse = (
            (t_array >= second_on)
            & (t_array < second_off)
        )

        return (
            first_pulse | second_pulse
        ).astype(float)

    return TreatmentSchedule(
        name="pulsed",
        function=exposure,
        description=(
            f"Exposure on [{first_on},{first_off}) and "
            f"[{second_on},{second_off})."
        ),
    )


# -------------------------------------------------------------------------
# Store all schedules in one dictionary.
# -------------------------------------------------------------------------
SCHEDULES = {
    schedule.name: schedule
    for schedule in [
        untreated_schedule(),
        early_continuous_schedule(),
        late_continuous_schedule(),
        pulsed_schedule(),
    ]
}

schedule_table = pd.DataFrame(
    {
        "Schedule": list(SCHEDULES.keys()),
        "Description": [
            schedule.description
            for schedule in SCHEDULES.values()
        ],
    }
)

schedule_table

# In[6]:


# =============================================================================
# CELL 5 — Deterministic AMR differential equations
# =============================================================================
# PURPOSE
# -------
# Implement the mean-field sensitive-resistant bacterial model.
#
# BIOLOGICAL INTERPRETATION
# -------------------------
# 1. Sensitive division without mutation increases S.
# 2. A mutated sensitive division creates a resistant offspring.
# 3. The sensitive parent remains present during the mutation event.
# 4. Resistant bacteria reproduce with a fitness cost.
# 5. Antibiotic exposure suppresses sensitive and resistant birth rates by
#    phenotype-specific amounts.
# 6. Both populations experience logistic crowding.
#
# No artificial clipping is performed inside the ODE because clipping would
# alter the mathematical system. Biological feasibility is tested separately.

def amr_rhs(
    t: float,
    state: np.ndarray,
    parameters: AMRParameters,
    schedule: TreatmentSchedule,
) -> np.ndarray:
    """
    Calculate dS/dt and dR/dt.

    Parameters
    ----------
    t:
        Current time.

    state:
        Array containing [S, R].

    parameters:
        Biological parameter object.

    schedule:
        Antibiotic-exposure function A(t).
    """

    sensitive, resistant = state

    total_population = sensitive + resistant

    antibiotic_exposure = float(schedule(t))

    # Drug-modified sensitive birth rate
    sensitive_birth_rate = (
        parameters.sensitive_birth_rate
        * (
            1.0
            - parameters.sensitive_drug_effect
            * antibiotic_exposure
        )
    )

    # Drug-modified resistant birth rate
    resistant_birth_rate = (
        parameters.resistant_birth_rate
        * (
            1.0
            - parameters.resistant_drug_effect
            * antibiotic_exposure
        )
    )

    # Logistic crowding factor
    crowding = (
        1.0
        - total_population
        / parameters.carrying_capacity
    )

    # Sensitive population equation
    d_sensitive = (
        sensitive_birth_rate
        * (1.0 - parameters.mutation_probability)
        * crowding
        * sensitive
        - parameters.sensitive_death_rate
        * sensitive
    )

    # Resistant population equation
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
        [d_sensitive, d_resistant],
        dtype=float,
    )


# -------------------------------------------------------------------------
# Inspect the initial derivatives for each schedule.
# At t=0 the treatment schedules are inactive, so the initial derivatives
# should be the same for all four experiments.
# -------------------------------------------------------------------------
initial_state = np.array(
    [
        CONFIG.initial_sensitive,
        CONFIG.initial_resistant,
    ],
    dtype=float,
)

for schedule_name, schedule in SCHEDULES.items():

    derivative = amr_rhs(
        CONFIG.start_time,
        initial_state,
        PARAMETERS,
        schedule,
    )

    print(
        f"{schedule_name:>18s}: "
        f"dS/dt={derivative[0]: .6f}, "
        f"dR/dt={derivative[1]: .6f}"
    )

# In[7]:


# =============================================================================
# CELL 6 — High-accuracy deterministic ODE solver
# =============================================================================
# PURPOSE
# -------
# Solve the AMR model using SciPy's adaptive DOP853 solver.
#
# OUTPUT
# ------
# A tidy dataframe containing:
# - time;
# - sensitive population;
# - resistant population;
# - total population;
# - resistant fraction;
# - antibiotic exposure;
# - treatment-schedule name.
#
# These dense trajectories will become the trusted ground truth for later PINN
# training and evaluation.

def solve_deterministic(
    parameters: AMRParameters,
    config: SimulationConfig,
    schedule: TreatmentSchedule,
) -> pd.DataFrame:
    """Solve the deterministic AMR model."""

    parameters.validate()
    config.validate(parameters.carrying_capacity)

    time_grid = np.linspace(
        config.start_time,
        config.final_time,
        config.dense_points,
        dtype=float,
    )

    solution = solve_ivp(
        fun=lambda t, state: amr_rhs(
            t,
            state,
            parameters,
            schedule,
        ),
        t_span=(
            config.start_time,
            config.final_time,
        ),
        y0=np.array(
            [
                config.initial_sensitive,
                config.initial_resistant,
            ],
            dtype=float,
        ),
        t_eval=time_grid,
        method=config.solver_method,
        rtol=config.relative_tolerance,
        atol=config.absolute_tolerance,
    )

    if not solution.success:
        raise RuntimeError(
            f"Integration failed for {schedule.name}: "
            f"{solution.message}"
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

    antibiotic = np.asarray(
        schedule(solution.t),
        dtype=float,
    )

    results = pd.DataFrame(
        {
            "time": solution.t,
            "sensitive": sensitive,
            "resistant": resistant,
            "total": total,
            "resistant_fraction": resistant_fraction,
            "antibiotic": antibiotic,
            "schedule": schedule.name,
        }
    )

    return results

# In[8]:


# =============================================================================
# CELL 7 — Biological and numerical validation
# =============================================================================
# PURPOSE
# -------
# Explicitly test whether every deterministic trajectory satisfies:
#
#     S(t) >= 0
#     R(t) >= 0
#     S(t) + R(t) <= K
#     0 <= R/(S+R) <= 1
#
# It also checks:
# - absence of NaN and infinite values;
# - correct initial conditions.
#
# This follows the important lesson from PINN-Proj and KKT-hPINN:
# prediction accuracy and constraint satisfaction should be measured
# independently.

def validate_trajectory(
    data: pd.DataFrame,
    parameters: AMRParameters,
    config: SimulationConfig,
    tolerance: float = 1.0e-6,
) -> Dict[str, float | bool]:
    """Validate one deterministic AMR trajectory."""

    numeric_columns = [
        "time",
        "sensitive",
        "resistant",
        "total",
        "resistant_fraction",
        "antibiotic",
    ]

    numeric_values = data[
        numeric_columns
    ].to_numpy(dtype=float)

    finite_values = bool(
        np.all(np.isfinite(numeric_values))
    )

    minimum_sensitive = float(
        data["sensitive"].min()
    )

    minimum_resistant = float(
        data["resistant"].min()
    )

    maximum_total = float(
        data["total"].max()
    )

    minimum_fraction = float(
        data["resistant_fraction"].min()
    )

    maximum_fraction = float(
        data["resistant_fraction"].max()
    )

    initial_sensitive_error = abs(
        float(data.iloc[0]["sensitive"])
        - config.initial_sensitive
    )

    initial_resistant_error = abs(
        float(data.iloc[0]["resistant"])
        - config.initial_resistant
    )

    report = {
        "finite_values": finite_values,
        "minimum_sensitive": minimum_sensitive,
        "minimum_resistant": minimum_resistant,
        "maximum_total": maximum_total,
        "minimum_resistant_fraction": minimum_fraction,
        "maximum_resistant_fraction": maximum_fraction,
        "initial_sensitive_error": initial_sensitive_error,
        "initial_resistant_error": initial_resistant_error,
        "nonnegative_populations": (
            minimum_sensitive >= -tolerance
            and minimum_resistant >= -tolerance
        ),
        "capacity_respected": (
            maximum_total
            <= parameters.carrying_capacity
            + tolerance
        ),
        "fraction_in_unit_interval": (
            minimum_fraction >= -tolerance
            and maximum_fraction <= 1.0 + tolerance
        ),
        "initial_conditions_respected": (
            initial_sensitive_error <= tolerance
            and initial_resistant_error <= tolerance
        ),
    }

    report["all_checks_passed"] = bool(
        report["finite_values"]
        and report["nonnegative_populations"]
        and report["capacity_respected"]
        and report["fraction_in_unit_interval"]
        and report["initial_conditions_respected"]
    )

    return report

# In[9]:


# =============================================================================
# CELL 8 — Run all four deterministic treatment scenarios
# =============================================================================
# PURPOSE
# -------
# Generate the dense reference trajectories for:
# 1. untreated control;
# 2. early continuous treatment;
# 3. late continuous treatment;
# 4. pulsed treatment.
#
# Every trajectory is validated and saved as a CSV file.

TRAJECTORIES: Dict[str, pd.DataFrame] = {}

VALIDATION_REPORTS: Dict[
    str,
    Dict[str, float | bool],
] = {}

summary_rows = []

for schedule_name, schedule in SCHEDULES.items():

    trajectory = solve_deterministic(
        PARAMETERS,
        CONFIG,
        schedule,
    )

    validation = validate_trajectory(
        trajectory,
        PARAMETERS,
        CONFIG,
    )

    if not validation["all_checks_passed"]:
        raise RuntimeError(
            f"Validation failed for {schedule_name}."
        )

    TRAJECTORIES[schedule_name] = trajectory
    VALIDATION_REPORTS[schedule_name] = validation

    # Save the complete dense trajectory
    trajectory.to_csv(
        DATA_DENSE
        / f"{schedule_name}_dense.csv",
        index=False,
    )

    summary_rows.append(
        {
            "schedule": schedule_name,
            "final_sensitive": float(
                trajectory.iloc[-1]["sensitive"]
            ),
            "final_resistant": float(
                trajectory.iloc[-1]["resistant"]
            ),
            "final_total": float(
                trajectory.iloc[-1]["total"]
            ),
            "final_resistant_fraction": float(
                trajectory.iloc[-1][
                    "resistant_fraction"
                ]
            ),
            "peak_total": float(
                trajectory["total"].max()
            ),
            "minimum_total": float(
                trajectory["total"].min()
            ),
            "all_checks_passed": bool(
                validation["all_checks_passed"]
            ),
        }
    )


SUMMARY = pd.DataFrame(summary_rows)

SUMMARY.to_csv(
    TABLE_DIR
    / "round1_deterministic_summary.csv",
    index=False,
)

with open(
    REPORT_DIR
    / "round1_validation_reports.json",
    "w",
) as file:

    json.dump(
        {
            "parameters": (
                PARAMETERS.as_dictionary()
            ),
            "simulation": asdict(CONFIG),
            "validation": VALIDATION_REPORTS,
        },
        file,
        indent=2,
    )

SUMMARY

# In[10]:


# =============================================================================
# CELL 9 — Sparse observations for the future inverse PINN
# =============================================================================
# PURPOSE
# -------
# Construct the central partially observed inverse problem:
#
# - total bacterial load measured at 21 time points;
# - resistant fraction measured at only three time points.
#
# This reflects the BINNs principle of learning biological dynamics from sparse
# measurements while creating an AMR-specific hidden-state problem.

def nearest_rows(
    data: pd.DataFrame,
    requested_times: Iterable[float],
) -> pd.DataFrame:
    """
    Select dataframe rows nearest to requested observation times.
    """

    available_times = data[
        "time"
    ].to_numpy(dtype=float)

    indices = [
        int(
            np.argmin(
                np.abs(
                    available_times
                    - requested_time
                )
            )
        )
        for requested_time in requested_times
    ]

    unique_indices = np.unique(indices)

    return (
        data.iloc[unique_indices]
        .copy()
        .reset_index(drop=True)
    )


SPARSE_DATASETS: Dict[
    str,
    Dict[str, pd.DataFrame],
] = {}


for schedule_name, trajectory in TRAJECTORIES.items():

    total_observation_times = np.linspace(
        CONFIG.start_time,
        CONFIG.final_time,
        CONFIG.sparse_total_points,
    )

    phenotype_observation_times = [
        7.0,
        12.0,
        18.0,
    ]

    total_observations = nearest_rows(
        trajectory,
        total_observation_times,
    )[
        [
            "time",
            "total",
            "antibiotic",
            "schedule",
        ]
    ]

    phenotype_observations = nearest_rows(
        trajectory,
        phenotype_observation_times,
    )[
        [
            "time",
            "resistant_fraction",
            "antibiotic",
            "schedule",
        ]
    ]

    total_observations.to_csv(
        DATA_SPARSE
        / f"{schedule_name}_total_load.csv",
        index=False,
    )

    phenotype_observations.to_csv(
        DATA_SPARSE
        / f"{schedule_name}_resistant_fraction.csv",
        index=False,
    )

    SPARSE_DATASETS[schedule_name] = {
        "total_load": total_observations,
        "resistant_fraction": (
            phenotype_observations
        ),
    }


print(
    "Early-treatment total-load observations:"
)

display(
    SPARSE_DATASETS[
        "early_continuous"
    ]["total_load"].head()
)

print(
    "\nEarly-treatment resistant-fraction observations:"
)

display(
    SPARSE_DATASETS[
        "early_continuous"
    ]["resistant_fraction"]
)

# In[11]:


# =============================================================================
# CELL 10 — Synthetic noisy observation generator
# =============================================================================
# PURPOSE
# -------
# Create reproducible noisy datasets for later robustness experiments.
#
# NOISE MODELS
# ------------
# Total bacterial load:
#     Multiplicative log-normal noise is used because bacterial counts remain
#     positive and measurement variability often scales with magnitude.
#
# Resistant fraction:
#     Noise is added on the logit scale and transformed back to [0,1].
#
# The dense ground-truth trajectory is never altered.

def add_measurement_noise(
    total_observations: pd.DataFrame,
    fraction_observations: pd.DataFrame,
    coefficient_of_variation: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add reproducible noise to sparse observations."""

    if coefficient_of_variation < 0:
        raise ValueError(
            "coefficient_of_variation "
            "must be non-negative."
        )

    rng = np.random.default_rng(seed)

    noisy_total = total_observations.copy()
    noisy_fraction = (
        fraction_observations.copy()
    )

    # No-noise reference case
    if coefficient_of_variation == 0:

        noisy_total[
            "total_observed"
        ] = noisy_total["total"]

        noisy_fraction[
            "resistant_fraction_observed"
        ] = noisy_fraction[
            "resistant_fraction"
        ]

        return noisy_total, noisy_fraction

    # ---------------------------------------------------------------------
    # Convert coefficient of variation to log-normal parameters.
    # ---------------------------------------------------------------------
    sigma_log = np.sqrt(
        np.log(
            1.0
            + coefficient_of_variation**2
        )
    )

    mu_log = -0.5 * sigma_log**2

    multiplicative_noise = rng.lognormal(
        mean=mu_log,
        sigma=sigma_log,
        size=len(noisy_total),
    )

    noisy_total[
        "total_observed"
    ] = (
        noisy_total["total"]
        * multiplicative_noise
    )

    # ---------------------------------------------------------------------
    # Add noise to resistant fractions on the logit scale.
    # ---------------------------------------------------------------------
    epsilon = 1.0e-8

    fraction = np.clip(
        noisy_fraction[
            "resistant_fraction"
        ].to_numpy(dtype=float),
        epsilon,
        1.0 - epsilon,
    )

    logits = np.log(
        fraction / (1.0 - fraction)
    )

    noisy_logits = logits + rng.normal(
        loc=0.0,
        scale=coefficient_of_variation,
        size=len(fraction),
    )

    noisy_fraction[
        "resistant_fraction_observed"
    ] = 1.0 / (
        1.0 + np.exp(-noisy_logits)
    )

    return noisy_total, noisy_fraction


NOISE_LEVELS = [
    0.00,
    0.02,
    0.05,
    0.10,
]


for noise_level in NOISE_LEVELS:

    noisy_total, noisy_fraction = (
        add_measurement_noise(
            SPARSE_DATASETS[
                "early_continuous"
            ]["total_load"],
            SPARSE_DATASETS[
                "early_continuous"
            ]["resistant_fraction"],
            coefficient_of_variation=(
                noise_level
            ),
            seed=(
                SEED
                + int(noise_level * 1000)
            ),
        )
    )

    noise_label = (
        f"{int(noise_level * 100):02d}pct"
    )

    noisy_total.to_csv(
        DATA_NOISY
        / (
            "early_continuous_"
            f"total_{noise_label}.csv"
        ),
        index=False,
    )

    noisy_fraction.to_csv(
        DATA_NOISY
        / (
            "early_continuous_"
            f"fraction_{noise_label}.csv"
        ),
        index=False,
    )


print(
    "Noisy observation files created "
    "for noise levels:",
    NOISE_LEVELS,
)

# In[12]:


# =============================================================================
# CELL 11 — Publication figures: state trajectories
# =============================================================================
# PURPOSE
# -------
# Generate one high-resolution independent figure for each treatment schedule.
#
# Each figure displays:
# - sensitive bacteria S(t);
# - resistant bacteria R(t);
# - total bacterial population S(t)+R(t).
#
# Every figure is saved as PNG, PDF, and SVG.

for schedule_name, data in TRAJECTORIES.items():

    fig, ax = plt.subplots(
        figsize=(8.6, 5.4)
    )

    ax.plot(
        data["time"],
        data["sensitive"],
        label=r"Sensitive, $S(t)$",
    )

    ax.plot(
        data["time"],
        data["resistant"],
        label=r"Resistant, $R(t)$",
    )

    ax.plot(
        data["time"],
        data["total"],
        label=r"Total, $S(t)+R(t)$",
    )

    ax.set_xlabel("Time")

    ax.set_ylabel(
        "Bacterial population"
    )

    ax.set_title(
        "Deterministic AMR dynamics: "
        + schedule_name.replace("_", " ")
    )

    ax.set_xlim(
        CONFIG.start_time,
        CONFIG.final_time,
    )

    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.25)

    ax.legend(loc="best")

    fig.tight_layout()

    save_figure(
        fig,
        f"figure_state_{schedule_name}",
    )

    plt.show()

# In[13]:


# =============================================================================
# CELL 12 — Publication figure: resistant fraction across treatments
# =============================================================================
# PURPOSE
# -------
# Compare the latent resistant composition:
#
#               q(t) = R(t) / [S(t)+R(t)]
#
# across all treatment schedules.
#
# This figure motivates the hidden-state inference problem because cultures
# with similar total load can have very different resistance compositions.

fig, ax = plt.subplots(
    figsize=(8.6, 5.4)
)

for schedule_name, data in TRAJECTORIES.items():

    ax.plot(
        data["time"],
        data["resistant_fraction"],
        label=schedule_name.replace(
            "_",
            " ",
        ),
    )

ax.set_xlabel("Time")

ax.set_ylabel(
    r"Resistant fraction, $R/(S+R)$"
)

ax.set_title(
    "Evolution of resistant composition "
    "under prescribed treatments"
)

ax.set_xlim(
    CONFIG.start_time,
    CONFIG.final_time,
)

ax.set_ylim(0, 1)

ax.grid(alpha=0.25)

ax.legend(loc="best")

fig.tight_layout()

save_figure(
    fig,
    "figure_resistant_fraction_comparison",
)

plt.show()

# In[14]:


# =============================================================================
# CELL 13 — Publication figure: total bacterial load across treatments
# =============================================================================
# PURPOSE
# -------
# Compare the main measurable quantity:
#
#               B(t) = S(t) + R(t)
#
# under all treatment schedules.
#
# Total load is the principal observation supplied to the future inverse PINN.

fig, ax = plt.subplots(
    figsize=(8.6, 5.4)
)

for schedule_name, data in TRAJECTORIES.items():

    ax.plot(
        data["time"],
        data["total"],
        label=schedule_name.replace(
            "_",
            " ",
        ),
    )

ax.set_xlabel("Time")

ax.set_ylabel(
    r"Total bacterial load, $B(t)$"
)

ax.set_title(
    "Total bacterial load under "
    "alternative antibiotic schedules"
)

ax.set_xlim(
    CONFIG.start_time,
    CONFIG.final_time,
)

ax.set_ylim(bottom=0)

ax.grid(alpha=0.25)

ax.legend(loc="best")

fig.tight_layout()

save_figure(
    fig,
    "figure_total_load_comparison",
)

plt.show()

# In[15]:


# =============================================================================
# CELL 14 — Publication figure: eco-evolutionary phase plane
# =============================================================================
# PURPOSE
# -------
# Plot resistant population R against sensitive population S.
#
# The phase plane shows how each treatment redirects the population through
# sensitive-resistant state space.
#
# Circular markers indicate initial states.
# Cross markers indicate final states.

fig, ax = plt.subplots(
    figsize=(7.2, 6.2)
)

for schedule_name, data in TRAJECTORIES.items():

    ax.plot(
        data["sensitive"],
        data["resistant"],
        label=schedule_name.replace(
            "_",
            " ",
        ),
    )

    # Initial state
    ax.scatter(
        data.iloc[0]["sensitive"],
        data.iloc[0]["resistant"],
        s=35,
        marker="o",
        zorder=3,
    )

    # Final state
    ax.scatter(
        data.iloc[-1]["sensitive"],
        data.iloc[-1]["resistant"],
        s=45,
        marker="x",
        zorder=3,
    )

ax.set_xlabel(
    r"Sensitive population, $S$"
)

ax.set_ylabel(
    r"Resistant population, $R$"
)

ax.set_title(
    "Eco-evolutionary phase trajectories"
)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

ax.grid(alpha=0.25)

ax.legend(loc="best")

fig.tight_layout()

save_figure(
    fig,
    "figure_phase_plane",
)

plt.show()

# In[16]:


# =============================================================================
# CELL 15 — Publication figures: observed load and hidden composition
# =============================================================================
# PURPOSE
# -------
# Show the partially observed inverse problem for early treatment.
#
# Figure 1:
# Dense total-load ground truth and 21 available total-load observations.
#
# Figure 2:
# Dense hidden resistant fraction and only three available phenotype
# observations.
#
# These figures directly illustrate the paper's central inference challenge.

early_data = TRAJECTORIES[
    "early_continuous"
]

early_total_observations = (
    SPARSE_DATASETS[
        "early_continuous"
    ]["total_load"]
)

early_fraction_observations = (
    SPARSE_DATASETS[
        "early_continuous"
    ]["resistant_fraction"]
)

# -------------------------------------------------------------------------
# Figure 1: total bacterial load
# -------------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8.6, 5.4)
)

ax.plot(
    early_data["time"],
    early_data["total"],
    label="Dense ground truth",
)

ax.scatter(
    early_total_observations["time"],
    early_total_observations["total"],
    s=38,
    label="Sparse total-load observations",
    zorder=3,
)

ax.set_xlabel("Time")

ax.set_ylabel(
    r"Total bacterial load, $B(t)$"
)

ax.set_title(
    "Sparse observation of total bacterial load"
)

ax.set_xlim(
    CONFIG.start_time,
    CONFIG.final_time,
)

ax.set_ylim(bottom=0)

ax.grid(alpha=0.25)

ax.legend(loc="best")

fig.tight_layout()

save_figure(
    fig,
    "figure_sparse_total_load",
)

plt.show()


# -------------------------------------------------------------------------
# Figure 2: resistant fraction
# -------------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8.6, 5.4)
)

ax.plot(
    early_data["time"],
    early_data["resistant_fraction"],
    label="Hidden resistant fraction",
)

ax.scatter(
    early_fraction_observations["time"],
    early_fraction_observations[
        "resistant_fraction"
    ],
    s=45,
    label="Available phenotype observations",
    zorder=3,
)

ax.set_xlabel("Time")

ax.set_ylabel(
    r"Resistant fraction, $R/(S+R)$"
)

ax.set_title(
    "Sparse phenotype information "
    "in the inverse problem"
)

ax.set_xlim(
    CONFIG.start_time,
    CONFIG.final_time,
)

ax.set_ylim(0, 1)

ax.grid(alpha=0.25)

ax.legend(loc="best")

fig.tight_layout()

save_figure(
    fig,
    "figure_sparse_resistant_fraction",
)

plt.show()

# In[17]:


# =============================================================================
# CELL 16 — Parameter sweep over mutation probability and fitness cost
# =============================================================================
# PURPOSE
# -------
# Examine how the planned inverse parameters influence resistant dynamics:
#
#     mutation probability, mu
#     resistance fitness cost, c_R
#
# This must be checked before PINN parameter inference.
# A parameter that produces almost no change in the observables will be
# difficult or impossible to estimate from sparse data.

mutation_grid = np.logspace(
    -5,
    -3,
    9,
)

fitness_cost_grid = np.linspace(
    0.0,
    0.20,
    9,
)

sweep_rows = []

for mutation_probability in mutation_grid:

    for fitness_cost in fitness_cost_grid:

        sweep_parameters = replace(
            PARAMETERS,
            mutation_probability=float(
                mutation_probability
            ),
            resistance_fitness_cost=float(
                fitness_cost
            ),
        )

        trajectory = solve_deterministic(
            sweep_parameters,
            CONFIG,
            SCHEDULES[
                "early_continuous"
            ],
        )

        validation = validate_trajectory(
            trajectory,
            sweep_parameters,
            CONFIG,
        )

        sweep_rows.append(
            {
                "mutation_probability": (
                    mutation_probability
                ),
                "fitness_cost": fitness_cost,
                "final_sensitive": (
                    trajectory.iloc[-1][
                        "sensitive"
                    ]
                ),
                "final_resistant": (
                    trajectory.iloc[-1][
                        "resistant"
                    ]
                ),
                "final_total": (
                    trajectory.iloc[-1][
                        "total"
                    ]
                ),
                "final_resistant_fraction": (
                    trajectory.iloc[-1][
                        "resistant_fraction"
                    ]
                ),
                "peak_resistant": (
                    trajectory[
                        "resistant"
                    ].max()
                ),
                "all_checks_passed": (
                    validation[
                        "all_checks_passed"
                    ]
                ),
            }
        )


SWEEP = pd.DataFrame(sweep_rows)

if not SWEEP[
    "all_checks_passed"
].all():

    raise RuntimeError(
        "At least one parameter-sweep "
        "simulation failed validation."
    )


SWEEP.to_csv(
    TABLE_DIR
    / "round1_parameter_sweep.csv",
    index=False,
)

SWEEP.head()

# In[18]:


# =============================================================================
# CELL 17 — Publication figure: parameter-response heat map
# =============================================================================
# PURPOSE
# -------
# Visualize how the final resistant fraction depends on:
#
# - mutation probability;
# - resistance fitness cost.
#
# The horizontal axis uses log10(mu) because mutation probabilities span
# several orders of magnitude.

heatmap_table = SWEEP.pivot(
    index="fitness_cost",
    columns="mutation_probability",
    values="final_resistant_fraction",
)

fig, ax = plt.subplots(
    figsize=(8.0, 6.2)
)

image = ax.imshow(
    heatmap_table.to_numpy(),
    origin="lower",
    aspect="auto",
    extent=[
        np.log10(
            heatmap_table.columns.min()
        ),
        np.log10(
            heatmap_table.columns.max()
        ),
        heatmap_table.index.min(),
        heatmap_table.index.max(),
    ],
)

ax.set_xlabel(
    r"$\log_{10}(\mu)$"
)

ax.set_ylabel(
    r"Resistance fitness cost, $c_R$"
)

ax.set_title(
    "Final resistant fraction under "
    "early continuous treatment"
)

colorbar = fig.colorbar(
    image,
    ax=ax,
)

colorbar.set_label(
    "Final resistant fraction"
)

fig.tight_layout()

save_figure(
    fig,
    "figure_parameter_heatmap",
)

plt.show()

# In[19]:


# =============================================================================
# CELL 18 — Local sensitivity analysis for inverse parameters
# =============================================================================
# PURPOSE
# -------
# Estimate finite-difference sensitivities of:
#
# - total bacterial load;
# - resistant fraction;
#
# with respect to:
#
# - mutation probability;
# - resistance fitness cost.
#
# These are preliminary identifiability diagnostics. Later rounds will include
# profile likelihood and bootstrap uncertainty analysis.

def observable_vector(
    parameters: AMRParameters,
    schedule: TreatmentSchedule,
    observation_times: np.ndarray,
) -> np.ndarray:
    """
    Generate one vector containing total-load and resistant-fraction values.
    """

    trajectory = solve_deterministic(
        parameters,
        CONFIG,
        schedule,
    )

    total_interpolator = interp1d(
        trajectory["time"],
        trajectory["total"],
        kind="cubic",
    )

    fraction_interpolator = interp1d(
        trajectory["time"],
        trajectory[
            "resistant_fraction"
        ],
        kind="cubic",
    )

    total_values = total_interpolator(
        observation_times
    )

    fraction_values = (
        fraction_interpolator(
            observation_times
        )
    )

    return np.concatenate(
        [
            total_values,
            fraction_values,
        ]
    )


def normalized_central_sensitivity(
    base_parameters: AMRParameters,
    parameter_name: str,
    relative_step: float = 0.01,
) -> np.ndarray:
    """
    Calculate a normalized central finite-difference sensitivity.

    S_ij = theta_j/y_i * dy_i/dtheta_j
    """

    base_value = getattr(
        base_parameters,
        parameter_name,
    )

    finite_difference_step = max(
        abs(base_value)
        * relative_step,
        1.0e-8,
    )

    lower_value = (
        base_value
        - finite_difference_step
    )

    upper_value = (
        base_value
        + finite_difference_step
    )

    bounded_names = {
        "mutation_probability",
        "resistance_fitness_cost",
        "sensitive_drug_effect",
        "resistant_drug_effect",
    }

    if parameter_name in bounded_names:

        lower_value = max(
            lower_value,
            1.0e-12,
        )

        upper_value = min(
            upper_value,
            1.0 - 1.0e-12,
        )

    lower_parameters = replace(
        base_parameters,
        **{
            parameter_name: lower_value
        },
    )

    upper_parameters = replace(
        base_parameters,
        **{
            parameter_name: upper_value
        },
    )

    observation_times = np.linspace(
        CONFIG.start_time,
        CONFIG.final_time,
        21,
    )

    lower_observables = observable_vector(
        lower_parameters,
        SCHEDULES[
            "early_continuous"
        ],
        observation_times,
    )

    upper_observables = observable_vector(
        upper_parameters,
        SCHEDULES[
            "early_continuous"
        ],
        observation_times,
    )

    base_observables = observable_vector(
        base_parameters,
        SCHEDULES[
            "early_continuous"
        ],
        observation_times,
    )

    derivative = (
        upper_observables
        - lower_observables
    ) / (
        upper_value
        - lower_value
    )

    normalized_sensitivity = (
        base_value
        * derivative
        / (
            np.abs(base_observables)
            + 1.0e-8
        )
    )

    return normalized_sensitivity


sensitivity_mutation = (
    normalized_central_sensitivity(
        PARAMETERS,
        "mutation_probability",
    )
)

sensitivity_fitness_cost = (
    normalized_central_sensitivity(
        PARAMETERS,
        "resistance_fitness_cost",
    )
)


sensitivity_summary = pd.DataFrame(
    {
        "parameter": [
            "mutation_probability",
            "resistance_fitness_cost",
        ],
        "mean_absolute_sensitivity": [
            np.mean(
                np.abs(
                    sensitivity_mutation
                )
            ),
            np.mean(
                np.abs(
                    sensitivity_fitness_cost
                )
            ),
        ],
        "maximum_absolute_sensitivity": [
            np.max(
                np.abs(
                    sensitivity_mutation
                )
            ),
            np.max(
                np.abs(
                    sensitivity_fitness_cost
                )
            ),
        ],
        "l2_sensitivity_norm": [
            np.linalg.norm(
                sensitivity_mutation
            ),
            np.linalg.norm(
                sensitivity_fitness_cost
            ),
        ],
    }
)

sensitivity_summary.to_csv(
    TABLE_DIR
    / "round1_local_sensitivity_summary.csv",
    index=False,
)

sensitivity_summary

# In[20]:


# =============================================================================
# CELL 19 — Publication figures: time-resolved parameter sensitivities
# =============================================================================
# PURPOSE
# -------
# Show when total load and resistant fraction are most informative about:
#
# - mutation probability;
# - resistance fitness cost.
#
# These figures can later guide the placement of phenotype measurements.

observation_times = np.linspace(
    CONFIG.start_time,
    CONFIG.final_time,
    21,
)

number_of_times = len(
    observation_times
)

sensitivity_figure_information = [
    (
        r"mutation probability, $\mu$",
        sensitivity_mutation,
        "figure_sensitivity_mutation",
    ),
    (
        r"resistance fitness cost, $c_R$",
        sensitivity_fitness_cost,
        "figure_sensitivity_fitness_cost",
    ),
]


for (
    parameter_label,
    sensitivity_values,
    output_stem,
) in sensitivity_figure_information:

    fig, ax = plt.subplots(
        figsize=(8.6, 5.4)
    )

    ax.plot(
        observation_times,
        sensitivity_values[
            :number_of_times
        ],
        label=(
            "Normalized sensitivity "
            "of total load"
        ),
    )

    ax.plot(
        observation_times,
        sensitivity_values[
            number_of_times:
        ],
        label=(
            "Normalized sensitivity "
            "of resistant fraction"
        ),
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel("Time")

    ax.set_ylabel(
        "Normalized sensitivity"
    )

    ax.set_title(
        "Observation sensitivity to "
        + parameter_label
    )

    ax.set_xlim(
        CONFIG.start_time,
        CONFIG.final_time,
    )

    ax.grid(alpha=0.25)

    ax.legend(loc="best")

    fig.tight_layout()

    save_figure(
        fig,
        output_stem,
    )

    plt.show()

# In[21]:


# =============================================================================
# CELL 20 — Numerical convergence study
# =============================================================================
# PURPOSE
# -------
# Verify that the selected ODE solver tolerances generate sufficiently accurate
# reference data.
#
# PROCEDURE
# ---------
# 1. Generate a highly accurate reference solution.
# 2. Solve the same model using several tolerance settings.
# 3. Interpolate each test solution onto the reference grid.
# 4. Calculate maximum absolute and relative L2 errors.
#
# This is necessary because PINN accuracy cannot be assessed against an
# unreliable numerical reference.

reference_config = replace(
    CONFIG,
    dense_points=2001,
    relative_tolerance=1.0e-12,
    absolute_tolerance=1.0e-14,
)

reference_solution = solve_deterministic(
    PARAMETERS,
    reference_config,
    SCHEDULES[
        "early_continuous"
    ],
)

convergence_settings = [
    (1.0e-6, 1.0e-8),
    (1.0e-8, 1.0e-10),
    (1.0e-9, 1.0e-11),
    (1.0e-10, 1.0e-12),
]

convergence_rows = []


for relative_tolerance, absolute_tolerance in convergence_settings:

    test_config = replace(
        CONFIG,
        dense_points=1001,
        relative_tolerance=(
            relative_tolerance
        ),
        absolute_tolerance=(
            absolute_tolerance
        ),
    )

    test_solution = solve_deterministic(
        PARAMETERS,
        test_config,
        SCHEDULES[
            "early_continuous"
        ],
    )

    sensitive_interpolated = interp1d(
        test_solution["time"],
        test_solution["sensitive"],
        kind="cubic",
    )(
        reference_solution["time"]
    )

    resistant_interpolated = interp1d(
        test_solution["time"],
        test_solution["resistant"],
        kind="cubic",
    )(
        reference_solution["time"]
    )

    sensitive_reference = (
        reference_solution[
            "sensitive"
        ].to_numpy()
    )

    resistant_reference = (
        reference_solution[
            "resistant"
        ].to_numpy()
    )

    sensitive_error = (
        sensitive_interpolated
        - sensitive_reference
    )

    resistant_error = (
        resistant_interpolated
        - resistant_reference
    )

    convergence_rows.append(
        {
            "rtol": relative_tolerance,
            "atol": absolute_tolerance,
            "max_abs_error_sensitive": (
                np.max(
                    np.abs(
                        sensitive_error
                    )
                )
            ),
            "max_abs_error_resistant": (
                np.max(
                    np.abs(
                        resistant_error
                    )
                )
            ),
            "relative_l2_sensitive": (
                np.linalg.norm(
                    sensitive_error
                )
                / np.linalg.norm(
                    sensitive_reference
                )
            ),
            "relative_l2_resistant": (
                np.linalg.norm(
                    resistant_error
                )
                / (
                    np.linalg.norm(
                        resistant_reference
                    )
                    + 1.0e-12
                )
            ),
        }
    )


CONVERGENCE = pd.DataFrame(
    convergence_rows
)

CONVERGENCE.to_csv(
    TABLE_DIR
    / "round1_solver_convergence.csv",
    index=False,
)

CONVERGENCE

# In[22]:


# =============================================================================
# CELL 21 — Publication figure: deterministic solver convergence
# =============================================================================
# PURPOSE
# -------
# Display relative L2 error as a function of solver tolerance.
#
# This figure documents that the selected reference trajectories are
# numerically stable and appropriate for later PINN comparisons.

fig, ax = plt.subplots(
    figsize=(8.0, 5.4)
)

ax.loglog(
    CONVERGENCE["rtol"],
    CONVERGENCE[
        "relative_l2_sensitive"
    ],
    marker="o",
    label="Sensitive population",
)

ax.loglog(
    CONVERGENCE["rtol"],
    CONVERGENCE[
        "relative_l2_resistant"
    ],
    marker="s",
    label="Resistant population",
)

ax.set_xlabel(
    "Relative solver tolerance"
)

ax.set_ylabel(
    r"Relative $L_2$ error"
)

ax.set_title(
    "Numerical convergence of "
    "the deterministic AMR solver"
)

ax.grid(alpha=0.25)

ax.legend(loc="best")

# Smaller tolerances appear toward the right after inversion.
ax.invert_xaxis()

fig.tight_layout()

save_figure(
    fig,
    "figure_solver_convergence",
)

plt.show()

# In[23]:


# =============================================================================
# CELL 22 — Analytical equilibrium consistency check
# =============================================================================
# PURPOSE
# -------
# Explain why the untreated population approaches approximately 900 rather
# than the carrying capacity K=1000.
#
# For a sensitive-dominated untreated population:
#
#     r_S(1-mu)(1-N*/K) - d_S = 0
#
# Therefore:
#
#     N* = K[1 - d_S/{r_S(1-mu)}]
#
# This check confirms that the numerical result has the expected biological
# equilibrium and is not a coding error.

analytical_equilibrium = (
    PARAMETERS.carrying_capacity
    * (
        1.0
        - PARAMETERS.sensitive_death_rate
        / (
            PARAMETERS.sensitive_birth_rate
            * (
                1.0
                - PARAMETERS.mutation_probability
            )
        )
    )
)

simulated_final_total = float(
    TRAJECTORIES[
        "untreated"
    ].iloc[-1]["total"]
)

relative_equilibrium_error = (
    abs(
        simulated_final_total
        - analytical_equilibrium
    )
    / analytical_equilibrium
)

equilibrium_report = pd.DataFrame(
    {
        "quantity": [
            (
                "analytical_sensitive_"
                "dominant_equilibrium"
            ),
            "simulated_final_total",
            "relative_error",
        ],
        "value": [
            analytical_equilibrium,
            simulated_final_total,
            relative_equilibrium_error,
        ],
    }
)

equilibrium_report.to_csv(
    TABLE_DIR
    / "round1_equilibrium_check.csv",
    index=False,
)

equilibrium_report

# In[24]:


# =============================================================================
# CELL 23 — Export a LaTeX results table
# =============================================================================
# PURPOSE
# -------
# Convert the deterministic outcome summary into LaTeX code that can be
# inserted directly into the paper.
#
# The generated file will be saved in:
#
#     outputs/tables/round1_deterministic_summary.tex

latex_summary = SUMMARY[
    [
        "schedule",
        "final_sensitive",
        "final_resistant",
        "final_total",
        "final_resistant_fraction",
    ]
].copy()

latex_summary[
    "schedule"
] = latex_summary[
    "schedule"
].str.replace(
    "_",
    " ",
)

latex_summary = latex_summary.rename(
    columns={
        "schedule": "Treatment",
        "final_sensitive": (
            r"Final $S$"
        ),
        "final_resistant": (
            r"Final $R$"
        ),
        "final_total": (
            r"Final $S+R$"
        ),
        "final_resistant_fraction": (
            r"Final $R/(S+R)$"
        ),
    }
)

latex_code = latex_summary.to_latex(
    index=False,
    float_format=lambda value: (
        f"{value:.4f}"
    ),
    escape=False,
    caption=(
        "Deterministic outcomes under "
        "the prescribed treatment schedules."
    ),
    label=(
        "tab:round1_deterministic_outcomes"
    ),
)

latex_path = (
    TABLE_DIR
    / "round1_deterministic_summary.tex"
)

latex_path.write_text(
    latex_code,
    encoding="utf-8",
)

print(latex_code)

# In[25]:


# =============================================================================
# CELL 24 — Final Round 1 reproducibility audit
# =============================================================================
# PURPOSE
# -------
# Confirm that:
#
# 1. every treatment trajectory passed all biological checks;
# 2. every parameter-sweep simulation passed;
# 3. all required tables and reports were generated;
# 4. a complete inventory of project files is available.
#
# Round 1 should be considered complete only if this cell executes without an
# exception.

assert SUMMARY[
    "all_checks_passed"
].all()

assert SWEEP[
    "all_checks_passed"
].all()


required_files = [
    TABLE_DIR
    / "round1_deterministic_summary.csv",

    TABLE_DIR
    / "round1_parameter_sweep.csv",

    TABLE_DIR
    / "round1_solver_convergence.csv",

    TABLE_DIR
    / "round1_local_sensitivity_summary.csv",

    TABLE_DIR
    / "round1_equilibrium_check.csv",

    TABLE_DIR
    / "round1_deterministic_summary.tex",

    REPORT_DIR
    / "round1_validation_reports.json",
]


missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "The following required outputs "
        f"are missing: {missing_files}"
    )


# -------------------------------------------------------------------------
# Create a complete file inventory for reproducibility.
# -------------------------------------------------------------------------
file_inventory = sorted(
    str(
        path.relative_to(
            PROJECT_ROOT
        )
    )
    for path in PROJECT_ROOT.rglob("*")
    if path.is_file()
)

inventory_path = (
    REPORT_DIR
    / "round1_file_inventory.txt"
)

inventory_path.write_text(
    "\n".join(file_inventory),
    encoding="utf-8",
)


print(
    "Round 1 completed successfully."
)

print(
    "Dense datasets:",
    len(
        list(
            DATA_DENSE.glob("*.csv")
        )
    ),
)

print(
    "Sparse datasets:",
    len(
        list(
            DATA_SPARSE.glob("*.csv")
        )
    ),
)

print(
    "Noisy datasets:",
    len(
        list(
            DATA_NOISY.glob("*.csv")
        )
    ),
)

print(
    "Figure files:",
    len(
        list(
            FIGURE_DIR.glob("*")
        )
    ),
)

print(
    "Table files:",
    len(
        list(
            TABLE_DIR.glob("*")
        )
    ),
)

print(
    "Report files:",
    len(
        list(
            REPORT_DIR.glob("*")
        )
    ),
)

print("\nNext computational round:")

print(
    "Implement the corrected Gillespie "
    "stochastic simulator."
)

print(
    "Compare stochastic ensemble means, "
    "variances, extinction, and resistant "
    "establishment with the deterministic "
    "reference trajectories."
)
