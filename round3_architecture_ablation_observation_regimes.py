#!/usr/bin/env python
# coding: utf-8

# Round 3

# In[1]:


# =============================================================================
# ROUND 3 — CELL 1
# Environment, reproducibility, GPU-aware folders, and execution budgets
# FINAL MANUSCRIPT-SCALE OBSERVATION-REGIME CONFIGURATION
# =============================================================================
# WHAT THIS CELL DOES
# -------------------
# 1. Configures Round 3 for the final O1–O4 observation-regime experiment.
# 2. Uses manuscript-scale training budgets.
# 3. Activates only RUN_OBSERVATION_REGIME_STUDY.
# 4. Keeps every other expensive Round 3 experiment disabled.
# 5. Creates the Round 3 final-output folders.
# 6. Performs configuration checks before training begins.
#
# SCIENTIFIC PURPOSE
# ------------------
# This experiment evaluates how the amount of phenotype information affects:
#
#     - reconstruction of the hidden sensitive population S(t);
#     - reconstruction of the hidden resistant population R(t);
#     - estimation of mutation probability mu;
#     - estimation of resistance fitness cost c_R;
#     - physics consistency;
#     - biological feasibility.
#
# The observation regimes are:
#
#     O1: total bacterial load only;
#     O2: total load + one resistant-fraction observation;
#     O3: total load + three resistant-fraction observations;
#     O4: fully phenotype-resolved observations.
#
# EXECUTION SEQUENCE
# ------------------
# 1. Run this corrected Cell 1.
# 2. Restart the kernel.
# 3. Run Round 3 Cells 1–13 sequentially.
# 4. Skip Cells 14–17.
# 5. Run Cell 18.
#
# IMPORTANT
# ---------
# QUICK_MODE=False is used because this is the final manuscript-scale run.
# Four hard-constrained PINNs will be trained, one for each regime O1–O4.

from pathlib import Path
import os
import sys
import platform
import random
import json
import math
import time
import copy
import gc
import warnings

from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 20260805

random.seed(SEED)
np.random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)


# =============================================================================
# SINGLE EXECUTION-MODE SWITCH
# =============================================================================
# True:
#     Reduced structural verification only.
#     Results must not be used in the manuscript.
#
# False:
#     Manuscript-scale training and evaluation.
#
# The current cell is configured for the final O1–O4 experiment.

QUICK_MODE: bool = False

if not isinstance(QUICK_MODE, bool):
    raise TypeError(
        "QUICK_MODE must be a Boolean value: either True or False."
    )

RUN_LABEL = "quick" if QUICK_MODE else "final"


# =============================================================================
# CURRENT EXPERIMENT
# =============================================================================

CURRENT_EXPERIMENT = "observation_regime_study"


# =============================================================================
# PROJECT AND OUTPUT FOLDERS
# =============================================================================
# The notebook may be stored:
#
# 1. directly inside the project root; or
# 2. inside a folder named 'notebooks'.

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent


# Quick and final outputs are isolated:
#
# project/
# └── round3/
#     ├── quick/
#     └── final/
#
# Because QUICK_MODE=False, the current outputs are written to:
#
# project/round3/final/

ROUND3_BASE_ROOT = PROJECT_ROOT / "round3"
ROUND3_ROOT = ROUND3_BASE_ROOT / RUN_LABEL

MODEL_DIR = ROUND3_ROOT / "models"
PREDICTION_DIR = ROUND3_ROOT / "predictions"
HISTORY_DIR = ROUND3_ROOT / "histories"
FIGURE_DIR_R3 = ROUND3_ROOT / "figures"
TABLE_DIR_R3 = ROUND3_ROOT / "tables"
REPORT_DIR_R3 = ROUND3_ROOT / "reports"
CACHE_DIR_R3 = ROUND3_ROOT / "cache"

ROUND3_DIRECTORIES = [
    MODEL_DIR,
    PREDICTION_DIR,
    HISTORY_DIR,
    FIGURE_DIR_R3,
    TABLE_DIR_R3,
    REPORT_DIR_R3,
    CACHE_DIR_R3,
]

for directory in ROUND3_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# TRAINING BUDGETS
# =============================================================================
# One O1–O4 model is trained at a time.
#
# The final configuration uses:
#
#     2,000 data-pretraining epochs;
#     up to 12,000 Adam epochs;
#     up to 1,000 L-BFGS iterations;
#     2,048 collocation points per treatment interval.
#
# Early stopping may terminate Adam before 12,000 epochs when the validation
# score does not improve for the specified patience period.

if QUICK_MODE:

    # -------------------------------------------------------------------------
    # Preliminary structural verification
    # -------------------------------------------------------------------------

    PRETRAIN_EPOCHS = 250
    ADAM_EPOCHS = 750
    LBFGS_MAX_ITER = 35

    COLLOCATION_POINTS_PER_INTERVAL = 256

    EARLY_STOPPING_PATIENCE = 300
    LOG_EVERY = 25

else:

    # -------------------------------------------------------------------------
    # Final manuscript-scale configuration
    # -------------------------------------------------------------------------

    PRETRAIN_EPOCHS = 2000
    ADAM_EPOCHS = 12000
    LBFGS_MAX_ITER = 1000

    COLLOCATION_POINTS_PER_INTERVAL = 2048

    EARLY_STOPPING_PATIENCE = 2500
    LOG_EVERY = 100


# =============================================================================
# EXPERIMENT-SUITE SWITCHES
# =============================================================================
# Only the O1–O4 observation-regime study is active.
#
# The primary single-model experiment and architecture ablation have already
# been completed and must not be retrained during this run.

RUN_SINGLE_MODEL_DEMO = False

RUN_PRIMARY_ABLATION = False

RUN_OBSERVATION_REGIME_STUDY = True

RUN_NOISE_ROBUSTNESS_STUDY = False

RUN_MULTI_SCHEDULE_STUDY = False

RUN_PROFILE_LIKELIHOOD = False

RUN_STOCHASTIC_BENCHMARK = False

RUN_STOCHASTIC_TRANSFER_VALIDATION = False


# =============================================================================
# EXPERIMENT-SWITCH INVENTORY
# =============================================================================

EXPERIMENT_SWITCHES = {
    "single_model_demo": RUN_SINGLE_MODEL_DEMO,
    "primary_ablation": RUN_PRIMARY_ABLATION,
    "observation_regime_study": RUN_OBSERVATION_REGIME_STUDY,
    "noise_robustness_study": RUN_NOISE_ROBUSTNESS_STUDY,
    "multi_schedule_study": RUN_MULTI_SCHEDULE_STUDY,
    "profile_likelihood": RUN_PROFILE_LIKELIHOOD,
    "stochastic_benchmark": RUN_STOCHASTIC_BENCHMARK,
    "stochastic_transfer_validation": (
        RUN_STOCHASTIC_TRANSFER_VALIDATION
    ),
}

active_experiments = [
    experiment_name
    for experiment_name, enabled in EXPERIMENT_SWITCHES.items()
    if enabled
]


# =============================================================================
# OBSERVATION-REGIME DEFINITIONS
# =============================================================================
# These labels are repeated here for configuration reporting. Cell 18 creates
# the actual datasets using build_observation_dataset().

OBSERVATION_REGIMES = {
    "O1": {
        "description": "Total bacterial load only",
        "phenotype_information": "No resistant-fraction measurement",
    },
    "O2": {
        "description": "Total load plus one phenotype measurement",
        "phenotype_information": "One resistant-fraction measurement",
    },
    "O3": {
        "description": "Total load plus three phenotype measurements",
        "phenotype_information": "Three resistant-fraction measurements",
    },
    "O4": {
        "description": "Fully phenotype-resolved observations",
        "phenotype_information": "Dense resistant-fraction measurements",
    },
}


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
# These checks prevent an accidental rerun of another expensive experiment.

if QUICK_MODE:
    warnings.warn(
        "QUICK_MODE=True. Results from this run are preliminary and must not "
        "be used as final manuscript results."
    )

if not active_experiments:
    raise RuntimeError(
        "No Round 3 experiment suite is active."
    )

if len(active_experiments) != 1:
    raise RuntimeError(
        "Exactly one Round 3 experiment suite must be active for this run. "
        f"Currently active: {active_experiments}"
    )

if active_experiments[0] != CURRENT_EXPERIMENT:
    raise RuntimeError(
        "The active experiment does not match CURRENT_EXPERIMENT. "
        f"Expected: {CURRENT_EXPERIMENT}; "
        f"active: {active_experiments}"
    )

if RUN_SINGLE_MODEL_DEMO:
    raise RuntimeError(
        "RUN_SINGLE_MODEL_DEMO must remain False during the dedicated "
        "observation-regime experiment."
    )

if RUN_PRIMARY_ABLATION:
    raise RuntimeError(
        "RUN_PRIMARY_ABLATION must remain False. The ablation experiment "
        "has already been completed and must not be retrained here."
    )

if not RUN_OBSERVATION_REGIME_STUDY:
    raise RuntimeError(
        "RUN_OBSERVATION_REGIME_STUDY must be True."
    )

if len(OBSERVATION_REGIMES) != 4:
    raise RuntimeError(
        "The observation-regime experiment must contain O1, O2, O3, and O4."
    )

if list(OBSERVATION_REGIMES.keys()) != ["O1", "O2", "O3", "O4"]:
    raise RuntimeError(
        "Observation regimes must be ordered as O1, O2, O3, and O4."
    )


# =============================================================================
# PRINT EXECUTION CONFIGURATION
# =============================================================================

print("=" * 82)
print("ROUND 3 EXECUTION CONFIGURATION")
print("=" * 82)

print("\nProject root:")
print(" ", PROJECT_ROOT.resolve())

print("\nRound 3 base root:")
print(" ", ROUND3_BASE_ROOT.resolve())

print("\nCurrent output root:")
print(" ", ROUND3_ROOT.resolve())

print("\nEnvironment")
print("  Python:", sys.version.split()[0])
print("  Operating system:", platform.platform())
print("  Random seed:", SEED)

print("\nExecution mode")
print("  QUICK_MODE:", QUICK_MODE)
print("  Run label:", RUN_LABEL)
print("  Current experiment:", CURRENT_EXPERIMENT)

print("\nTraining budgets")
print("  Pretraining epochs:", PRETRAIN_EPOCHS)
print("  Adam epochs:", ADAM_EPOCHS)
print("  L-BFGS maximum iterations:", LBFGS_MAX_ITER)
print(
    "  Collocation points per treatment interval:",
    COLLOCATION_POINTS_PER_INTERVAL,
)
print(
    "  Early-stopping patience:",
    EARLY_STOPPING_PATIENCE,
)
print(
    "  Logging interval:",
    LOG_EVERY,
)

print("\nExperiment switches")

for experiment_name, enabled in EXPERIMENT_SWITCHES.items():
    print(
        f"  {experiment_name:35s}: {enabled}"
    )

print("\nActive experiment suites:")
print(" ", active_experiments)

print("\nObservation regimes")

for regime_name, regime_information in OBSERVATION_REGIMES.items():

    print(
        f"  {regime_name}: "
        f"{regime_information['description']}"
    )

print("\nExpected experiment")
print(
    "  Final manuscript-scale O1–O4 observation-regime study"
)

print("\nRequired execution sequence")
print("  1. Restart the kernel.")
print("  2. Run Round 3 Cells 1–13 sequentially.")
print("  3. Skip Cells 14–17.")
print("  4. Run Cell 18.")
print("  5. Inspect and save the O1–O4 metrics.")
print("  6. Run the observation-regime plotting cell, when available.")

print("\nExpected Cell 18 model labels")
print("  round3_observation_O1")
print("  round3_observation_O2")
print("  round3_observation_O3")
print("  round3_observation_O4")

print("\nExpected output table")
print(
    " ",
    TABLE_DIR_R3 / "round3_observation_regime_metrics.csv",
)

print("=" * 82)


# =============================================================================
# FINAL CONFIGURATION ASSERTIONS
# =============================================================================
# These assertions verify that Cell 1 is configured exactly for the final
# observation-regime experiment.

assert QUICK_MODE is False, (
    "This Cell 1 is configured for manuscript-scale execution. "
    "QUICK_MODE must be False."
)

assert RUN_SINGLE_MODEL_DEMO is False, (
    "RUN_SINGLE_MODEL_DEMO must be False during the dedicated "
    "observation-regime run."
)

assert RUN_PRIMARY_ABLATION is False, (
    "RUN_PRIMARY_ABLATION must be False because the architecture ablation "
    "has already been completed."
)

assert RUN_OBSERVATION_REGIME_STUDY is True, (
    "RUN_OBSERVATION_REGIME_STUDY must be True."
)

assert active_experiments == ["observation_regime_study"], (
    "Only observation_regime_study should be active. "
    f"Currently active: {active_experiments}"
)

assert RUN_LABEL == "final", (
    "The observation-regime manuscript run must use RUN_LABEL='final'."
)

assert PRETRAIN_EPOCHS == 2000
assert ADAM_EPOCHS == 12000
assert LBFGS_MAX_ITER == 1000
assert COLLOCATION_POINTS_PER_INTERVAL == 2048

print(
    "\nCell 1 completed successfully: final manuscript-scale "
    "O1–O4 observation-regime configuration is active."
)

# In[2]:


# ROUND 3 — CELL 2
# Scientific imports, PyTorch precision, deterministic settings, and figures
# =============================================================================
# Why double precision is used
# ----------------------------
# PINNs differentiate network outputs with respect to time and optimize small
# mutation probabilities. Float64 substantially improves residual and parameter
# stability. The network and collocation sets are small enough that an 8 GB GPU
# can use float64 without difficulty.
#
# Figure policy
# -------------
# Every manuscript figure is saved as:
# - PNG at 400 dpi for convenient viewing;
# - PDF for vector-quality publication;
# - SVG for editing in Inkscape/Illustrator.

import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore', category=UserWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64
# Float64 is selected explicitly rather than through torch.set_default_dtype so
# external libraries and integer tensors are not affected unexpectedly.

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 400,
    'font.family': 'sans-serif',
    'font.size': 10.5,
    'axes.labelsize': 11.5,
    'axes.titlesize': 12.5,
    'legend.fontsize': 9.0,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def save_figure_r3(fig: plt.Figure, stem: str, close: bool = False) -> None:
    """Save a Round 3 figure in high-resolution raster and vector formats."""
    clean = stem.strip().replace(' ', '_')
    fig.savefig(FIGURE_DIR_R3 / f'{clean}.png', dpi=400,
                bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURE_DIR_R3 / f'{clean}.pdf',
                bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURE_DIR_R3 / f'{clean}.svg',
                bbox_inches='tight', facecolor='white')
    if close:
        plt.close(fig)


def clear_gpu_memory() -> None:
    """Release unused Python and CUDA memory between sequential experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('PyTorch:', torch.__version__)
print('SciPy:', scipy.__version__)
print('Matplotlib:', matplotlib.__version__)
print('Device:', DEVICE)
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print('GPU:', properties.name)
    print(f'GPU memory: {properties.total_memory / 1024**3:.2f} GB')
print('Tensor dtype:', DTYPE)

# In[3]:


# ROUND 3 — CELL 3
# Repository provenance, novelty boundary, and reproducibility manifest
# =============================================================================
# This cell documents exactly how each audited repository influenced the code.
# It is important for discussions with a supervisor or reviewer because it
# separates reusable published ideas from the proposed AMR-specific synthesis.

REPOSITORY_PROVENANCE = {
    'DrugRes_StructPop': {
        'used': [
            'sensitive/resistant stochastic reaction structure',
            'logistic birth, death, mutation during sensitive division',
            'treatment-switch handling and evolutionary-rescue terminology',
        ],
        'not_claimed_as_novel': 'The AMR population model and Gillespie method.',
        'round3_role': 'Provides the deterministic mean-field equations and later stochastic transfer test.',
    },
    'BINNs': {
        'used': [
            'sparse biological inverse-learning philosophy',
            'explicit non-negative output mappings as a baseline',
            'separation of data-fit and mechanistic residual losses',
            'careful output scaling and validation tracking',
        ],
        'not_claimed_as_novel': 'Learning biological dynamics from sparse data.',
        'round3_role': 'Motivates sparse total-load and phenotype-fraction inference.',
    },
    'PINN-Proj': {
        'used': [
            'ordinary versus soft versus exact-constraint comparisons',
            'separate reporting of prediction and constraint errors',
            'L-BFGS refinement after first-order optimization',
        ],
        'not_claimed_as_novel': 'Differentiable projection for conservation constraints.',
        'round3_role': 'Motivates the controlled constraint ablation suite.',
    },
    'Control-PINNs': {
        'used': [
            'treatment represented explicitly as an external system input',
            'hierarchical state representations as a conceptual reference',
        ],
        'excluded': 'No adjoint, stationarity equation, or treatment optimization is used in Paper 1.',
        'round3_role': 'Confirms that prescribed-schedule inference is distinct from optimal control.',
    },
    'evo_dm': {
        'used': [
            'fitness-cost interpretation',
            'evaluation across multiple treatment environments',
        ],
        'excluded': 'No reinforcement-learning or drug-selection policy is trained in Round 3.',
        'round3_role': 'Motivates schedule diversity without broadening Paper 1 into control.',
    },
    'KKT_hPINN': {
        'used': [
            'hard versus soft feasibility comparison',
            'training through an exact differentiable constraint mapping',
            'constraint audits independent of predictive accuracy',
        ],
        'not_claimed_as_novel': 'Hard linear equality constraints using KKT projection.',
        'round3_role': 'Motivates exact feasibility but cannot directly encode the AMR feasible set.',
    },
}

NOVELTY_BOUNDARY = {
    'proposed_contribution': [
        'latent separation of sensitive and resistant populations from total load plus sparse phenotype fractions',
        'coupled total-abundance sigmoid and phenotype softmax representation',
        'exact non-negativity, phenotype simplex, carrying capacity, initial state, and treatment-interface continuity',
        'identifiability-aware inference of mutation probability and resistance fitness cost',
        'transfer of deterministic inverse estimates to independent Gillespie outcome validation',
    ],
    'not_claimed': [
        'first PINN', 'first AMR model', 'first hard-constrained neural network',
        'first Gillespie AMR simulator', 'first antibiotic-control or reinforcement-learning framework',
    ],
}

with open(REPORT_DIR_R3 / 'round3_repository_provenance.json', 'w', encoding='utf-8') as file:
    json.dump({'repositories': REPOSITORY_PROVENANCE,
               'novelty_boundary': NOVELTY_BOUNDARY}, file, indent=2)

pd.DataFrame([
    {'repository': name,
     'round3_role': details.get('round3_role', ''),
     'excluded_or_not_novel': details.get('excluded', details.get('not_claimed_as_novel', ''))}
    for name, details in REPOSITORY_PROVENANCE.items()
])

# In[4]:


# ROUND 3 — CELL 4
# Eco-evolutionary AMR parameters, treatment schedules, and deterministic solver
# =============================================================================
# The state is normalized by carrying capacity during PINN training:
#     s = S/K, r = R/K.
# This keeps both states and ODE residuals near order one and improves PINN
# optimization. Dimensional bacterial counts are restored for all reports.

@dataclass(frozen=True)
class AMRParametersR3:
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
        return self.sensitive_birth_rate * (1.0 - self.resistance_fitness_cost)

    def validate(self) -> None:
        if self.carrying_capacity <= 0 or self.sensitive_birth_rate <= 0:
            raise ValueError('Capacity and sensitive birth rate must be positive.')
        for name, value in {
            'resistance_fitness_cost': self.resistance_fitness_cost,
            'mutation_probability': self.mutation_probability,
            'sensitive_drug_effect': self.sensitive_drug_effect,
            'resistant_drug_effect': self.resistant_drug_effect,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0,1].')


@dataclass(frozen=True)
class ScheduleR3:
    name: str
    switch_times: Tuple[float, ...]
    levels: Tuple[float, ...]
    description: str

    def __post_init__(self):
        if len(self.levels) != len(self.switch_times) + 1:
            raise ValueError('A piecewise schedule requires one more level than switches.')
        if tuple(sorted(self.switch_times)) != self.switch_times:
            raise ValueError('switch_times must be strictly ordered.')
        if any((level < 0 or level > 1) for level in self.levels):
            raise ValueError('Exposure levels must remain in [0,1].')

    def exposure_numpy(self, t: np.ndarray | float) -> np.ndarray | float:
        arr = np.asarray(t, dtype=float)
        indices = np.searchsorted(np.asarray(self.switch_times), arr, side='right')
        values = np.asarray(self.levels, dtype=float)[indices]
        return float(values) if np.isscalar(t) else values

    def exposure_torch(self, t: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(t, float(self.levels[0]))
        for switch, level in zip(self.switch_times, self.levels[1:]):
            result = torch.where(t >= float(switch),
                                 torch.full_like(t, float(level)), result)
        return result

    def intervals(self, start: float, final: float) -> List[Tuple[float, float, float]]:
        boundaries = [start] + [x for x in self.switch_times if start < x < final] + [final]
        intervals = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            midpoint = 0.5 * (left + right)
            intervals.append((left, right, float(self.exposure_numpy(midpoint))))
        return intervals


TRUE_PARAMETERS = AMRParametersR3()
TRUE_PARAMETERS.validate()
INITIAL_S = 100.0
INITIAL_R = 0.0
T_START = 0.0
T_FINAL = 20.0
DENSE_POINTS = 1001

SCHEDULES_R3 = {
    'untreated': ScheduleR3('untreated', (), (0.0,), 'No antibiotic exposure.'),
    'early_continuous': ScheduleR3('early_continuous', (5.0,), (0.0, 1.0),
                                   'Continuous exposure beginning at t=5.'),
    'late_continuous': ScheduleR3('late_continuous', (10.0,), (0.0, 1.0),
                                  'Continuous exposure beginning at t=10.'),
    'pulsed': ScheduleR3('pulsed', (5.0, 8.0, 12.0, 15.0),
                         (0.0, 1.0, 0.0, 1.0, 0.0),
                         'Exposure on [5,8) and [12,15).'),
}


def normalized_amr_rhs_numpy(t: float, state: np.ndarray,
                             parameters: AMRParametersR3,
                             schedule: ScheduleR3) -> np.ndarray:
    """Mean-field ODE for normalized sensitive and resistant abundances."""
    s, r = np.asarray(state, dtype=float)
    exposure = float(schedule.exposure_numpy(t))
    b_s = parameters.sensitive_birth_rate * (1.0 - parameters.sensitive_drug_effect * exposure)
    b_r = parameters.resistant_birth_rate * (1.0 - parameters.resistant_drug_effect * exposure)
    crowding = 1.0 - s - r
    ds = b_s * (1.0 - parameters.mutation_probability) * crowding * s \
         - parameters.sensitive_death_rate * s
    dr = b_s * parameters.mutation_probability * crowding * s \
         + b_r * crowding * r - parameters.resistant_death_rate * r
    return np.array([ds, dr], dtype=float)


def solve_reference_r3(parameters: AMRParametersR3,
                       schedule: ScheduleR3,
                       times: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Solve each treatment interval separately to respect coefficient jumps."""
    if times is None:
        times = np.linspace(T_START, T_FINAL, DENSE_POINTS)
    times = np.asarray(times, dtype=float)
    current_state = np.array([INITIAL_S / parameters.carrying_capacity,
                              INITIAL_R / parameters.carrying_capacity], dtype=float)
    out_s = np.empty_like(times)
    out_r = np.empty_like(times)

    for interval_index, (left, right, _) in enumerate(schedule.intervals(T_START, T_FINAL)):
        is_last = interval_index == len(schedule.intervals(T_START, T_FINAL)) - 1
        mask = (times >= left - 1e-12) & ((times <= right + 1e-12) if is_last else (times < right - 1e-12))
        interval_times = times[mask]
        eval_times = np.unique(np.concatenate(([left], interval_times, [right])))
        sol = solve_ivp(
            lambda t, y: normalized_amr_rhs_numpy(t, y, parameters, schedule),
            (left, right), current_state, t_eval=eval_times,
            method='DOP853', rtol=1e-10, atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        if interval_times.size:
            indices = np.searchsorted(eval_times, interval_times)
            out_s[mask] = sol.y[0, indices]
            out_r[mask] = sol.y[1, indices]
        current_state = sol.y[:, -1]

    S = parameters.carrying_capacity * out_s
    R = parameters.carrying_capacity * out_r
    B = S + R
    q = np.divide(R, B, out=np.zeros_like(R), where=B > 0)
    return pd.DataFrame({
        'time': times, 'sensitive': S, 'resistant': R, 'total': B,
        'resistant_fraction': q,
        'antibiotic': schedule.exposure_numpy(times),
        'schedule': schedule.name,
    })

print(pd.DataFrame([asdict(TRUE_PARAMETERS) | {'resistant_birth_rate': TRUE_PARAMETERS.resistant_birth_rate}]))
print(pd.DataFrame([{'schedule': s.name, 'description': s.description}
                    for s in SCHEDULES_R3.values()]))

# In[5]:


# ROUND 3 — CELL 5
# Load Round 1 data, verify it, and construct O1–O4 observation regimes
# =============================================================================
# Data loading strategy
# ---------------------
# The cell first searches for the dense CSV files produced in Round 1. When a
# file is unavailable, it regenerates the same high-accuracy deterministic
# trajectory. Therefore, Round 3 remains reproducible even after a kernel restart.
#
# Observation regimes
# -------------------
# O1: 21 total-load values, no phenotype fraction.
# O2: 21 total-load values, one phenotype fraction at t=12.
# O3: 21 total-load values, phenotype fractions at t=7,12,18.
# O4: 21 total-load values and phenotype fractions at all 21 times.

@dataclass
class ObservationDataset:
    schedule_name: str
    regime: str
    noise_level: float
    total_time: np.ndarray
    total_observed: np.ndarray
    fraction_time: np.ndarray
    fraction_observed: np.ndarray
    total_train_mask: np.ndarray
    total_validation_mask: np.ndarray
    fraction_train_mask: np.ndarray
    fraction_validation_mask: np.ndarray

    def summary(self) -> Dict[str, Any]:
        return {
            'schedule': self.schedule_name,
            'regime': self.regime,
            'noise_level': self.noise_level,
            'total_points': len(self.total_time),
            'fraction_points': len(self.fraction_time),
            'total_train': int(self.total_train_mask.sum()),
            'total_validation': int(self.total_validation_mask.sum()),
            'fraction_train': int(self.fraction_train_mask.sum()),
            'fraction_validation': int(self.fraction_validation_mask.sum()),
        }


def nearest_values(frame: pd.DataFrame, requested_times: np.ndarray,
                   column: str) -> np.ndarray:
    grid = frame['time'].to_numpy(float)
    values = frame[column].to_numpy(float)
    indices = [int(np.argmin(np.abs(grid - time_value))) for time_value in requested_times]
    return values[np.asarray(indices, dtype=int)]


def load_or_generate_dense(schedule_name: str) -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / 'data' / 'dense' / f'{schedule_name}_dense.csv',
        PROJECT_ROOT / 'AMR_PINN_Round1_Comprehensive' / 'data' / 'dense' / f'{schedule_name}_dense.csv',
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            required = {'time','sensitive','resistant','total','resistant_fraction','antibiotic'}
            if required.issubset(frame.columns):
                return frame
    frame = solve_reference_r3(TRUE_PARAMETERS, SCHEDULES_R3[schedule_name])
    fallback_dir = CACHE_DIR_R3 / 'dense_reference'
    fallback_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(fallback_dir / f'{schedule_name}_dense.csv', index=False)
    return frame


DENSE_REFERENCE = {name: load_or_generate_dense(name) for name in SCHEDULES_R3}


def build_observation_dataset(schedule_name: str, regime: str,
                              noise_level: float = 0.0,
                              seed: int = SEED) -> ObservationDataset:
    if regime not in {'O1','O2','O3','O4'}:
        raise ValueError('regime must be O1, O2, O3, or O4.')
    if noise_level < 0:
        raise ValueError('noise_level cannot be negative.')

    dense = DENSE_REFERENCE[schedule_name]
    total_time = np.linspace(T_START, T_FINAL, 21)
    true_total = nearest_values(dense, total_time, 'total')

    fraction_times_by_regime = {
        'O1': np.array([], dtype=float),
        'O2': np.array([12.0], dtype=float),
        'O3': np.array([7.0, 12.0, 18.0], dtype=float),
        'O4': total_time.copy(),
    }
    fraction_time = fraction_times_by_regime[regime]
    true_fraction = nearest_values(dense, fraction_time, 'resistant_fraction') \
                    if len(fraction_time) else np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    if noise_level == 0:
        total_observed = true_total.copy()
        fraction_observed = true_fraction.copy()
    else:
        sigma_log = np.sqrt(np.log1p(noise_level**2))
        total_observed = true_total * rng.lognormal(-0.5*sigma_log**2,
                                                    sigma_log, len(true_total))
        eps = 1e-8
        clipped = np.clip(true_fraction, eps, 1-eps)
        logits = np.log(clipped / (1-clipped))
        fraction_observed = 1 / (1 + np.exp(-(logits + rng.normal(0, noise_level, len(logits)))))

    # Hold out every fifth total observation for validation, but always retain
    # the initial point in training. With only one phenotype measurement (O2),
    # it is retained for training and validation is based on total load.
    total_validation_mask = (np.arange(len(total_time)) % 5 == 4)
    total_validation_mask[0] = False
    total_train_mask = ~total_validation_mask

    fraction_validation_mask = np.zeros(len(fraction_time), dtype=bool)
    if len(fraction_time) >= 3:
        fraction_validation_mask[-1] = True
    fraction_train_mask = ~fraction_validation_mask

    return ObservationDataset(
        schedule_name=schedule_name, regime=regime, noise_level=noise_level,
        total_time=total_time, total_observed=total_observed,
        fraction_time=fraction_time, fraction_observed=fraction_observed,
        total_train_mask=total_train_mask,
        total_validation_mask=total_validation_mask,
        fraction_train_mask=fraction_train_mask,
        fraction_validation_mask=fraction_validation_mask,
    )

OBSERVATION_EXAMPLES = {
    regime: build_observation_dataset('early_continuous', regime, 0.0, SEED)
    for regime in ['O1','O2','O3','O4']
}

summary = pd.DataFrame([dataset.summary() for dataset in OBSERVATION_EXAMPLES.values()])
summary.to_csv(TABLE_DIR_R3 / 'round3_observation_regime_definitions.csv', index=False)
summary

# In[6]:


# ROUND 3 — CELL 6
# Publication figure: observation design and hidden-state ambiguity
# =============================================================================
# This figure communicates the inverse problem before presenting any model:
# total bacterial load is observed frequently, while phenotype composition is
# observed sparsely. The shaded region shows how much of the total population is
# resistant even though only the total curve is continuously visible.

example_dense = DENSE_REFERENCE['early_continuous']
example_obs = OBSERVATION_EXAMPLES['O3']

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

ax = axes[0,0]
ax.plot(example_dense['time'], example_dense['sensitive'], label='Hidden sensitive $S(t)$')
ax.plot(example_dense['time'], example_dense['resistant'], label='Hidden resistant $R(t)$')
ax.fill_between(example_dense['time'], 0, example_dense['resistant'], alpha=0.18)
ax.axvline(5.0, linestyle='--', linewidth=1.4, label='Treatment begins')
ax.set(xlabel='Time', ylabel='Bacterial abundance', title='A. Hidden phenotype trajectories')
ax.grid(alpha=0.22); ax.legend(loc='best')

ax = axes[0,1]
ax.plot(example_dense['time'], example_dense['total'], label='True total $B(t)$')
ax.scatter(example_obs.total_time[example_obs.total_train_mask],
           example_obs.total_observed[example_obs.total_train_mask], s=38,
           label='Training total-load measurements', zorder=4)
ax.scatter(example_obs.total_time[example_obs.total_validation_mask],
           example_obs.total_observed[example_obs.total_validation_mask], s=55,
           marker='D', label='Withheld validation measurements', zorder=5)
ax.axvline(5.0, linestyle='--', linewidth=1.4)
ax.set(xlabel='Time', ylabel='Total load', title='B. Aggregated observations')
ax.grid(alpha=0.22); ax.legend(loc='best')

ax = axes[1,0]
ax.plot(example_dense['time'], example_dense['resistant_fraction'],
        label='True resistant fraction $q(t)$')
ax.scatter(example_obs.fraction_time[example_obs.fraction_train_mask],
           example_obs.fraction_observed[example_obs.fraction_train_mask],
           s=55, label='Sparse phenotype measurements', zorder=5)
if example_obs.fraction_validation_mask.any():
    ax.scatter(example_obs.fraction_time[example_obs.fraction_validation_mask],
               example_obs.fraction_observed[example_obs.fraction_validation_mask],
               s=60, marker='D', label='Withheld phenotype value', zorder=5)
ax.set(xlabel='Time', ylabel='Resistant fraction', ylim=(-0.03,1.03),
       title='C. Sparse phenotype information')
ax.grid(alpha=0.22); ax.legend(loc='best')

ax = axes[1,1]
regimes = ['O1','O2','O3','O4']
counts = [len(OBSERVATION_EXAMPLES[r].fraction_time) for r in regimes]
ax.bar(regimes, counts)
for x, count in enumerate(counts):
    ax.text(x, count + 0.35, str(count), ha='center', va='bottom')
ax.set(xlabel='Observation regime', ylabel='Phenotype-fraction measurements',
       title='D. Progressively informative regimes')
ax.set_ylim(0, max(counts)+4); ax.grid(axis='y', alpha=0.22)

save_figure_r3(fig, 'round3_observation_design')
plt.show()

# In[7]:


# ROUND 3 — CELL 7
# Tensor conversion, treatment-interface collocation, and reproducible sampling
# =============================================================================
# Collocation points are sampled separately inside every treatment interval.
# Points exactly at a coefficient discontinuity are excluded. This prevents the
# network from being penalized for a derivative jump that is physically expected
# when treatment changes instantaneously.


def tensor(data: np.ndarray | List[float], requires_grad: bool = False) -> torch.Tensor:
    value = torch.as_tensor(np.asarray(data), dtype=DTYPE, device=DEVICE)
    if value.ndim == 1:
        value = value[:, None]
    value.requires_grad_(requires_grad)
    return value


def sample_interface_aware_collocation(schedule: ScheduleR3,
                                       points_per_interval: int,
                                       seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    samples = []
    margin = 1e-6
    for left, right, _ in schedule.intervals(T_START, T_FINAL):
        if right - left <= 2*margin:
            continue
        # Latin-hypercube-like stratification in each interval gives more even
        # temporal coverage than independent uniform samples.
        bins = (np.arange(points_per_interval) + rng.random(points_per_interval)) \
               / points_per_interval
        points = left + margin + bins * (right-left-2*margin)
        rng.shuffle(points)
        samples.append(points)
    merged = np.concatenate(samples)
    return tensor(merged, requires_grad=True)


def observation_to_tensors(dataset: ObservationDataset, split: str = 'train') -> Dict[str, torch.Tensor]:
    if split not in {'train','validation','all'}:
        raise ValueError('split must be train, validation, or all.')
    if split == 'train':
        total_mask = dataset.total_train_mask
        fraction_mask = dataset.fraction_train_mask
    elif split == 'validation':
        total_mask = dataset.total_validation_mask
        fraction_mask = dataset.fraction_validation_mask
    else:
        total_mask = np.ones(len(dataset.total_time), dtype=bool)
        fraction_mask = np.ones(len(dataset.fraction_time), dtype=bool)

    return {
        'total_time': tensor(dataset.total_time[total_mask]),
        'total_observed': tensor(dataset.total_observed[total_mask] / TRUE_PARAMETERS.carrying_capacity),
        'fraction_time': tensor(dataset.fraction_time[fraction_mask]),
        'fraction_observed': tensor(dataset.fraction_observed[fraction_mask]),
    }

# Verify that no sampled point is placed directly on a treatment switch.
check_points = sample_interface_aware_collocation(SCHEDULES_R3['pulsed'], 20, SEED).detach().cpu().numpy().ravel()
for switch in SCHEDULES_R3['pulsed'].switch_times:
    assert np.min(np.abs(check_points-switch)) > 1e-7
print('Interface-aware collocation test passed with', len(check_points), 'points.')

# In[8]:


# ROUND 3 — CELL 8
# Neural building blocks and bounded inverse parameters
# =============================================================================
# The unknown Tier-I parameters are:
#     mutation probability mu in [1e-6, 1e-2] on a logarithmic scale;
#     resistance fitness cost c_R in [0, 0.40].
# Parameter transforms make every optimizer iterate biologically admissible.

class TanhMLP(nn.Module):
    """Four-layer tanh MLP with Xavier initialization."""
    def __init__(self, input_dim: int, output_dim: int,
                 width: int = 64, depth: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for _ in range(depth):
            linear = nn.Linear(previous, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.Tanh()]
            previous = width
        output = nn.Linear(previous, output_dim)
        nn.init.xavier_normal_(output.weight, gain=0.5)
        nn.init.zeros_(output.bias)
        layers.append(output)
        # Construct all trainable layers directly in float64. Calling model.to(DEVICE)
        # changes the device but not the dtype, so this explicit conversion prevents
        # Float/Double mismatches during automatic differentiation.
        self.network = nn.Sequential(*layers).to(dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def inverse_sigmoid(value: float) -> float:
    value = float(np.clip(value, 1e-8, 1-1e-8))
    return math.log(value/(1-value))


class TierIInverseParameters(nn.Module):
    """Bounded trainable mutation probability and resistance fitness cost."""
    def __init__(self, initial_mu: float = 3e-4, initial_cost: float = 0.15,
                 trainable: bool = True):
        super().__init__()
        self.log10_mu_min = -6.0
        self.log10_mu_max = -2.0
        mu_position = (math.log10(initial_mu)-self.log10_mu_min) \
                      / (self.log10_mu_max-self.log10_mu_min)
        cost_position = initial_cost / 0.40
        self.raw_mu = nn.Parameter(torch.tensor(inverse_sigmoid(mu_position), dtype=DTYPE),
                                   requires_grad=trainable)
        self.raw_cost = nn.Parameter(torch.tensor(inverse_sigmoid(cost_position), dtype=DTYPE),
                                     requires_grad=trainable)

    def mutation_probability(self) -> torch.Tensor:
        log10_mu = self.log10_mu_min + (self.log10_mu_max-self.log10_mu_min) \
                   * torch.sigmoid(self.raw_mu)
        return torch.pow(torch.tensor(10.0, dtype=DTYPE, device=self.raw_mu.device), log10_mu)

    def resistance_fitness_cost(self) -> torch.Tensor:
        return 0.40 * torch.sigmoid(self.raw_cost)

    def as_floats(self) -> Dict[str, float]:
        return {
            'mutation_probability': float(self.mutation_probability().detach().cpu()),
            'resistance_fitness_cost': float(self.resistance_fitness_cost().detach().cpu()),
        }

# In[9]:


# ROUND 3 — CELL 9
# Treatment-interface state network and all architectural ablations
# =============================================================================
# Available model kinds
# ---------------------
# data_only : unconstrained state network, no ODE residual.
# vanilla   : unconstrained PINN with exact IC/interface continuity only.
# soft      : vanilla PINN plus penalties for negativity and capacity excess.
# positive  : softplus S and R, but total abundance is not bounded by K.
# simplex   : positive total and exact phenotype fractions, but total is unbounded.
# hard      : proposed bounded total + phenotype softmax; exact feasible set.
#
# Every model uses one subnetwork per treatment interval. The state at the end
# of one interval becomes the exact anchor of the next interval. This gives
# continuous S and R while allowing their derivatives to respond immediately
# to the changed treatment coefficient.

class InterfaceStateNetwork(nn.Module):
    def __init__(self, schedule: ScheduleR3, model_kind: str,
                 initial_s: float, initial_r: float,
                 width: int = 64, depth: int = 4, tau_h: float = 0.35):
        super().__init__()
        allowed = {'data_only','vanilla','soft','positive','simplex','hard'}
        if model_kind not in allowed:
            raise ValueError(f'Unknown model_kind {model_kind}.')
        self.schedule = schedule
        self.model_kind = model_kind
        self.initial_s = float(initial_s)
        self.initial_r = float(initial_r)
        self.tau_h = float(tau_h)
        self.intervals = schedule.intervals(T_START, T_FINAL)
        output_dim = 3 if model_kind in {'simplex','hard'} else 2
        self.segment_networks = nn.ModuleList([
            TanhMLP(1, output_dim, width=width, depth=depth)
            for _ in self.intervals
        ])

    def _scaled_local_time(self, t: torch.Tensor, left: float, right: float) -> torch.Tensor:
        return 2.0 * (t-left)/(right-left) - 1.0

    def _transform(self, raw: torch.Tensor, local_elapsed: torch.Tensor,
                   anchor: torch.Tensor) -> torch.Tensor:
        h = 1.0 - torch.exp(-torch.clamp(local_elapsed, min=0.0)/self.tau_h)
        anchor = anchor.expand(raw.shape[0], -1)

        if self.model_kind in {'data_only','vanilla','soft'}:
            candidate = raw[:, :2]
            return (1-h)*anchor + h*candidate

        if self.model_kind == 'positive':
            candidate = F.softplus(raw[:, :2])
            return (1-h)*anchor + h*candidate

        anchor_total = torch.clamp(anchor.sum(dim=1, keepdim=True), min=1e-12)
        anchor_fraction = anchor / anchor_total
        candidate_fraction = torch.softmax(raw[:, 1:3], dim=1)
        fraction = (1-h)*anchor_fraction + h*candidate_fraction

        if self.model_kind == 'simplex':
            candidate_total = F.softplus(raw[:, 0:1])
        else:  # proposed hard model
            candidate_total = torch.sigmoid(raw[:, 0:1])
        total = (1-h)*anchor_total + h*candidate_total
        return total * fraction

    def _anchors(self) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        anchor = torch.tensor([[self.initial_s, self.initial_r]], dtype=DTYPE, device=device)
        anchors = [anchor]
        for index in range(1, len(self.intervals)):
            previous_left, previous_right, _ = self.intervals[index-1]
            switch_tensor = torch.tensor([[previous_right]], dtype=DTYPE, device=device)
            scaled = self._scaled_local_time(switch_tensor, previous_left, previous_right)
            raw = self.segment_networks[index-1](scaled)
            anchor = self._transform(raw, switch_tensor-previous_left, anchor)
            anchors.append(anchor)
        return anchors

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        anchors = self._anchors()
        output = torch.zeros((len(t),2), dtype=t.dtype, device=t.device)
        for index, (left, right, _) in enumerate(self.intervals):
            is_last = index == len(self.intervals)-1
            mask = (t[:,0] >= left-1e-12) & ((t[:,0] <= right+1e-12) if is_last else (t[:,0] < right-1e-12))
            if mask.any():
                local_t = t[mask]
                scaled = self._scaled_local_time(local_t, left, right)
                raw = self.segment_networks[index](scaled)
                output[mask] = self._transform(raw, local_t-left, anchors[index])
        return output


class AMRInverseModel(nn.Module):
    """One or more experiment-specific state networks sharing Tier-I parameters."""
    def __init__(self, schedules: Dict[str, ScheduleR3], model_kind: str,
                 width: int = 64, depth: int = 4,
                 parameter_initialization: Tuple[float,float] = (3e-4,0.15)):
        super().__init__()
        self.model_kind = model_kind
        self.physics_enabled = model_kind != 'data_only'
        self.state_networks = nn.ModuleDict({
            name: InterfaceStateNetwork(schedule, model_kind,
                                        INITIAL_S/TRUE_PARAMETERS.carrying_capacity,
                                        INITIAL_R/TRUE_PARAMETERS.carrying_capacity,
                                        width=width, depth=depth)
            for name, schedule in schedules.items()
        })
        self.inverse_parameters = TierIInverseParameters(
            parameter_initialization[0], parameter_initialization[1],
            trainable=self.physics_enabled,
        )

    def forward(self, t: torch.Tensor, schedule_name: str) -> torch.Tensor:
        return self.state_networks[schedule_name](t)

    def inferred_parameters(self) -> Dict[str,float]:
        if not self.physics_enabled:
            return {'mutation_probability': np.nan,
                    'resistance_fitness_cost': np.nan}
        return self.inverse_parameters.as_floats()

# Architectural guarantees are verified before training.
for kind in ['data_only','vanilla','soft','positive','simplex','hard']:
    trial = AMRInverseModel({'early_continuous': SCHEDULES_R3['early_continuous']},
                            kind, width=16, depth=2).to(DEVICE)
    t_test = tensor(np.linspace(0,20,101))
    with torch.no_grad():
        state = trial(t_test, 'early_continuous').cpu().numpy()
    assert np.max(np.abs(state[0]-np.array([0.1,0.0]))) < 1e-12
    if kind in {'positive','simplex','hard'}:
        assert state.min() >= -1e-12
    if kind == 'hard':
        assert state.sum(axis=1).max() <= 1+1e-12
    del trial
clear_gpu_memory()
print('All untrained architecture guarantee tests passed.')

# In[10]:


# ROUND 3 — CELL 10
# Physics residuals, observation likelihoods, and soft-constraint penalties
# =============================================================================
# Total-load data are compared on a log scale because bacterial abundance spans
# a wide range. Phenotype fractions are compared directly in [0,1].
# The proposed hard model receives no positivity or capacity penalties because
# those properties are exact consequences of its output mapping.

@dataclass
class LossWeights:
    total_data: float = 10.0
    fraction_data: float = 5.0
    physics: float = 1.0
    soft_constraint: float = 10.0
    weight_decay: float = 1e-8

LOSS_WEIGHTS = LossWeights()


def physics_residual(model: AMRInverseModel, t: torch.Tensor,
                     schedule_name: str) -> Tuple[torch.Tensor,torch.Tensor]:
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)
    state = model(t, schedule_name)
    s = state[:,0:1]
    r = state[:,1:2]
    ds_dt = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True,
                                retain_graph=True)[0]
    dr_dt = torch.autograd.grad(r, t, torch.ones_like(r), create_graph=True,
                                retain_graph=True)[0]

    mu = model.inverse_parameters.mutation_probability()
    cost = model.inverse_parameters.resistance_fitness_cost()
    schedule = SCHEDULES_R3[schedule_name]
    exposure = schedule.exposure_torch(t)
    b_s = TRUE_PARAMETERS.sensitive_birth_rate * \
          (1.0-TRUE_PARAMETERS.sensitive_drug_effect*exposure)
    b_r = TRUE_PARAMETERS.sensitive_birth_rate * (1.0-cost) * \
          (1.0-TRUE_PARAMETERS.resistant_drug_effect*exposure)
    crowding = 1.0-s-r
    rhs_s = b_s*(1.0-mu)*crowding*s - TRUE_PARAMETERS.sensitive_death_rate*s
    rhs_r = b_s*mu*crowding*s + b_r*crowding*r - TRUE_PARAMETERS.resistant_death_rate*r
    return ds_dt-rhs_s, dr_dt-rhs_r


def data_loss_for_dataset(model: AMRInverseModel, dataset: ObservationDataset,
                          split: str = 'train') -> Tuple[torch.Tensor,torch.Tensor]:
    values = observation_to_tensors(dataset, split)
    eps = torch.tensor(1e-10, dtype=DTYPE, device=DEVICE)

    if len(values['total_time']):
        state_total = model(values['total_time'], dataset.schedule_name)
        predicted_total = state_total.sum(dim=1, keepdim=True)
        # Unconstrained baselines can temporarily predict a non-positive total.
        # A lower clamp keeps the logarithmic likelihood finite; feasibility is
        # still measured separately and the soft baseline receives explicit
        # negativity/capacity penalties. The proposed hard model never uses it.
        predicted_total_for_log = torch.clamp(predicted_total, min=float(eps))
        total_loss = torch.mean((torch.log(predicted_total_for_log)
                                 - torch.log(values['total_observed']+eps))**2)
    else:
        total_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)

    if len(values['fraction_time']):
        state_fraction = model(values['fraction_time'], dataset.schedule_name)
        denominator = torch.clamp(state_fraction.sum(dim=1, keepdim=True), min=float(eps))
        predicted_fraction = state_fraction[:,1:2] / denominator
        # Clipping is used only in the observation operator for invalid baseline
        # states. Raw S/R predictions remain unchanged for constraint auditing.
        predicted_fraction = torch.clamp(predicted_fraction, min=0.0, max=1.0)
        fraction_loss = torch.mean((predicted_fraction-values['fraction_observed'])**2)
    else:
        fraction_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
    return total_loss, fraction_loss


def regularization_loss(model: nn.Module) -> torch.Tensor:
    value = torch.zeros((), dtype=DTYPE, device=DEVICE)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.ndim >= 2:
            value = value + torch.mean(parameter**2)
    return value


def soft_constraint_loss(model: AMRInverseModel,
                         collocation: Dict[str,torch.Tensor]) -> torch.Tensor:
    if model.model_kind != 'soft' or not collocation:
        # During data-only pretraining no collocation points are supplied, so
        # the soft-constraint term is intentionally inactive.
        return torch.zeros((), dtype=DTYPE, device=DEVICE)
    penalties = []
    for schedule_name, times in collocation.items():
        state = model(times, schedule_name)
        negative = F.relu(-state)
        capacity = F.relu(state.sum(dim=1,keepdim=True)-1.0)
        penalties.append(torch.mean(negative**2)+torch.mean(capacity**2))
    return torch.stack(penalties).mean()


def complete_loss(model: AMRInverseModel,
                  datasets: Dict[str,ObservationDataset],
                  collocation: Dict[str,torch.Tensor],
                  physics_weight: float,
                  split: str = 'train') -> Tuple[torch.Tensor,Dict[str,float]]:
    total_terms, fraction_terms, physics_terms = [], [], []
    for schedule_name, dataset in datasets.items():
        load_loss, fraction_loss = data_loss_for_dataset(model, dataset, split)
        total_terms.append(load_loss)
        fraction_terms.append(fraction_loss)
        if model.physics_enabled and schedule_name in collocation:
            residual_s, residual_r = physics_residual(model, collocation[schedule_name], schedule_name)
            physics_terms.append(torch.mean(residual_s**2)+torch.mean(residual_r**2))

    load_loss = torch.stack(total_terms).mean()
    fraction_loss = torch.stack(fraction_terms).mean()
    phys_loss = torch.stack(physics_terms).mean() if physics_terms \
                else torch.zeros((),dtype=DTYPE,device=DEVICE)
    constraint_loss = soft_constraint_loss(model, collocation)
    reg_loss = regularization_loss(model)

    total = LOSS_WEIGHTS.total_data*load_loss \
            + LOSS_WEIGHTS.fraction_data*fraction_loss \
            + physics_weight*LOSS_WEIGHTS.physics*phys_loss \
            + LOSS_WEIGHTS.soft_constraint*constraint_loss \
            + LOSS_WEIGHTS.weight_decay*reg_loss
    components = {
        'total': float(total.detach().cpu()),
        'load': float(load_loss.detach().cpu()),
        'fraction': float(fraction_loss.detach().cpu()),
        'physics': float(phys_loss.detach().cpu()),
        'constraint': float(constraint_loss.detach().cpu()),
        'regularization': float(reg_loss.detach().cpu()),
    }
    return total, components

# In[11]:


# ROUND 3 — CELL 11
# Three-stage optimizer, early stopping, checkpointing, and training history
# =============================================================================
# Stage I  : data pretraining.
# Stage II : joint AdamW optimization with periodically resampled collocation.
# Stage III: full-batch L-BFGS refinement using fixed collocation points.
#
# Memory policy for 8 GB GPU
# --------------------------
# Only one model is trained at a time. The best state_dict is moved to CPU, and
# CUDA cache is cleared after each experiment. No mixed precision is used because
# float64 derivatives and very small mutation probabilities are more stable.

@dataclass
class TrainingResult:
    label: str
    model: AMRInverseModel
    history: pd.DataFrame
    elapsed_seconds: float
    best_validation_score: float


def make_collocation_bundle(datasets: Dict[str,ObservationDataset], seed: int,
                            points_per_interval: Optional[int] = None) -> Dict[str,torch.Tensor]:
    count = points_per_interval or COLLOCATION_POINTS_PER_INTERVAL
    return {
        name: sample_interface_aware_collocation(SCHEDULES_R3[name], count,
                                                 seed+1009*index)
        for index, name in enumerate(datasets)
    }


def validation_score(model: AMRInverseModel,
                     datasets: Dict[str,ObservationDataset],
                     seed: int) -> float:
    collocation = make_collocation_bundle(datasets, seed,
                                          max(32, COLLOCATION_POINTS_PER_INTERVAL//4))
    _, components = complete_loss(model, datasets, collocation,
                                  physics_weight=1.0 if model.physics_enabled else 0.0,
                                  split='validation')
    # When a validation split has no phenotype point, fraction loss is zero.
    return components['load'] + components['fraction'] + 0.1*components['physics']


def train_inverse_model(model: AMRInverseModel,
                        datasets: Dict[str,ObservationDataset],
                        label: str,
                        seed: int = SEED) -> TrainingResult:
    model = model.to(device=DEVICE, dtype=DTYPE)
    start_time = time.time()
    history_rows: List[Dict[str,Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = np.inf
    stale_epochs = 0

    # ----------------------------- Stage I -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    empty_collocation: Dict[str,torch.Tensor] = {}
    for epoch in range(PRETRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        total, components = complete_loss(model, datasets, empty_collocation,
                                          physics_weight=0.0, split='train')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if epoch % LOG_EVERY == 0 or epoch == PRETRAIN_EPOCHS-1:
            score = validation_score(model, datasets, seed+epoch)
            history_rows.append({'stage':'pretrain','epoch':epoch,
                                 'validation_score':score, **components,
                                 **model.inferred_parameters()})
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    # ----------------------------- Stage II ----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(50, ADAM_EPOCHS//20),
        min_lr=1e-5,
    )
    for epoch in range(ADAM_EPOCHS):
        collocation = make_collocation_bundle(datasets, seed+10_000+epoch)
        optimizer.zero_grad(set_to_none=True)
        total, components = complete_loss(model, datasets, collocation,
                                          physics_weight=1.0, split='train')
        if not torch.isfinite(total):
            raise FloatingPointError(f'Non-finite loss in {label} at Adam epoch {epoch}.')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == ADAM_EPOCHS-1:
            score = validation_score(model, datasets, seed+20_000+epoch)
            scheduler.step(score)
            row = {'stage':'adam','epoch':epoch,
                   'validation_score':score,
                   'learning_rate':optimizer.param_groups[0]['lr'],
                   **components, **model.inferred_parameters()}
            history_rows.append(row)
            if score < best_score*(1-1e-6):
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += LOG_EVERY
            if epoch % max(LOG_EVERY, ADAM_EPOCHS//10) == 0:
                print(f'{label:28s} | Adam {epoch:6d}/{ADAM_EPOCHS} | '
                      f'loss={components["total"]:.3e} | val={score:.3e} | '
                      f'mu={model.inferred_parameters()["mutation_probability"]:.3e} | '
                      f'c={model.inferred_parameters()["resistance_fitness_cost"]:.4f}')
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print(f'{label}: Adam early stopping at epoch {epoch}.')
                break

    model.load_state_dict(best_state)

    # ----------------------------- Stage III ---------------------------
    fixed_collocation = make_collocation_bundle(datasets, seed+90_000,
                                                 COLLOCATION_POINTS_PER_INTERVAL)
    if LBFGS_MAX_ITER > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=0.8, max_iter=LBFGS_MAX_ITER,
            max_eval=max(LBFGS_MAX_ITER*2, 10), tolerance_grad=1e-10,
            tolerance_change=1e-12, history_size=50, line_search_fn='strong_wolfe',
        )
        closure_count = {'value':0}
        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            total, _ = complete_loss(model, datasets, fixed_collocation,
                                     physics_weight=1.0 if model.physics_enabled else 0.0,
                                     split='train')
            total.backward()
            closure_count['value'] += 1
            return total
        optimizer_lbfgs.step(closure)
        total, components = complete_loss(model, datasets, fixed_collocation,
                                          physics_weight=1.0 if model.physics_enabled else 0.0,
                                          split='train')
        score = validation_score(model, datasets, seed+99_000)
        history_rows.append({'stage':'lbfgs','epoch':closure_count['value'],
                             'validation_score':score, **components,
                             **model.inferred_parameters()})
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    elapsed = time.time()-start_time
    history = pd.DataFrame(history_rows)
    history.to_csv(HISTORY_DIR/f'{label}_history.csv', index=False)
    torch.save({'model_state': model.state_dict(),
                'model_kind': model.model_kind,
                'schedules': list(datasets),
                'inferred_parameters': model.inferred_parameters(),
                'training_budget': {
                    'quick_mode': QUICK_MODE,
                    'pretrain_epochs': PRETRAIN_EPOCHS,
                    'adam_epochs': ADAM_EPOCHS,
                    'lbfgs_max_iter': LBFGS_MAX_ITER,
                    'collocation_points_per_interval': COLLOCATION_POINTS_PER_INTERVAL,
                }}, MODEL_DIR/f'{label}.pt')
    print(f'{label} completed in {elapsed/60:.2f} min; best validation score={best_score:.3e}.')
    return TrainingResult(label, model, history, elapsed, best_score)

# In[12]:


# ROUND 3 — CELL 12
# Dense evaluation, state/parameter metrics, constraint audit, and inference time
# =============================================================================


def predict_dense(model: AMRInverseModel, schedule_name: str,
                  dense: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    dense = DENSE_REFERENCE[schedule_name] if dense is None else dense
    t_values = dense['time'].to_numpy(float)
    t_tensor = tensor(t_values)
    model.eval()
    with torch.no_grad():
        prediction = model(t_tensor, schedule_name).detach().cpu().numpy()
    S = TRUE_PARAMETERS.carrying_capacity*prediction[:,0]
    R = TRUE_PARAMETERS.carrying_capacity*prediction[:,1]
    B = S+R
    q = np.divide(R,B,out=np.zeros_like(R),where=B>0)
    frame = dense.copy()
    frame['predicted_sensitive'] = S
    frame['predicted_resistant'] = R
    frame['predicted_total'] = B
    frame['predicted_resistant_fraction'] = q
    return frame


def relative_l2(predicted: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(predicted-true)/(np.linalg.norm(true)+1e-12))


def residual_rmse_dense(model: AMRInverseModel, schedule_name: str) -> float:
    schedule = SCHEDULES_R3[schedule_name]
    points = []
    for left,right,_ in schedule.intervals(T_START,T_FINAL):
        points.append(np.linspace(left+1e-5,right-1e-5,150))
    t_res = tensor(np.concatenate(points), requires_grad=True)
    if not model.physics_enabled:
        return np.nan
    residual_s,residual_r = physics_residual(model,t_res,schedule_name)
    value = torch.sqrt(torch.mean(residual_s**2+residual_r**2))
    return float(value.detach().cpu())


def inference_milliseconds(model: AMRInverseModel, schedule_name: str,
                           n_points: int = 1001, repeats: int = 20) -> float:
    t_eval = tensor(np.linspace(T_START,T_FINAL,n_points))
    with torch.no_grad():
        _ = model(t_eval,schedule_name)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(repeats):
            _=model(t_eval,schedule_name)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    return 1000*(time.perf_counter()-start)/repeats


def evaluate_trained_model(result: TrainingResult,
                           schedule_name: str) -> Tuple[Dict[str,Any],pd.DataFrame]:
    frame = predict_dense(result.model,schedule_name)
    metrics: Dict[str,Any] = {
        'label':result.label, 'model_kind':result.model.model_kind,
        'schedule':schedule_name, 'training_seconds':result.elapsed_seconds,
        'best_validation_score':result.best_validation_score,
    }
    pairs = {
        'sensitive':('sensitive','predicted_sensitive'),
        'resistant':('resistant','predicted_resistant'),
        'total':('total','predicted_total'),
        'fraction':('resistant_fraction','predicted_resistant_fraction'),
    }
    for name,(true_col,pred_col) in pairs.items():
        true=frame[true_col].to_numpy(float); pred=frame[pred_col].to_numpy(float)
        metrics[f'rmse_{name}']=float(np.sqrt(mean_squared_error(true,pred)))
        metrics[f'mae_{name}']=float(mean_absolute_error(true,pred))
        metrics[f'r2_{name}']=float(r2_score(true,pred))
        metrics[f'relative_l2_{name}']=relative_l2(pred,true)

    S=frame['predicted_sensitive'].to_numpy(float)
    R=frame['predicted_resistant'].to_numpy(float)
    metrics.update({
        'maximum_negative_violation':float(max(0,-S.min(),-R.min())),
        'maximum_capacity_violation':float(max(0,(S+R).max()-TRUE_PARAMETERS.carrying_capacity)),
        'fraction_points_negative':float(np.mean((S<0)|(R<0))),
        'fraction_points_over_capacity':float(np.mean(S+R>TRUE_PARAMETERS.carrying_capacity+1e-8)),
        'physics_residual_rmse':residual_rmse_dense(result.model,schedule_name),
        'inference_ms_1001_points':inference_milliseconds(result.model,schedule_name),
        **result.model.inferred_parameters(),
    })
    if result.model.physics_enabled:
        metrics['relative_error_mu']=abs(metrics['mutation_probability']-TRUE_PARAMETERS.mutation_probability)/TRUE_PARAMETERS.mutation_probability
        metrics['relative_error_cost']=abs(metrics['resistance_fitness_cost']-TRUE_PARAMETERS.resistance_fitness_cost)/TRUE_PARAMETERS.resistance_fitness_cost
    else:
        metrics['relative_error_mu']=np.nan; metrics['relative_error_cost']=np.nan
    frame.to_csv(PREDICTION_DIR/f'{result.label}_{schedule_name}_predictions.csv',index=False)
    return metrics,frame

# In[13]:


# ROUND 3 — CELL 13
# Classical ODE nonlinear least-squares baseline with multiple starting points
# =============================================================================
# This baseline asks how well conventional parameter fitting performs when the
# states are generated by numerical integration rather than a neural network.
# It is essential because a PINN should not be compared only with other PINNs.


def ode_prediction_for_parameters(schedule_name: str, times: np.ndarray,
                                  log10_mu: float, cost: float) -> pd.DataFrame:
    parameters = replace(TRUE_PARAMETERS,
                         mutation_probability=10.0**log10_mu,
                         resistance_fitness_cost=float(cost))
    return solve_reference_r3(parameters,SCHEDULES_R3[schedule_name],times)


def ode_residual_vector(x: np.ndarray,
                       datasets: Dict[str,ObservationDataset]) -> np.ndarray:
    log10_mu,cost=x
    residuals=[]
    for schedule_name,dataset in datasets.items():
        all_times=np.unique(np.concatenate([dataset.total_time,dataset.fraction_time]))
        pred=ode_prediction_for_parameters(schedule_name,all_times,log10_mu,cost)
        pred_total=nearest_values(pred,dataset.total_time,'total')
        residuals.extend(np.log(pred_total+1e-8)-np.log(dataset.total_observed+1e-8))
        if len(dataset.fraction_time):
            pred_fraction=nearest_values(pred,dataset.fraction_time,'resistant_fraction')
            residuals.extend(2.0*(pred_fraction-dataset.fraction_observed))
    return np.asarray(residuals,float)


def fit_ode_baseline(datasets: Dict[str,ObservationDataset],
                     label: str, seed: int=SEED,
                     starts: int=12) -> Dict[str,Any]:
    rng=np.random.default_rng(seed)
    initial_points=[np.array([-4.0,0.10])]
    initial_points += [np.array([rng.uniform(-5.8,-2.2),rng.uniform(0.005,0.38)])
                       for _ in range(starts-1)]
    best=None
    start_time=time.time()
    for initial in initial_points:
        fit=least_squares(lambda x: ode_residual_vector(x,datasets),initial,
                          bounds=([-6.0,0.0],[-2.0,0.40]),
                          xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=400)
        if best is None or np.sum(fit.fun**2)<np.sum(best.fun**2):
            best=fit
    elapsed=time.time()-start_time
    result={
        'label':label,'model_kind':'ode_fit',
        'mutation_probability':10.0**best.x[0],
        'resistance_fitness_cost':best.x[1],
        'objective':float(np.sum(best.fun**2)),
        'nfev':int(best.nfev),'training_seconds':elapsed,
        'relative_error_mu':abs(10.0**best.x[0]-TRUE_PARAMETERS.mutation_probability)/TRUE_PARAMETERS.mutation_probability,
        'relative_error_cost':abs(best.x[1]-TRUE_PARAMETERS.resistance_fitness_cost)/TRUE_PARAMETERS.resistance_fitness_cost,
    }
    with open(REPORT_DIR_R3/f'{label}_ode_fit.json','w') as file:
        json.dump(result,file,indent=2)
    return result

print('ODE baseline functions are ready.')

# In[14]:


# ROUND 3 — CELL 14
# Single proposed-model demonstration: early treatment, O3, noise-free data
# =============================================================================
# This is the first cell to perform neural training. It should be run before the
# larger experiment suites. In QUICK_MODE it verifies the complete pipeline. In
# final mode it produces the principal proposed-model reconstruction.

PRIMARY_DATASET = build_observation_dataset('early_continuous','O3',0.0,SEED)
PRIMARY_BUNDLE = {'early_continuous':PRIMARY_DATASET}
PRIMARY_HARD_RESULT = None
PRIMARY_HARD_METRICS = None
PRIMARY_HARD_PREDICTION = None

if RUN_SINGLE_MODEL_DEMO:
    primary_model = AMRInverseModel(
        {'early_continuous':SCHEDULES_R3['early_continuous']},
        model_kind='hard', width=64, depth=4,
        parameter_initialization=(3e-4,0.15),
    )
    PRIMARY_HARD_RESULT = train_inverse_model(primary_model,PRIMARY_BUNDLE,
                                               'round3_primary_hard_O3',SEED)
    PRIMARY_HARD_METRICS,PRIMARY_HARD_PREDICTION = evaluate_trained_model(
        PRIMARY_HARD_RESULT,'early_continuous')
    pd.DataFrame([PRIMARY_HARD_METRICS]).to_csv(
        TABLE_DIR_R3/'round3_primary_hard_metrics.csv',index=False)
    display(pd.DataFrame([PRIMARY_HARD_METRICS]).T.rename(columns={0:'value'}))
else:
    print('RUN_SINGLE_MODEL_DEMO=False; no model was trained in this cell.')

# In[15]:


# ROUND 3 — CELL 15
# Publication figure: proposed-model reconstruction, errors, and parameters
# =============================================================================

if PRIMARY_HARD_RESULT is not None:
    frame=PRIMARY_HARD_PREDICTION
    obs=PRIMARY_DATASET
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.6),constrained_layout=True)

    LEGEND_STYLE = dict(fontsize=9, framealpha=0.95, borderaxespad=0.6)

    # --- A: peaks mid-left, empty in the upper-right as the curves decay ---
    ax=axes[0,0]
    ax.plot(frame.time,frame.sensitive,label='Reference sensitive')
    ax.plot(frame.time,frame.predicted_sensitive,linestyle='--',label='PINN sensitive')
    ax.plot(frame.time,frame.resistant,label='Reference resistant')
    ax.plot(frame.time,frame.predicted_resistant,linestyle='--',label='PINN resistant')
    ax.axvline(5,linestyle=':',linewidth=1.5)
    ax.set(xlabel='Time',ylabel='Bacterial abundance',title='A. Hidden-state reconstruction')
    ax.set_ylim(top=ax.get_ylim()[1]*1.18)  # headroom so legend clears the peak
    ax.grid(alpha=.22); ax.legend(loc='upper right', ncol=2, **LEGEND_STYLE)

    # --- B: same shape as A, empty upper-right ---
    ax=axes[0,1]
    ax.plot(frame.time,frame.total,label='Reference total')
    ax.plot(frame.time,frame.predicted_total,linestyle='--',label='PINN total')
    ax.scatter(obs.total_time[obs.total_train_mask],obs.total_observed[obs.total_train_mask],s=30,label='Training data')
    ax.scatter(obs.total_time[obs.total_validation_mask],obs.total_observed[obs.total_validation_mask],s=45,marker='D',label='Validation data')
    ax.set(xlabel='Time',ylabel='Total bacterial load',title='B. Aggregated-load fit')
    ax.set_ylim(top=ax.get_ylim()[1]*1.18)
    ax.grid(alpha=.22); ax.legend(loc='upper right', **LEGEND_STYLE)

    # --- C: fraction stays near 0 for most of the range, top half is empty ---
    ax=axes[1,0]
    ax.plot(frame.time,frame.resistant_fraction,label='Reference fraction')
    ax.plot(frame.time,frame.predicted_resistant_fraction,linestyle='--',label='PINN fraction')
    ax.scatter(obs.fraction_time[obs.fraction_train_mask],obs.fraction_observed[obs.fraction_train_mask],s=45,label='Training phenotype')
    if obs.fraction_validation_mask.any():
        ax.scatter(obs.fraction_time[obs.fraction_validation_mask],obs.fraction_observed[obs.fraction_validation_mask],s=55,marker='D',label='Validation phenotype')
    ax.set(xlabel='Time',ylabel='Resistant fraction',ylim=(-.03,1.03),title='C. Latent phenotype composition')
    ax.grid(alpha=.22); ax.legend(loc='upper left', **LEGEND_STYLE)

    # --- D: log scale — push ylim way up so legend/text sit in headroom the curves never reach ---
    ax=axes[1,1]
    error_s=np.abs(frame.predicted_sensitive-frame.sensitive)
    error_r=np.abs(frame.predicted_resistant-frame.resistant)
    ax.semilogy(frame.time,error_s+1e-10,label='$|S_\theta-S^*|$')
    ax.semilogy(frame.time,error_r+1e-10,label='$|R_\theta-R^*|$')
    ax.set(xlabel='Time',ylabel='Absolute error',title='D. Pointwise reconstruction error')
    ax.set_ylim(1e-10, 1e4)  # headroom above the curves (which top out ~1e1)
    ax.grid(alpha=.22,which='both')
    ax.legend(loc='upper left', **LEGEND_STYLE)
    text=(f"$\\hat{{\\mu}}$={PRIMARY_HARD_METRICS['mutation_probability']:.2e}\n"
          f"$\\hat{{c}}_R$={PRIMARY_HARD_METRICS['resistance_fitness_cost']:.3f}\n"
          f"$V_K$={PRIMARY_HARD_METRICS['maximum_capacity_violation']:.1e}")
    ax.text(.98,.98,text,transform=ax.transAxes,ha='right',va='top',fontsize=9,
            bbox=dict(boxstyle='round,pad=.4',facecolor='white',alpha=.95))

    save_figure_r3(fig,'round3_primary_hard_reconstruction')
    plt.show()
else:
    print('Run Cell 14 with RUN_SINGLE_MODEL_DEMO=True before plotting.')

# In[16]:


print("RUN_PRIMARY_ABLATION:", RUN_PRIMARY_ABLATION)
print("RUN_SINGLE_MODEL_DEMO:", RUN_SINGLE_MODEL_DEMO)
print("PRIMARY_DATASET exists:", "PRIMARY_DATASET" in globals())
print("PRIMARY_BUNDLE exists:", "PRIMARY_BUNDLE" in globals())
print("ABLATION_METRICS exists:", "ABLATION_METRICS" in globals())
print("ABLATION_PREDICTIONS exists:", "ABLATION_PREDICTIONS" in globals())

# In[18]:


# =============================================================================
# TEMPORARY SWITCH OVERRIDE FOR CORRECTED CELL 16
# =============================================================================

RUN_SINGLE_MODEL_DEMO = False
RUN_PRIMARY_ABLATION = True
RUN_OBSERVATION_REGIME_STUDY = False
RUN_NOISE_ROBUSTNESS_STUDY = False
RUN_MULTI_SCHEDULE_STUDY = False
RUN_PROFILE_LIKELIHOOD = False
RUN_STOCHASTIC_BENCHMARK = False
RUN_STOCHASTIC_TRANSFER_VALIDATION = False

CURRENT_EXPERIMENT = "primary_ablation"

EXPERIMENT_SWITCHES = {
    "single_model_demo": RUN_SINGLE_MODEL_DEMO,
    "primary_ablation": RUN_PRIMARY_ABLATION,
    "observation_regime_study": RUN_OBSERVATION_REGIME_STUDY,
    "noise_robustness_study": RUN_NOISE_ROBUSTNESS_STUDY,
    "multi_schedule_study": RUN_MULTI_SCHEDULE_STUDY,
    "profile_likelihood": RUN_PROFILE_LIKELIHOOD,
    "stochastic_benchmark": RUN_STOCHASTIC_BENCHMARK,
    "stochastic_transfer_validation": (
        RUN_STOCHASTIC_TRANSFER_VALIDATION
    ),
}

active_experiments = [
    name
    for name, enabled in EXPERIMENT_SWITCHES.items()
    if enabled
]

print("QUICK_MODE:", QUICK_MODE)
print("RUN_PRIMARY_ABLATION:", RUN_PRIMARY_ABLATION)
print(
    "RUN_OBSERVATION_REGIME_STUDY:",
    RUN_OBSERVATION_REGIME_STUDY,
)
print("Active experiments:", active_experiments)
print("PRIMARY_DATASET exists:", "PRIMARY_DATASET" in globals())
print("PRIMARY_BUNDLE exists:", "PRIMARY_BUNDLE" in globals())

assert QUICK_MODE is False
assert RUN_PRIMARY_ABLATION is True
assert RUN_OBSERVATION_REGIME_STUDY is False
assert active_experiments == ["primary_ablation"]
assert "PRIMARY_DATASET" in globals()
assert "PRIMARY_BUNDLE" in globals()

print(
    "\nTemporary primary-ablation configuration is active. "
    "You may now run corrected Cell 16."
)

# In[19]:


# =============================================================================
# ROUND 3 — CELL 16 (CORRECTED FAIR-COMPARISON VERSION)
# Architecture ablation: six neural models + matched ODE benchmark
# =============================================================================
# Corrections:
# 1. ODE fitting uses only the same TRAINING observations used by the PINNs.
# 2. Held-out validation observations are not used during ODE optimization.
# 3. Dense hidden S/R trajectories are used only after fitting for evaluation.
# 4. The ODE optimizer does not start from the exact true parameter pair.
# 5. ODE physics-residual RMSE is reported as N/A rather than zero.
# 6. A common held-out observation score is calculated for every method.
# 7. Existing neural results can be reused, avoiding several hours of retraining.
#
# Current recommendation:
# Keep REUSE_EXISTING_NEURAL_RESULTS=True because the six neural models have
# already been trained successfully. This cell will then refit only the fair ODE
# benchmark and rebuild the comparison tables.
# =============================================================================

from datetime import datetime
from pathlib import Path
import copy
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REUSE_EXISTING_NEURAL_RESULTS = True
SCHEDULE_NAME = "early_continuous"
MODEL_KINDS = ["data_only", "vanilla", "soft", "positive", "simplex", "hard"]


# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------
required_objects = [
    "PRIMARY_BUNDLE", "PRIMARY_DATASET", "TRUE_PARAMETERS", "SCHEDULES_R3",
    "DENSE_REFERENCE", "LOSS_WEIGHTS", "AMRInverseModel", "train_inverse_model",
    "evaluate_trained_model", "solve_reference_r3", "nearest_values",
    "relative_l2", "clear_gpu_memory",
]
missing = [name for name in required_objects if name not in globals()]
if missing:
    raise RuntimeError(f"Run Round 3 Cells 1–14 first. Missing: {missing}")


# -----------------------------------------------------------------------------
# Common held-out observation score for all methods
# -----------------------------------------------------------------------------
def common_validation_metrics(frame, dataset):
    """Evaluate every method on the same held-out observations only."""
    total_mask = np.asarray(dataset.total_validation_mask, dtype=bool)
    frac_mask = np.asarray(dataset.fraction_validation_mask, dtype=bool)

    if total_mask.any():
        times = dataset.total_time[total_mask]
        observed = dataset.total_observed[total_mask]
        predicted = nearest_values(frame, times, "predicted_total")
        total_log_mse = float(np.mean(
            (np.log(np.clip(predicted, 1e-10, None))
             - np.log(np.clip(observed, 1e-10, None)))**2
        ))
    else:
        total_log_mse = 0.0

    if frac_mask.any():
        times = dataset.fraction_time[frac_mask]
        observed = dataset.fraction_observed[frac_mask]
        predicted = nearest_values(frame, times, "predicted_resistant_fraction")
        fraction_mse = float(np.mean(
            (np.clip(predicted, 0.0, 1.0) - observed)**2
        ))
    else:
        fraction_mse = 0.0

    return {
        "heldout_total_log_mse": total_log_mse,
        "heldout_fraction_mse": fraction_mse,
        "heldout_observation_score": total_log_mse + fraction_mse,
    }


# -----------------------------------------------------------------------------
# Fair ODE objective: use TRAINING masks only
# -----------------------------------------------------------------------------
def fair_ode_residual_vector(x, datasets):
    log10_mu, cost = x
    blocks = []

    for schedule_name, dataset in datasets.items():
        total_mask = np.asarray(dataset.total_train_mask, dtype=bool)
        frac_mask = np.asarray(dataset.fraction_train_mask, dtype=bool)

        total_times = dataset.total_time[total_mask]
        total_obs = dataset.total_observed[total_mask]
        frac_times = dataset.fraction_time[frac_mask]
        frac_obs = dataset.fraction_observed[frac_mask]

        all_times = np.unique(np.concatenate([total_times, frac_times]))
        parameters = replace(
            TRUE_PARAMETERS,
            mutation_probability=10.0**float(log10_mu),
            resistance_fitness_cost=float(cost),
        )
        pred = solve_reference_r3(
            parameters,
            SCHEDULES_R3[schedule_name],
            all_times,
        )

        if total_times.size:
            pred_total = nearest_values(pred, total_times, "total")
            residual = (
                np.log(np.clip(pred_total, 1e-10, None))
                - np.log(np.clip(total_obs, 1e-10, None))
            )
            scale = math.sqrt(
                float(LOSS_WEIGHTS.total_data) / max(len(residual), 1)
            )
            blocks.append(scale * residual)

        if frac_times.size:
            pred_fraction = nearest_values(
                pred, frac_times, "resistant_fraction"
            )
            residual = np.clip(pred_fraction, 0.0, 1.0) - frac_obs
            scale = math.sqrt(
                float(LOSS_WEIGHTS.fraction_data) / max(len(residual), 1)
            )
            blocks.append(scale * residual)

    return np.concatenate(blocks).astype(float)


def fit_fair_ode_baseline(datasets, label, seed, starts):
    """
    Fit the ODE from the same non-oracle initialization used by the PINNs,
    followed by random multi-starts. The exact true pair is not inserted.
    """
    rng = np.random.default_rng(seed)
    lower = np.array([-6.0, 0.0])
    upper = np.array([-2.0, 0.40])

    initial_points = [np.array([np.log10(3e-4), 0.15])]
    initial_points += [
        np.array([rng.uniform(-5.8, -2.2), rng.uniform(0.005, 0.38)])
        for _ in range(starts - 1)
    ]

    true_pair = np.array([
        np.log10(TRUE_PARAMETERS.mutation_probability),
        TRUE_PARAMETERS.resistance_fitness_cost,
    ])
    assert not np.allclose(initial_points[0], true_pair, atol=1e-14, rtol=0.0)

    best = None
    audit = []
    start_time = time.time()

    for start_index, initial in enumerate(initial_points):
        fit = least_squares(
            lambda z: fair_ode_residual_vector(z, datasets),
            x0=initial,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=500 if QUICK_MODE else 1500,
        )
        objective = float(np.sum(fit.fun**2))
        audit.append({
            "start_index": start_index,
            "initial_mu": float(10.0**initial[0]),
            "initial_cost": float(initial[1]),
            "fitted_mu": float(10.0**fit.x[0]),
            "fitted_cost": float(fit.x[1]),
            "objective": objective,
            "nfev": int(fit.nfev),
            "success": bool(fit.success),
        })
        if best is None or objective < float(np.sum(best.fun**2)):
            best = fit

    elapsed = time.time() - start_time
    result = {
        "label": label,
        "model_kind": "ode_fit",
        "method_class": "matched_mechanistic_benchmark",
        "schedule": SCHEDULE_NAME,
        "training_seconds": float(elapsed),
        "best_validation_score": np.nan,
        "mutation_probability": float(10.0**best.x[0]),
        "resistance_fitness_cost": float(best.x[1]),
        "relative_error_mu": float(
            abs(10.0**best.x[0] - TRUE_PARAMETERS.mutation_probability)
            / TRUE_PARAMETERS.mutation_probability
        ),
        "relative_error_cost": float(
            abs(best.x[1] - TRUE_PARAMETERS.resistance_fitness_cost)
            / TRUE_PARAMETERS.resistance_fitness_cost
        ),
        "objective": float(np.sum(best.fun**2)),
        "nfev": int(best.nfev),
        "fit_scope": "training observations only",
        "hidden_states_used_for_fit": False,
        "exact_true_initialization_used": False,
        "multistart_count": int(len(initial_points)),
    }
    return result, pd.DataFrame(audit)


# -----------------------------------------------------------------------------
# Dense post-fit evaluation of the ODE benchmark
# -----------------------------------------------------------------------------
def evaluate_fair_ode(ode_metrics):
    fitted = replace(
        TRUE_PARAMETERS,
        mutation_probability=ode_metrics["mutation_probability"],
        resistance_fitness_cost=ode_metrics["resistance_fitness_cost"],
    )

    true = DENSE_REFERENCE[SCHEDULE_NAME].copy()
    times = true.time.to_numpy(float)
    pred = solve_reference_r3(fitted, SCHEDULES_R3[SCHEDULE_NAME], times)

    frame = true.copy()
    frame["predicted_sensitive"] = pred.sensitive.to_numpy(float)
    frame["predicted_resistant"] = pred.resistant.to_numpy(float)
    frame["predicted_total"] = pred.total.to_numpy(float)
    frame["predicted_resistant_fraction"] = (
        pred.resistant_fraction.to_numpy(float)
    )

    pairs = {
        "sensitive": ("sensitive", "predicted_sensitive"),
        "resistant": ("resistant", "predicted_resistant"),
        "total": ("total", "predicted_total"),
        "fraction": ("resistant_fraction", "predicted_resistant_fraction"),
    }
    for name, (true_col, pred_col) in pairs.items():
        y_true = frame[true_col].to_numpy(float)
        y_pred = frame[pred_col].to_numpy(float)
        ode_metrics[f"rmse_{name}"] = float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        )
        ode_metrics[f"mae_{name}"] = float(
            mean_absolute_error(y_true, y_pred)
        )
        ode_metrics[f"r2_{name}"] = float(r2_score(y_true, y_pred))
        ode_metrics[f"relative_l2_{name}"] = relative_l2(y_pred, y_true)

    S = frame.predicted_sensitive.to_numpy(float)
    R = frame.predicted_resistant.to_numpy(float)
    total = S + R
    K = float(TRUE_PARAMETERS.carrying_capacity)

    ode_metrics.update({
        "maximum_negative_violation": float(max(0.0, -S.min(), -R.min())),
        "maximum_capacity_violation": float(max(0.0, total.max() - K)),
        "fraction_points_negative": float(np.mean((S < 0) | (R < 0))),
        "fraction_points_over_capacity": float(np.mean(total > K + 1e-8)),
        # N/A: the neural autograd residual metric is not directly comparable
        # with a trajectory obtained by direct numerical integration.
        "physics_residual_rmse": np.nan,
        "constraint_interpretation": (
            "empirical mechanistic-trajectory audit; not a neural guarantee"
        ),
    })
    ode_metrics.update(common_validation_metrics(frame, PRIMARY_DATASET))

    # Comparable prediction time at 1001 requested points
    eval_times = np.linspace(T_START, T_FINAL, 1001)
    repeats = 5 if QUICK_MODE else 10
    _ = solve_reference_r3(fitted, SCHEDULES_R3[SCHEDULE_NAME], eval_times)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = solve_reference_r3(
            fitted, SCHEDULES_R3[SCHEDULE_NAME], eval_times
        )
    ode_metrics["inference_ms_1001_points"] = float(
        1000.0 * (time.perf_counter() - start) / repeats
    )
    return ode_metrics, frame


# =============================================================================
# Execute the corrected ablation
# =============================================================================
if RUN_PRIMARY_ABLATION:

    previous_metrics = (
        ABLATION_METRICS.copy()
        if "ABLATION_METRICS" in globals()
        and isinstance(ABLATION_METRICS, pd.DataFrame)
        else None
    )
    previous_predictions = (
        copy.copy(ABLATION_PREDICTIONS)
        if "ABLATION_PREDICTIONS" in globals()
        and isinstance(ABLATION_PREDICTIONS, dict)
        else {}
    )

    ABLATION_RESULTS = {}
    ABLATION_METRICS_ROWS = []
    ABLATION_PREDICTIONS = {}

    combined_path = (
        TABLE_DIR_R3 / "round3_architecture_ablation_metrics.csv"
    )
    neural_path = (
        TABLE_DIR_R3 / "round3_architecture_ablation_neural_metrics.csv"
    )
    ode_path = (
        TABLE_DIR_R3 / "round3_matched_ode_benchmark_metrics.csv"
    )
    multistart_path = (
        TABLE_DIR_R3 / "round3_matched_ode_multistart_audit.csv"
    )

    # -------------------------------------------------------------------------
    # Reuse or retrain neural models
    # -------------------------------------------------------------------------
    candidate = previous_metrics
    if candidate is None and combined_path.exists():
        candidate = pd.read_csv(combined_path)

    can_reuse = False
    if REUSE_EXISTING_NEURAL_RESULTS and candidate is not None:
        candidate = candidate[candidate.model_kind.isin(MODEL_KINDS)].copy()
        if candidate.model_kind.nunique() == len(MODEL_KINDS):
            can_reuse = True
            for kind in MODEL_KINDS:
                if kind in previous_predictions:
                    pred = previous_predictions[kind].copy()
                else:
                    pred_path = PREDICTION_DIR / (
                        f"round3_ablation_{kind}_O3_"
                        f"{SCHEDULE_NAME}_predictions.csv"
                    )
                    if not pred_path.exists():
                        can_reuse = False
                        break
                    pred = pd.read_csv(pred_path)
                ABLATION_PREDICTIONS[kind] = pred

    if can_reuse:
        print(
            "Reusing completed neural results; refitting only the corrected "
            "ODE benchmark."
        )
        for _, series in candidate.iterrows():
            row = series.to_dict()
            kind = row["model_kind"]
            row["method_class"] = (
                "proposed_neural_method"
                if kind == "hard"
                else "neural_baseline"
            )
            row["fit_scope"] = "training observations only"
            row["hidden_states_used_for_fit"] = False
            row.update(
                common_validation_metrics(
                    ABLATION_PREDICTIONS[kind],
                    PRIMARY_DATASET,
                )
            )
            ABLATION_METRICS_ROWS.append(row)
    else:
        if REUSE_EXISTING_NEURAL_RESULTS:
            print(
                "Reusable neural files were incomplete; retraining all six "
                "neural models."
            )
        for index, kind in enumerate(MODEL_KINDS):
            label = f"round3_ablation_{kind}_O3"
            model = AMRInverseModel(
                {SCHEDULE_NAME: SCHEDULES_R3[SCHEDULE_NAME]},
                model_kind=kind,
                width=64,
                depth=4,
                parameter_initialization=(3e-4, 0.15),
            )
            result = train_inverse_model(
                model, PRIMARY_BUNDLE, label, SEED + 1000 * index
            )
            metrics, pred = evaluate_trained_model(
                result, SCHEDULE_NAME
            )
            metrics["method_class"] = (
                "proposed_neural_method"
                if kind == "hard"
                else "neural_baseline"
            )
            metrics["fit_scope"] = "training observations only"
            metrics["hidden_states_used_for_fit"] = False
            metrics.update(
                common_validation_metrics(pred, PRIMARY_DATASET)
            )
            ABLATION_RESULTS[kind] = result
            ABLATION_METRICS_ROWS.append(metrics)
            ABLATION_PREDICTIONS[kind] = pred
            result.model.to("cpu")
            clear_gpu_memory()

    # -------------------------------------------------------------------------
    # Corrected ODE benchmark
    # -------------------------------------------------------------------------
    ode_metrics, multistart_audit = fit_fair_ode_baseline(
        PRIMARY_BUNDLE,
        label="round3_ablation_ode_fit_fair_O3",
        seed=SEED + 880_000,
        starts=8 if QUICK_MODE else 24,
    )
    ode_metrics, ode_pred = evaluate_fair_ode(ode_metrics)

    ode_pred.to_csv(
        PREDICTION_DIR
        / "round3_ablation_ode_fit_fair_O3_predictions.csv",
        index=False,
    )
    multistart_audit.to_csv(multistart_path, index=False)
    ABLATION_PREDICTIONS["ode_fit"] = ode_pred
    ABLATION_METRICS_ROWS.append(ode_metrics)

    # -------------------------------------------------------------------------
    # Save separate and combined tables
    # -------------------------------------------------------------------------
    ABLATION_METRICS = pd.DataFrame(ABLATION_METRICS_ROWS)
    order_map = {
        "ode_fit": 0, "data_only": 1, "vanilla": 2, "soft": 3,
        "positive": 4, "simplex": 5, "hard": 6,
    }
    ABLATION_METRICS["display_order"] = (
        ABLATION_METRICS.model_kind.map(order_map)
    )
    ABLATION_METRICS = (
        ABLATION_METRICS
        .sort_values("display_order")
        .reset_index(drop=True)
    )

    if combined_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(
            combined_path,
            TABLE_DIR_R3
            / f"round3_architecture_ablation_metrics_before_fair_fix_{stamp}.csv",
        )

    neural_table = ABLATION_METRICS[
        ABLATION_METRICS.model_kind.isin(MODEL_KINDS)
    ].copy()
    ode_table = ABLATION_METRICS[
        ABLATION_METRICS.model_kind == "ode_fit"
    ].copy()

    ABLATION_METRICS.to_csv(combined_path, index=False)
    neural_table.to_csv(neural_path, index=False)
    ode_table.to_csv(ode_path, index=False)

    fairness_audit = {
        "ode_fit_scope": "training observations only",
        "validation_points_used_during_ode_fit": False,
        "dense_hidden_states_used_during_ode_fit": False,
        "dense_hidden_states_used_for_postfit_evaluation": True,
        "exact_true_initialization_used": False,
        "shared_non_oracle_initialization": {
            "mutation_probability": 3e-4,
            "resistance_fitness_cost": 0.15,
        },
        "ode_physics_residual_rmse": "N/A",
        "common_validation_metric": (
            "heldout total log-MSE + heldout resistant-fraction MSE"
        ),
    }
    with open(
        REPORT_DIR_R3 / "round3_ablation_fairness_audit.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(fairness_audit, file, indent=2)

    summary_columns = [
        "method_class", "model_kind",
        "rmse_sensitive", "rmse_resistant",
        "heldout_observation_score",
        "relative_error_mu", "relative_error_cost",
        "maximum_negative_violation",
        "maximum_capacity_violation",
        "physics_residual_rmse",
        "training_seconds",
    ]
    summary_columns = [
        c for c in summary_columns if c in ABLATION_METRICS.columns
    ]

    display(ABLATION_METRICS[summary_columns])

    print("\nFair-comparison Cell 16 completed successfully.")
    print("ODE fitting used training observations only.")
    print("The exact true parameter pair was not used as an initialization.")
    print("Dense hidden states were used only for post-fit evaluation.")
    print("ODE and neural results were saved separately:")
    print(" ", neural_path)
    print(" ", ode_path)
    print("Combined table:")
    print(" ", combined_path)

else:
    print(
        "Set RUN_PRIMARY_ABLATION=True in Cell 1 and rerun this cell."
    )

# In[17]:


# =============================================================================
# LOAD COMPLETED ABLATION RESULTS FOR PLOTTING
# =============================================================================
# Plotting should not depend on RUN_PRIMARY_ABLATION because the expensive
# training has already been completed and saved by Cell 16.

ABLATION_TABLE_PATH = (
    TABLE_DIR_R3
    / "round3_architecture_ablation_metrics.csv"
)

if (
    "ABLATION_METRICS" not in globals()
    or not isinstance(ABLATION_METRICS, pd.DataFrame)
    or ABLATION_METRICS.empty
):
    if ABLATION_TABLE_PATH.exists():
        ABLATION_METRICS = pd.read_csv(
            ABLATION_TABLE_PATH
        )

        print(
            "Loaded completed ablation metrics from:"
        )
        print(" ", ABLATION_TABLE_PATH)

    else:
        raise FileNotFoundError(
            "The completed Cell 16 ablation table was not found:\n"
            f"  {ABLATION_TABLE_PATH}\n"
            "Run corrected Cell 16 once before generating the figures."
        )

print(
    f"Available ablation methods: "
    f"{len(ABLATION_METRICS)}"
)

# In[65]:


# ROUND 3 — CELL 17
# FINAL PUBLICATION-READY ABLATION FIGURES
# =============================================================================
# This cell does not retrain any model.
#
# It:
#   1. Loads the completed Cell 16 ablation table when needed.
#   2. Generates a neural-only state-error heatmap.
#   3. Generates a two-panel accuracy-feasibility comparison without
#      overlapping zero-violation points.
#   4. Generates a clearer parameter-recovery figure with the matched ODE
#      benchmark visually separated from the neural methods.
#   5. Uses explicit decimal values instead of computer-style "e" notation.
# =============================================================================

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace


# =============================================================================
# STANDALONE ROUND-3 PATH / HELPER RECOVERY
# =============================================================================
# This makes Cell 17 safe to run even after a Jupyter kernel restart.
# If the original Round-3 setup variables already exist, they are reused.
# Otherwise, the canonical project folders are reconstructed from Path.home().
# NO TRAINING IS PERFORMED HERE.

ROUND3_ROOT = Path(
    globals().get(
        "ROUND3_ROOT",
        Path.home() / "round3" / "final",
    )
)

TABLE_DIR_R3 = Path(
    globals().get(
        "TABLE_DIR_R3",
        ROUND3_ROOT / "tables",
    )
)

FIGURE_DIR_R3 = Path(
    globals().get(
        "FIGURE_DIR_R3",
        ROUND3_ROOT / "figures",
    )
)

TABLE_DIR_R3.mkdir(parents=True, exist_ok=True)
FIGURE_DIR_R3.mkdir(parents=True, exist_ok=True)

# Recover the two data-generating parameters used throughout the final
# Round-3/Round-4 manuscript experiments if the original parameter container
# is not currently present in memory.
if "TRUE_PARAMETERS" not in globals():
    TRUE_PARAMETERS = SimpleNamespace(
        mutation_probability=1.0e-4,
        resistance_fitness_cost=0.10,
    )

# Recover the figure-saving helper if the notebook setup cell has not been run
# in the current kernel.  The same basename is written in vector and raster
# formats for manuscript and presentation use.
if "save_figure_r3" not in globals():

    def save_figure_r3(fig, stem):
        stem = str(stem)

        pdf_path = FIGURE_DIR_R3 / f"{stem}.pdf"
        png_path = FIGURE_DIR_R3 / f"{stem}.png"
        svg_path = FIGURE_DIR_R3 / f"{stem}.svg"

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor="white",
        )

        print("Saved figure files:")
        print(" ", pdf_path.resolve())
        print(" ", png_path.resolve())
        print(" ", svg_path.resolve())


print("Round-3 Cell 17 paths:")
print("  Tables :", TABLE_DIR_R3.resolve())
print("  Figures:", FIGURE_DIR_R3.resolve())


# =============================================================================
# LOAD COMPLETED ABLATION RESULTS
# =============================================================================

ABLATION_TABLE_PATH = (
    TABLE_DIR_R3
    / "round3_architecture_ablation_metrics.csv"
)

if (
    "ABLATION_METRICS" not in globals()
    or not isinstance(ABLATION_METRICS, pd.DataFrame)
    or ABLATION_METRICS.empty
):
    if ABLATION_TABLE_PATH.exists():
        ABLATION_METRICS = pd.read_csv(
            ABLATION_TABLE_PATH
        )

        print("Loaded completed ablation metrics from:")
        print(" ", ABLATION_TABLE_PATH)

    else:
        raise FileNotFoundError(
            "The completed Cell 16 ablation table was not found:\n"
            f"  {ABLATION_TABLE_PATH}\n"
            "Run corrected Cell 16 once before generating these figures."
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def explicit_decimal(value, significant_digits=3):
    """
    Format a number without computer-style e notation.

    Examples
    --------
    154.0       -> 154
    1.538       -> 1.54
    0.517       -> 0.517
    0.00466     -> 0.00466
    0.000705    -> 0.000705
    3.081e-7    -> 0.000000308
    """
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if value == 0.0:
        return "0"

    absolute_value = abs(value)

    if absolute_value >= 100:
        decimal_places = 0
    elif absolute_value >= 10:
        decimal_places = 1
    elif absolute_value >= 1:
        decimal_places = 2
    else:
        exponent = math.floor(math.log10(absolute_value))
        decimal_places = max(
            significant_digits,
            -exponent + significant_digits - 1,
        )
        decimal_places = min(decimal_places, 12)

    formatted = f"{value:.{decimal_places}f}"

    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")

    return formatted


def set_black_frame(axis):
    """Give every panel a complete black frame."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def add_panel_label(axis, label):
    axis.text(
        -0.10,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# =============================================================================
# PREPARE AND ORDER THE METRICS
# =============================================================================

metrics = ABLATION_METRICS.copy()

display_name_map = {
    "ode_fit": "Matched ODE benchmark",
    "data_only": "Data-only NN",
    "vanilla": "Vanilla PINN",
    "soft": "Soft-constrained PINN",
    "positive": "Positive-output PINN",
    "simplex": "Simplex-only PINN",
    "hard": "Proposed hard PINN",
}

method_order = [
    "ode_fit",
    "data_only",
    "vanilla",
    "soft",
    "positive",
    "simplex",
    "hard",
]

metrics["display_name"] = (
    metrics["model_kind"]
    .map(display_name_map)
    .fillna(metrics["model_kind"])
)

metrics["display_order"] = metrics["model_kind"].map(
    {name: index for index, name in enumerate(method_order)}
)

metrics = (
    metrics
    .sort_values("display_order")
    .reset_index(drop=True)
)

ode_metrics = (
    metrics[
        metrics["model_kind"] == "ode_fit"
    ]
    .copy()
    .reset_index(drop=True)
)

neural_metrics = (
    metrics[
        metrics["model_kind"] != "ode_fit"
    ]
    .copy()
    .reset_index(drop=True)
)

# Save transparent audit tables.
neural_metrics.to_csv(
    TABLE_DIR_R3
    / "round3_architecture_ablation_neural_only_metrics.csv",
    index=False,
)

ode_metrics.to_csv(
    TABLE_DIR_R3
    / "round3_matched_ode_benchmark_metrics.csv",
    index=False,
)

print(
    "Cell 17: generating final publication-ready figures from "
    f"{len(neural_metrics)} neural methods and "
    f"{len(ode_metrics)} matched ODE benchmark."
)


# =============================================================================
# DEFAULT MATPLOTLIB COLOR CYCLE
# =============================================================================
# The default cycle is used consistently rather than defining custom colors.

default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

method_color = {
    "data_only": default_cycle[0],
    "vanilla": default_cycle[1],
    "soft": default_cycle[2],
    "positive": default_cycle[3],
    "simplex": default_cycle[4],
    "hard": default_cycle[5],
    "ode_fit": default_cycle[0],
}


# =============================================================================
# FIGURE A — NEURAL-ONLY STATE-ERROR HEATMAP
# =============================================================================

heat_columns = [
    "rmse_sensitive",
    "rmse_resistant",
    "rmse_total",
    "rmse_fraction",
]

heat_labels = [
    r"RMSE $S$",
    r"RMSE $R$",
    r"RMSE $S+R$",
    r"RMSE $q_R$",
]

heat_values = (
    neural_metrics[heat_columns]
    .astype(float)
    .to_numpy()
)

heat_log10 = np.log10(
    np.maximum(heat_values, 1e-12)
)

finite_heat = heat_log10[
    np.isfinite(heat_log10)
]

if finite_heat.size == 0:
    raise RuntimeError(
        "No finite neural RMSE values are available for the heatmap."
    )

heat_norm = mcolors.Normalize(
    vmin=float(np.nanmin(finite_heat)),
    vmax=float(np.nanmax(finite_heat)),
)

fig, ax = plt.subplots(
    figsize=(10.4, 5.5)
)

image = ax.imshow(
    heat_log10,
    aspect="auto",
    norm=heat_norm,
    interpolation="nearest",
)

ax.set_xticks(
    np.arange(len(heat_labels))
)

ax.set_xticklabels(
    heat_labels,
    fontsize=11,
)

ax.set_yticks(
    np.arange(len(neural_metrics))
)

ax.set_yticklabels(
    neural_metrics["display_name"],
    fontsize=10.8,
)

ax.set_title(
    "Neural architecture ablation: state-reconstruction accuracy",
    fontsize=15,
    fontweight="bold",
    pad=14,
)

ax.set_xlabel(
    "Reconstruction metric",
    fontsize=11.5,
)

ax.set_ylabel(
    "Inference method",
    fontsize=11.5,
)

# Add cell boundaries.
ax.set_xticks(
    np.arange(-0.5, len(heat_labels), 1),
    minor=True,
)

ax.set_yticks(
    np.arange(-0.5, len(neural_metrics), 1),
    minor=True,
)

ax.grid(
    which="minor",
    linewidth=1.2,
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

# Add explicit decimal values.
for row_index in range(heat_values.shape[0]):
    for column_index in range(heat_values.shape[1]):

        original_value = heat_values[
            row_index,
            column_index,
        ]

        normalized_value = heat_norm(
            heat_log10[
                row_index,
                column_index,
            ]
        )

        text_color = (
            "white"
            if normalized_value < 0.42
            else "black"
        )

        ax.text(
            column_index,
            row_index,
            explicit_decimal(original_value),
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight="semibold",
            color=text_color,
        )

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.038,
    pad=0.025,
)

colorbar.set_label(
    r"$\log_{10}(\mathrm{RMSE})$ used for colour only",
    fontsize=10.5,
)

set_black_frame(ax)

fig.tight_layout()

save_figure_r3(
    fig,
    "round3_ablation_error_heatmap_neural_only_final",
)

plt.show()


# =============================================================================
# MATCHED ODE BENCHMARK — FORMATTED SEPARATE TABLE
# =============================================================================

if not ode_metrics.empty:

    ode_row = ode_metrics.iloc[0]

    ode_display = pd.DataFrame({
        "Method": [
            "Matched ODE benchmark"
        ],
        "RMSE S": [
            explicit_decimal(
                ode_row["rmse_sensitive"]
            )
        ],
        "RMSE R": [
            explicit_decimal(
                ode_row["rmse_resistant"]
            )
        ],
        "RMSE total": [
            explicit_decimal(
                ode_row["rmse_total"]
            )
        ],
        "RMSE fraction": [
            explicit_decimal(
                ode_row["rmse_fraction"]
            )
        ],
        "Estimated mu": [
            explicit_decimal(
                ode_row["mutation_probability"]
            )
        ],
        "Estimated c_R": [
            explicit_decimal(
                ode_row["resistance_fitness_cost"]
            )
        ],
    })

    print(
        "\nMatched ODE benchmark "
        "(reported separately from the neural architecture ablation):"
    )

    display(ode_display)


# =============================================================================
# =============================================================================
# FIGURE B — FINAL PUBLICATION ACCURACY–FEASIBILITY COMPARISON
# =============================================================================
# Scientific purpose
# ------------------
# Panel A shows neural methods that exhibit nonzero state-feasibility
# violations on the evaluation grid. The plotted vertical quantity is the
# maximum negative-state violation, V_neg, because the completed ablation has
# zero carrying-capacity violation for every method.
#
# Panel B shows neural methods for which no violation was observed on the
# evaluation grid. Marker SHAPE encodes the architectural guarantee:
#   square   -> positivity only
#   triangle -> positivity + simplex
#   star     -> full feasible set
#
# Important interpretation
# ------------------------
# Zero observed violation on a finite evaluation grid is not equivalent to a
# guarantee by construction. The proposed hard PINN is the only architecture
# in this comparison that guarantees the complete feasible set by design.
#
# No model is retrained in this section.
# =============================================================================

required_feasibility_columns = [
    "rmse_resistant",
    "maximum_negative_violation",
    "maximum_capacity_violation",
]

missing_feasibility_columns = [
    column
    for column in required_feasibility_columns
    if column not in neural_metrics.columns
]

if missing_feasibility_columns:
    raise KeyError(
        "Missing feasibility columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_feasibility_columns
        )
    )

# -----------------------------------------------------------------------------
# Classify methods by whether ANY feasibility violation was observed.
# -----------------------------------------------------------------------------

neural_metrics["total_feasibility_violation"] = (
    neural_metrics["maximum_negative_violation"].astype(float)
    + neural_metrics["maximum_capacity_violation"].astype(float)
)

violation_tolerance = 1e-12

nonzero_violation = (
    neural_metrics[
        neural_metrics["total_feasibility_violation"] > violation_tolerance
    ]
    .copy()
    .sort_values("rmse_resistant")
    .reset_index(drop=True)
)

zero_violation = (
    neural_metrics[
        neural_metrics["total_feasibility_violation"] <= violation_tolerance
    ]
    .copy()
    .sort_values(
        "rmse_resistant",
        ascending=False,
    )
    .reset_index(drop=True)
)

if nonzero_violation.empty:
    raise RuntimeError(
        "No nonzero-violation neural methods are available for Panel A."
    )

if zero_violation.empty:
    raise RuntimeError(
        "No zero-observed-violation neural methods are available for Panel B."
    )

# In the completed ablation, Panel A violations are negativity violations.
if np.any(
    nonzero_violation["maximum_negative_violation"].to_numpy(float)
    <= violation_tolerance
):
    raise RuntimeError(
        "At least one nonzero-violation method has no positive negativity "
        "violation. Revisit the Panel A metric before plotting."
    )

# -----------------------------------------------------------------------------
# Publication layout
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.6, 5.2),
    gridspec_kw={
        "width_ratios": [1.08, 1.0],
    },
)

# =============================================================================
# PANEL A — METHODS WITH OBSERVED CONSTRAINT VIOLATION
# =============================================================================

axis = axes[0]

nonzero_x_values = (
    nonzero_violation["rmse_resistant"]
    .to_numpy(float)
)

nonzero_y_values = (
    nonzero_violation["maximum_negative_violation"]
    .to_numpy(float)
)

axis.set_xscale("log")
axis.set_yscale("log")

axis.set_xlim(
    max(np.min(nonzero_x_values) / 1.80, 1e-12),
    np.max(nonzero_x_values) * 1.28,
)

axis.set_ylim(
    max(np.min(nonzero_y_values) / 1.55, 1e-12),
    np.max(nonzero_y_values) * 1.55,
)

# Hand-tuned offsets to avoid clipping and overlap
annotation_offsets = {
    "data_only": (-12, 8),   # move inside from right edge
    "vanilla": (10, 10),     # move above, not below axis
    "soft": (10, 8),
}

for _, row in nonzero_violation.iterrows():

    model_kind = row["model_kind"]
    rmse_value = float(row["rmse_resistant"])
    violation_value = float(row["maximum_negative_violation"])

    axis.scatter(
        rmse_value,
        violation_value,
        s=118,
        marker="o",
        color=method_color.get(model_kind, None),
        edgecolor="black",
        linewidth=0.85,
        zorder=4,
    )

    offset_x, offset_y = annotation_offsets.get(
        model_kind,
        (9, 7),
    )

    axis.annotate(
        row["display_name"],
        (rmse_value, violation_value),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        fontsize=9.0,
        ha=("right" if offset_x < 0 else "left"),
        va="center",
        annotation_clip=False,
        clip_on=False,
        zorder=5,
    )

axis.set_xlabel(
    r"Resistant-state error, $\mathrm{RMSE}_R$",
    fontsize=11.1,
    labelpad=6,
)

axis.set_ylabel(
    r"Observed feasibility violation",
    fontsize=11.1,
    labelpad=7,
)

axis.set_title(
    "Methods with nonzero violation",
    fontsize=12.0,
    fontweight="semibold",
    pad=10,
)

axis.grid(
    alpha=0.20,
    which="major",
    linestyle="--",
    linewidth=0.75,
)

axis.grid(
    alpha=0.11,
    which="minor",
    linestyle="--",
    linewidth=0.55,
)

axis.tick_params(
    axis="both",
    which="major",
    labelsize=9.6,
    width=0.9,
    length=4.2,
    direction="out",
)

axis.tick_params(
    axis="both",
    which="minor",
    width=0.7,
    length=2.5,
    direction="out",
)

set_black_frame(axis)
add_panel_label(axis, "A")

# =============================================================================
# PANEL B — METHODS WITH ZERO OBSERVED CONSTRAINT VIOLATION
# =============================================================================

axis = axes[1]

zero_x_values = (
    zero_violation["rmse_resistant"]
    .to_numpy(float)
)

axis.set_xscale("log")

# Publication-safe horizontal limits.
# Extra right-side headroom keeps the rightmost square and its numeric
# annotation comfortably inside the plotting frame after PDF export.
axis.set_xlim(
    max(np.min(zero_x_values) / 1.25, 1e-12),
    np.max(zero_x_values) * 1.28,
)

zero_y = np.arange(len(zero_violation))

marker_map = {
    "positive": "s",
    "simplex": "^",
    "hard": "*",
}

guarantee_text = {
    "positive": "Positivity only",
    "simplex": "Positivity + simplex",
    "hard": "Full feasible set",
}

marker_size = {
    "positive": 170,
    "simplex": 190,
    "hard": 285,
}

# Explicit method-specific numeric-label placement.
#
# The positive-output PINN is the rightmost point. Its value is therefore
# forced to the LEFT of the square. The other two values are placed to the
# RIGHT of their markers. This deterministic placement is more robust than
# threshold-based logic and prevents labels from leaving the axes after
# raster/PDF export or journal down-scaling.
value_label_style = {
    "positive": {
        "xytext": (-12, 0),
        "ha": "right",
    },
    "simplex": {
        "xytext": (10, 0),
        "ha": "left",
    },
    "hard": {
        "xytext": (10, 0),
        "ha": "left",
    },
}

for row_index, row in zero_violation.iterrows():

    model_kind = row["model_kind"]
    rmse_value = float(row["rmse_resistant"])

    axis.scatter(
        rmse_value,
        row_index,
        s=marker_size.get(model_kind, 170),
        marker=marker_map.get(model_kind, "o"),
        color=method_color.get(model_kind, None),
        edgecolor="black",
        linewidth=0.9,
        zorder=4,
    )

    label_style = value_label_style.get(
        model_kind,
        {
            "xytext": (10, 0),
            "ha": "left",
        },
    )

    axis.annotate(
        explicit_decimal(rmse_value),
        (rmse_value, row_index),
        xytext=label_style["xytext"],
        textcoords="offset points",
        fontsize=9.1,
        ha=label_style["ha"],
        va="center",
        annotation_clip=True,
        clip_on=True,
        zorder=5,
    )

axis.set_yticks(zero_y)
axis.set_yticklabels(
    zero_violation["display_name"],
    fontsize=9.7,
)

axis.set_xlabel(
    r"Resistant-state error, $\mathrm{RMSE}_R$",
    fontsize=11.1,
    labelpad=6,
)

axis.set_title(
    "Zero observed violation on the evaluation grid",
    fontsize=12.0,
    fontweight="semibold",
    pad=10,
)

axis.grid(
    alpha=0.18,
    axis="x",
    which="major",
    linestyle="--",
    linewidth=0.75,
)

axis.grid(
    alpha=0.09,
    axis="x",
    which="minor",
    linestyle="--",
    linewidth=0.55,
)

axis.set_ylim(
    len(zero_violation) - 0.52,
    -0.48,
)

axis.tick_params(
    axis="x",
    which="major",
    labelsize=9.6,
    width=0.9,
    length=4.2,
    direction="out",
)

axis.tick_params(
    axis="x",
    which="minor",
    width=0.7,
    length=2.5,
    direction="out",
)

axis.tick_params(
    axis="y",
    which="major",
    width=0.9,
    length=4.0,
    direction="out",
)

set_black_frame(axis)
add_panel_label(axis, "B")

# -----------------------------------------------------------------------------
# Guarantee-by-construction legend
# -----------------------------------------------------------------------------

legend_handles = [
    Line2D(
        [0],
        [0],
        marker=marker_map["positive"],
        linestyle="None",
        markerfacecolor=method_color["positive"],
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=8.0,
        label=guarantee_text["positive"],
    ),
    Line2D(
        [0],
        [0],
        marker=marker_map["simplex"],
        linestyle="None",
        markerfacecolor=method_color["simplex"],
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=8.4,
        label=guarantee_text["simplex"],
    ),
    Line2D(
        [0],
        [0],
        marker=marker_map["hard"],
        linestyle="None",
        markerfacecolor=method_color["hard"],
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=11.0,
        label=guarantee_text["hard"],
    ),
]

legend = axis.legend(
    handles=legend_handles,
    loc="lower right",
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    edgecolor="black",
    title="Architectural guarantee",
    fontsize=8.7,
    title_fontsize=9.0,
    borderpad=0.65,
    labelspacing=0.55,
    handletextpad=0.70,
)

legend.get_frame().set_linewidth(0.8)
legend.get_title().set_fontweight("semibold")

# =============================================================================
# FINAL LAYOUT AND SAVE
# =============================================================================

fig.suptitle(
    "Accuracy-feasibility comparison across neural inference methods",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.5,
    0.03,
    "Zero observed violation does not imply the same architectural guarantee; "
    "only the proposed hard PINN guarantees the complete feasible set by construction.",
    ha="center",
    va="center",
    fontsize=10.0,
)

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.16,
    top=0.83,
    wspace=0.34,
)

save_figure_r3(
    fig,
    "round3_ablation_accuracy_feasibility_final_v2",
)

plt.show()


# =============================================================================
# FIGURE C — EVOLUTIONARY-PARAMETER RECOVERY
# =============================================================================
# The matched ODE benchmark is retained because it is a meaningful parameter-
# estimation baseline. It is visually separated from the neural methods.
# =============================================================================

parameter_metrics = (
    metrics[
        metrics["mutation_probability"].notna()
        &
        metrics["resistance_fitness_cost"].notna()
    ]
    .copy()
)

parameter_order = [
    "ode_fit",
    "vanilla",
    "soft",
    "positive",
    "simplex",
    "hard",
]

parameter_metrics["parameter_order"] = (
    parameter_metrics["model_kind"].map(
        {
            name: index
            for index, name in enumerate(
                parameter_order
            )
        }
    )
)

parameter_metrics = (
    parameter_metrics
    .sort_values("parameter_order")
    .reset_index(drop=True)
)

parameter_y = np.arange(
    len(parameter_metrics)
)

true_mutation = float(
    TRUE_PARAMETERS.mutation_probability
)

true_cost = float(
    TRUE_PARAMETERS.resistance_fitness_cost
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.8, 5.8),
    sharey=True,
)

parameter_marker = {
    "ode_fit": "D",
    "vanilla": "o",
    "soft": "o",
    "positive": "o",
    "simplex": "o",
    "hard": "*",
}

parameter_size = {
    "ode_fit": 130,
    "vanilla": 105,
    "soft": 105,
    "positive": 105,
    "simplex": 105,
    "hard": 180,
}

# -----------------------------------------------------------------------------
# Panel A: mutation probability
# -----------------------------------------------------------------------------

axis = axes[0]

mutation_estimates = (
    parameter_metrics["mutation_probability"].to_numpy(float)
)

mutation_min = min(
    float(np.min(mutation_estimates)),
    true_mutation,
)

mutation_max = max(
    float(np.max(mutation_estimates)),
    true_mutation,
)

mutation_mid = math.sqrt(
    mutation_min * mutation_max
)

axis.set_xscale("log")
axis.set_xlim(
    mutation_min / 1.75,
    mutation_max * 1.85,
)

for row_index, row in parameter_metrics.iterrows():

    model_kind = row["model_kind"]
    estimate = float(
        row["mutation_probability"]
    )

    axis.scatter(
        estimate,
        row_index,
        s=parameter_size[model_kind],
        marker=parameter_marker[model_kind],
        facecolors=(
            "none"
            if model_kind == "ode_fit"
            else None
        ),
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    if estimate > mutation_mid:
        label_offset = (-8, 0)
        label_alignment = "right"
    else:
        label_offset = (8, 0)
        label_alignment = "left"

    axis.annotate(
        explicit_decimal(estimate),
        (
            estimate,
            row_index,
        ),
        xytext=label_offset,
        textcoords="offset points",
        fontsize=8.7,
        ha=label_alignment,
        va="center",
        annotation_clip=True,
        clip_on=True,
    )

axis.axvline(
    true_mutation,
    linestyle="--",
    linewidth=1.6,
    color="black",
    label=(
        "True value = "
        + explicit_decimal(
            true_mutation
        )
    ),
)

axis.set_yticks(
    parameter_y
)

axis.set_yticklabels(
    parameter_metrics["display_name"],
    fontsize=10,
)

axis.set_ylim(
    len(parameter_metrics) - 0.50,
    -0.85,
)

axis.text(
    true_mutation,
    -0.58,
    "True value = " + explicit_decimal(true_mutation),
    fontsize=8.8,
    ha="center",
    va="center",
)

axis.set_xlabel(
    r"Estimated mutation probability, $\hat{\mu}$",
    fontsize=11,
)

axis.set_title(
    "Mutation-probability recovery",
    fontsize=12.2,
    fontweight="semibold",
    pad=10,
)

axis.grid(
    alpha=0.24,
    which="both",
    linestyle="--",
)

set_black_frame(axis)
add_panel_label(axis, "A")

# -----------------------------------------------------------------------------
# Panel B: resistance fitness cost
# -----------------------------------------------------------------------------

axis = axes[1]

cost_estimates = (
    parameter_metrics["resistance_fitness_cost"].to_numpy(float)
)

cost_min = min(
    float(np.min(cost_estimates)),
    true_cost,
)

cost_max = max(
    float(np.max(cost_estimates)),
    true_cost,
)

cost_span = max(
    cost_max - cost_min,
    0.02,
)

cost_mid = 0.5 * (cost_min + cost_max)

axis.set_xlim(
    max(0.0, cost_min - (0.08 * cost_span + 0.004)),
    cost_max + (0.18 * cost_span + 0.008),
)

for row_index, row in parameter_metrics.iterrows():

    model_kind = row["model_kind"]
    estimate = float(
        row["resistance_fitness_cost"]
    )

    axis.scatter(
        estimate,
        row_index,
        s=parameter_size[model_kind],
        marker=parameter_marker[model_kind],
        facecolors=(
            "none"
            if model_kind == "ode_fit"
            else None
        ),
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    if estimate > cost_mid:
        label_offset = (-8, 0)
        label_alignment = "right"
    else:
        label_offset = (8, 0)
        label_alignment = "left"

    axis.annotate(
        explicit_decimal(estimate),
        (
            estimate,
            row_index,
        ),
        xytext=label_offset,
        textcoords="offset points",
        fontsize=8.7,
        ha=label_alignment,
        va="center",
        annotation_clip=True,
        clip_on=True,
    )

axis.axvline(
    true_cost,
    linestyle="--",
    linewidth=1.6,
    color="black",
    label=(
        "True value = "
        + explicit_decimal(
            true_cost
        )
    ),
)

axis.set_ylim(
    len(parameter_metrics) - 0.50,
    -0.85,
)

axis.text(
    true_cost,
    -0.58,
    "True value = " + explicit_decimal(true_cost),
    fontsize=8.8,
    ha="center",
    va="center",
)

axis.set_xlabel(
    r"Estimated resistance fitness cost, $\hat{c}_R$",
    fontsize=11,
)

axis.set_title(
    "Resistance-fitness-cost recovery",
    fontsize=12.2,
    fontweight="semibold",
    pad=10,
)

axis.grid(
    alpha=0.24,
    linestyle="--",
)

set_black_frame(axis)
add_panel_label(axis, "B")

# Separate the ODE benchmark row from neural methods.
for axis in axes:
    axis.axhline(
        0.5,
        color="black",
        linewidth=1.0,
        linestyle=":",
        alpha=0.75,
    )

benchmark_handles = [
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        label="Matched ODE benchmark",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=11,
        label="Proposed hard PINN",
    ),
    Line2D(
        [0],
        [0],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="True parameter value",
    ),
]

fig.legend(
    handles=benchmark_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.045),
    ncol=3,
    frameon=True,
    fontsize=8.7,
)

fig.suptitle(
    "Evolutionary-parameter recovery across inference methods",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

fig.text(
    0.5,
    0.005,
    "The hollow diamond denotes the matched mechanistic ODE benchmark; "
    "the rows below the dotted separator are neural inference architectures.",
    ha="center",
    va="bottom",
    fontsize=9.1,
)

fig.tight_layout(
    rect=[0.0, 0.12, 1.0, 0.98],
    w_pad=2.5,
)

save_figure_r3(
    fig,
    "round3_ablation_parameter_recovery_final_v2",
)

plt.show()


# =============================================================================
# FINAL CONFIRMATION
# =============================================================================

print(
    "\nCell 17 completed successfully."
)

print(
    "Generated final publication figures:"
)

print(
    " ",
    FIGURE_DIR_R3
    / "round3_ablation_error_heatmap_neural_only_final",
)

print(
    " ",
    FIGURE_DIR_R3
    / "round3_ablation_accuracy_feasibility_final_v2",
)

print(
    " ",
    FIGURE_DIR_R3
    / "round3_ablation_parameter_recovery_final_v2",
)

# In[14]:


# =============================================================================
# ROUND 3 — CELL 18 (FIVE-SEED, PAIRED, RESUME-SAFE VERSION)
# Observation-regime experiment: O1, O2, O3, and O4 with the hard model
# =============================================================================
# Five replicate seeds are used. Within each replicate, O1–O4 use the same
# dataset seed and the same training seed, so phenotype information is the
# controlled change. Results are saved after every completed model.
# =============================================================================

import gc
import random
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REGIMES = ["O1", "O2", "O3", "O4"]
SEED_IDS = [1, 2, 3, 4, 5]

FORCE_RERUN = False
KEEP_RESULTS_IN_MEMORY = True

RAW_RESULTS_PATH = (
    TABLE_DIR_R3 / "round3_observation_regime_all_seeds_metrics.csv"
)
SUMMARY_RESULTS_PATH = (
    TABLE_DIR_R3 / "round3_observation_regime_summary_mean_std.csv"
)
MANUSCRIPT_TABLE_PATH = (
    TABLE_DIR_R3 / "round3_observation_regime_manuscript_mean_std.csv"
)

OBSERVATION_STUDY_RESULTS: Dict[str, Any] = {}
OBSERVATION_STUDY_ROWS = []


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Save one unique row per regime and seed
# -----------------------------------------------------------------------------
def save_raw_rows(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame = (
        frame
        .drop_duplicates(subset=["regime", "seed_id"], keep="last")
        .copy()
    )

    order_map = {regime: i for i, regime in enumerate(REGIMES)}
    frame["_order"] = frame["regime"].map(order_map)

    frame = (
        frame
        .sort_values(["_order", "seed_number"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    frame.to_csv(RAW_RESULTS_PATH, index=False)
    return frame


# -----------------------------------------------------------------------------
# Build numeric mean/SD and manuscript-ready "mean ± SD" tables
# -----------------------------------------------------------------------------
def build_mean_sd_tables(all_runs: pd.DataFrame):
    group_columns = ["regime", "phenotype_points"]

    excluded_numeric = {
        "seed_number",
        "dataset_seed",
        "training_seed",
        "phenotype_points",
    }

    numeric_metrics = [
        column
        for column in all_runs.select_dtypes(include=[np.number]).columns
        if column not in excluded_numeric
    ]

    grouped = all_runs.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    summary = grouped[numeric_metrics].agg(["mean", "std"]).reset_index()

    summary.columns = [
        column
        if isinstance(column, str)
        else "_".join(
            str(part) for part in column if str(part) not in {"", "None"}
        )
        for column in summary.columns
    ]

    counts = grouped.size().rename("n_seeds").reset_index()

    summary = counts.merge(
        summary,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    order_map = {regime: i for i, regime in enumerate(REGIMES)}
    summary["_order"] = summary["regime"].map(order_map)

    summary = (
        summary
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    def format_mean_sd(mean_value, std_value, metric):
        if pd.isna(mean_value):
            return "N/A"

        if pd.isna(std_value):
            return f"{mean_value:.4g} ± N/A"

        if metric.endswith("_percent"):
            return f"{mean_value:.1f} ± {std_value:.1f}"

        if metric == "training_minutes":
            return f"{mean_value:.2f} ± {std_value:.2f}"

        magnitude = abs(float(mean_value))

        if (
            metric == "mutation_probability"
            or (magnitude != 0.0 and magnitude < 1e-3)
            or magnitude >= 1e3
        ):
            return f"{mean_value:.3e} ± {std_value:.2e}"

        return f"{mean_value:.4f} ± {std_value:.4f}"

    manuscript_metrics = [
        "rmse_sensitive",
        "rmse_resistant",
        "rmse_total",
        "rmse_fraction",
        "relative_l2_sensitive",
        "relative_l2_resistant",
        "mutation_probability",
        "mu_error_percent",
        "resistance_fitness_cost",
        "cost_error_percent",
        "physics_residual_rmse",
        "fraction_points_negative",
        "fraction_points_over_capacity",
        "training_minutes",
    ]

    manuscript_rows = []

    for _, row in summary.iterrows():
        output_row = {
            "regime": row["regime"],
            "phenotype_points": int(row["phenotype_points"]),
            "n_seeds": int(row["n_seeds"]),
        }

        for metric in manuscript_metrics:
            mean_column = f"{metric}_mean"
            std_column = f"{metric}_std"

            if mean_column in summary.columns:
                output_row[metric] = format_mean_sd(
                    row[mean_column],
                    row[std_column],
                    metric,
                )

        manuscript_rows.append(output_row)

    manuscript = pd.DataFrame(manuscript_rows)

    return summary, manuscript


# =============================================================================
# Run the five-seed study
# =============================================================================
if RUN_OBSERVATION_REGIME_STUDY:

    if QUICK_MODE:
        print(
            "WARNING: QUICK_MODE=True. These are preliminary runs, "
            "not final manuscript results."
        )

    # -------------------------------------------------------------------------
    # Resume from the new five-seed raw-results file
    # -------------------------------------------------------------------------
    if RAW_RESULTS_PATH.exists() and not FORCE_RERUN:
        existing = pd.read_csv(RAW_RESULTS_PATH)

        required_columns = {"regime", "seed_id", "seed_number"}

        if required_columns.issubset(existing.columns):
            OBSERVATION_STUDY_ROWS = existing.to_dict("records")
            print(
                f"Resuming from {len(existing)} completed regime-seed runs."
            )
        else:
            print(
                "Existing file has an incompatible schema and will not be "
                "used for resume."
            )

    completed_pairs = {
        (str(row["regime"]), str(row["seed_id"]))
        for row in OBSERVATION_STUDY_ROWS
        if "regime" in row and "seed_id" in row
    }

    total_runs = len(REGIMES) * len(SEED_IDS)

    print(
        f"Running {total_runs} models: "
        f"{len(REGIMES)} regimes × {len(SEED_IDS)} seeds."
    )

    # -------------------------------------------------------------------------
    # Paired design: the same seeds are reused across O1–O4
    # -------------------------------------------------------------------------
    for seed_number in SEED_IDS:

        seed_id = f"seed{seed_number}"

        dataset_seed = int(SEED + 10_000 * seed_number)
        training_seed = int(SEED + 100_000 + 10_000 * seed_number)

        print("\n" + "=" * 78)
        print(
            f"{seed_id}: dataset_seed={dataset_seed}, "
            f"training_seed={training_seed}"
        )
        print("=" * 78)

        for regime in REGIMES:

            pair = (regime, seed_id)
            label = f"round3_observation_{regime}_{seed_id}"

            if pair in completed_pairs and not FORCE_RERUN:
                print(f"Skipping completed run: {label}")
                continue

            print(f"\nTraining {label}")

            try:
                # Use the same dataset randomness across O1–O4 for this seed.
                set_reproducible_seed(dataset_seed)

                dataset = build_observation_dataset(
                    "early_continuous",
                    regime,
                    0.0,
                    dataset_seed,
                )

                bundle = {"early_continuous": dataset}

                # Use the same network/training randomness across O1–O4.
                set_reproducible_seed(training_seed)

                model = AMRInverseModel(
                    {
                        "early_continuous":
                        SCHEDULES_R3["early_continuous"]
                    },
                    "hard",
                    width=64,
                    depth=4,
                    parameter_initialization=(3e-4, 0.15),
                )

                result = train_inverse_model(
                    model,
                    bundle,
                    label,
                    training_seed,
                )

                metrics, prediction = evaluate_trained_model(
                    result,
                    "early_continuous",
                )

                relative_error_mu = metrics.get(
                    "relative_error_mu",
                    np.nan,
                )
                relative_error_cost = metrics.get(
                    "relative_error_cost",
                    np.nan,
                )
                training_seconds = metrics.get(
                    "training_seconds",
                    np.nan,
                )

                metrics.update({
                    "regime": regime,
                    "phenotype_points": int(len(dataset.fraction_time)),
                    "seed_id": seed_id,
                    "seed_number": int(seed_number),
                    "dataset_seed": int(dataset_seed),
                    "training_seed": int(training_seed),
                    "mu_error_percent": (
                        100.0 * float(relative_error_mu)
                        if pd.notna(relative_error_mu)
                        else np.nan
                    ),
                    "cost_error_percent": (
                        100.0 * float(relative_error_cost)
                        if pd.notna(relative_error_cost)
                        else np.nan
                    ),
                    "training_minutes": (
                        float(training_seconds) / 60.0
                        if pd.notna(training_seconds)
                        else np.nan
                    ),
                })

                # Save each dense prediction for later polished plotting.
                prediction_path = (
                    PREDICTION_DIR
                    / f"{label}_early_continuous_predictions.csv"
                )
                prediction.to_csv(prediction_path, index=False)

                if KEEP_RESULTS_IN_MEMORY:
                    OBSERVATION_STUDY_RESULTS[label] = result

                OBSERVATION_STUDY_ROWS.append(metrics)
                completed_pairs.add(pair)

                current_raw = save_raw_rows(
                    OBSERVATION_STUDY_ROWS
                )

                print(
                    f"Saved {label}: "
                    f"{len(current_raw)}/{total_runs} runs completed."
                )

                result.model.to("cpu")

                del model, dataset, bundle, prediction
                gc.collect()
                clear_gpu_memory()

            except Exception:
                save_raw_rows(OBSERVATION_STUDY_ROWS)
                clear_gpu_memory()

                print(
                    f"\nThe run failed at {label}. All completed runs "
                    f"remain saved in:\n  {RAW_RESULTS_PATH}"
                )
                raise

    # -------------------------------------------------------------------------
    # Final tables
    # -------------------------------------------------------------------------
    OBSERVATION_STUDY_ALL_RUNS = save_raw_rows(
        OBSERVATION_STUDY_ROWS
    )

    (
        OBSERVATION_STUDY_SUMMARY,
        OBSERVATION_STUDY_MANUSCRIPT,
    ) = build_mean_sd_tables(
        OBSERVATION_STUDY_ALL_RUNS
    )

    OBSERVATION_STUDY_SUMMARY.to_csv(
        SUMMARY_RESULTS_PATH,
        index=False,
    )

    OBSERVATION_STUDY_MANUSCRIPT.to_csv(
        MANUSCRIPT_TABLE_PATH,
        index=False,
    )

    # Backward-compatible name.
    OBSERVATION_STUDY = OBSERVATION_STUDY_ALL_RUNS

    print("\n" + "=" * 78)
    print("FIVE-SEED OBSERVATION-REGIME STUDY COMPLETED")
    print("=" * 78)

    print("\nRaw per-seed table:")
    print(" ", RAW_RESULTS_PATH)

    print("\nNumeric mean/SD table:")
    print(" ", SUMMARY_RESULTS_PATH)

    print("\nManuscript-ready mean ± SD table:")
    print(" ", MANUSCRIPT_TABLE_PATH)

    print("\nPer-seed results:")
    display(OBSERVATION_STUDY_ALL_RUNS)

    print("\nMean ± sample SD results:")
    display(OBSERVATION_STUDY_MANUSCRIPT)

    completed_counts = (
        OBSERVATION_STUDY_ALL_RUNS
        .groupby("regime")["seed_id"]
        .nunique()
        .reindex(REGIMES, fill_value=0)
    )

    if (completed_counts == len(SEED_IDS)).all():
        print(
            "\nAll O1–O4 regimes contain five paired seeds. "
            "SD is the sample standard deviation (ddof=1)."
        )
    else:
        print(
            "\nWARNING: The study is incomplete. Completed seeds:\n"
            + completed_counts.to_string()
        )

else:
    print(
        "Set RUN_OBSERVATION_REGIME_STUDY=True in Cell 1 "
        "to run the five-seed O1–O4 study."
    )

# In[69]:


# ROUND 3 — CELL 19 (FINAL REFINED VERSION)
# OVERLAP-FREE, MORE POLISHED PUBLICATION FIGURES
# =============================================================================
# This revision specifically fixes:
# 1. title / legend overlap,
# 2. crowded x-axis labels,
# 3. figure-bottom note overlap,
# 4. paired-seed legend collision,
# 5. overly dense panel spacing.
#
# It does NOT retrain any model. It only reads the saved five-seed results.
# =============================================================================

from pathlib import Path
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


# =============================================================================
# STANDALONE ROUND-3 PATH / HELPER RECOVERY
# =============================================================================
# Cell 19 is a post-processing cell. It does not retrain any model.
# If the Round-3 setup variables are already present, they are reused.
# Otherwise, the canonical final-result folders are reconstructed so that
# the saved five-seed tables can be read directly after a kernel restart.
# =============================================================================

ROUND3_ROOT = Path(
    globals().get(
        "ROUND3_ROOT",
        Path.home() / "round3" / "final",
    )
)

TABLE_DIR_R3 = Path(
    globals().get(
        "TABLE_DIR_R3",
        ROUND3_ROOT / "tables",
    )
)

FIGURE_DIR_R3 = Path(
    globals().get(
        "FIGURE_DIR_R3",
        ROUND3_ROOT / "figures",
    )
)

TABLE_DIR_R3.mkdir(parents=True, exist_ok=True)
FIGURE_DIR_R3.mkdir(parents=True, exist_ok=True)

if "TRUE_PARAMETERS" not in globals():
    TRUE_PARAMETERS = SimpleNamespace(
        mutation_probability=1.0e-4,
        resistance_fitness_cost=0.10,
    )

if "save_figure_r3" not in globals():

    def save_figure_r3(fig, stem):
        """Save one Round-3 publication figure as PDF, PNG, and SVG."""
        stem = str(stem)

        pdf_path = FIGURE_DIR_R3 / f"{stem}.pdf"
        png_path = FIGURE_DIR_R3 / f"{stem}.png"
        svg_path = FIGURE_DIR_R3 / f"{stem}.svg"

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor="white",
        )

        print("Saved figure files:")
        print(" ", pdf_path.resolve())
        print(" ", png_path.resolve())
        print(" ", svg_path.resolve())


print("Round-3 Cell 19 paths:")
print("  Tables :", TABLE_DIR_R3.resolve())
print("  Figures:", FIGURE_DIR_R3.resolve())


# =============================================================================
# CONFIGURATION
# =============================================================================

REGIME_ORDER = ["O1", "O2", "O3", "O4"]

REGIME_DESCRIPTIONS = {
    "O1": "Total load only",
    "O2": "Total load + 1 phenotype measurement",
    "O3": "Total load + 3 phenotype measurements",
    "O4": "Fully phenotype-resolved observations",
}

EXPECTED_PHENOTYPE_POINTS = {
    "O1": 0,
    "O2": 1,
    "O3": 3,
    "O4": 21,
}

RAW_RESULTS_PATH = TABLE_DIR_R3 / "round3_observation_regime_all_seeds_metrics.csv"
FINAL_TABLE_PATH = TABLE_DIR_R3 / "round3_observation_regime_final_manuscript_table_v5.csv"
NUMERIC_SUMMARY_PATH = TABLE_DIR_R3 / "round3_observation_regime_final_numeric_summary_v5.csv"
FEASIBILITY_SUMMARY_PATH = TABLE_DIR_R3 / "round3_observation_regime_feasibility_summary_v5.csv"
EFFECT_SIZE_PATH = TABLE_DIR_R3 / "round3_observation_regime_effect_sizes_v5.csv"

GENERATE_PAIRED_SEED_FIGURE = True
GENERATE_EFFECT_SIZE_FIGURE = True


# =============================================================================
# PUBLICATION STYLE
# =============================================================================

PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 11.7,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 9.1,
    "ytick.labelsize": 9.4,
    "legend.fontsize": 8.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "grid.linewidth": 0.65,
}

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

REGIME_COLORS = {
    regime: DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
    for index, regime in enumerate(REGIME_ORDER)
}

REGIME_MARKERS = {
    "O1": "o",
    "O2": "s",
    "O3": "^",
    "O4": "D",
}


# =============================================================================
# LOAD RAW FIVE-SEED RESULTS
# =============================================================================

if (
    "OBSERVATION_STUDY_ALL_RUNS" in globals()
    and isinstance(OBSERVATION_STUDY_ALL_RUNS, pd.DataFrame)
    and not OBSERVATION_STUDY_ALL_RUNS.empty
):
    observation_runs = OBSERVATION_STUDY_ALL_RUNS.copy()
    print("Using OBSERVATION_STUDY_ALL_RUNS from memory.")
elif RAW_RESULTS_PATH.exists():
    observation_runs = pd.read_csv(RAW_RESULTS_PATH)
    print("Loaded five-seed results from:")
    print(" ", RAW_RESULTS_PATH)
else:
    raise FileNotFoundError(
        "Five-seed raw-results table not found. Expected:\n"
        f"  {RAW_RESULTS_PATH}\n"
        "Run the five-seed Cell 18 first."
    )

required_columns = [
    "regime",
    "seed_id",
    "phenotype_points",
    "rmse_sensitive",
    "rmse_resistant",
    "rmse_total",
    "rmse_fraction",
    "relative_l2_sensitive",
    "relative_l2_resistant",
    "mutation_probability",
    "mu_error_percent",
    "resistance_fitness_cost",
    "cost_error_percent",
    "physics_residual_rmse",
    "fraction_points_negative",
    "fraction_points_over_capacity",
    "training_minutes",
]

missing_columns = [column for column in required_columns if column not in observation_runs.columns]
if missing_columns:
    raise KeyError(
        "The five-seed table is missing required columns:\n"
        + "\n".join(f"  - {column}" for column in missing_columns)
    )

observation_runs = observation_runs[
    observation_runs["regime"].astype(str).isin(REGIME_ORDER)
].copy()

observation_runs = (
    observation_runs
    .drop_duplicates(subset=["regime", "seed_id"], keep="last")
    .reset_index(drop=True)
)

observation_runs["regime"] = pd.Categorical(
    observation_runs["regime"],
    categories=REGIME_ORDER,
    ordered=True,
)

observation_runs = (
    observation_runs
    .sort_values(["regime", "seed_id"])
    .reset_index(drop=True)
)

seed_counts = (
    observation_runs
    .groupby("regime", observed=True)["seed_id"]
    .nunique()
    .reindex(REGIME_ORDER, fill_value=0)
)

print("\nCompleted seed counts:")
print(seed_counts.to_string())

if not (seed_counts == 5).all():
    warnings.warn(
        "At least one regime does not contain five unique seeds. "
        "Figures will use the available completed runs."
    )


# =============================================================================
# CORRECT SUMMARY TABLES
# =============================================================================

metrics_for_summary = [
    "rmse_sensitive",
    "rmse_resistant",
    "rmse_total",
    "rmse_fraction",
    "relative_l2_sensitive",
    "relative_l2_resistant",
    "mutation_probability",
    "mu_error_percent",
    "resistance_fitness_cost",
    "cost_error_percent",
    "physics_residual_rmse",
    "fraction_points_negative",
    "fraction_points_over_capacity",
    "training_minutes",
]

summary_mean = observation_runs.groupby("regime", observed=True)[metrics_for_summary].mean()
summary_std = observation_runs.groupby("regime", observed=True)[metrics_for_summary].std(ddof=1)
summary_count = observation_runs.groupby("regime", observed=True).size().rename("n_seeds")
phenotype_points = observation_runs.groupby("regime", observed=True)["phenotype_points"].first().rename("phenotype_points")

numeric_summary = pd.concat(
    [
        phenotype_points,
        summary_count,
        summary_mean.add_prefix("mean_"),
        summary_std.add_prefix("std_"),
    ],
    axis=1,
).reset_index()

numeric_summary["regime"] = pd.Categorical(
    numeric_summary["regime"],
    categories=REGIME_ORDER,
    ordered=True,
)

numeric_summary = numeric_summary.sort_values("regime").reset_index(drop=True)
numeric_summary.to_csv(NUMERIC_SUMMARY_PATH, index=False)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def seed_sort_key(seed_id):
    match = re.search(r"(\d+)$", str(seed_id))
    if match:
        return int(match.group(1))
    return str(seed_id)


def compact_regime_labels():
    return [
        "O1\n(0 pts)",
        "O2\n(1 pt)",
        "O3\n(3 pts)",
        "O4\n(21 pts)",
    ]


def summary_value(regime, metric, statistic="mean"):
    row = numeric_summary[numeric_summary["regime"].astype(str) == regime]
    if row.empty:
        return np.nan
    column = f"{statistic}_{metric}"
    return float(row.iloc[0][column])


def raw_values(regime, metric):
    return (
        observation_runs.loc[
            observation_runs["regime"].astype(str) == regime,
            metric,
        ]
        .astype(float)
        .dropna()
        .to_numpy()
    )


def format_value(value, metric):
    if pd.isna(value):
        return "N/A"
    value = float(value)

    if metric in {"mutation_probability", "rmse_fraction", "physics_residual_rmse"}:
        if abs(value) < 0.001:
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")

    if metric in {"mu_error_percent", "cost_error_percent"}:
        return f"{value:.1f}%"

    if metric == "training_minutes":
        return f"{value:.1f}"

    if abs(value) >= 100:
        return f"{value:.0f}"

    if abs(value) >= 1:
        return f"{value:.2f}"

    return f"{value:.3f}".rstrip("0").rstrip(".")



def format_annotation_value(value, metric):
    """
    Format values shown inside figure annotation boxes.

    Resistant-fraction RMSE is shown in scientific notation for consistent
    journal-scale readability; other metrics retain the compact decimal style.
    """
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if metric == "rmse_fraction" and value != 0.0:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10.0 ** exponent)
        return rf"${mantissa:.2f}\times10^{{{exponent}}}$"

    return format_value(value, metric)


def format_mean_sd(mean_value, sd_value, metric):
    if pd.isna(mean_value):
        return "N/A"
    if pd.isna(sd_value):
        return f"{format_value(mean_value, metric)} ± N/A"

    if metric in {"mutation_probability", "rmse_fraction", "physics_residual_rmse"}:
        return f"{format_value(mean_value, metric)} ± {format_value(sd_value, metric)}"

    if metric in {"mu_error_percent", "cost_error_percent"}:
        return f"{mean_value:.1f} ± {sd_value:.1f}"

    if metric == "training_minutes":
        return f"{mean_value:.2f} ± {sd_value:.2f}"

    return f"{mean_value:.4f} ± {sd_value:.4f}"


def set_black_frame(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def add_panel_label(axis, label):
    axis.text(
        -0.09,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_footer_note(figure, text, y=0.045):
    figure.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.6,
        color="black",
    )


def annotation_layout_for_regime(regime_index, total_regimes):
    # Place numeric labels slightly to the side of the markers so they do not
    # sit on the mean-connecting line or on top of the error bars.
    if total_regimes <= 4:
        layout_map = {
            0: dict(offset=(12, 8), ha="left"),
            1: dict(offset=(12, 8), ha="left"),
            2: dict(offset=(-12, 8), ha="right"),
            3: dict(offset=(-12, 8), ha="right"),
        }
        return layout_map.get(regime_index, dict(offset=(10, 8), ha="left"))

    return dict(offset=(10, 8), ha="left")


def annotate_mean_value(axis, x_value, y_value, metric, regime_index=None, total_regimes=4):
    """Add a compact publication-style label to the across-seed mean."""
    layout = annotation_layout_for_regime(regime_index, total_regimes)

    # Final manuscript micro-adjustment:
    # In the resistant-fraction panel, place the O3 mean label on the
    # RIGHT side and ABOVE the connecting line for maximum clarity.
    vertical_alignment = "bottom"
    if metric == "rmse_fraction" and regime_index == 2:
        layout = dict(offset=(14, 10), ha="left")
        vertical_alignment = "bottom"

    axis.annotate(
        format_annotation_value(y_value, metric),
        (x_value, y_value),
        xytext=layout["offset"],
        textcoords="offset points",
        ha=layout["ha"],
        va=vertical_alignment,
        fontsize=8.3,
        fontweight="semibold",
        annotation_clip=True,
        clip_on=True,
        bbox=dict(
            boxstyle="round,pad=0.14",
            facecolor="white",
            edgecolor="0.40",
            linewidth=0.60,
            alpha=0.94,
        ),
        zorder=7,
    )


def plot_distribution_panel(
    axis,
    metric,
    ylabel,
    title,
    *,
    log_scale=False,
    true_value=None,
    annotate_means=True,
    lower_is_better=False,
    nonnegative_metric=False,
):
    # Retained for compatibility with existing calls; interpretation belongs
    # in the Results and Discussion rather than inside the figure panels.
    _ = lower_is_better

    x = np.arange(len(REGIME_ORDER), dtype=float)

    means = np.array([summary_value(regime, metric, "mean") for regime in REGIME_ORDER])
    standard_deviations = np.array([summary_value(regime, metric, "std") for regime in REGIME_ORDER])

    for regime_index, regime in enumerate(REGIME_ORDER):
        values = raw_values(regime, metric)
        if len(values) == 0:
            continue

        jitter = np.array([0.0]) if len(values) == 1 else np.linspace(-0.12, 0.12, len(values))

        axis.scatter(
            np.full(len(values), x[regime_index]) + jitter,
            values,
            s=28,
            marker=REGIME_MARKERS[regime],
            facecolor=REGIME_COLORS[regime],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.38,
            zorder=2,
        )

    axis.plot(
        x,
        means,
        color="black",
        linewidth=1.35,
        alpha=0.78,
        zorder=3,
    )

    for regime_index, regime in enumerate(REGIME_ORDER):

        mean_value = float(means[regime_index])
        sd_value = float(standard_deviations[regime_index])

        # RMSE and related error metrics are nonnegative by definition.
        # For those panels, retain the full +1 SD upper bar but truncate only
        # the visual lower extent at zero. This avoids displaying an impossible
        # negative RMSE while preserving the reported mean and sample SD.
        if nonnegative_metric and not log_scale:
            lower_error = min(sd_value, max(mean_value, 0.0))
            upper_error = sd_value

            y_error = np.array(
                [
                    [lower_error],
                    [upper_error],
                ],
                dtype=float,
            )

        else:
            y_error = sd_value

        axis.errorbar(
            x[regime_index],
            means[regime_index],
            yerr=y_error,
            fmt=REGIME_MARKERS[regime],
            markersize=7.8,
            markerfacecolor=REGIME_COLORS[regime],
            markeredgecolor="black",
            markeredgewidth=0.85,
            ecolor="black",
            elinewidth=1.15,
            capsize=4.0,
            capthick=1.0,
            zorder=5,
        )

    if true_value is not None:
        axis.axhline(
            true_value,
            linestyle="--",
            linewidth=1.35,
            color="black",
            zorder=1,
            label="True value = " + format_value(true_value, metric),
        )

    if log_scale:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10))
        axis.yaxis.set_minor_formatter(NullFormatter())

    elif nonnegative_metric:
        # RMSE cannot be negative. Keep the automatically determined upper
        # limit but anchor the visible axis at the physically valid boundary.
        _, current_upper = axis.get_ylim()
        axis.set_ylim(
            bottom=0.0,
            top=current_upper,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(compact_regime_labels())
    axis.set_xlim(-0.45, len(REGIME_ORDER) - 0.55)

    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="semibold", pad=7)
    axis.grid(True, axis="y")
    axis.set_axisbelow(True)
    set_black_frame(axis)

    if annotate_means:
        for regime_index, mean_value in enumerate(means):
            if pd.isna(mean_value):
                continue
            annotate_mean_value(axis, x[regime_index], mean_value, metric, regime_index=regime_index, total_regimes=len(REGIME_ORDER))

    return means, standard_deviations


# =============================================================================
# FIGURE 1 — FINAL MANUSCRIPT STATE-RECONSTRUCTION FIGURE
# =============================================================================
# Main O1--O4 state-observability figure.
#
# Publication decisions:
#   - no large in-figure title;
#   - no explanatory footer inside the artwork;
#   - concise, neutral panel titles;
#   - individual seed results retained;
#   - across-seed mean ± sample SD retained;
#   - log scale retained for resistant-fraction RMSE;
#   - compact mean-value labels;
#   - scientific notation used consistently in Panel D;
#   - RMSE axes remain physically valid (no negative displayed RMSE).
# =============================================================================

with plt.rc_context(PUBLICATION_RC):

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.4, 8.35),
    )

    state_panels = [
        (
            "rmse_sensitive",
            axes[0, 0],
            r"Sensitive-state RMSE",
            r"Sensitive state, $S(t)$",
            False,
        ),
        (
            "rmse_resistant",
            axes[0, 1],
            r"Resistant-state RMSE",
            r"Resistant state, $R(t)$",
            False,
        ),
        (
            "rmse_total",
            axes[1, 0],
            r"Total-population RMSE",
            r"Total population, $S(t)+R(t)$",
            False,
        ),
        (
            "rmse_fraction",
            axes[1, 1],
            r"Resistant-fraction RMSE",
            r"Resistant fraction, $q_R(t)$",
            True,
        ),
    ]

    for panel_label, (
        metric,
        axis,
        ylabel,
        title,
        use_log,
    ) in zip(
        ["A", "B", "C", "D"],
        state_panels,
    ):

        plot_distribution_panel(
            axis,
            metric,
            ylabel,
            title,
            log_scale=use_log,
            annotate_means=True,
            lower_is_better=True,
            nonnegative_metric=True,
        )

        add_panel_label(
            axis,
            panel_label,
        )

        axis.set_title(
            title,
            loc="left",
            fontsize=11.8,
            fontweight="semibold",
            pad=8,
        )

        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            width=0.85,
            length=3.8,
        )

    figure.align_ylabels(
        axes[:, 0]
    )
    figure.align_ylabels(
        axes[:, 1]
    )

    # No figure-level title and no explanatory footer.
    # These belong in the manuscript caption / Results text.

    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.955,
        wspace=0.27,
        hspace=0.38,
    )

    save_figure_r3(
        figure,
        "round3_observation_regime_state_distributions_v5",
    )

    plt.show()


# =============================================================================
# FIGURE 2 — PARAMETER RECOVERY AND IDENTIFIABILITY
# =============================================================================

with plt.rc_context(PUBLICATION_RC):

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.1))

    parameter_panels = [
        ("mutation_probability", axes[0, 0], r"Estimated mutation probability, $\hat{\mu}$", "Mutation-probability recovery", True, float(TRUE_PARAMETERS.mutation_probability), False),
        ("resistance_fitness_cost", axes[0, 1], r"Estimated resistance fitness cost, $\hat{c}_R$", "Resistance-fitness-cost recovery", False, float(TRUE_PARAMETERS.resistance_fitness_cost), False),
        ("mu_error_percent", axes[1, 0], r"Relative error in $\mu$ (%)", "Mutation-probability identifiability", True, None, True),
        ("cost_error_percent", axes[1, 1], r"Relative error in $c_R$ (%)", "Fitness-cost identifiability", False, None, True),
    ]

    for panel_label, (metric, axis, ylabel, title, use_log, true_value, lower_is_better) in zip(
        ["A", "B", "C", "D"], parameter_panels
    ):
        plot_distribution_panel(
            axis,
            metric,
            ylabel,
            title,
            log_scale=use_log,
            true_value=true_value,
            annotate_means=True,
            lower_is_better=lower_is_better,
        )
        if true_value is not None:
            axis.legend(loc="upper right", frameon=True, fontsize=8.2)
        add_panel_label(axis, panel_label)

    figure.suptitle(
        "State observability and evolutionary-parameter identifiability are distinct",
        fontsize=15.7,
        fontweight="bold",
        y=0.975,
    )

    add_footer_note(
        figure,
        "Even when hidden-state reconstruction improves, parameter recovery can remain substantially harder.",
        y=0.038,
    )

    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.11,
        top=0.90,
        wspace=0.25,
        hspace=0.35,
    )

    save_figure_r3(figure, "round3_observation_regime_parameter_distributions_v5")
    plt.show()


# =============================================================================
# FIGURE 3 — PHYSICS CONSISTENCY AND COMPUTATIONAL COST
# =============================================================================

with plt.rc_context(PUBLICATION_RC):

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    plot_distribution_panel(
        axes[0],
        "physics_residual_rmse",
        "Physics-residual RMSE",
        "Dynamical consistency",
        log_scale=True,
        annotate_means=True,
        lower_is_better=True,
    )

    plot_distribution_panel(
        axes[1],
        "training_minutes",
        "Training time (min)",
        "Computational cost",
        log_scale=False,
        annotate_means=True,
        lower_is_better=False,
    )

    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")

    figure.suptitle(
        "Physics consistency and computational cost",
        fontsize=15.5,
        fontweight="bold",
        y=0.96,
    )

    add_footer_note(
        figure,
        "O3 is computationally cheapest on average, while O4 attains the best resistant-state reconstruction.",
        y=0.06,
    )

    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.18,
        top=0.84,
        wspace=0.26,
    )

    save_figure_r3(figure, "round3_observation_regime_physics_cost_v5")
    plt.show()


# =============================================================================
# FIGURE 4 — INFORMATION GAIN RELATIVE TO O1
# =============================================================================

if GENERATE_EFFECT_SIZE_FIGURE:

    baseline_metrics = {
        metric: summary_value("O1", metric, "mean")
        for metric in ["rmse_resistant", "rmse_fraction", "mu_error_percent", "cost_error_percent"]
    }

    effect_rows = []
    for regime in REGIME_ORDER:
        row = {"regime": regime, "phenotype_points": EXPECTED_PHENOTYPE_POINTS[regime]}
        for metric, baseline in baseline_metrics.items():
            current = summary_value(regime, metric, "mean")
            row[f"{metric}_improvement_percent"] = 100.0 * (baseline - current) / baseline
        effect_rows.append(row)

    effect_sizes = pd.DataFrame(effect_rows)
    effect_sizes.to_csv(EFFECT_SIZE_PATH, index=False)

    with plt.rc_context(PUBLICATION_RC):

        figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
        x = np.arange(len(REGIME_ORDER))
        width = 0.34

        resistant_improvement = effect_sizes["rmse_resistant_improvement_percent"].to_numpy(float)
        fraction_improvement = effect_sizes["rmse_fraction_improvement_percent"].to_numpy(float)

        axes[0].bar(x - width / 2, resistant_improvement, width, label=r"RMSE $R$ reduction")
        axes[0].bar(x + width / 2, fraction_improvement, width, label=r"RMSE $q_R$ reduction")
        axes[0].axhline(0.0, color="black", linewidth=1.0)
        axes[0].set_ylabel("Improvement relative to O1 (%)")
        axes[0].set_title("Gain in resistant-state observability", loc="left", fontweight="semibold")
        axes[0].legend(loc="upper left", frameon=True)

        mu_improvement = effect_sizes["mu_error_percent_improvement_percent"].to_numpy(float)
        cost_improvement = effect_sizes["cost_error_percent_improvement_percent"].to_numpy(float)

        axes[1].bar(x - width / 2, mu_improvement, width, label=r"Error reduction in $\mu$")
        axes[1].bar(x + width / 2, cost_improvement, width, label=r"Error reduction in $c_R$")
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_ylabel("Improvement relative to O1 (%)")
        axes[1].set_title("Gain in parameter identifiability", loc="left", fontweight="semibold")
        axes[1].legend(loc="upper left", frameon=True)

        for panel_label, axis in zip(["A", "B"], axes):
            axis.set_xticks(x)
            axis.set_xticklabels(compact_regime_labels())
            axis.grid(True, axis="y")
            axis.set_axisbelow(True)
            set_black_frame(axis)
            add_panel_label(axis, panel_label)

        figure.suptitle(
            "Information gain from additional phenotype observations",
            fontsize=15.5,
            fontweight="bold",
            y=0.96,
        )

        add_footer_note(
            figure,
            "O4 provides the largest overall gain; O2 can help some quantities while degrading others.",
            y=0.065,
        )

        figure.subplots_adjust(
            left=0.09,
            right=0.985,
            bottom=0.18,
            top=0.84,
            wspace=0.28,
        )

        save_figure_r3(figure, "round3_observation_regime_information_gain_v5")
        plt.show()


# =============================================================================
# FIGURE 5 — PAIRED-SEED STABILITY
# =============================================================================

if GENERATE_PAIRED_SEED_FIGURE:

    with plt.rc_context(PUBLICATION_RC):

        figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
        x = np.arange(len(REGIME_ORDER))

        seed_ids = sorted(observation_runs["seed_id"].astype(str).unique(), key=seed_sort_key)

        seed_colors = {
            seed_id: DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
            for index, seed_id in enumerate(seed_ids)
        }

        for seed_id in seed_ids:
            seed_frame = observation_runs[observation_runs["seed_id"].astype(str) == seed_id].copy()
            seed_frame["regime"] = pd.Categorical(seed_frame["regime"], categories=REGIME_ORDER, ordered=True)
            seed_frame = seed_frame.sort_values("regime").reset_index(drop=True)

            if len(seed_frame) != len(REGIME_ORDER):
                continue

            axes[0].plot(
                x,
                seed_frame["rmse_resistant"].to_numpy(float),
                marker="o",
                markersize=4.8,
                linewidth=1.05,
                alpha=0.50,
                color=seed_colors[seed_id],
                label=seed_id,
            )

            axes[1].plot(
                x,
                seed_frame["mu_error_percent"].to_numpy(float),
                marker="o",
                markersize=4.8,
                linewidth=1.05,
                alpha=0.50,
                color=seed_colors[seed_id],
                label=seed_id,
            )

        resistant_mean = np.array([summary_value(regime, "rmse_resistant", "mean") for regime in REGIME_ORDER])
        mu_error_mean = np.array([summary_value(regime, "mu_error_percent", "mean") for regime in REGIME_ORDER])

        axes[0].plot(
            x,
            resistant_mean,
            color="black",
            marker="D",
            markersize=6.8,
            linewidth=2.25,
            label="Mean",
            zorder=6,
        )

        axes[1].plot(
            x,
            mu_error_mean,
            color="black",
            marker="D",
            markersize=6.8,
            linewidth=2.25,
            label="Mean",
            zorder=6,
        )

        axes[0].set_ylabel(r"Resistant-state RMSE")
        axes[0].set_title("Seed-level resistant-state reconstruction", loc="left", fontweight="semibold")

        axes[1].set_yscale("log")
        axes[1].set_ylabel(r"Relative error in $\mu$ (%)")
        axes[1].set_title("Seed-level mutation-identification error", loc="left", fontweight="semibold")

        for panel_label, axis in zip(["A", "B"], axes):
            axis.set_xticks(x)
            axis.set_xticklabels(compact_regime_labels())
            axis.set_xlim(-0.25, len(REGIME_ORDER) - 0.75)
            axis.grid(True, axis="y")
            axis.set_axisbelow(True)
            set_black_frame(axis)
            add_panel_label(axis, panel_label)

        legend_handles, legend_labels = axes[1].get_legend_handles_labels()

        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=3,
            frameon=True,
            columnspacing=1.2,
            handlelength=2.0,
        )

        figure.suptitle(
            "Paired-seed stability across observation regimes",
            fontsize=15.7,
            fontweight="bold",
            y=0.975,
        )

        add_footer_note(
            figure,
            "Thin colored lines show paired runs for each seed; the black line shows the across-seed mean.",
            y=0.065,
        )

        figure.subplots_adjust(
            left=0.08,
            right=0.985,
            bottom=0.18,
            top=0.77,
            wspace=0.28,
        )

        save_figure_r3(figure, "round3_observation_regime_paired_seed_stability_v5")
        plt.show()


# =============================================================================
# MANUSCRIPT TABLE
# =============================================================================

table_metrics = [
    "rmse_sensitive",
    "rmse_resistant",
    "rmse_total",
    "rmse_fraction",
    "mutation_probability",
    "mu_error_percent",
    "resistance_fitness_cost",
    "cost_error_percent",
    "physics_residual_rmse",
    "training_minutes",
]

table_rows = []
for regime in REGIME_ORDER:
    row = {
        "Regime": regime,
        "Phenotype information": REGIME_DESCRIPTIONS[regime],
        "Phenotype points": EXPECTED_PHENOTYPE_POINTS[regime],
        "Seeds": int(seed_counts.loc[regime]),
    }
    for metric in table_metrics:
        row[metric] = format_mean_sd(
            summary_value(regime, metric, "mean"),
            summary_value(regime, metric, "std"),
            metric,
        )
    table_rows.append(row)

final_manuscript_table = (
    pd.DataFrame(table_rows)
    .rename(
        columns={
            "rmse_sensitive": "RMSE S",
            "rmse_resistant": "RMSE R",
            "rmse_total": "RMSE total",
            "rmse_fraction": "RMSE fraction",
            "mutation_probability": "Estimated mu",
            "mu_error_percent": "Mu error (%)",
            "resistance_fitness_cost": "Estimated c_R",
            "cost_error_percent": "c_R error (%)",
            "physics_residual_rmse": "Physics RMSE",
            "training_minutes": "Training time (min)",
        }
    )
)

final_manuscript_table.to_csv(FINAL_TABLE_PATH, index=False)


# =============================================================================
# FEASIBILITY TABLE
# =============================================================================

feasibility_rows = []
for regime in REGIME_ORDER:
    feasibility_rows.append({
        "Regime": regime,
        "Negative-point fraction": format_mean_sd(
            summary_value(regime, "fraction_points_negative", "mean"),
            summary_value(regime, "fraction_points_negative", "std"),
            "fraction_points_negative",
        ),
        "Over-capacity fraction": format_mean_sd(
            summary_value(regime, "fraction_points_over_capacity", "mean"),
            summary_value(regime, "fraction_points_over_capacity", "std"),
            "fraction_points_over_capacity",
        ),
    })

feasibility_summary = pd.DataFrame(feasibility_rows)
feasibility_summary.to_csv(FEASIBILITY_SUMMARY_PATH, index=False)


# =============================================================================
# FINAL REPORT
# =============================================================================

o1_rmse_r = summary_value("O1", "rmse_resistant", "mean")
o4_rmse_r = summary_value("O4", "rmse_resistant", "mean")
o1_rmse_fraction = summary_value("O1", "rmse_fraction", "mean")
o4_rmse_fraction = summary_value("O4", "rmse_fraction", "mean")

resistant_reduction_percent = 100.0 * (o1_rmse_r - o4_rmse_r) / o1_rmse_r
fraction_reduction_percent = 100.0 * (o1_rmse_fraction - o4_rmse_fraction) / o1_rmse_fraction

print("\n" + "=" * 78)
print("REFINED OBSERVATION-REGIME FIGURES COMPLETED")
print("=" * 78)

print("\nGenerated figures:")
print("  round3_observation_regime_state_distributions_v5")
print("  round3_observation_regime_parameter_distributions_v5")
print("  round3_observation_regime_physics_cost_v5")
if GENERATE_EFFECT_SIZE_FIGURE:
    print("  round3_observation_regime_information_gain_v5")
if GENERATE_PAIRED_SEED_FIGURE:
    print("  round3_observation_regime_paired_seed_stability_v5")

print("\nGenerated tables:")
print(" ", FINAL_TABLE_PATH)
print(" ", NUMERIC_SUMMARY_PATH)
print(" ", FEASIBILITY_SUMMARY_PATH)
if GENERATE_EFFECT_SIZE_FIGURE:
    print(" ", EFFECT_SIZE_PATH)

print("\nKey O4 improvements relative to O1:")
print(f"  Resistant-state RMSE reduction: {resistant_reduction_percent:.1f}%")
print(f"  Resistant-fraction RMSE reduction: {fraction_reduction_percent:.1f}%")

print("\nFinal manuscript-ready table:")
display(final_manuscript_table)

print("\nFeasibility summary:")
display(feasibility_summary)

print("\nNo retraining was performed in this cell.")
print("Main manuscript state figure updated: round3_observation_regime_state_distributions_v5")
