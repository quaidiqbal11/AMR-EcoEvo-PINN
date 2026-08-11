#!/usr/bin/env python
# coding: utf-8

# Round 4

# In[1]:


# =============================================================================
# ROUND 4 — CELL 1
# Final primary multi-schedule experiment configuration for an 8 GB GPU
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# Round 4 tests whether combining several prescribed antibiotic-treatment
# schedules improves practical identifiability of the shared eco-evolutionary
# parameters:
#
#     mutation probability, mu,
#     resistance fitness cost, c_R.
#
# Every treatment schedule has its own hard-constrained neural state model:
#
#     S_j(t), R_j(t),
#
# while the biological parameters mu and c_R are shared across all schedules
# included in the same experimental design.
#
# The antibiotic treatment is prescribed and is NOT optimized in Round 4.
#
# FINAL PRIMARY EXPERIMENT
# ------------------------
# This configuration activates the five-seed M1–M5 study:
#
#     M1: early-continuous schedule, O3 observations
#     M2: early + late schedules, O3 observations
#     M3: early + late + pulsed schedules, O3 observations
#     M4: untreated + early + late + pulsed schedules, O3 observations
#     M5: all four schedules, O4 observations
#
# Number of primary models:
#
#     5 experimental designs × 5 paired seeds = 25 trained models
#
# EXECUTION POLICY FOR AN 8 GB GPU
# ---------------------------------
# 1. Train only one design-seed model at a time.
# 2. Use float64 for stable derivatives and small mutation probabilities.
# 3. Move completed models to CPU before starting the next model.
# 4. Clear unused Python and CUDA memory after every completed run.
# 5. Save each completed design-seed result immediately.
# 6. Resume from saved results after interruption.
# 7. Activate only one expensive Round 4 suite at a time.
#
# IMPORTANT EXECUTION INSTRUCTION
# -------------------------------
# After replacing the earlier quick-mode Cell 1 with this final version:
#
#     1. Save the notebook.
#     2. Restart the kernel.
#     3. Run Cells 1–14 sequentially.
#     4. Skip Cell 15 because the structural smoke test is disabled.
#     5. Skip Cell 16 because the single-design demonstration is disabled.
#     6. Run Cell 17 to train the 25 primary models.
#     7. Run Cells 18–20 after Cell 17 finishes.
#
# Do not activate the optional ODE, identifiability, profile, transfer, or
# plotting-only suites during this primary training run.
# =============================================================================

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

from dataclasses import dataclass, asdict, replace, field
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Iterable,
    Any,
    Sequence,
)

import numpy as np
import pandas as pd


# =============================================================================
# GLOBAL REPRODUCIBILITY SEED
# =============================================================================

SEED = 20260806

random.seed(SEED)
np.random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)


# =============================================================================
# EXECUTION MODE
# =============================================================================
# False activates the full manuscript-scale budgets and five paired seeds.

QUICK_MODE: bool = False

if not isinstance(QUICK_MODE, bool):
    raise TypeError(
        "QUICK_MODE must be either True or False."
    )

RUN_LABEL = (
    "quick"
    if QUICK_MODE
    else "final"
)


# =============================================================================
# PROJECT AND OUTPUT FOLDERS
# =============================================================================

PROJECT_ROOT = Path.cwd()

# When the notebook is stored in a project/notebooks folder, use the parent
# directory as the project root.
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

ROUND4_BASE_ROOT = (
    PROJECT_ROOT
    / "round4"
)

ROUND4_ROOT = (
    ROUND4_BASE_ROOT
    / RUN_LABEL
)

MODEL_DIR_R4 = (
    ROUND4_ROOT
    / "models"
)

PREDICTION_DIR_R4 = (
    ROUND4_ROOT
    / "predictions"
)

HISTORY_DIR_R4 = (
    ROUND4_ROOT
    / "histories"
)

FIGURE_DIR_R4 = (
    ROUND4_ROOT
    / "figures"
)

TABLE_DIR_R4 = (
    ROUND4_ROOT
    / "tables"
)

REPORT_DIR_R4 = (
    ROUND4_ROOT
    / "reports"
)

CACHE_DIR_R4 = (
    ROUND4_ROOT
    / "cache"
)

ROUND4_DIRECTORIES = [
    MODEL_DIR_R4,
    PREDICTION_DIR_R4,
    HISTORY_DIR_R4,
    FIGURE_DIR_R4,
    TABLE_DIR_R4,
    REPORT_DIR_R4,
    CACHE_DIR_R4,
]

for directory in ROUND4_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# GPU-AWARE TRAINING BUDGETS
# =============================================================================
# The quick budgets remain defined for future debugging, but this final Cell 1
# uses the manuscript-scale branch because QUICK_MODE=False.

if QUICK_MODE:

    # -------------------------------------------------------------------------
    # Structural verification only
    # -------------------------------------------------------------------------

    PRETRAIN_EPOCHS_R4 = 15

    PHYSICS_WARMUP_EPOCHS_R4 = 15

    ADAM_EPOCHS_R4 = 30

    LBFGS_MAX_ITER_R4 = 0

    COLLOCATION_POINTS_PER_INTERVAL_R4 = 32

    VALIDATION_POINTS_PER_INTERVAL_R4 = 16

    LBFGS_POINTS_PER_INTERVAL_R4 = 24

    EARLY_STOPPING_PATIENCE_R4 = 30

    LOG_EVERY_R4 = 5

    RESAMPLE_EVERY_R4 = 5

    PRIMARY_SEED_IDS_R4 = [1]

else:

    # -------------------------------------------------------------------------
    # Final manuscript-scale experiment for an 8 GB GPU
    # -------------------------------------------------------------------------

    PRETRAIN_EPOCHS_R4 = 1500

    PHYSICS_WARMUP_EPOCHS_R4 = 1000

    ADAM_EPOCHS_R4 = 9000

    LBFGS_MAX_ITER_R4 = 300

    COLLOCATION_POINTS_PER_INTERVAL_R4 = 768

    VALIDATION_POINTS_PER_INTERVAL_R4 = 128

    LBFGS_POINTS_PER_INTERVAL_R4 = 256

    EARLY_STOPPING_PATIENCE_R4 = 1800

    LOG_EVERY_R4 = 100

    RESAMPLE_EVERY_R4 = 20

    PRIMARY_SEED_IDS_R4 = [
        1,
        2,
        3,
        4,
        5,
    ]


# =============================================================================
# NEURAL-NETWORK AND OPTIMIZER SETTINGS
# =============================================================================

NETWORK_WIDTH_R4 = 64

NETWORK_DEPTH_R4 = 4

# State-network learning rate
STATE_LEARNING_RATE_R4 = 8.0e-4

# Shared biological-parameter learning rate
PARAMETER_LEARNING_RATE_R4 = 2.0e-3

# Protects against occasional unstable gradient steps
GRADIENT_CLIP_NORM_R4 = 10.0


# =============================================================================
# PRIMARY EXPERIMENT SWITCHES
# =============================================================================
# Only the final five-seed M1–M5 primary experiment is activated here.

RUN_STRUCTURAL_SMOKE_TEST = False

RUN_SINGLE_DESIGN_DEMO = False

RUN_PRIMARY_MULTI_SCHEDULE_STUDY = True

RUN_MATCHED_ODE_BASELINES = False

RUN_IDENTIFIABILITY_GEOMETRY = False

RUN_LEAVE_ONE_SCHEDULE_OUT = False

RUN_PROFILE_SURFACES = False

RUN_PARAMETER_TRANSFER_TEST = False

RUN_PLOTTING_ONLY = False


# =============================================================================
# EXPERIMENT-SWITCH REGISTRY
# =============================================================================

EXPERIMENT_SWITCHES_R4 = {
    "structural_smoke_test":
        RUN_STRUCTURAL_SMOKE_TEST,

    "single_design_demo":
        RUN_SINGLE_DESIGN_DEMO,

    "primary_multi_schedule_study":
        RUN_PRIMARY_MULTI_SCHEDULE_STUDY,

    "matched_ode_baselines":
        RUN_MATCHED_ODE_BASELINES,

    "identifiability_geometry":
        RUN_IDENTIFIABILITY_GEOMETRY,

    "leave_one_schedule_out":
        RUN_LEAVE_ONE_SCHEDULE_OUT,

    "profile_surfaces":
        RUN_PROFILE_SURFACES,

    "parameter_transfer_test":
        RUN_PARAMETER_TRANSFER_TEST,

    "plotting_only":
        RUN_PLOTTING_ONLY,
}

active_experiments_r4 = [
    experiment_name
    for experiment_name, enabled
    in EXPERIMENT_SWITCHES_R4.items()
    if enabled
]


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

if len(active_experiments_r4) == 0:
    raise RuntimeError(
        "No Round 4 experiment is active. "
        "Enable exactly one experiment switch."
    )

if len(active_experiments_r4) > 1:
    raise RuntimeError(
        "More than one Round 4 experiment is active:\n"
        f"  {active_experiments_r4}\n"
        "Activate only one expensive suite at a time on the 8 GB GPU."
    )

CURRENT_EXPERIMENT_R4 = (
    active_experiments_r4[0]
)

if QUICK_MODE and RUN_PRIMARY_MULTI_SCHEDULE_STUDY:
    warnings.warn(
        "The primary multi-schedule study is active while QUICK_MODE=True. "
        "This would produce only preliminary debugging results."
    )

if (
    not QUICK_MODE
    and RUN_STRUCTURAL_SMOKE_TEST
):
    warnings.warn(
        "The structural smoke test is active while QUICK_MODE=False. "
        "The smoke test should normally use QUICK_MODE=True."
    )

if (
    RUN_PRIMARY_MULTI_SCHEDULE_STUDY
    and QUICK_MODE
):
    raise RuntimeError(
        "The final primary study requires QUICK_MODE=False."
    )

if (
    RUN_PRIMARY_MULTI_SCHEDULE_STUDY
    and PRIMARY_SEED_IDS_R4
    != [1, 2, 3, 4, 5]
):
    raise RuntimeError(
        "The final primary experiment must use five paired seeds: "
        "[1, 2, 3, 4, 5]."
    )


# =============================================================================
# EXPECTED PRIMARY-STUDY SIZE
# =============================================================================

PRIMARY_DESIGN_IDS_R4 = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
]

EXPECTED_PRIMARY_MODELS_R4 = (
    len(PRIMARY_DESIGN_IDS_R4)
    * len(PRIMARY_SEED_IDS_R4)
)


# =============================================================================
# RESUME-SAFE PRIMARY RESULTS PATH
# =============================================================================

PRIMARY_RESULTS_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_all_seeds_metrics.csv"
)

PRIMARY_SUMMARY_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_summary_mean_std.csv"
)

PRIMARY_MANUSCRIPT_TABLE_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_manuscript_table.csv"
)


# =============================================================================
# COMPATIBILITY ALIASES
# =============================================================================
# Some reusable utilities use the generic directory names below. They are
# mapped to the active Round 4 final-output folders.

MODEL_DIR = MODEL_DIR_R4

PREDICTION_DIR = PREDICTION_DIR_R4

HISTORY_DIR = HISTORY_DIR_R4

FIGURE_DIR = FIGURE_DIR_R4

TABLE_DIR = TABLE_DIR_R4

REPORT_DIR = REPORT_DIR_R4

CACHE_DIR = CACHE_DIR_R4


# =============================================================================
# PRINT AND VERIFY THE FINAL CONFIGURATION
# =============================================================================

print("=" * 78)

print(
    "ROUND 4 EXECUTION CONFIGURATION"
)

print("=" * 78)

print(
    "Project root:",
    PROJECT_ROOT.resolve(),
)

print(
    "Round 4 base root:",
    ROUND4_BASE_ROOT.resolve(),
)

print(
    "Output root:",
    ROUND4_ROOT.resolve(),
)

print(
    "Python:",
    sys.version.split()[0],
)

print(
    "Operating system:",
    platform.platform(),
)

print(
    "Random seed:",
    SEED,
)

print(
    "QUICK_MODE:",
    QUICK_MODE,
)

print(
    "Run label:",
    RUN_LABEL,
)

print(
    "Current experiment:",
    CURRENT_EXPERIMENT_R4,
)

print(
    "Primary designs:",
    PRIMARY_DESIGN_IDS_R4,
)

print(
    "Primary seeds:",
    PRIMARY_SEED_IDS_R4,
)

print(
    "Expected primary models:",
    EXPECTED_PRIMARY_MODELS_R4,
)

print(
    "Pretraining epochs:",
    PRETRAIN_EPOCHS_R4,
)

print(
    "Physics warm-up epochs:",
    PHYSICS_WARMUP_EPOCHS_R4,
)

print(
    "Joint Adam epochs:",
    ADAM_EPOCHS_R4,
)

print(
    "L-BFGS maximum iterations:",
    LBFGS_MAX_ITER_R4,
)

print(
    "Collocation points per interval:",
    COLLOCATION_POINTS_PER_INTERVAL_R4,
)

print(
    "Validation points per interval:",
    VALIDATION_POINTS_PER_INTERVAL_R4,
)

print(
    "L-BFGS points per interval:",
    LBFGS_POINTS_PER_INTERVAL_R4,
)

print(
    "Early-stopping patience:",
    EARLY_STOPPING_PATIENCE_R4,
)

print(
    "Logging interval:",
    LOG_EVERY_R4,
)

print(
    "Collocation-resampling interval:",
    RESAMPLE_EVERY_R4,
)

print(
    "Network width:",
    NETWORK_WIDTH_R4,
)

print(
    "Network depth:",
    NETWORK_DEPTH_R4,
)

print(
    "State learning rate:",
    STATE_LEARNING_RATE_R4,
)

print(
    "Parameter learning rate:",
    PARAMETER_LEARNING_RATE_R4,
)

print(
    "Gradient clipping norm:",
    GRADIENT_CLIP_NORM_R4,
)

print(
    "Active suites:",
    active_experiments_r4,
)

print(
    "Resume-safe primary table:",
    PRIMARY_RESULTS_PATH_R4.resolve(),
)

print("=" * 78)


# =============================================================================
# FINAL SAFETY ASSERTIONS
# =============================================================================

assert QUICK_MODE is False

assert RUN_STRUCTURAL_SMOKE_TEST is False

assert RUN_SINGLE_DESIGN_DEMO is False

assert RUN_PRIMARY_MULTI_SCHEDULE_STUDY is True

assert RUN_MATCHED_ODE_BASELINES is False

assert RUN_IDENTIFIABILITY_GEOMETRY is False

assert RUN_LEAVE_ONE_SCHEDULE_OUT is False

assert RUN_PROFILE_SURFACES is False

assert RUN_PARAMETER_TRANSFER_TEST is False

assert RUN_PLOTTING_ONLY is False

assert active_experiments_r4 == [
    "primary_multi_schedule_study"
]

assert PRIMARY_SEED_IDS_R4 == [
    1,
    2,
    3,
    4,
    5,
]

assert RUN_LABEL == "final"

assert ROUND4_ROOT.name == "final"


print(
    "\nCell 1 completed successfully."
)

print(
    "Final manuscript-scale M1–M5 primary configuration is active."
)

print(
    "Restarted kernel confirmed: run Cells 2–14 sequentially, "
    "skip Cells 15–16, and then run Cell 17."
)

# In[2]:



# =============================================================================
# ROUND 4 — CELL 2
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
from IPython.display import display
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


def save_figure_r4(fig: plt.Figure, stem: str, close: bool = False) -> None:
    """Save a Round 4 figure in high-resolution raster and vector formats."""
    clean = stem.strip().replace(' ', '_')
    fig.savefig(FIGURE_DIR_R4 / f'{clean}.png', dpi=400,
                bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURE_DIR_R4 / f'{clean}.pdf',
                bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURE_DIR_R4 / f'{clean}.svg',
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


# =============================================================================
# ROUND 4 — CELL 3
# Six-repository provenance, scientific boundaries, and Round 4 hypotheses
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# The implementation deliberately reuses verified ideas without claiming those
# ideas as new. The proposed contribution is their AMR-specific synthesis:
# exact pointwise biological feasibility plus shared-parameter inference across
# heterogeneous prescribed treatment environments.

REPOSITORY_PROVENANCE_R4 = {
    "DrugRes_StructPop": {
        "used": [
            "sensitive/resistant birth, death, mutation, and crowding structure",
            "piecewise treatment changes applied exactly at switch times",
            "evolutionary-rescue interpretation and later stochastic validation",
        ],
        "not_claimed_as_novel": "The AMR population process and Gillespie method.",
        "round4_role": (
            "Supplies the deterministic mean-field biology and motivates "
            "schedule-specific treatment interfaces."
        ),
    },
    "BINNs": {
        "used": [
            "inverse biological learning from sparse observations",
            "separation of observation loss and mechanistic residual loss",
            "explicit normalization and held-out validation",
        ],
        "not_claimed_as_novel": "Sparse biological inverse learning.",
        "round4_role": (
            "Motivates joint inference from total burden and sparse phenotype fractions."
        ),
    },
    "PINN-Proj": {
        "used": [
            "hard constraints implemented through differentiable output mappings",
            "separate predictive and feasibility audits",
            "first-order optimization followed by optional L-BFGS refinement",
        ],
        "not_claimed_as_novel": "Projection-based exact conservation in PINNs.",
        "round4_role": (
            "Motivates exact feasibility and controlled comparisons, but no "
            "integral projection is copied into the AMR state mapping."
        ),
    },
    "Control-PINNs": {
        "used": [
            "treatment represented explicitly as an input to the dynamics",
            "one model containing multiple logically coupled components",
        ],
        "excluded": [
            "adjoint equations",
            "stationarity residuals",
            "learned or optimized treatment controls",
        ],
        "round4_role": (
            "Clarifies that Round 4 is multi-environment inverse inference, "
            "not antibiotic optimal control."
        ),
    },
    "evo_dm": {
        "used": [
            "schedule/environment diversity as evolutionary excitation",
            "fitness-cost interpretation across drug environments",
        ],
        "excluded": [
            "DQN training",
            "MDP policy optimization",
            "genotype-level policy claims",
        ],
        "round4_role": (
            "Motivates heterogeneous prescribed schedules as complementary "
            "experiments without importing the RL architecture."
        ),
    },
    "KKT_hPINN": {
        "used": [
            "training through an exact differentiable feasibility layer",
            "hard versus soft constraint reasoning",
            "independent reporting of feasibility residuals",
        ],
        "not_claimed_as_novel": "KKT projection for affine equality constraints.",
        "round4_role": (
            "Motivates exact enforcement, while the AMR inequalities and "
            "simplex/capacity structure require a different nonlinear mapping."
        ),
    },
}

ROUND4_HYPOTHESES = {
    "H1": (
        "Joint inference across heterogeneous schedules increases the weakest "
        "parameter-sensitivity direction relative to early treatment alone."
    ),
    "H2": (
        "Shared-parameter multi-schedule fitting reduces relative error in mu "
        "and c_R across paired seeds."
    ),
    "H3": (
        "The hard output mapping preserves S>=0, R>=0, and S+R<=K for every "
        "schedule, design, seed, and dense evaluation point."
    ),
    "H4": (
        "Leave-one-schedule-out and parameter-transfer tests reveal which "
        "treatment environments contribute complementary information."
    ),
}

NOVELTY_BOUNDARY_R4 = {
    "proposed_round4_contribution": [
        "shared eco-evolutionary parameter inference across prescribed treatment schedules",
        "schedule-specific interface networks with one shared bounded parameter module",
        "paired-seed comparison of nested schedule designs",
        "sensitivity-SVD, profile geometry, leave-one-out, and transfer analyses",
        "exact pointwise positivity, phenotype simplex, capacity, initial state, and interface continuity",
    ],
    "not_claimed": [
        "first PINN",
        "first AMR model",
        "first hard-constrained neural network",
        "first multi-task neural network",
        "first antibiotic RL or optimal-control method",
    ],
}

with open(
    REPORT_DIR_R4 / "round4_repository_provenance.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "repositories": REPOSITORY_PROVENANCE_R4,
            "hypotheses": ROUND4_HYPOTHESES,
            "novelty_boundary": NOVELTY_BOUNDARY_R4,
        },
        file,
        indent=2,
    )

provenance_table_r4 = pd.DataFrame([
    {
        "repository": name,
        "round4_role": details.get("round4_role", ""),
        "excluded_or_not_novel": "; ".join(details.get(
            "excluded",
            [details.get("not_claimed_as_novel", "")]
        )),
    }
    for name, details in REPOSITORY_PROVENANCE_R4.items()
])

display(provenance_table_r4)

# In[4]:


# ROUND 4 — CELL 4
# Eco-evolutionary AMR parameters, treatment schedules, and deterministic solver
# =============================================================================
# The state is normalized by carrying capacity during PINN training:
#     s = S/K, r = R/K.
# This keeps both states and ODE residuals near order one and improves PINN
# optimization. Dimensional bacterial counts are restored for all reports.

@dataclass(frozen=True)
class AMRParametersR4:
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
class ScheduleR4:
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


TRUE_PARAMETERS = AMRParametersR4()
TRUE_PARAMETERS.validate()
INITIAL_S = 100.0
INITIAL_R = 0.0
T_START = 0.0
T_FINAL = 20.0
DENSE_POINTS = 1001

SCHEDULES_R4 = {
    'untreated': ScheduleR4('untreated', (), (0.0,), 'No antibiotic exposure.'),
    'early_continuous': ScheduleR4('early_continuous', (5.0,), (0.0, 1.0),
                                   'Continuous exposure beginning at t=5.'),
    'late_continuous': ScheduleR4('late_continuous', (10.0,), (0.0, 1.0),
                                  'Continuous exposure beginning at t=10.'),
    'pulsed': ScheduleR4('pulsed', (5.0, 8.0, 12.0, 15.0),
                         (0.0, 1.0, 0.0, 1.0, 0.0),
                         'Exposure on [5,8) and [12,15).'),
}


def normalized_amr_rhs_numpy(t: float, state: np.ndarray,
                             parameters: AMRParametersR4,
                             schedule: ScheduleR4) -> np.ndarray:
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


def solve_reference_r4(parameters: AMRParametersR4,
                       schedule: ScheduleR4,
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
                    for s in SCHEDULES_R4.values()]))

# In[5]:


# ROUND 4 — CELL 5
# Load Round 1 data, verify it, and construct O1–O4 observation regimes
# =============================================================================
# Data loading strategy
# ---------------------
# The cell first searches for the dense CSV files produced in Round 1. When a
# file is unavailable, it regenerates the same high-accuracy deterministic
# trajectory. Therefore, Round 4 remains reproducible even after a kernel restart.
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
    frame = solve_reference_r4(TRUE_PARAMETERS, SCHEDULES_R4[schedule_name])
    fallback_dir = CACHE_DIR_R4 / 'dense_reference'
    fallback_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(fallback_dir / f'{schedule_name}_dense.csv', index=False)
    return frame


DENSE_REFERENCE = {name: load_or_generate_dense(name) for name in SCHEDULES_R4}


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
summary.to_csv(TABLE_DIR_R4 / 'round4_observation_regime_definitions.csv', index=False)
summary


# In[61]:


# =============================================================================
# ROUND 4 — CELL 6
# Nested multi-schedule designs, schedule-aware observations,
# and publication-ready experimental-design figure
# =============================================================================
#
# SCIENTIFIC PURPOSE
# ------------------
# Round 4 evaluates whether treatment-environment diversity improves
# practical identifiability of the shared eco-evolutionary parameters.
#
# M1 : early-continuous treatment only, sparse O3 phenotype observations
# M2 : early + late continuous treatment, O3
# M3 : early + late + pulsed treatment, O3
# M4 : untreated + early + late + pulsed treatment, O3
# M5 : same four environments as M4, but phenotype-rich O4 observations
#
# M1--M4 isolate the effect of schedule diversity while keeping phenotype
# observation richness fixed.
#
# M5 is an upper-information benchmark and must NOT be interpreted as a
# direct sparse-data replacement for M1--M4.
#
# O3 resistant-fraction observation times are schedule aware and are chosen
# away from treatment-coefficient discontinuities.
#
# This cell:
#   1. defines the Round-4 nested designs;
#   2. constructs schedule-aware O3/O4 observation datasets;
#   3. constructs design bundles;
#   4. saves the design-definition table; and
#   5. generates the final publication-ready treatment-design figure.
#
# NO MODEL TRAINING IS PERFORMED IN THIS CELL.
# =============================================================================


# =============================================================================
# 1. ROUND-4 DESIGN CONTAINER
# =============================================================================

@dataclass(frozen=True)
class Round4Design:
    """
    Definition of one nested Round-4 experimental design.
    """

    design_id: str
    schedule_names: Tuple[str, ...]
    observation_regime: str
    description: str

    @property
    def schedule_count(self) -> int:
        return len(self.schedule_names)


# =============================================================================
# 2. CANONICAL SCHEDULE ORDER
# =============================================================================
# Used consistently for:
#   - seed offsets,
#   - design construction,
#   - table ordering,
#   - figure-panel ordering.
# =============================================================================

SCHEDULE_ORDER_R4 = (
    "untreated",
    "early_continuous",
    "late_continuous",
    "pulsed",
)


SCHEDULE_SEED_OFFSET_R4 = {
    schedule_name: index
    for index, schedule_name in enumerate(
        SCHEDULE_ORDER_R4
    )
}


# =============================================================================
# 3. SPARSE O3 RESISTANT-FRACTION OBSERVATION TIMES
# =============================================================================
# Three phenotype observations are used per schedule.
#
# These times intentionally avoid known treatment-switch discontinuities.
# =============================================================================

PHENOTYPE_TIMES_O3_R4 = {

    "untreated":
        np.array(
            [4.0, 10.0, 16.0],
            dtype=float,
        ),

    "early_continuous":
        np.array(
            [4.0, 8.0, 16.0],
            dtype=float,
        ),

    "late_continuous":
        np.array(
            [5.0, 12.0, 17.0],
            dtype=float,
        ),

    "pulsed":
        np.array(
            [4.0, 6.5, 13.5],
            dtype=float,
        ),
}


# Basic structural audit.
for schedule_name_r4, times_r4 in PHENOTYPE_TIMES_O3_R4.items():

    if schedule_name_r4 not in SCHEDULES_R4:
        raise KeyError(
            f"O3 observation schedule '{schedule_name_r4}' "
            "is not defined in SCHEDULES_R4."
        )

    if len(times_r4) != 3:
        raise ValueError(
            f"{schedule_name_r4} must contain exactly "
            "three O3 phenotype-observation times."
        )

    if np.any(times_r4 < T_START) or np.any(times_r4 > T_FINAL):
        raise ValueError(
            f"O3 observation times for {schedule_name_r4} "
            "fall outside the simulation interval."
        )


# =============================================================================
# 4. NESTED MULTI-SCHEDULE DESIGNS
# =============================================================================

ROUND4_DESIGNS = {

    "M1":
        Round4Design(
            design_id="M1",
            schedule_names=(
                "early_continuous",
            ),
            observation_regime="O3",
            description=(
                "Single early-continuous treatment environment "
                "with three resistant-fraction observations."
            ),
        ),

    "M2":
        Round4Design(
            design_id="M2",
            schedule_names=(
                "early_continuous",
                "late_continuous",
            ),
            observation_regime="O3",
            description=(
                "Early- and late-continuous treatment environments "
                "with shared inverse parameters."
            ),
        ),

    "M3":
        Round4Design(
            design_id="M3",
            schedule_names=(
                "early_continuous",
                "late_continuous",
                "pulsed",
            ),
            observation_regime="O3",
            description=(
                "Three treated environments with shared "
                "inverse parameters."
            ),
        ),

    "M4":
        Round4Design(
            design_id="M4",
            schedule_names=(
                "untreated",
                "early_continuous",
                "late_continuous",
                "pulsed",
            ),
            observation_regime="O3",
            description=(
                "Untreated plus three treatment environments "
                "with sparse phenotype observations."
            ),
        ),

    "M5":
        Round4Design(
            design_id="M5",
            schedule_names=(
                "untreated",
                "early_continuous",
                "late_continuous",
                "pulsed",
            ),
            observation_regime="O4",
            description=(
                "Four-environment phenotype-rich "
                "upper-information benchmark."
            ),
        ),
}


DESIGN_ORDER_R4 = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
]


# =============================================================================
# 5. ROUND-4 OBSERVATION-DATASET CONSTRUCTOR
# =============================================================================

def build_round4_dataset(
    schedule_name: str,
    regime: str,
    noise_level: float = 0.0,
    seed: int = SEED,
) -> ObservationDataset:
    """
    Construct a schedule-aware Round-4 observation dataset.

    O3
    --
    21 total-load observations
    + 3 schedule-aware resistant-fraction observations.

    O4
    --
    21 total-load observations
    + resistant-fraction observations at all 21 observation times.
    """

    # ---------------------------------------------------------------------
    # Input checks
    # ---------------------------------------------------------------------

    if schedule_name not in SCHEDULES_R4:
        raise KeyError(
            f"Unknown Round-4 schedule: {schedule_name}"
        )

    if regime not in {
        "O3",
        "O4",
    }:
        raise ValueError(
            "Round-4 primary designs use only O3 or O4 observations."
        )

    if noise_level < 0:
        raise ValueError(
            "noise_level cannot be negative."
        )


    # ---------------------------------------------------------------------
    # Dense reference trajectory
    # ---------------------------------------------------------------------

    dense = DENSE_REFERENCE[
        schedule_name
    ]


    # ---------------------------------------------------------------------
    # Total-load observations
    # ---------------------------------------------------------------------

    total_time = np.linspace(
        T_START,
        T_FINAL,
        21,
    )


    true_total = nearest_values(
        dense,
        total_time,
        "total",
    )


    # ---------------------------------------------------------------------
    # Resistant-fraction observations
    # ---------------------------------------------------------------------

    if regime == "O3":

        fraction_time = (
            PHENOTYPE_TIMES_O3_R4[
                schedule_name
            ]
            .copy()
        )

    else:

        # O4: phenotype-resolved information
        # at every observation time.
        fraction_time = (
            total_time.copy()
        )


    true_fraction = nearest_values(
        dense,
        fraction_time,
        "resistant_fraction",
    )


    # ---------------------------------------------------------------------
    # Observation noise
    # ---------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )


    if noise_level == 0.0:

        total_observed = (
            true_total.copy()
        )

        fraction_observed = (
            true_fraction.copy()
        )

    else:

        # -------------------------------------------------------------
        # Positive multiplicative noise for total load
        # -------------------------------------------------------------

        sigma_log = np.sqrt(
            np.log1p(
                noise_level ** 2
            )
        )


        total_observed = (
            true_total
            *
            rng.lognormal(
                -0.5 * sigma_log ** 2,
                sigma_log,
                len(
                    true_total
                ),
            )
        )


        # -------------------------------------------------------------
        # Logit-space noise for the resistant fraction
        # -------------------------------------------------------------

        eps = 1.0e-8


        clipped = np.clip(
            true_fraction,
            eps,
            1.0 - eps,
        )


        logits = np.log(
            clipped
            /
            (
                1.0
                -
                clipped
            )
        )


        fraction_observed = (
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -(
                        logits
                        +
                        rng.normal(
                            0.0,
                            noise_level,
                            len(
                                logits
                            ),
                        )
                    )
                )
            )
        )


    # ---------------------------------------------------------------------
    # Total-load train / validation split
    # ---------------------------------------------------------------------
    # The same held-out indices are used for every schedule and seed.
    # ---------------------------------------------------------------------

    total_validation_mask = (
        np.arange(
            len(
                total_time
            )
        )
        % 5
        == 4
    )


    # Keep the initial observation in training.
    total_validation_mask[
        0
    ] = False


    total_train_mask = (
        ~total_validation_mask
    )


    # ---------------------------------------------------------------------
    # Resistant-fraction train / validation split
    # ---------------------------------------------------------------------

    if regime == "O3":

        # Three phenotype observations:
        # first two for training,
        # last one held out for validation.
        fraction_validation_mask = np.array(
            [
                False,
                False,
                True,
            ],
            dtype=bool,
        )

    else:

        # O4:
        # hold out every fifth phenotype observation.
        fraction_validation_mask = (
            np.arange(
                len(
                    fraction_time
                )
            )
            % 5
            == 4
        )


        # Preserve the initial phenotype observation
        # in the training set.
        fraction_validation_mask[
            0
        ] = False


    fraction_train_mask = (
        ~fraction_validation_mask
    )


    # ---------------------------------------------------------------------
    # Final dataset
    # ---------------------------------------------------------------------

    return ObservationDataset(

        schedule_name=
            schedule_name,

        regime=
            regime,

        noise_level=
            noise_level,

        total_time=
            total_time,

        total_observed=
            total_observed,

        fraction_time=
            fraction_time,

        fraction_observed=
            fraction_observed,

        total_train_mask=
            total_train_mask,

        total_validation_mask=
            total_validation_mask,

        fraction_train_mask=
            fraction_train_mask,

        fraction_validation_mask=
            fraction_validation_mask,
    )


# =============================================================================
# 6. MULTI-SCHEDULE DESIGN-BUNDLE CONSTRUCTOR
# =============================================================================

def build_design_bundle_r4(
    design_id: str,
    seed: int,
    noise_level: float = 0.0,
) -> Dict[str, ObservationDataset]:
    """
    Construct all schedule-specific observations belonging to one
    Round-4 nested design.
    """

    if design_id not in ROUND4_DESIGNS:
        raise KeyError(
            f"Unknown Round-4 design: {design_id}"
        )


    design = ROUND4_DESIGNS[
        design_id
    ]


    bundle = {}


    for schedule_name in design.schedule_names:

        schedule_seed = (
            seed
            +
            1009
            *
            SCHEDULE_SEED_OFFSET_R4[
                schedule_name
            ]
        )


        bundle[
            schedule_name
        ] = build_round4_dataset(

            schedule_name=
                schedule_name,

            regime=
                design.observation_regime,

            noise_level=
                noise_level,

            seed=
                schedule_seed,
        )


    return bundle


# =============================================================================
# 7. SAVE ROUND-4 DESIGN-DEFINITION TABLE
# =============================================================================

design_definition_rows = []


for design_id in DESIGN_ORDER_R4:

    design = ROUND4_DESIGNS[
        design_id
    ]


    design_definition_rows.append(
        {
            "design":
                design_id,

            "schedule_count":
                design.schedule_count,

            "schedules":
                ", ".join(
                    design.schedule_names
                ),

            "regime":
                design.observation_regime,

            "phenotype_points_per_schedule":
                (
                    3
                    if
                    design.observation_regime
                    == "O3"
                    else
                    21
                ),

            "description":
                design.description,
        }
    )


ROUND4_DESIGN_TABLE = pd.DataFrame(
    design_definition_rows
)


ROUND4_DESIGN_TABLE.to_csv(
    TABLE_DIR_R4
    /
    "round4_design_definitions.csv",
    index=False,
)


display(
    ROUND4_DESIGN_TABLE
)


# =============================================================================
# =============================================================================
# 8. FINAL PUBLICATION FIGURE
# =============================================================================
#
# Publication design:
#
#   - Four vertically aligned treatment environments.
#   - Shared time axis for direct schedule comparison.
#   - Treatment exposure shown by a step function.
#   - O3 resistant-fraction observation times shown in a dedicated
#     upper observation band INSIDE each panel.
#   - Short panel titles.
#   - Compact, borderless legend separated from panel (a).
#   - No internal figure title; the manuscript caption provides context.
#
# NO SCIENTIFIC VALUES ARE CHANGED.
# =============================================================================


# -----------------------------------------------------------------------------
# Canonical plotting order
# -----------------------------------------------------------------------------

schedule_order_plot_r4 = [
    "untreated",
    "early_continuous",
    "late_continuous",
    "pulsed",
]


# -----------------------------------------------------------------------------
# Concise publication panel titles
# -----------------------------------------------------------------------------

panel_titles_r4 = {
    "untreated":
        "(a) Untreated",

    "early_continuous":
        "(b) Early continuous",

    "late_continuous":
        "(c) Late continuous",

    "pulsed":
        "(d) Pulsed",
}


# -----------------------------------------------------------------------------
# Treatment-switch annotations
# -----------------------------------------------------------------------------
# Kept separate from the panel titles to avoid overly long headings.
# -----------------------------------------------------------------------------

switch_annotations_r4 = {

    "untreated":
        None,

    "early_continuous":
        r"$t_{\mathrm{on}}=5$",

    "late_continuous":
        r"$t_{\mathrm{on}}=10$",

    "pulsed":
        r"ON: $5\!-\!8,\;12\!-\!15$",
}


# -----------------------------------------------------------------------------
# Dense time axis used only for plotting
# -----------------------------------------------------------------------------

time_plot_r4 = np.linspace(
    T_START,
    T_FINAL,
    1201,
)


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
# Increased height is intentional:
# each panel now has adequate breathing room.
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(9.0, 8.0),
    sharex=True,
)


# =============================================================================
# 9. PLOT EACH TREATMENT ENVIRONMENT
# =============================================================================

for axis, schedule_name in zip(
    axes,
    schedule_order_plot_r4,
):

    schedule = SCHEDULES_R4[
        schedule_name
    ]


    exposure = schedule.exposure_numpy(
        time_plot_r4
    )


    phenotype_times = (
        PHENOTYPE_TIMES_O3_R4[
            schedule_name
        ]
    )


    # -------------------------------------------------------------------------
    # Prescribed antibiotic exposure
    # -------------------------------------------------------------------------

    exposure_line = axis.step(
        time_plot_r4,
        exposure,
        where="post",
        linewidth=2.2,
        zorder=3,
        label=r"Exposure $A(t)$",
    )[0]


    exposure_color_r4 = (
        exposure_line.get_color()
    )


    # -------------------------------------------------------------------------
    # Very light treatment-ON shading
    # -------------------------------------------------------------------------
    # Provides quick visual separation between treated and untreated intervals
    # without obscuring the step function.
    # -------------------------------------------------------------------------

    axis.fill_between(
        time_plot_r4,
        0.0,
        exposure,
        step="post",
        color=exposure_color_r4,
        alpha=0.055,
        linewidth=0.0,
        zorder=0,
    )


    # -------------------------------------------------------------------------
    # O3 observation guides
    # -------------------------------------------------------------------------

    for sample_time in phenotype_times:

        axis.axvline(
            sample_time,
            ymin=0.0,
            ymax=0.91,
            linestyle="--",
            linewidth=0.95,
            color=exposure_color_r4,
            alpha=0.50,
            zorder=1,
        )


    # -------------------------------------------------------------------------
    # Dedicated phenotype-observation band
    # -------------------------------------------------------------------------
    #
    # y = 1.14 is deliberately above the exposure maximum A(t)=1,
    # but remains INSIDE the plotting area.
    #
    # This prevents the diamonds from colliding with panel titles.
    # The y-axis still displays only the physically meaningful ticks 0 and 1.
    # -------------------------------------------------------------------------

    observation_y_r4 = 1.14


    axis.scatter(
        phenotype_times,
        np.full(
            len(
                phenotype_times
            ),
            observation_y_r4,
        ),
        marker="D",
        s=52,
        facecolor=exposure_color_r4,
        edgecolor="black",
        linewidth=0.65,
        zorder=5,
        label=r"O3 $q_R$ observation",
    )


    # -------------------------------------------------------------------------
    # Panel limits
    # -------------------------------------------------------------------------
    # Additional headroom separates the observation markers from A(t)=1.
    # -------------------------------------------------------------------------

    axis.set_xlim(
        T_START,
        T_FINAL,
    )


    axis.set_ylim(
        -0.055,
        1.25,
    )


    axis.set_yticks(
        [0, 1]
    )


    # -------------------------------------------------------------------------
    # Panel title
    # -------------------------------------------------------------------------

    axis.set_title(
        panel_titles_r4[
            schedule_name
        ],
        loc="left",
        fontsize=12.0,
        fontweight="semibold",
        pad=10,
    )


    # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
    # Small treatment-timing annotation
    # -------------------------------------------------------------------------

    timing_text_r4 = (
        switch_annotations_r4[
            schedule_name
        ]
    )

    if timing_text_r4 is not None:

        axis.text(
            0.995,
            1.075,
            timing_text_r4,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.3,
            clip_on=False,
        )


    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    axis.grid(
        axis="x",
        linewidth=0.55,
        alpha=0.13,
    )


    axis.set_axisbelow(
        True
    )


    # -------------------------------------------------------------------------
    # Tick appearance
    # -------------------------------------------------------------------------

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=9.5,
        width=0.8,
        length=4.0,
        direction="out",
    )


    # -------------------------------------------------------------------------
    # Journal-style frame
    # -------------------------------------------------------------------------

    for spine in axis.spines.values():

        spine.set_linewidth(
            0.85
        )


# =============================================================================
# 10. SHARED AXIS LABELS
# =============================================================================

axes[-1].set_xlabel(
    "Time",
    fontsize=11.5,
    labelpad=7,
)


fig.supylabel(
    r"Antibiotic exposure, $A(t)$",
    fontsize=11.5,
    x=0.035,
)


# =============================================================================
# 11. CLEAN SHARED LEGEND
# =============================================================================
#
# Short labels are used here.
# The manuscript caption defines q_R explicitly as the resistant fraction.
# =============================================================================

handles_r4, labels_r4 = (
    axes[0]
    .get_legend_handles_labels()
)


legend_r4 = fig.legend(
    handles_r4,
    labels_r4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=2,
    frameon=False,
    fontsize=10.5,
    handlelength=2.7,
    handletextpad=0.65,
    columnspacing=2.2,
)


# =============================================================================
# 12. FINAL PUBLICATION LAYOUT
# =============================================================================
#
# Large vertical separation is intentional.
#
# top=0.89:
#   reserves a dedicated band for the legend.
#
# hspace=0.68:
#   prevents panel titles from appearing attached to the preceding subplot.
#
# =============================================================================

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    bottom=0.085,
    top=0.885,
    hspace=0.68,
)


# =============================================================================
# 13. SAVE FINAL FIGURE
# =============================================================================
#
# Same filename as before:
# downstream LaTeX does not need to change.
# =============================================================================

save_figure_r4(
    fig,
    "round4_schedule_and_observation_design",
)


plt.show()


print(
    "\n"
    + "=" * 84
)


print(
    "ROUND 4 CELL 6 PUBLICATION FIGURE UPDATED SUCCESSFULLY"
)


print(
    "=" * 84
)


print(
    "\nFigure:"
)


print(
    "  round4_schedule_and_observation_design"
)


print(
    "\nNO MODEL TRAINING WAS PERFORMED."
)

# In[7]:


# =============================================================================
# ROUND 4 — CELL 7
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


def sample_interface_aware_collocation(schedule: ScheduleR4,
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
check_points = sample_interface_aware_collocation(SCHEDULES_R4['pulsed'], 20, SEED).detach().cpu().numpy().ravel()
for switch in SCHEDULES_R4['pulsed'].switch_times:
    assert np.min(np.abs(check_points-switch)) > 1e-7
print('Interface-aware collocation test passed with', len(check_points), 'points.')

# In[8]:


# =============================================================================
# ROUND 4 — CELL 8
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


# =============================================================================
# ROUND 4 — CELL 9
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
    def __init__(self, schedule: ScheduleR4, model_kind: str,
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
    def __init__(self, schedules: Dict[str, ScheduleR4], model_kind: str,
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
    trial = AMRInverseModel({'early_continuous': SCHEDULES_R4['early_continuous']},
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


# =============================================================================
# ROUND 4 — CELL 10
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
    schedule = SCHEDULES_R4[schedule_name]
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

# In[12]:


# =============================================================================
# ROUND 4 — CELL 11
# Four-stage shared-parameter optimizer with 8 GB GPU memory management
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# The optimization is staged to reduce compensation between flexible state
# networks and the two biological parameters.
#
# Stage I   : observation pretraining with mu and c_R frozen.
# Stage II  : physics warm-up with mu and c_R still frozen.
# Stage III : joint AdamW optimization with separate state/parameter learning rates.
# Stage IV  : optional full-batch L-BFGS refinement on reduced fixed collocation.
#
# The model is trained one design/seed at a time. The best checkpoint is copied
# to CPU, histories are saved immediately, and CUDA cache is explicitly cleared.

@dataclass
class TrainingResultR4:
    label: str
    design_id: str
    model: AMRInverseModel
    history: pd.DataFrame
    elapsed_seconds: float
    best_validation_score: float
    seed: int


def set_seed_r4(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict_r4(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Copy a model checkpoint to CPU so it does not consume GPU memory."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def state_parameters_r4(model: AMRInverseModel) -> List[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("inverse_parameters.")
    ]


def parameter_parameters_r4(model: AMRInverseModel) -> List[nn.Parameter]:
    return list(model.inverse_parameters.parameters())


def set_inverse_trainable_r4(
    model: AMRInverseModel,
    trainable: bool,
) -> None:
    for parameter in model.inverse_parameters.parameters():
        parameter.requires_grad_(trainable)


def make_collocation_bundle_r4(
    datasets: Dict[str, ObservationDataset],
    seed: int,
    points_per_interval: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    count = (
        COLLOCATION_POINTS_PER_INTERVAL_R4
        if points_per_interval is None
        else int(points_per_interval)
    )
    return {
        schedule_name: sample_interface_aware_collocation(
            SCHEDULES_R4[schedule_name],
            count,
            seed + 1009 * schedule_index,
        )
        for schedule_index, schedule_name in enumerate(datasets)
    }


def validation_score_r4(
    model: AMRInverseModel,
    datasets: Dict[str, ObservationDataset],
    seed: int,
    physics_weight: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    collocation = make_collocation_bundle_r4(
        datasets,
        seed,
        VALIDATION_POINTS_PER_INTERVAL_R4,
    )
    total, components = complete_loss(
        model,
        datasets,
        collocation,
        physics_weight=physics_weight,
        split="validation",
    )
    score = (
        components["load"]
        + components["fraction"]
        + 0.10 * physics_weight * components["physics"]
    )
    del total, collocation
    clear_gpu_memory()
    return float(score), components


def append_history_row_r4(
    history_rows: List[Dict[str, Any]],
    model: AMRInverseModel,
    stage: str,
    epoch: int,
    score: float,
    components: Dict[str, float],
    learning_rate_state: float = np.nan,
    learning_rate_parameter: float = np.nan,
) -> None:
    history_rows.append({
        "stage": stage,
        "epoch": int(epoch),
        "validation_score": float(score),
        "learning_rate_state": learning_rate_state,
        "learning_rate_parameter": learning_rate_parameter,
        **components,
        **model.inferred_parameters(),
        "gpu_allocated_gb": (
            float(torch.cuda.memory_allocated() / 1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "gpu_reserved_gb": (
            float(torch.cuda.memory_reserved() / 1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        ),
    })


def train_shared_parameter_model_r4(
    model: AMRInverseModel,
    datasets: Dict[str, ObservationDataset],
    design_id: str,
    label: str,
    seed: int,
) -> TrainingResultR4:
    """Train one nested multi-schedule design using one shared parameter pair."""
    set_seed_r4(seed)
    model = model.to(device=DEVICE, dtype=DTYPE)
    start_time = time.time()
    history_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Stage I: observation pretraining, inverse parameters frozen
    # ------------------------------------------------------------------
    set_inverse_trainable_r4(model, False)
    optimizer = torch.optim.AdamW(
        state_parameters_r4(model),
        lr=2.0e-3,
        weight_decay=0.0,
    )

    stage_best_score = np.inf
    stage_best_state = cpu_state_dict_r4(model)
    empty_collocation: Dict[str, torch.Tensor] = {}

    for epoch in range(PRETRAIN_EPOCHS_R4):
        optimizer.zero_grad(set_to_none=True)
        total, components = complete_loss(
            model,
            datasets,
            empty_collocation,
            physics_weight=0.0,
            split="train",
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            state_parameters_r4(model),
            max_norm=GRADIENT_CLIP_NORM_R4,
        )
        optimizer.step()

        if (
            epoch % LOG_EVERY_R4 == 0
            or epoch == PRETRAIN_EPOCHS_R4 - 1
        ):
            score, _ = validation_score_r4(
                model,
                datasets,
                seed + 10_000 + epoch,
                physics_weight=0.0,
            )
            append_history_row_r4(
                history_rows,
                model,
                "pretrain",
                epoch,
                score,
                components,
                optimizer.param_groups[0]["lr"],
                0.0,
            )
            if score < stage_best_score:
                stage_best_score = score
                stage_best_state = cpu_state_dict_r4(model)

    model.load_state_dict(stage_best_state)

    # ------------------------------------------------------------------
    # Stage II: physics warm-up, inverse parameters remain frozen
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        state_parameters_r4(model),
        lr=1.0e-3,
        weight_decay=0.0,
    )

    stage_best_score = np.inf
    stage_best_state = cpu_state_dict_r4(model)

    collocation = None
    for epoch in range(PHYSICS_WARMUP_EPOCHS_R4):
        if epoch % RESAMPLE_EVERY_R4 == 0 or collocation is None:
            if collocation is not None:
                del collocation
            collocation = make_collocation_bundle_r4(
                datasets,
                seed + 30_000 + epoch,
            )

        optimizer.zero_grad(set_to_none=True)
        total, components = complete_loss(
            model,
            datasets,
            collocation,
            physics_weight=1.0,
            split="train",
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite warm-up loss in {label} at epoch {epoch}."
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            state_parameters_r4(model),
            max_norm=GRADIENT_CLIP_NORM_R4,
        )
        optimizer.step()

        if (
            epoch % LOG_EVERY_R4 == 0
            or epoch == PHYSICS_WARMUP_EPOCHS_R4 - 1
        ):
            score, _ = validation_score_r4(
                model,
                datasets,
                seed + 40_000 + epoch,
                physics_weight=1.0,
            )
            append_history_row_r4(
                history_rows,
                model,
                "physics_warmup",
                epoch,
                score,
                components,
                optimizer.param_groups[0]["lr"],
                0.0,
            )
            if score < stage_best_score:
                stage_best_score = score
                stage_best_state = cpu_state_dict_r4(model)

    if collocation is not None:
        del collocation
    model.load_state_dict(stage_best_state)
    clear_gpu_memory()

    # ------------------------------------------------------------------
    # Stage III: joint AdamW with separate parameter learning rate
    # ------------------------------------------------------------------
    set_inverse_trainable_r4(model, True)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": state_parameters_r4(model),
                "lr": STATE_LEARNING_RATE_R4,
            },
            {
                "params": parameter_parameters_r4(model),
                "lr": PARAMETER_LEARNING_RATE_R4,
            },
        ],
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(50, ADAM_EPOCHS_R4 // 20),
        min_lr=1.0e-6,
    )

    best_score = np.inf
    best_state = cpu_state_dict_r4(model)
    stale_epochs = 0
    collocation = None

    for epoch in range(ADAM_EPOCHS_R4):
        if epoch % RESAMPLE_EVERY_R4 == 0 or collocation is None:
            if collocation is not None:
                del collocation
            collocation = make_collocation_bundle_r4(
                datasets,
                seed + 50_000 + epoch,
            )

        optimizer.zero_grad(set_to_none=True)
        total, components = complete_loss(
            model,
            datasets,
            collocation,
            physics_weight=1.0,
            split="train",
        )

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite joint loss in {label} at epoch {epoch}."
            )

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM_R4,
        )
        optimizer.step()

        if epoch % LOG_EVERY_R4 == 0 or epoch == ADAM_EPOCHS_R4 - 1:
            score, _ = validation_score_r4(
                model,
                datasets,
                seed + 60_000 + epoch,
                physics_weight=1.0,
            )
            scheduler.step(score)

            append_history_row_r4(
                history_rows,
                model,
                "joint_adam",
                epoch,
                score,
                components,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
            )

            inferred = model.inferred_parameters()
            print(
                f"{label:34s} | Adam {epoch:5d}/{ADAM_EPOCHS_R4} | "
                f"loss={components['total']:.3e} | val={score:.3e} | "
                f"mu={inferred['mutation_probability']:.3e} | "
                f"c={inferred['resistance_fitness_cost']:.4f}"
            )

            if score < best_score * (1.0 - 1.0e-6):
                best_score = score
                best_state = cpu_state_dict_r4(model)
                stale_epochs = 0
            else:
                stale_epochs += LOG_EVERY_R4

            if stale_epochs >= EARLY_STOPPING_PATIENCE_R4:
                print(
                    f"{label}: joint Adam early stopping at epoch {epoch}."
                )
                break

    if collocation is not None:
        del collocation
    model.load_state_dict(best_state)
    clear_gpu_memory()

    # ------------------------------------------------------------------
    # Stage IV: reduced fixed-collocation L-BFGS refinement
    # ------------------------------------------------------------------
    if LBFGS_MAX_ITER_R4 > 0:
        fixed_collocation = make_collocation_bundle_r4(
            datasets,
            seed + 90_000,
            LBFGS_POINTS_PER_INTERVAL_R4,
        )

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.75,
            max_iter=LBFGS_MAX_ITER_R4,
            max_eval=max(2 * LBFGS_MAX_ITER_R4, 20),
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=40,
            line_search_fn="strong_wolfe",
        )

        closure_counter = {"value": 0}

        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            total, _ = complete_loss(
                model,
                datasets,
                fixed_collocation,
                physics_weight=1.0,
                split="train",
            )
            total.backward()
            closure_counter["value"] += 1
            return total

        optimizer_lbfgs.step(closure)

        total, components = complete_loss(
            model,
            datasets,
            fixed_collocation,
            physics_weight=1.0,
            split="train",
        )

        score, _ = validation_score_r4(
            model,
            datasets,
            seed + 99_000,
            physics_weight=1.0,
        )

        append_history_row_r4(
            history_rows,
            model,
            "lbfgs",
            closure_counter["value"],
            score,
            components,
        )

        if score < best_score:
            best_score = score
            best_state = cpu_state_dict_r4(model)

        del fixed_collocation, total
        clear_gpu_memory()

    model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    history = pd.DataFrame(history_rows)

    history_path = HISTORY_DIR_R4 / f"{label}_history.csv"
    model_path = MODEL_DIR_R4 / f"{label}.pt"

    history.to_csv(history_path, index=False)

    checkpoint = {
        "model_state": cpu_state_dict_r4(model),
        "model_kind": model.model_kind,
        "design_id": design_id,
        "schedules": list(datasets),
        "inferred_parameters": model.inferred_parameters(),
        "seed": seed,
        "training_budget": {
            "quick_mode": QUICK_MODE,
            "pretrain_epochs": PRETRAIN_EPOCHS_R4,
            "physics_warmup_epochs": PHYSICS_WARMUP_EPOCHS_R4,
            "adam_epochs": ADAM_EPOCHS_R4,
            "lbfgs_max_iter": LBFGS_MAX_ITER_R4,
            "collocation_points_per_interval": (
                COLLOCATION_POINTS_PER_INTERVAL_R4
            ),
        },
    }

    torch.save(checkpoint, model_path)

    print(
        f"{label} completed in {elapsed / 60.0:.2f} min; "
        f"best validation score={best_score:.3e}."
    )

    return TrainingResultR4(
        label=label,
        design_id=design_id,
        model=model,
        history=history,
        elapsed_seconds=elapsed,
        best_validation_score=float(best_score),
        seed=int(seed),
    )


# In[13]:


# =============================================================================
# ROUND 4 — CELL 12
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
    schedule = SCHEDULES_R4[schedule_name]
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


def evaluate_trained_model(result: TrainingResultR4,
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


# -----------------------------------------------------------------------------
# Round 4 design-level aggregation
# -----------------------------------------------------------------------------

def evaluate_design_r4(
    result: TrainingResultR4,
    datasets: Dict[str, ObservationDataset],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate every schedule and return both schedule-level and design-level rows.
    Design-level RMSE values are calculated after concatenating dense trajectories,
    so every evaluated time point contributes directly.
    """
    schedule_rows: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []

    for schedule_name in datasets:
        metrics, frame = evaluate_trained_model(
            result,
            schedule_name,
        )
        metrics.update({
            "design_id": result.design_id,
            "seed": result.seed,
            "row_type": "schedule",
        })
        schedule_rows.append(metrics)

        frame = frame.copy()
        frame["schedule"] = schedule_name
        frame["design_id"] = result.design_id
        frame["seed"] = result.seed
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    design_metrics: Dict[str, Any] = {
        "label": result.label,
        "model_kind": result.model.model_kind,
        "design_id": result.design_id,
        "seed": result.seed,
        "schedule": "ALL",
        "row_type": "design",
        "schedule_count": len(datasets),
        "observation_regime": ROUND4_DESIGNS[
            result.design_id
        ].observation_regime,
        "training_seconds": result.elapsed_seconds,
        "training_minutes": result.elapsed_seconds / 60.0,
        "best_validation_score": result.best_validation_score,
        **result.model.inferred_parameters(),
    }

    pairs = {
        "sensitive": (
            "sensitive",
            "predicted_sensitive",
        ),
        "resistant": (
            "resistant",
            "predicted_resistant",
        ),
        "total": (
            "total",
            "predicted_total",
        ),
        "fraction": (
            "resistant_fraction",
            "predicted_resistant_fraction",
        ),
    }

    for name, (true_column, predicted_column) in pairs.items():
        true_values = combined[true_column].to_numpy(float)
        predicted_values = combined[predicted_column].to_numpy(float)

        design_metrics[f"rmse_{name}"] = float(
            np.sqrt(
                mean_squared_error(
                    true_values,
                    predicted_values,
                )
            )
        )
        design_metrics[f"mae_{name}"] = float(
            mean_absolute_error(
                true_values,
                predicted_values,
            )
        )
        design_metrics[f"relative_l2_{name}"] = relative_l2(
            predicted_values,
            true_values,
        )

    sensitive_prediction = combined[
        "predicted_sensitive"
    ].to_numpy(float)
    resistant_prediction = combined[
        "predicted_resistant"
    ].to_numpy(float)

    design_metrics.update({
        "maximum_negative_violation": float(
            max(
                0.0,
                -sensitive_prediction.min(),
                -resistant_prediction.min(),
            )
        ),
        "maximum_capacity_violation": float(
            max(
                0.0,
                (
                    sensitive_prediction
                    + resistant_prediction
                ).max()
                - TRUE_PARAMETERS.carrying_capacity,
            )
        ),
        "fraction_points_negative": float(
            np.mean(
                (sensitive_prediction < 0.0)
                | (resistant_prediction < 0.0)
            )
        ),
        "fraction_points_over_capacity": float(
            np.mean(
                sensitive_prediction
                + resistant_prediction
                > TRUE_PARAMETERS.carrying_capacity
                + 1.0e-8
            )
        ),
        "physics_residual_rmse": float(
            np.sqrt(
                np.mean([
                    row["physics_residual_rmse"] ** 2
                    for row in schedule_rows
                ])
            )
        ),
        "relative_error_mu": abs(
            design_metrics["mutation_probability"]
            - TRUE_PARAMETERS.mutation_probability
        )
        / TRUE_PARAMETERS.mutation_probability,
        "relative_error_cost": abs(
            design_metrics["resistance_fitness_cost"]
            - TRUE_PARAMETERS.resistance_fitness_cost
        )
        / TRUE_PARAMETERS.resistance_fitness_cost,
    })

    all_rows = pd.DataFrame(
        schedule_rows + [design_metrics]
    )

    all_rows.to_csv(
        TABLE_DIR_R4
        / f"{result.label}_schedule_and_design_metrics.csv",
        index=False,
    )

    combined.to_csv(
        PREDICTION_DIR_R4
        / f"{result.label}_all_schedule_predictions.csv",
        index=False,
    )

    return design_metrics, all_rows


# In[14]:


# =============================================================================
# ROUND 4 — CELL 13
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
    return solve_reference_r4(parameters,SCHEDULES_R4[schedule_name],times)


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
    with open(REPORT_DIR_R4/f'{label}_ode_fit.json','w') as file:
        json.dump(result,file,indent=2)
    return result

print('ODE baseline functions are ready.')


# -----------------------------------------------------------------------------
# Round 4 matched ODE evaluation on a complete design
# -----------------------------------------------------------------------------

def evaluate_ode_design_r4(
    design_id: str,
    datasets: Dict[str, ObservationDataset],
    fit_result: Dict[str, Any],
) -> Dict[str, Any]:
    parameters = replace(
        TRUE_PARAMETERS,
        mutation_probability=fit_result["mutation_probability"],
        resistance_fitness_cost=fit_result["resistance_fitness_cost"],
    )

    frames = []
    for schedule_name in datasets:
        prediction = solve_reference_r4(
            parameters,
            SCHEDULES_R4[schedule_name],
            DENSE_REFERENCE[schedule_name]["time"].to_numpy(float),
        )
        true = DENSE_REFERENCE[schedule_name]
        frame = true.copy()
        frame["predicted_sensitive"] = prediction["sensitive"].to_numpy(float)
        frame["predicted_resistant"] = prediction["resistant"].to_numpy(float)
        frame["predicted_total"] = prediction["total"].to_numpy(float)
        frame["predicted_resistant_fraction"] = prediction[
            "resistant_fraction"
        ].to_numpy(float)
        frame["schedule"] = schedule_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    row = {
        **fit_result,
        "design_id": design_id,
        "schedule_count": len(datasets),
        "observation_regime": ROUND4_DESIGNS[design_id].observation_regime,
    }

    for name, true_column, predicted_column in [
        ("sensitive", "sensitive", "predicted_sensitive"),
        ("resistant", "resistant", "predicted_resistant"),
        ("total", "total", "predicted_total"),
        (
            "fraction",
            "resistant_fraction",
            "predicted_resistant_fraction",
        ),
    ]:
        true_values = combined[true_column].to_numpy(float)
        predicted_values = combined[predicted_column].to_numpy(float)
        row[f"rmse_{name}"] = float(
            np.sqrt(
                mean_squared_error(
                    true_values,
                    predicted_values,
                )
            )
        )
        row[f"relative_l2_{name}"] = relative_l2(
            predicted_values,
            true_values,
        )

    return row

# In[15]:


# =============================================================================
# ROUND 4 — CELL 14
# Design-level sensitivity SVD, Fisher geometry, and confounding diagnostics
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# Neural optimization variability is not the same as information content.
# This cell differentiates the deterministic observation map with respect to
#
#     theta = [log10(mu), c_R]
#
# and analyzes its normalized sensitivity matrix. The smallest singular value
# measures the weakest observable parameter direction; the condition number
# measures confounding. This analysis depends on the sampling design, not on a
# trained neural-network checkpoint.


def observable_vector_from_design_r4(
    datasets: Dict[str, ObservationDataset],
    log10_mu: float,
    cost: float,
) -> np.ndarray:
    parameters = replace(
        TRUE_PARAMETERS,
        mutation_probability=10.0 ** float(log10_mu),
        resistance_fitness_cost=float(cost),
    )

    values: List[float] = []

    for schedule_name, dataset in datasets.items():
        all_times = np.unique(
            np.concatenate([
                dataset.total_time,
                dataset.fraction_time,
            ])
        )

        prediction = solve_reference_r4(
            parameters,
            SCHEDULES_R4[schedule_name],
            all_times,
        )

        predicted_total = nearest_values(
            prediction,
            dataset.total_time,
            "total",
        ) / TRUE_PARAMETERS.carrying_capacity

        # Log total load mirrors the training observation operator.
        values.extend(
            np.log(predicted_total + 1.0e-8)
        )

        if len(dataset.fraction_time):
            values.extend(
                nearest_values(
                    prediction,
                    dataset.fraction_time,
                    "resistant_fraction",
                )
            )

    return np.asarray(values, dtype=float)


def normalized_sensitivity_geometry_r4(
    datasets: Dict[str, ObservationDataset],
) -> Dict[str, Any]:
    theta = np.array([
        math.log10(TRUE_PARAMETERS.mutation_probability),
        TRUE_PARAMETERS.resistance_fitness_cost,
    ])

    steps = np.array([
        1.0e-3,
        1.0e-4,
    ])

    baseline = observable_vector_from_design_r4(
        datasets,
        theta[0],
        theta[1],
    )

    columns = []

    for parameter_index in range(2):
        plus = theta.copy()
        minus = theta.copy()
        plus[parameter_index] += steps[parameter_index]
        minus[parameter_index] -= steps[parameter_index]

        derivative = (
            observable_vector_from_design_r4(
                datasets,
                plus[0],
                plus[1],
            )
            - observable_vector_from_design_r4(
                datasets,
                minus[0],
                minus[1],
            )
        ) / (2.0 * steps[parameter_index])

        # Prevent near-zero fraction observations from receiving unbounded weight.
        output_scale = np.maximum(
            np.abs(baseline),
            1.0e-3,
        )
        columns.append(
            derivative / output_scale
        )

    sensitivity = np.column_stack(columns)
    _, singular_values, right_vectors = np.linalg.svd(
        sensitivity,
        full_matrices=False,
    )

    fisher = sensitivity.T @ sensitivity
    covariance_proxy = np.linalg.pinv(fisher)

    denominator = math.sqrt(
        max(covariance_proxy[0, 0], 0.0)
        * max(covariance_proxy[1, 1], 0.0)
    )
    correlation = (
        covariance_proxy[0, 1] / denominator
        if denominator > 0
        else np.nan
    )

    return {
        "matrix": sensitivity,
        "singular_values": singular_values,
        "right_vectors": right_vectors,
        "fisher": fisher,
        "covariance_proxy": covariance_proxy,
        "condition_number": float(
            singular_values[0]
            / max(singular_values[-1], 1.0e-15)
        ),
        "parameter_correlation": float(correlation),
    }


SENSITIVITY_RESULTS_R4: Dict[str, Dict[str, Any]] = {}
sensitivity_rows_r4 = []

for design_id in DESIGN_ORDER_R4:
    bundle = build_design_bundle_r4(
        design_id,
        seed=SEED,
        noise_level=0.0,
    )
    geometry = normalized_sensitivity_geometry_r4(bundle)
    SENSITIVITY_RESULTS_R4[design_id] = geometry

    sensitivity_rows_r4.append({
        "design_id": design_id,
        "schedule_count": ROUND4_DESIGNS[design_id].schedule_count,
        "observation_regime": ROUND4_DESIGNS[
            design_id
        ].observation_regime,
        "largest_singular_value": float(
            geometry["singular_values"][0]
        ),
        "smallest_singular_value": float(
            geometry["singular_values"][-1]
        ),
        "condition_number": geometry["condition_number"],
        "parameter_correlation": geometry[
            "parameter_correlation"
        ],
        "weak_direction_log10_mu": float(
            geometry["right_vectors"][-1, 0]
        ),
        "weak_direction_cost": float(
            geometry["right_vectors"][-1, 1]
        ),
    })

SENSITIVITY_SUMMARY_R4 = pd.DataFrame(
    sensitivity_rows_r4
)

SENSITIVITY_SUMMARY_R4.to_csv(
    TABLE_DIR_R4
    / "round4_design_identifiability_geometry.csv",
    index=False,
)

display(SENSITIVITY_SUMMARY_R4)

# In[42]:


# =============================================================================
# ROUND 4 — CELL 15
# Structural smoke test: hard constraints, shared parameters, loss, and training
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# This is a full pipeline check using a tiny M2 model. It verifies:
# - one shared mu/c_R pair across two schedule-specific state networks;
# - exact initial states and treatment-interface continuity;
# - positivity and carrying-capacity compliance;
# - finite observation and physics losses;
# - successful staged optimization and dense evaluation.
#
# It is not a manuscript result.

ROUND4_SMOKE_RESULT = None
ROUND4_SMOKE_METRICS = None

if RUN_STRUCTURAL_SMOKE_TEST:
    smoke_design_id = "M2"
    smoke_bundle = build_design_bundle_r4(
        smoke_design_id,
        seed=SEED + 700,
        noise_level=0.0,
    )

    smoke_schedules = {
        name: SCHEDULES_R4[name]
        for name in smoke_bundle
    }

    smoke_model = AMRInverseModel(
        smoke_schedules,
        model_kind="hard",
        width=16,
        depth=2,
        parameter_initialization=(3.0e-4, 0.15),
    ).to(DEVICE)

    dense_test = tensor(
        np.linspace(
            T_START,
            T_FINAL,
            1001,
        )
    )

    with torch.no_grad():
        for schedule_name in smoke_bundle:
            state = smoke_model(
                dense_test,
                schedule_name,
            ).detach().cpu().numpy()

            assert state.min() >= -1.0e-12
            assert state.sum(axis=1).max() <= 1.0 + 1.0e-12
            assert np.max(
                np.abs(
                    state[0]
                    - np.array([
                        INITIAL_S
                        / TRUE_PARAMETERS.carrying_capacity,
                        INITIAL_R
                        / TRUE_PARAMETERS.carrying_capacity,
                    ])
                )
            ) <= 1.0e-12

    smoke_collocation = make_collocation_bundle_r4(
        smoke_bundle,
        seed=SEED + 701,
        points_per_interval=16,
    )

    smoke_loss, smoke_components = complete_loss(
        smoke_model,
        smoke_bundle,
        smoke_collocation,
        physics_weight=1.0,
        split="train",
    )

    assert torch.isfinite(smoke_loss)

    smoke_loss.backward()

    gradient_norm = torch.sqrt(
        sum(
            torch.sum(parameter.grad ** 2)
            for parameter in smoke_model.parameters()
            if parameter.grad is not None
        )
    )

    assert torch.isfinite(gradient_norm)

    del smoke_model, smoke_collocation, smoke_loss
    clear_gpu_memory()

    ROUND4_SMOKE_RESULT = train_shared_parameter_model_r4(
        AMRInverseModel(
            smoke_schedules,
            model_kind="hard",
            width=16,
            depth=2,
            parameter_initialization=(3.0e-4, 0.15),
        ),
        smoke_bundle,
        design_id=smoke_design_id,
        label="round4_smoke_M2",
        seed=SEED + 702,
    )

    ROUND4_SMOKE_METRICS, smoke_rows = evaluate_design_r4(
        ROUND4_SMOKE_RESULT,
        smoke_bundle,
    )

    print("\nRound 4 structural smoke test passed.")
    display(
        pd.DataFrame([
            ROUND4_SMOKE_METRICS
        ])
    )
else:
    print(
        "Set RUN_STRUCTURAL_SMOKE_TEST=True for the reduced pipeline check."
    )


# In[43]:


# =============================================================================
# ROUND 4 — CELL 16
# Optional single-design demonstration: M4 with four schedules and shared parameters
# =============================================================================
# This cell is useful before committing to the complete 25-model paired-seed
# suite. It trains one M4 model, evaluates every schedule, and saves all dense
# predictions. Keep it disabled during the main primary experiment.

ROUND4_DEMO_RESULT = None
ROUND4_DEMO_METRICS = None

if RUN_SINGLE_DESIGN_DEMO:
    demo_design_id = "M4"
    demo_seed = SEED + 400_000
    demo_bundle = build_design_bundle_r4(
        demo_design_id,
        seed=demo_seed,
        noise_level=0.0,
    )

    demo_model = AMRInverseModel(
        {
            name: SCHEDULES_R4[name]
            for name in demo_bundle
        },
        model_kind="hard",
        width=NETWORK_WIDTH_R4,
        depth=NETWORK_DEPTH_R4,
        parameter_initialization=(3.0e-4, 0.15),
    )

    ROUND4_DEMO_RESULT = train_shared_parameter_model_r4(
        demo_model,
        demo_bundle,
        design_id=demo_design_id,
        label="round4_demo_M4",
        seed=demo_seed,
    )

    ROUND4_DEMO_METRICS, demo_rows = evaluate_design_r4(
        ROUND4_DEMO_RESULT,
        demo_bundle,
    )

    display(demo_rows)
else:
    print(
        "RUN_SINGLE_DESIGN_DEMO=False; the optional M4 demonstration is skipped."
    )


# In[16]:


# =============================================================================
# ROUND 4 — CELL 17
# Primary paired-seed nested-design experiment: M1--M5
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# This is the main Round 4 experiment.
#
# For every seed, the nested designs reuse the same schedule-specific synthetic
# observations and the same training seed. Therefore, differences between M1,
# M2, M3, and M4 are attributable primarily to added treatment environments.
# M5 uses O4 observations and is interpreted as an upper-information benchmark.
#
# RESUME SAFETY
# -------------
# One design/seed row is saved immediately after completion. Rerunning the cell
# skips completed pairs unless FORCE_RERUN_PRIMARY_R4=True.

PRIMARY_RAW_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_all_seeds_metrics.csv"
)

PRIMARY_SCHEDULE_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_schedule_metrics.csv"
)

FORCE_RERUN_PRIMARY_R4 = False

ROUND4_PRIMARY_RESULTS: Dict[str, TrainingResultR4] = {}
ROUND4_PRIMARY_ROWS: List[Dict[str, Any]] = []
ROUND4_PRIMARY_SCHEDULE_ROWS: List[Dict[str, Any]] = []


def save_primary_progress_r4() -> None:
    if ROUND4_PRIMARY_ROWS:
        frame = (
            pd.DataFrame(ROUND4_PRIMARY_ROWS)
            .drop_duplicates(
                subset=["design_id", "seed_id"],
                keep="last",
            )
        )
        order_map = {
            design_id: index
            for index, design_id in enumerate(
                DESIGN_ORDER_R4
            )
        }
        frame["_order"] = frame["design_id"].map(order_map)
        frame = (
            frame
            .sort_values(["_order", "seed_number"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )
        frame.to_csv(
            PRIMARY_RAW_PATH_R4,
            index=False,
        )

    if ROUND4_PRIMARY_SCHEDULE_ROWS:
        schedule_frame = (
            pd.DataFrame(
                ROUND4_PRIMARY_SCHEDULE_ROWS
            )
            .drop_duplicates(
                subset=[
                    "design_id",
                    "seed_id",
                    "schedule",
                    "row_type",
                ],
                keep="last",
            )
        )
        schedule_frame.to_csv(
            PRIMARY_SCHEDULE_PATH_R4,
            index=False,
        )


if RUN_PRIMARY_MULTI_SCHEDULE_STUDY:
    if (
        PRIMARY_RAW_PATH_R4.exists()
        and not FORCE_RERUN_PRIMARY_R4
    ):
        existing = pd.read_csv(
            PRIMARY_RAW_PATH_R4
        )
        required_resume = {
            "design_id",
            "seed_id",
            "seed_number",
        }
        if required_resume.issubset(existing.columns):
            ROUND4_PRIMARY_ROWS = existing.to_dict(
                "records"
            )
            print(
                f"Resuming from {len(existing)} completed design-seed runs."
            )

    if (
        PRIMARY_SCHEDULE_PATH_R4.exists()
        and not FORCE_RERUN_PRIMARY_R4
    ):
        existing_schedule = pd.read_csv(
            PRIMARY_SCHEDULE_PATH_R4
        )
        ROUND4_PRIMARY_SCHEDULE_ROWS = (
            existing_schedule.to_dict("records")
        )

    completed_pairs = {
        (
            str(row["design_id"]),
            str(row["seed_id"]),
        )
        for row in ROUND4_PRIMARY_ROWS
    }

    total_runs = (
        len(DESIGN_ORDER_R4)
        * len(PRIMARY_SEED_IDS_R4)
    )

    print(
        f"Primary Round 4 suite: {len(DESIGN_ORDER_R4)} designs "
        f"x {len(PRIMARY_SEED_IDS_R4)} paired seeds = {total_runs} models."
    )

    for seed_number in PRIMARY_SEED_IDS_R4:
        seed_id = f"seed{seed_number}"

        # The same seeds are reused across every nested design.
        dataset_seed = (
            SEED
            + 10_000 * seed_number
        )
        training_seed = (
            SEED
            + 100_000
            + 10_000 * seed_number
        )

        print("\n" + "=" * 78)
        print(
            f"{seed_id}: dataset_seed={dataset_seed}, "
            f"training_seed={training_seed}"
        )
        print("=" * 78)

        for design_id in DESIGN_ORDER_R4:
            pair = (
                design_id,
                seed_id,
            )

            label = (
                f"round4_{design_id}_{seed_id}"
            )

            if (
                pair in completed_pairs
                and not FORCE_RERUN_PRIMARY_R4
            ):
                print(
                    f"Skipping completed run: {label}"
                )
                continue

            bundle = build_design_bundle_r4(
                design_id,
                seed=dataset_seed,
                noise_level=0.0,
            )

            model = AMRInverseModel(
                {
                    name: SCHEDULES_R4[name]
                    for name in bundle
                },
                model_kind="hard",
                width=NETWORK_WIDTH_R4,
                depth=NETWORK_DEPTH_R4,
                parameter_initialization=(
                    3.0e-4,
                    0.15,
                ),
            )

            try:
                result = train_shared_parameter_model_r4(
                    model,
                    bundle,
                    design_id=design_id,
                    label=label,
                    seed=training_seed,
                )

                design_metrics, all_rows = evaluate_design_r4(
                    result,
                    bundle,
                )

                design_metrics.update({
                    "seed_id": seed_id,
                    "seed_number": int(seed_number),
                    "dataset_seed": int(dataset_seed),
                    "training_seed": int(training_seed),
                    "mu_error_percent": (
                        100.0
                        * design_metrics[
                            "relative_error_mu"
                        ]
                    ),
                    "cost_error_percent": (
                        100.0
                        * design_metrics[
                            "relative_error_cost"
                        ]
                    ),
                })

                all_rows = all_rows.copy()
                all_rows["seed_id"] = seed_id
                all_rows["seed_number"] = int(
                    seed_number
                )
                all_rows["dataset_seed"] = int(
                    dataset_seed
                )
                all_rows["training_seed"] = int(
                    training_seed
                )

                ROUND4_PRIMARY_ROWS.append(
                    design_metrics
                )
                ROUND4_PRIMARY_SCHEDULE_ROWS.extend(
                    all_rows.to_dict("records")
                )
                completed_pairs.add(pair)

                ROUND4_PRIMARY_RESULTS[label] = result
                save_primary_progress_r4()

                completed_count = len(
                    {
                        (
                            str(row["design_id"]),
                            str(row["seed_id"]),
                        )
                        for row in ROUND4_PRIMARY_ROWS
                    }
                )

                print(
                    f"Saved {label}: "
                    f"{completed_count}/{total_runs} completed."
                )

                result.model.to("cpu")
                del model, bundle
                clear_gpu_memory()

            except Exception:
                save_primary_progress_r4()
                clear_gpu_memory()
                print(
                    f"\nRun failed at {label}. "
                    "All previously completed runs remain saved."
                )
                raise

    save_primary_progress_r4()

    ROUND4_PRIMARY_ALL_RUNS = pd.read_csv(
        PRIMARY_RAW_PATH_R4
    )

    print("\nPrimary Round 4 suite completed.")
    display(ROUND4_PRIMARY_ALL_RUNS)

else:
    print(
        "Set RUN_PRIMARY_MULTI_SCHEDULE_STUDY=True to run M1--M5 "
        "across the paired seed set."
    )

# In[17]:


# =============================================================================
# ROUND 4 — CELL 18
# Mean ± sample SD tables, paired effects, and manuscript-ready summaries
# =============================================================================
# This cell is plotting/training independent. It reloads the primary CSV when
# necessary and calculates mean ± sample SD using only observed design groups.

if (
    "ROUND4_PRIMARY_ALL_RUNS" not in globals()
    or not isinstance(
        ROUND4_PRIMARY_ALL_RUNS,
        pd.DataFrame,
    )
    or ROUND4_PRIMARY_ALL_RUNS.empty
):
    if PRIMARY_RAW_PATH_R4.exists():
        ROUND4_PRIMARY_ALL_RUNS = pd.read_csv(
            PRIMARY_RAW_PATH_R4
        )
    else:
        ROUND4_PRIMARY_ALL_RUNS = pd.DataFrame()

PRIMARY_SUMMARY_R4 = pd.DataFrame()
PRIMARY_MANUSCRIPT_TABLE_R4 = pd.DataFrame()
PRIMARY_PAIRED_EFFECTS_R4 = pd.DataFrame()

if not ROUND4_PRIMARY_ALL_RUNS.empty:
    raw = ROUND4_PRIMARY_ALL_RUNS.copy()

    numeric_metrics = [
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
        "maximum_negative_violation",
        "maximum_capacity_violation",
        "training_minutes",
        "best_validation_score",
    ]

    grouped = raw.groupby(
        "design_id",
        observed=True,
        sort=False,
    )

    means = grouped[numeric_metrics].mean()
    standard_deviations = grouped[
        numeric_metrics
    ].std(ddof=1)
    counts = grouped.size().rename("n_seeds")

    PRIMARY_SUMMARY_R4 = pd.concat(
        [
            counts,
            means.add_prefix("mean_"),
            standard_deviations.add_prefix("std_"),
        ],
        axis=1,
    ).reset_index()

    order_map = {
        design_id: index
        for index, design_id in enumerate(
            DESIGN_ORDER_R4
        )
    }

    PRIMARY_SUMMARY_R4["_order"] = (
        PRIMARY_SUMMARY_R4["design_id"].map(
            order_map
        )
    )

    PRIMARY_SUMMARY_R4 = (
        PRIMARY_SUMMARY_R4
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    PRIMARY_SUMMARY_R4.to_csv(
        TABLE_DIR_R4
        / "round4_primary_summary_mean_sd.csv",
        index=False,
    )

    def format_mean_sd_r4(
        mean_value: float,
        std_value: float,
        metric: str,
    ) -> str:
        if pd.isna(mean_value):
            return "N/A"
        if pd.isna(std_value):
            return f"{mean_value:.4g} ± N/A"

        if metric == "mutation_probability":
            return (
                f"{mean_value:.3e} ± "
                f"{std_value:.2e}"
            )
        if metric in {
            "mu_error_percent",
            "cost_error_percent",
        }:
            return (
                f"{mean_value:.1f} ± "
                f"{std_value:.1f}"
            )
        if metric in {
            "rmse_fraction",
            "physics_residual_rmse",
        }:
            return (
                f"{mean_value:.3e} ± "
                f"{std_value:.2e}"
            )
        if metric == "training_minutes":
            return (
                f"{mean_value:.2f} ± "
                f"{std_value:.2f}"
            )
        return (
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
        )

    manuscript_rows = []

    for _, summary_row in PRIMARY_SUMMARY_R4.iterrows():
        design_id = summary_row["design_id"]
        design = ROUND4_DESIGNS[design_id]

        output_row = {
            "Design": design_id,
            "Schedules": design.schedule_count,
            "Observation regime": design.observation_regime,
            "Seeds": int(summary_row["n_seeds"]),
        }

        for metric in [
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
        ]:
            output_row[metric] = format_mean_sd_r4(
                summary_row[f"mean_{metric}"],
                summary_row[f"std_{metric}"],
                metric,
            )

        manuscript_rows.append(output_row)

    PRIMARY_MANUSCRIPT_TABLE_R4 = (
        pd.DataFrame(manuscript_rows)
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

    PRIMARY_MANUSCRIPT_TABLE_R4.to_csv(
        TABLE_DIR_R4
        / "round4_primary_manuscript_mean_sd.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Paired percentage effects relative to M1
    # ---------------------------------------------------------------
    paired_rows = []

    for metric in [
        "rmse_resistant",
        "rmse_fraction",
        "mu_error_percent",
        "cost_error_percent",
    ]:
        pivot = raw.pivot(
            index="seed_id",
            columns="design_id",
            values=metric,
        )

        for design_id in DESIGN_ORDER_R4[1:]:
            paired = pivot[["M1", design_id]].dropna()
            improvements = (
                100.0
                * (
                    paired["M1"]
                    - paired[design_id]
                )
                / paired["M1"]
            )

            paired_rows.append({
                "metric": metric,
                "comparison": f"{design_id} vs M1",
                "n_pairs": len(improvements),
                "mean_improvement_percent": float(
                    improvements.mean()
                ),
                "std_improvement_percent": float(
                    improvements.std(ddof=1)
                ),
                "median_improvement_percent": float(
                    improvements.median()
                ),
            })

    PRIMARY_PAIRED_EFFECTS_R4 = pd.DataFrame(
        paired_rows
    )

    PRIMARY_PAIRED_EFFECTS_R4.to_csv(
        TABLE_DIR_R4
        / "round4_primary_paired_effects.csv",
        index=False,
    )

    print("Primary mean ± SD table:")
    display(PRIMARY_MANUSCRIPT_TABLE_R4)

    print("\nPaired effects relative to M1:")
    display(PRIMARY_PAIRED_EFFECTS_R4)

else:
    print(
        "Primary results were not found. Run Cell 17 or place the saved "
        "round4_primary_multischedule_all_seeds_metrics.csv in the table folder."
    )

# In[71]:


# ROUND 4 — CELL 19
# Final publication figures for the completed five-seed M1–M5 experiment
# =============================================================================
# This cell performs NO TRAINING. It uses Cell 17 results and Cell 18 summaries.
# It generates four manuscript-oriented figures with consistent styling.
# =============================================================================

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, LogLocator

# -----------------------------------------------------------------------------
# Reload completed data if the notebook was reopened.
# -----------------------------------------------------------------------------
if "PRIMARY_RAW_PATH_R4" not in globals():
    PRIMARY_RAW_PATH_R4 = TABLE_DIR_R4 / "round4_primary_multischedule_all_seeds_metrics.csv"

if (
    "ROUND4_PRIMARY_ALL_RUNS" not in globals()
    or not isinstance(ROUND4_PRIMARY_ALL_RUNS, pd.DataFrame)
    or ROUND4_PRIMARY_ALL_RUNS.empty
):
    if PRIMARY_RAW_PATH_R4.exists():
        ROUND4_PRIMARY_ALL_RUNS = pd.read_csv(PRIMARY_RAW_PATH_R4)
    else:
        raise FileNotFoundError(
            "Round 4 primary results were not found. Run Cell 17 first."
        )

if "DESIGN_ORDER_R4" not in globals():
    DESIGN_ORDER_R4 = ["M1", "M2", "M3", "M4", "M5"]

# Rebuild Cell 18 numerical summary only if necessary.
if (
    "PRIMARY_SUMMARY_R4" not in globals()
    or not isinstance(PRIMARY_SUMMARY_R4, pd.DataFrame)
    or PRIMARY_SUMMARY_R4.empty
):
    metrics = [
        "rmse_sensitive", "rmse_resistant", "rmse_total", "rmse_fraction",
        "mutation_probability", "mu_error_percent",
        "resistance_fitness_cost", "cost_error_percent",
        "physics_residual_rmse", "training_minutes",
    ]

    grouped = ROUND4_PRIMARY_ALL_RUNS.groupby(
        "design_id",
        observed=True,
        sort=False,
    )

    PRIMARY_SUMMARY_R4 = pd.concat(
        [
            grouped.size().rename("n_seeds"),
            grouped[metrics].mean().add_prefix("mean_"),
            grouped[metrics].std(ddof=1).add_prefix("std_"),
        ],
        axis=1,
    ).reset_index()

    order_map = {
        d: i
        for i, d in enumerate(DESIGN_ORDER_R4)
    }

    PRIMARY_SUMMARY_R4["_order"] = (
        PRIMARY_SUMMARY_R4["design_id"].map(order_map)
    )

    PRIMARY_SUMMARY_R4 = (
        PRIMARY_SUMMARY_R4
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

# -----------------------------------------------------------------------------
# Validate paired-seed counts.
# -----------------------------------------------------------------------------
seed_counts_r4 = (
    ROUND4_PRIMARY_ALL_RUNS
    .groupby("design_id", observed=True)["seed_id"]
    .nunique()
    .reindex(DESIGN_ORDER_R4, fill_value=0)
)

print("\nSeed counts used for Cell 19:")
print(seed_counts_r4.to_string())

if not (seed_counts_r4 == 5).all():
    print(
        "WARNING: at least one design does not contain five unique seeds. "
        "The plots will use the available completed runs."
    )

# -----------------------------------------------------------------------------
# Design metadata.
# -----------------------------------------------------------------------------
FALLBACK_META_R4 = {
    "M1": (1, "O3"),
    "M2": (2, "O3"),
    "M3": (3, "O3"),
    "M4": (4, "O3"),
    "M5": (4, "O4"),
}

def schedule_count_r4(design_id: str) -> int:
    if "ROUND4_DESIGNS" in globals() and design_id in ROUND4_DESIGNS:
        return int(
            ROUND4_DESIGNS[design_id].schedule_count
        )
    return FALLBACK_META_R4[design_id][0]

def observation_regime_r4(design_id: str) -> str:
    if "ROUND4_DESIGNS" in globals() and design_id in ROUND4_DESIGNS:
        return str(
            ROUND4_DESIGNS[design_id].observation_regime
        )
    return FALLBACK_META_R4[design_id][1]

def design_tick_labels_r4() -> List[str]:
    """
    Compact two-line labels used in the publication figures.

    The design ID remains visually dominant, while the second line reports
    schedule count and observation regime without crowding the x-axis.
    """
    labels = []

    for d in DESIGN_ORDER_R4:
        n = schedule_count_r4(d)
        schedule_text = "sched." if n == 1 else "scheds."

        labels.append(
            f"{d}\n{n} {schedule_text}, {observation_regime_r4(d)}"
        )

    return labels

# -----------------------------------------------------------------------------
# Publication style.
# -----------------------------------------------------------------------------
PUBLICATION_RC_R4 = {
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 11.8,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.3,
    "legend.fontsize": 8.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]

DESIGN_COLORS_R4 = {
    d: colors[i % len(colors)]
    for i, d in enumerate(DESIGN_ORDER_R4)
}

DESIGN_MARKERS_R4 = {
    "M1": "o",
    "M2": "s",
    "M3": "^",
    "M4": "D",
    "M5": "P",
}

# -----------------------------------------------------------------------------
# Fallback figure saver.
# -----------------------------------------------------------------------------
if "save_figure_r4" not in globals():

    def save_figure_r4(fig, stem):

        FIGURE_DIR_R4.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.svg",
            bbox_inches="tight",
            facecolor="white",
        )

# -----------------------------------------------------------------------------
# Helper functions.
# -----------------------------------------------------------------------------
def summary_value_r4(
    design_id: str,
    metric: str,
    statistic: str,
) -> float:

    row = PRIMARY_SUMMARY_R4[
        PRIMARY_SUMMARY_R4["design_id"] == design_id
    ]

    return float(
        row.iloc[0][f"{statistic}_{metric}"]
    )

def set_panel_frame_r4(ax):

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

def add_panel_label_r4(ax, label):

    ax.text(
        -0.10,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

def distribution_panel_r4(
    ax,
    metric,
    title,
    ylabel,
    *,
    log_scale=False,
    true_value: Optional[float] = None,
):

    x = np.arange(
        len(DESIGN_ORDER_R4),
        dtype=float,
    )

    means = np.array([
        summary_value_r4(
            d,
            metric,
            "mean",
        )
        for d in DESIGN_ORDER_R4
    ])

    stds = np.array([
        summary_value_r4(
            d,
            metric,
            "std",
        )
        for d in DESIGN_ORDER_R4
    ])

    # -------------------------------------------------------------------------
    # Individual five-seed values.
    # -------------------------------------------------------------------------
    for i, d in enumerate(DESIGN_ORDER_R4):

        values = (
            ROUND4_PRIMARY_ALL_RUNS.loc[
                ROUND4_PRIMARY_ALL_RUNS[
                    "design_id"
                ] == d,
                metric,
            ]
            .astype(float)
            .dropna()
            .to_numpy()
        )

        jitter = (
            np.array([0.0])
            if len(values) <= 1
            else np.linspace(
                -0.115,
                0.115,
                len(values),
            )
        )

        ax.scatter(
            np.full(
                len(values),
                x[i],
            ) + jitter,
            values,
            s=29,
            marker=DESIGN_MARKERS_R4[d],
            facecolor=DESIGN_COLORS_R4[d],
            edgecolor="black",
            linewidth=0.34,
            alpha=0.27,
            zorder=2,
        )

    # -------------------------------------------------------------------------
    # Across-seed mean trajectory.
    # -------------------------------------------------------------------------
    ax.plot(
        x,
        means,
        color="black",
        linewidth=1.35,
        alpha=0.78,
        zorder=3,
    )

    # -------------------------------------------------------------------------
    # Mean ± sample SD.
    # -------------------------------------------------------------------------
    for i, d in enumerate(DESIGN_ORDER_R4):

        ax.errorbar(
            x[i],
            means[i],
            yerr=stds[i],
            fmt=DESIGN_MARKERS_R4[d],
            markersize=8.2,
            markerfacecolor=DESIGN_COLORS_R4[d],
            markeredgecolor="black",
            markeredgewidth=0.85,
            ecolor="black",
            elinewidth=1.18,
            capsize=4.2,
            capthick=1.0,
            zorder=5,
        )

    # -------------------------------------------------------------------------
    # True parameter value, when relevant.
    # -------------------------------------------------------------------------
    if true_value is not None:

        ax.axhline(
            true_value,
            linestyle="--",
            linewidth=1.35,
            color="black",
            label="True value",
            zorder=1,
        )

    # -------------------------------------------------------------------------
    # Log axis where required.
    # -------------------------------------------------------------------------
    if log_scale:

        ax.set_yscale("log")

        ax.yaxis.set_major_locator(
            LogLocator(base=10)
        )

        ax.yaxis.set_minor_formatter(
            NullFormatter()
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        design_tick_labels_r4()
    )

    ax.set_xlim(
        -0.38,
        len(DESIGN_ORDER_R4) - 0.62,
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="semibold",
        pad=8,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        which="major",
        alpha=0.20,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    set_panel_frame_r4(ax)

# -----------------------------------------------------------------------------
# Common legend handles.
# -----------------------------------------------------------------------------
SEED_HANDLE_R4 = Line2D(
    [0],
    [0],
    marker="o",
    linestyle="None",
    markerfacecolor="0.65",
    markeredgecolor="black",
    alpha=0.30,
    markersize=5.8,
    label="Individual seed",
)

MEAN_HANDLE_R4 = Line2D(
    [0],
    [0],
    marker="D",
    linestyle="-",
    color="black",
    markerfacecolor="white",
    markeredgecolor="black",
    markersize=6.5,
    linewidth=1.2,
    label="Mean ± sample SD",
)

# =============================================================================
# FIGURE 1
# State reconstruction and physics consistency
# =============================================================================
with plt.rc_context(PUBLICATION_RC_R4):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.6, 8.8),
    )

    panels = [
        (
            axes[0, 0],
            "rmse_sensitive",
            r"Sensitive-state reconstruction, $S(t)$",
            "Sensitive-state RMSE",
            False,
        ),
        (
            axes[0, 1],
            "rmse_resistant",
            r"Resistant-state reconstruction, $R(t)$",
            "Resistant-state RMSE",
            False,
        ),
        (
            axes[1, 0],
            "rmse_fraction",
            r"Phenotype-composition reconstruction, $q_R(t)$",
            "Resistant-fraction RMSE",
            True,
        ),
        (
            axes[1, 1],
            "physics_residual_rmse",
            "Dynamical consistency",
            "Physics-residual RMSE",
            True,
        ),
    ]

    for label, (
        ax,
        metric,
        title,
        ylabel,
        log_scale,
    ) in zip(
        ["A", "B", "C", "D"],
        panels,
    ):

        distribution_panel_r4(
            ax,
            metric,
            title,
            ylabel,
            log_scale=log_scale,
        )

        add_panel_label_r4(
            ax,
            label,
        )

    fig.suptitle(
        "Schedule diversity and hidden-state reconstruction",
        fontsize=15.7,
        fontweight="bold",
        y=0.985,
    )

    fig.legend(
        handles=[
            SEED_HANDLE_R4,
            MEAN_HANDLE_R4,
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.095,
        top=0.885,
        wspace=0.27,
        hspace=0.36,
    )

    save_figure_r4(
        fig,
        "round4_primary_state_and_physics_final",
    )

    plt.show()

# =============================================================================
# FIGURE 2
# FINAL MANUSCRIPT FIGURE:
# Shared-parameter recovery and practical identifiability
# =============================================================================
#
# Publication decisions:
#   - no large in-figure suptitle; the manuscript caption carries the message;
#   - x-axis shows only M1--M5; design details remain in the caption/table;
#   - individual seeds remain visible but visually secondary;
#   - mean ± sample SD remains the dominant statistical summary;
#   - dashed truth lines are identified by compact in-panel annotations rather
#     than redundant legend boxes;
#   - concise panel titles and aligned y-axis labels;
#   - no model training or numerical recomputation is performed here.
# =============================================================================

with plt.rc_context(PUBLICATION_RC_R4):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 8.15),
    )

    panels = [
        (
            axes[0, 0],
            "mutation_probability",
            "Mutation-probability recovery",
            r"Estimated mutation probability, $\hat{\mu}$",
            True,
            float(
                TRUE_PARAMETERS.mutation_probability
            ),
        ),
        (
            axes[0, 1],
            "resistance_fitness_cost",
            "Resistance-fitness-cost recovery",
            r"Estimated resistance fitness cost, $\hat{c}_R$",
            False,
            float(
                TRUE_PARAMETERS.resistance_fitness_cost
            ),
        ),
        (
            axes[1, 0],
            "mu_error_percent",
            "Mutation-probability error",
            r"Relative error in $\mu$ (%)",
            True,
            None,
        ),
        (
            axes[1, 1],
            "cost_error_percent",
            "Fitness-cost error",
            r"Relative error in $c_R$ (%)",
            False,
            None,
        ),
    ]

    for label, (
        ax,
        metric,
        title,
        ylabel,
        log_scale,
        true_value,
    ) in zip(
        ["A", "B", "C", "D"],
        panels,
    ):

        distribution_panel_r4(
            ax,
            metric,
            title,
            ylabel,
            log_scale=log_scale,
            true_value=true_value,
        )

        add_panel_label_r4(
            ax,
            label,
        )

        # Final manuscript refinement:
        # Keep the parameter-recovery figure visually compact by showing only
        # the formal nested-design identifiers on the x-axis. The schedule
        # count and O3/O4 definitions are reported in the experimental-design
        # figure/table and explained again in the manuscript caption.
        ax.set_xticks(
            np.arange(
                len(DESIGN_ORDER_R4)
            )
        )

        ax.set_xticklabels(
            DESIGN_ORDER_R4,
            fontsize=9.6,
        )

        # Slightly stronger panel-title hierarchy without an overall headline.
        ax.set_title(
            title,
            loc="left",
            fontsize=11.9,
            fontweight="semibold",
            pad=8,
        )

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            width=0.85,
            length=3.8,
        )

    # -------------------------------------------------------------------------
    # Compact truth annotations.
    # -------------------------------------------------------------------------
    # The dashed horizontal lines are already drawn by distribution_panel_r4.
    # These text labels replace the two redundant "True value" legend boxes.
    # -------------------------------------------------------------------------

    axes[0, 0].text(
        0.975,
        0.955,
        r"Truth: $\mu^\star=10^{-4}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8.9,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.55,
            alpha=0.92,
        ),
        zorder=8,
    )

    axes[0, 1].text(
        0.975,
        0.955,
        r"Truth: $c_R^\star=0.10$",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8.9,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.55,
            alpha=0.92,
        ),
        zorder=8,
    )

    # -------------------------------------------------------------------------
    # Shared statistical legend.
    # -------------------------------------------------------------------------
    # Kept outside the data panels so that no observations or error bars are
    # obscured. The legend is intentionally compact for journal reduction.
    # -------------------------------------------------------------------------

    fig.legend(
        handles=[
            SEED_HANDLE_R4,
            MEAN_HANDLE_R4,
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=2,
        frameon=False,
        fontsize=9.0,
        handletextpad=0.55,
        columnspacing=1.6,
    )

    # Align labels for a cleaner 2 x 2 manuscript layout.
    fig.align_ylabels(
        axes[:, 0]
    )
    fig.align_ylabels(
        axes[:, 1]
    )

    # No figure-level title by design.
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.925,
        wspace=0.27,
        hspace=0.39,
    )

    save_figure_r4(
        fig,
        "round4_primary_parameter_recovery_final",
    )

    plt.show()

# =============================================================================
# FIGURE 3
# Paired information gain relative to M1
# =============================================================================
with plt.rc_context(PUBLICATION_RC_R4):

    effect_metrics = [
        "rmse_resistant",
        "rmse_fraction",
        "mu_error_percent",
        "cost_error_percent",
    ]

    effect_labels = [
        r"RMSE $R$",
        r"RMSE $q_R$",
        r"Error in $\mu$",
        r"Error in $c_R$",
    ]

    effect_matrix = np.full(
        (4, 4),
        np.nan,
    )

    for i, design_id in enumerate(
        DESIGN_ORDER_R4[1:]
    ):

        for j, metric in enumerate(
            effect_metrics
        ):

            pivot = ROUND4_PRIMARY_ALL_RUNS.pivot(
                index="seed_id",
                columns="design_id",
                values=metric,
            )

            paired = pivot[
                ["M1", design_id]
            ].dropna()

            effect_matrix[i, j] = float(
                (
                    100.0
                    * (
                        paired["M1"]
                        - paired[design_id]
                    )
                    / paired["M1"]
                ).mean()
            )

    fig, ax = plt.subplots(
        figsize=(9.3, 5.4),
    )

    limit = max(
        float(
            np.nanmax(
                np.abs(effect_matrix)
            )
        ),
        1.0,
    )

    image = ax.imshow(
        effect_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_xticks(
        np.arange(4)
    )

    ax.set_xticklabels(
        effect_labels
    )

    ax.set_yticks(
        np.arange(4)
    )

    ax.set_yticklabels([
        f"{d} vs M1"
        for d in DESIGN_ORDER_R4[1:]
    ])

    for i in range(4):

        for j in range(4):

            value = effect_matrix[
                i,
                j,
            ]

            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="semibold",
                color=(
                    "white"
                    if abs(value) > 0.55 * limit
                    else "black"
                ),
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.025,
    )

    cbar.set_label(
        "Mean paired improvement relative to M1 (%)"
    )

    ax.set_title(
        "Information gain from schedule diversity and phenotype observations",
        fontsize=14.2,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Evaluation quantity"
    )

    ax.set_ylabel(
        "Nested design comparison"
    )

    set_panel_frame_r4(ax)

    fig.tight_layout()

    save_figure_r4(
        fig,
        "round4_primary_information_gain_heatmap_final",
    )

    plt.show()

# =============================================================================
# FIGURE 4
# Accuracy-identifiability trade-off
# =============================================================================
with plt.rc_context(PUBLICATION_RC_R4):

    fig, ax = plt.subplots(
        figsize=(8.5, 5.9),
    )

    offsets = {
        "M1": (10, 9),
        "M2": (10, -14),
        "M3": (10, 9),
        "M4": (-10, 9),
        "M5": (-10, -15),
    }

    for d in DESIGN_ORDER_R4:

        row = PRIMARY_SUMMARY_R4[
            PRIMARY_SUMMARY_R4[
                "design_id"
            ] == d
        ].iloc[0]

        x = float(
            row["mean_rmse_resistant"]
        )

        y = float(
            row["mean_mu_error_percent"]
        )

        n = schedule_count_r4(d)

        ax.scatter(
            x,
            y,
            s=95 + 46 * n,
            marker=DESIGN_MARKERS_R4[d],
            facecolor=DESIGN_COLORS_R4[d],
            edgecolor="black",
            linewidth=0.85,
            zorder=4,
        )

        dx, dy = offsets[d]

        ax.annotate(
            d,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=(
                "right"
                if dx < 0
                else "left"
            ),
            va=(
                "bottom"
                if dy >= 0
                else "top"
            ),
            fontsize=9.7,
            fontweight="semibold",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    xs = np.array([
        summary_value_r4(
            d,
            "rmse_resistant",
            "mean",
        )
        for d in DESIGN_ORDER_R4
    ])

    ys = np.array([
        summary_value_r4(
            d,
            "mu_error_percent",
            "mean",
        )
        for d in DESIGN_ORDER_R4
    ])

    ax.set_xlim(
        np.min(xs) / 1.35,
        np.max(xs) * 1.40,
    )

    ax.set_ylim(
        np.min(ys) / 1.45,
        np.max(ys) * 1.45,
    )

    ax.set_xlabel(
        "Mean resistant-state RMSE"
    )

    ax.set_ylabel(
        r"Mean relative error in $\mu$ (%)"
    )

    ax.set_title(
        "State accuracy versus mutation-parameter identifiability",
        fontsize=14.2,
        fontweight="bold",
        pad=12,
    )

    ax.grid(
        which="major",
        alpha=0.20,
        linestyle="--",
    )

    set_panel_frame_r4(
        ax
    )

    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6 + 1.2 * n,
            label=(
                f"{n} schedule"
                if n == 1
                else f"{n} schedules"
            ),
        )
        for n in [1, 2, 3, 4]
    ]

    ax.legend(
        handles=size_handles,
        title="Schedule count",
        loc="upper right",
        frameon=True,
        fontsize=8.3,
        title_fontsize=8.6,
    )

    fig.tight_layout()

    save_figure_r4(
        fig,
        "round4_primary_accuracy_identifiability_tradeoff_final",
    )

    plt.show()

# =============================================================================
# FINAL CONFIRMATION
# =============================================================================
print(
    "\n" + "=" * 78
)

print(
    "ROUND 4 CELL 19 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 78
)

print(
    "Generated figures:"
)

print(
    "  1. round4_primary_state_and_physics_final"
)

print(
    "  2. round4_primary_parameter_recovery_final"
)

print(
    "  3. round4_primary_information_gain_heatmap_final"
)

print(
    "  4. round4_primary_accuracy_identifiability_tradeoff_final"
)

print(
    "Figure directory:",
    FIGURE_DIR_R4.resolve(),
)

print(
    "No model was retrained."
)

# In[72]:


# =============================================================================
# ROUND 4 — CELL 20
# Publication-ready trajectory validation
#   Figure A: objectively selected representative M4 reconstruction
#   Figure B: five-seed M4-versus-M5 trajectory comparison (mean ± sample SD)
# =============================================================================
# This cell performs NO TRAINING. It reuses the completed metrics and dense
# prediction CSV files saved by Cell 17.
#
# M4 and M5 use the same four treatment schedules. M4 uses O3 observations,
# whereas M5 uses O4 phenotype-resolved observations. The second figure thus
# helps isolate the effect of richer phenotype information after schedule
# diversity has already been introduced.
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# -----------------------------------------------------------------------------
# 1. Reload completed primary results if required
# -----------------------------------------------------------------------------
if (
    "ROUND4_PRIMARY_ALL_RUNS" not in globals()
    or not isinstance(ROUND4_PRIMARY_ALL_RUNS, pd.DataFrame)
    or ROUND4_PRIMARY_ALL_RUNS.empty
):
    primary_path_r4 = TABLE_DIR_R4 / "round4_primary_multischedule_all_seeds_metrics.csv"
    if not primary_path_r4.exists():
        raise FileNotFoundError("Round 4 primary results are unavailable. Run Cell 17 first.")
    ROUND4_PRIMARY_ALL_RUNS = pd.read_csv(primary_path_r4)
    print("Reloaded Round 4 primary results from:", primary_path_r4)

# -----------------------------------------------------------------------------
# 2. Fallback figure helpers
# -----------------------------------------------------------------------------
if "set_panel_frame_r4" not in globals():
    def set_panel_frame_r4(axis):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

if "save_figure_r4" not in globals():
    def save_figure_r4(fig, stem):
        """Save a publication figure in raster and vector formats."""
        FIGURE_DIR_R4.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.svg",
            bbox_inches="tight",
            facecolor="white",
        )

# -----------------------------------------------------------------------------
# 3. Consistent publication style
# -----------------------------------------------------------------------------
CELL20_RC_R4 = {
    "font.family": "sans-serif",
    "font.size": 10.2,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.7,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

SCHEDULE_ORDER_R4_CELL20 = [
    "untreated",
    "early_continuous",
    "late_continuous",
    "pulsed",
]

SCHEDULE_DISPLAY_R4_CELL20 = {
    "untreated": "Untreated",
    "early_continuous": "Early continuous",
    "late_continuous": "Late continuous",
    "pulsed": "Pulsed",
}

ROW_PANEL_LABELS_R4 = ["A", "B", "C", "D"]

# -----------------------------------------------------------------------------
# 4. Axis formatting and treatment shading
# -----------------------------------------------------------------------------
def format_numeric_axis_r4(axis, *, resistant_fraction=False):
    """Compact numerical formatting with a limited number of y ticks."""

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(
        (-3, 3) if resistant_fraction else (-4, 4)
    )

    axis.yaxis.set_major_formatter(formatter)

    # IMPORTANT FIX:
    # Matplotlib uses 'nbins' (plural), not 'nbin'.
    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=5)
    )


def apply_common_axis_style_r4(axis, *, resistant_fraction=False):

    axis.grid(
        axis="both",
        alpha=0.16,
        linestyle="--",
        linewidth=0.65,
    )

    axis.set_axisbelow(True)

    set_panel_frame_r4(axis)

    format_numeric_axis_r4(
        axis,
        resistant_fraction=resistant_fraction,
    )


def shade_treatment_intervals_r4(axis, schedule_name):
    """Shade intervals for which prescribed antibiotic treatment is ON."""

    schedule = SCHEDULES_R4[schedule_name]

    for left, right, level in schedule.intervals(
        T_START,
        T_FINAL,
    ):
        if level > 0:
            axis.axvspan(
                left,
                right,
                facecolor="0.82",
                edgecolor="none",
                alpha=0.30,
                zorder=0,
            )

# -----------------------------------------------------------------------------
# 5. Objective representative M4 seed selection
# -----------------------------------------------------------------------------
m4_rows_r4 = (
    ROUND4_PRIMARY_ALL_RUNS[
        ROUND4_PRIMARY_ALL_RUNS["design_id"] == "M4"
    ]
    .copy()
    .reset_index(drop=True)
)

if m4_rows_r4.empty:
    raise RuntimeError("No completed M4 rows were found.")

selection_metrics_r4 = [
    "rmse_resistant",
    "rmse_fraction",
    "mu_error_percent",
    "cost_error_percent",
]

missing = [
    metric
    for metric in selection_metrics_r4
    if metric not in m4_rows_r4.columns
]

if missing:
    raise KeyError(
        f"Missing M4 columns needed for representative-seed selection: {missing}"
    )

representative_distance_r4 = np.zeros(
    len(m4_rows_r4),
    dtype=float,
)

for metric in selection_metrics_r4:

    values = (
        m4_rows_r4[metric]
        .astype(float)
        .to_numpy()
    )

    median = np.median(values)

    mad = np.median(
        np.abs(
            values - median
        )
    )

    scale = max(
        float(mad),
        1.0e-12,
    )

    representative_distance_r4 += (
        (values - median) / scale
    ) ** 2

representative_index_r4 = int(
    np.argmin(
        representative_distance_r4
    )
)

REPRESENTATIVE_M4_ROW_R4 = (
    m4_rows_r4.iloc[
        representative_index_r4
    ].copy()
)

representative_label_r4 = str(
    REPRESENTATIVE_M4_ROW_R4["label"]
)

representative_seed_r4 = str(
    REPRESENTATIVE_M4_ROW_R4["seed_id"]
)

representative_prediction_path_r4 = (
    PREDICTION_DIR_R4
    / f"{representative_label_r4}_all_schedule_predictions.csv"
)

if not representative_prediction_path_r4.exists():
    raise FileNotFoundError(
        "Representative M4 prediction file was not found:\n"
        f"{representative_prediction_path_r4}"
    )

representative_prediction_r4 = pd.read_csv(
    representative_prediction_path_r4
)

# Save an auditable record of why this seed was selected.
representative_audit_r4 = m4_rows_r4[
    [
        "label",
        "seed_id",
        "rmse_resistant",
        "rmse_fraction",
        "mu_error_percent",
        "cost_error_percent",
    ]
].copy()

representative_audit_r4[
    "median_distance_score"
] = representative_distance_r4

representative_audit_r4[
    "selected_representative"
] = (
    representative_audit_r4["label"]
    == representative_label_r4
)

representative_audit_path_r4 = (
    TABLE_DIR_R4
    / "round4_M4_representative_seed_selection.csv"
)

representative_audit_r4.to_csv(
    representative_audit_path_r4,
    index=False,
)

# =============================================================================
# FIGURE A — FINAL REPRESENTATIVE FOUR-SCHEDULE M4 RECONSTRUCTION
# =============================================================================
# Publication design:
#   - no large internal title; representative-seed information belongs in caption;
#   - one compact legend above the grid;
#   - column headings shown once, outside the axes;
#   - schedule names shown as clean row headers;
#   - "Resistant fraction" terminology used consistently throughout manuscript;
#   - shared column-level y labels reduce repetition;
#   - treatment shading retained;
#   - time label retained only on bottom row;
#   - all trajectories and numerical values are unchanged.
# =============================================================================

with plt.rc_context(CELL20_RC_R4):

    fig_a, axes_a = plt.subplots(
        4,
        3,
        figsize=(12.4, 9.35),
        sharex=True,
    )

    reference_color = None
    pinn_color = None

    for row_index, schedule_name in enumerate(
        SCHEDULE_ORDER_R4_CELL20
    ):

        frame = (
            representative_prediction_r4[
                representative_prediction_r4[
                    "schedule"
                ] == schedule_name
            ]
            .copy()
            .sort_values("time")
        )

        if frame.empty:
            raise RuntimeError(
                f"No representative trajectory data found for {schedule_name}."
            )

        # ---------------------------------------------------------------------
        # Sensitive population
        # ---------------------------------------------------------------------
        ref_s, = axes_a[
            row_index,
            0,
        ].plot(
            frame["time"],
            frame["sensitive"],
            linewidth=2.05,
            label="Reference",
            zorder=3,
        )

        pred_s, = axes_a[
            row_index,
            0,
        ].plot(
            frame["time"],
            frame["predicted_sensitive"],
            linestyle="--",
            linewidth=1.85,
            label="Hard PINN",
            zorder=4,
        )

        # ---------------------------------------------------------------------
        # Resistant population
        # ---------------------------------------------------------------------
        axes_a[
            row_index,
            1,
        ].plot(
            frame["time"],
            frame["resistant"],
            color=ref_s.get_color(),
            linewidth=2.05,
            zorder=3,
        )

        axes_a[
            row_index,
            1,
        ].plot(
            frame["time"],
            frame["predicted_resistant"],
            color=pred_s.get_color(),
            linestyle="--",
            linewidth=1.85,
            zorder=4,
        )

        # ---------------------------------------------------------------------
        # Resistant fraction
        # ---------------------------------------------------------------------
        axes_a[
            row_index,
            2,
        ].plot(
            frame["time"],
            frame["resistant_fraction"],
            color=ref_s.get_color(),
            linewidth=2.05,
            zorder=3,
        )

        axes_a[
            row_index,
            2,
        ].plot(
            frame["time"],
            frame["predicted_resistant_fraction"],
            color=pred_s.get_color(),
            linestyle="--",
            linewidth=1.85,
            zorder=4,
        )

        if row_index == 0:
            reference_color = ref_s.get_color()
            pinn_color = pred_s.get_color()

        # ---------------------------------------------------------------------
        # Common axis formatting and treatment shading
        # ---------------------------------------------------------------------
        for column_index in range(3):

            axis = axes_a[
                row_index,
                column_index,
            ]

            shade_treatment_intervals_r4(
                axis,
                schedule_name,
            )

            apply_common_axis_style_r4(
                axis,
                resistant_fraction=(
                    column_index == 2
                ),
            )

            # Per-panel y labels are intentionally removed. Clean shared
            # column-level labels are added below after final layout.
            axis.set_ylabel("")

    # Time is shown only on the bottom row.
    for axis in axes_a[-1, :]:
        axis.set_xlabel(
            "Time"
        )

    # -------------------------------------------------------------------------
    # Reserve clean space for legend, column headings, and row headers.
    # -------------------------------------------------------------------------
    fig_a.subplots_adjust(
        left=0.095,
        right=0.992,
        bottom=0.070,
        top=0.855,
        wspace=0.245,
        hspace=0.430,
    )

    # -------------------------------------------------------------------------
    # Column headings — shown once for the complete grid.
    # -------------------------------------------------------------------------
    column_titles = [
        r"Sensitive population, $S(t)$",
        r"Resistant population, $R(t)$",
        r"Resistant fraction, $q_R(t)$",
    ]

    for column_index, column_title in enumerate(
        column_titles
    ):

        position = axes_a[
            0,
            column_index,
        ].get_position()

        x_center = 0.5 * (
            position.x0 + position.x1
        )

        fig_a.text(
            x_center,
            0.895,
            column_title,
            ha="center",
            va="center",
            fontsize=11.7,
            fontweight="semibold",
        )

    # -------------------------------------------------------------------------
    # Row headers — deliberately separated from the column headings.
    # -------------------------------------------------------------------------
    for row_index, schedule_name in enumerate(
        SCHEDULE_ORDER_R4_CELL20
    ):

        position = axes_a[
            row_index,
            0,
        ].get_position()

        fig_a.text(
            position.x0,
            position.y1 + 0.010,
            (
                f"{ROW_PANEL_LABELS_R4[row_index]}  "
                f"{SCHEDULE_DISPLAY_R4_CELL20[schedule_name]}"
            ),
            ha="left",
            va="bottom",
            fontsize=10.9,
            fontweight="bold",
        )

    # -------------------------------------------------------------------------
    # Shared column-level y labels.
    # -------------------------------------------------------------------------
    shared_y_labels = [
        "Population",
        "Population",
        "Resistant fraction",
    ]

    for column_index, shared_label in enumerate(
        shared_y_labels
    ):

        column_position = axes_a[
            1,
            column_index,
        ].get_position()

        x_label = (
            column_position.x0 - 0.038
            if column_index == 0
            else column_position.x0 - 0.032
        )

        fig_a.text(
            x_label,
            0.465,
            shared_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=10.6,
        )

    # -------------------------------------------------------------------------
    # Compact shared legend.
    # -------------------------------------------------------------------------
    fig_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=reference_color,
                linewidth=2.05,
                label="Reference",
            ),
            Line2D(
                [0],
                [0],
                color=pinn_color,
                linestyle="--",
                linewidth=1.85,
                label="Hard PINN",
            ),
            Patch(
                facecolor="0.82",
                edgecolor="none",
                alpha=0.50,
                label="Treatment ON",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.7,
        handlelength=2.7,
        fontsize=9.0,
    )

    # No figure-level title by design. The caption should state that this is
    # the objectively selected representative M4 run (seed3 in the final run).

    save_figure_r4(
        fig_a,
        "round4_representative_M4_trajectories_refined",
    )

    plt.show()

print(
    "\\nObjectively selected representative M4 run:"
)

print(
    "  Label:",
    representative_label_r4,
)

print(
    "  Seed:",
    representative_seed_r4,
)

print(
    "  Selection audit:",
    representative_audit_path_r4,
)

display(
    pd.DataFrame([
        REPRESENTATIVE_M4_ROW_R4
    ])
)

# -----------------------------------------------------------------------------
# 6. Load all saved prediction files for one design
# -----------------------------------------------------------------------------
def load_design_prediction_ensemble_r4(
    design_id
):

    design_rows = (
        ROUND4_PRIMARY_ALL_RUNS[
            ROUND4_PRIMARY_ALL_RUNS[
                "design_id"
            ] == design_id
        ]
        .copy()
        .sort_values("seed_id")
    )

    if design_rows.empty:
        raise RuntimeError(
            f"No completed rows found for design {design_id}."
        )

    frames = []

    for _, row in design_rows.iterrows():

        label = str(
            row["label"]
        )

        seed_id = str(
            row["seed_id"]
        )

        prediction_file = (
            PREDICTION_DIR_R4
            / f"{label}_all_schedule_predictions.csv"
        )

        if not prediction_file.exists():
            print(
                "WARNING: missing prediction file:",
                prediction_file,
            )
            continue

        frame = pd.read_csv(
            prediction_file
        )

        frame["design_id"] = design_id
        frame["seed_id"] = seed_id

        frames.append(
            frame
        )

    if not frames:
        raise RuntimeError(
            f"No usable prediction files were available for {design_id}."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )

m4_prediction_ensemble_r4 = (
    load_design_prediction_ensemble_r4(
        "M4"
    )
)

m5_prediction_ensemble_r4 = (
    load_design_prediction_ensemble_r4(
        "M5"
    )
)

print(
    "\nPrediction ensemble sizes:"
)

print(
    "  M4 seeds:",
    m4_prediction_ensemble_r4[
        "seed_id"
    ].nunique(),
)

print(
    "  M5 seeds:",
    m5_prediction_ensemble_r4[
        "seed_id"
    ].nunique(),
)

# -----------------------------------------------------------------------------
# 7. Aggregate mean ± sample SD trajectories across seeds
# -----------------------------------------------------------------------------
def sample_sd_r4(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0
    )


def aggregate_prediction_ensemble_r4(
    frame
):

    return (
        frame.groupby(
            [
                "schedule",
                "time",
            ],
            observed=True,
            sort=True,
        )
        .agg(
            resistant_reference=(
                "resistant",
                "mean",
            ),
            fraction_reference=(
                "resistant_fraction",
                "mean",
            ),
            resistant_mean=(
                "predicted_resistant",
                "mean",
            ),
            resistant_sd=(
                "predicted_resistant",
                sample_sd_r4,
            ),
            fraction_mean=(
                "predicted_resistant_fraction",
                "mean",
            ),
            fraction_sd=(
                "predicted_resistant_fraction",
                sample_sd_r4,
            ),
            n_seeds=(
                "seed_id",
                "nunique",
            ),
        )
        .reset_index()
    )

m4_trajectory_summary_r4 = (
    aggregate_prediction_ensemble_r4(
        m4_prediction_ensemble_r4
    )
)

m5_trajectory_summary_r4 = (
    aggregate_prediction_ensemble_r4(
        m5_prediction_ensemble_r4
    )
)

# =============================================================================
# FIGURE B — FINAL FIVE-SEED M4 VERSUS M5 TRAJECTORY COMPARISON
# =============================================================================
# This supporting figure isolates the effect of phenotype-rich O4 information
# after four-schedule diversity is already present. It is intended primarily
# for Supplementary Material.
# =============================================================================

with plt.rc_context(CELL20_RC_R4):

    fig_b, axes_b = plt.subplots(
        4,
        2,
        figsize=(10.9, 9.15),
        sharex=True,
    )

    legend_handles = None

    for row_index, schedule_name in enumerate(
        SCHEDULE_ORDER_R4_CELL20
    ):

        m4 = (
            m4_trajectory_summary_r4[
                m4_trajectory_summary_r4[
                    "schedule"
                ] == schedule_name
            ]
            .copy()
            .sort_values("time")
        )

        m5 = (
            m5_trajectory_summary_r4[
                m5_trajectory_summary_r4[
                    "schedule"
                ] == schedule_name
            ]
            .copy()
            .sort_values("time")
        )

        if m4.empty or m5.empty:
            raise RuntimeError(
                f"Missing M4 or M5 trajectory summary for {schedule_name}."
            )

        ax_r = axes_b[
            row_index,
            0,
        ]

        ax_q = axes_b[
            row_index,
            1,
        ]

        # ---------------------------------------------------------------------
        # Resistant population
        # ---------------------------------------------------------------------
        ref_line, = ax_r.plot(
            m4["time"],
            m4["resistant_reference"],
            linewidth=2.10,
            label="Reference",
            zorder=5,
        )

        m4_line, = ax_r.plot(
            m4["time"],
            m4["resistant_mean"],
            linestyle="--",
            linewidth=1.85,
            label="M4 mean",
            zorder=4,
        )

        m5_line, = ax_r.plot(
            m5["time"],
            m5["resistant_mean"],
            linestyle="-.",
            linewidth=1.85,
            label="M5 mean",
            zorder=4,
        )

        m4_mean = m4[
            "resistant_mean"
        ].to_numpy(float)

        m4_sd = m4[
            "resistant_sd"
        ].to_numpy(float)

        m5_mean = m5[
            "resistant_mean"
        ].to_numpy(float)

        m5_sd = m5[
            "resistant_sd"
        ].to_numpy(float)

        ax_r.fill_between(
            m4["time"].to_numpy(float),
            np.maximum(
                0.0,
                m4_mean - m4_sd,
            ),
            m4_mean + m4_sd,
            alpha=0.13,
            color=m4_line.get_color(),
            linewidth=0,
            zorder=2,
        )

        ax_r.fill_between(
            m5["time"].to_numpy(float),
            np.maximum(
                0.0,
                m5_mean - m5_sd,
            ),
            m5_mean + m5_sd,
            alpha=0.13,
            color=m5_line.get_color(),
            linewidth=0,
            zorder=2,
        )

        # ---------------------------------------------------------------------
        # Resistant fraction
        # ---------------------------------------------------------------------
        ax_q.plot(
            m4["time"],
            m4["fraction_reference"],
            color=ref_line.get_color(),
            linewidth=2.10,
            zorder=5,
        )

        m4_q_line, = ax_q.plot(
            m4["time"],
            m4["fraction_mean"],
            color=m4_line.get_color(),
            linestyle="--",
            linewidth=1.85,
            zorder=4,
        )

        m5_q_line, = ax_q.plot(
            m5["time"],
            m5["fraction_mean"],
            color=m5_line.get_color(),
            linestyle="-.",
            linewidth=1.85,
            zorder=4,
        )

        m4_q_mean = m4[
            "fraction_mean"
        ].to_numpy(float)

        m4_q_sd = m4[
            "fraction_sd"
        ].to_numpy(float)

        m5_q_mean = m5[
            "fraction_mean"
        ].to_numpy(float)

        m5_q_sd = m5[
            "fraction_sd"
        ].to_numpy(float)

        ax_q.fill_between(
            m4["time"].to_numpy(float),
            np.maximum(
                0.0,
                m4_q_mean - m4_q_sd,
            ),
            m4_q_mean + m4_q_sd,
            alpha=0.13,
            color=m4_q_line.get_color(),
            linewidth=0,
            zorder=2,
        )

        ax_q.fill_between(
            m5["time"].to_numpy(float),
            np.maximum(
                0.0,
                m5_q_mean - m5_q_sd,
            ),
            m5_q_mean + m5_q_sd,
            alpha=0.13,
            color=m5_q_line.get_color(),
            linewidth=0,
            zorder=2,
        )

        # Treatment shading + common style.
        for axis in (
            ax_r,
            ax_q,
        ):
            shade_treatment_intervals_r4(
                axis,
                schedule_name,
            )

        apply_common_axis_style_r4(
            ax_r,
            resistant_fraction=False,
        )

        apply_common_axis_style_r4(
            ax_q,
            resistant_fraction=True,
        )

        ax_r.set_ylabel("")
        ax_q.set_ylabel("")

        if row_index == 0:
            legend_handles = [
                ref_line,
                m4_line,
                m5_line,
                Patch(
                    facecolor="0.82",
                    edgecolor="none",
                    alpha=0.50,
                    label="Treatment ON",
                ),
            ]

    for axis in axes_b[-1, :]:
        axis.set_xlabel(
            "Time"
        )

    # Reserve explicit space for hierarchy outside the axes.
    fig_b.subplots_adjust(
        left=0.105,
        right=0.992,
        bottom=0.070,
        top=0.850,
        wspace=0.245,
        hspace=0.430,
    )

    # Column headings.
    comparison_titles = [
        r"Resistant population, $R(t)$",
        r"Resistant fraction, $q_R(t)$",
    ]

    for column_index, column_title in enumerate(
        comparison_titles
    ):

        position = axes_b[
            0,
            column_index,
        ].get_position()

        fig_b.text(
            0.5 * (
                position.x0 + position.x1
            ),
            0.895,
            column_title,
            ha="center",
            va="center",
            fontsize=11.7,
            fontweight="semibold",
        )

    # Row headers.
    for row_index, schedule_name in enumerate(
        SCHEDULE_ORDER_R4_CELL20
    ):

        position = axes_b[
            row_index,
            0,
        ].get_position()

        fig_b.text(
            position.x0,
            position.y1 + 0.010,
            (
                f"{ROW_PANEL_LABELS_R4[row_index]}  "
                f"{SCHEDULE_DISPLAY_R4_CELL20[schedule_name]}"
            ),
            ha="left",
            va="bottom",
            fontsize=10.9,
            fontweight="bold",
        )

    # Shared y labels.
    first_position = axes_b[
        1,
        0,
    ].get_position()

    second_position = axes_b[
        1,
        1,
    ].get_position()

    fig_b.text(
        first_position.x0 - 0.045,
        0.465,
        "Population",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.6,
    )

    fig_b.text(
        second_position.x0 - 0.038,
        0.465,
        "Resistant fraction",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.6,
    )

    # Compact shared legend.
    fig_b.legend(
        handles=legend_handles,
        labels=[
            "Reference",
            "M4 mean ± SD",
            "M5 mean ± SD",
            "Treatment ON",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.5,
        fontsize=8.9,
    )

    # No internal suptitle. The Supplementary caption should explain that
    # M4 and M5 share the four schedules and differ in O3 versus O4 information.

    save_figure_r4(
        fig_b,
        "round4_M4_vs_M5_five_seed_trajectory_comparison",
    )

    plt.show()

# -----------------------------------------------------------------------------
# 8. Save five-seed trajectory summaries
# -----------------------------------------------------------------------------
m4_trajectory_summary_path_r4 = (
    TABLE_DIR_R4
    / "round4_M4_five_seed_trajectory_summary.csv"
)

m5_trajectory_summary_path_r4 = (
    TABLE_DIR_R4
    / "round4_M5_five_seed_trajectory_summary.csv"
)

m4_trajectory_summary_r4.to_csv(
    m4_trajectory_summary_path_r4,
    index=False,
)

m5_trajectory_summary_r4.to_csv(
    m5_trajectory_summary_path_r4,
    index=False,
)

# -----------------------------------------------------------------------------
# 9. Final report
# -----------------------------------------------------------------------------
print(
    "\n" + "=" * 82
)

print(
    "ROUND 4 CELL 20 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 82
)

print(
    "\nRepresentative M4 run:"
)

print(
    "  Label:",
    representative_label_r4,
)

print(
    "  Seed:",
    representative_seed_r4,
)

print(
    "\nGenerated Figure A:"
)

print(
    "  round4_representative_M4_trajectories_refined"
)

print(
    "\nGenerated Figure B:"
)

print(
    "  round4_M4_vs_M5_five_seed_trajectory_comparison"
)

print(
    "\nFive-seed trajectory summaries:"
)

print(
    " ",
    m4_trajectory_summary_path_r4
)

print(
    " ",
    m5_trajectory_summary_path_r4
)

print(
    "\nRepresentative-selection audit:"
)

print(
    " ",
    representative_audit_path_r4
)

print(
    "\nNo model was retrained."
)

print(
    "All figures were generated from prediction files already saved during Cell 17."
)

# In[24]:


# =============================================================================
# ROUND 4 — TEMPORARY SWITCH
# Run matched mechanistic ODE baselines only
# =============================================================================

RUN_STRUCTURAL_SMOKE_TEST = False
RUN_SINGLE_DESIGN_DEMO = False
RUN_PRIMARY_MULTI_SCHEDULE_STUDY = False

RUN_MATCHED_ODE_BASELINES = True

RUN_IDENTIFIABILITY_GEOMETRY = False
RUN_LEAVE_ONE_SCHEDULE_OUT = False
RUN_PROFILE_SURFACES = False
RUN_PARAMETER_TRANSFER_TEST = False
RUN_PLOTTING_ONLY = False

CURRENT_EXPERIMENT_R4 = "matched_ode_baselines"

EXPERIMENT_SWITCHES_R4 = {
    "structural_smoke_test": RUN_STRUCTURAL_SMOKE_TEST,
    "single_design_demo": RUN_SINGLE_DESIGN_DEMO,
    "primary_multi_schedule_study": RUN_PRIMARY_MULTI_SCHEDULE_STUDY,
    "matched_ode_baselines": RUN_MATCHED_ODE_BASELINES,
    "identifiability_geometry": RUN_IDENTIFIABILITY_GEOMETRY,
    "leave_one_schedule_out": RUN_LEAVE_ONE_SCHEDULE_OUT,
    "profile_surfaces": RUN_PROFILE_SURFACES,
    "parameter_transfer_test": RUN_PARAMETER_TRANSFER_TEST,
    "plotting_only": RUN_PLOTTING_ONLY,
}

active_experiments_r4 = [
    name
    for name, enabled in EXPERIMENT_SWITCHES_R4.items()
    if enabled
]

print("Current experiment:", CURRENT_EXPERIMENT_R4)
print("Active experiments:", active_experiments_r4)

assert RUN_MATCHED_ODE_BASELINES is True
assert RUN_PRIMARY_MULTI_SCHEDULE_STUDY is False
assert active_experiments_r4 == ["matched_ode_baselines"]

print("\nMatched ODE-baseline configuration is active.")
print("Run the ODE-baseline cell now.")

# In[25]:


# =============================================================================
# ROUND 4 — CELL 21
# Matched mechanistic ODE baselines for M1--M5
# =============================================================================
# The synthetic observations are generated from the same ODE family. Therefore,
# this baseline is expected to be extremely accurate in the noise-free matched-
# model setting. It is reported separately from the neural design comparison.

ODE_BASELINE_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_matched_ode_design_metrics.csv"
)

ODE_BASELINE_ROWS_R4: List[Dict[str, Any]] = []

if RUN_MATCHED_ODE_BASELINES:
    for design_index, design_id in enumerate(
        DESIGN_ORDER_R4
    ):
        bundle = build_design_bundle_r4(
            design_id,
            seed=SEED,
            noise_level=0.0,
        )

        fit = fit_ode_baseline(
            bundle,
            label=f"round4_ode_{design_id}",
            seed=SEED + 800_000 + design_index,
            starts=(5 if QUICK_MODE else 16),
        )

        row = evaluate_ode_design_r4(
            design_id,
            bundle,
            fit,
        )

        ODE_BASELINE_ROWS_R4.append(
            row
        )

    ODE_BASELINE_RESULTS_R4 = pd.DataFrame(
        ODE_BASELINE_ROWS_R4
    )

    ODE_BASELINE_RESULTS_R4.to_csv(
        ODE_BASELINE_PATH_R4,
        index=False,
    )

    display(ODE_BASELINE_RESULTS_R4)
else:
    if ODE_BASELINE_PATH_R4.exists():
        ODE_BASELINE_RESULTS_R4 = pd.read_csv(
            ODE_BASELINE_PATH_R4
        )
        print(
            "Loaded the existing matched ODE benchmark table."
        )
    else:
        ODE_BASELINE_RESULTS_R4 = pd.DataFrame()
        print(
            "Set RUN_MATCHED_ODE_BASELINES=True to fit the separate "
            "matched mechanistic benchmark."
        )

# In[74]:


# =============================================================================
# ROUND 4 — TEMPORARY SWITCH FOR CELL 22
# Identifiability-geometry / sensitivity analysis only
# =============================================================================

RUN_STRUCTURAL_SMOKE_TEST = False
RUN_SINGLE_DESIGN_DEMO = False
RUN_PRIMARY_MULTI_SCHEDULE_STUDY = False
RUN_MATCHED_ODE_BASELINES = False

RUN_IDENTIFIABILITY_GEOMETRY = True

RUN_LEAVE_ONE_SCHEDULE_OUT = False
RUN_PROFILE_SURFACES = False
RUN_PARAMETER_TRANSFER_TEST = False
RUN_PLOTTING_ONLY = False

CURRENT_EXPERIMENT_R4 = "identifiability_geometry"

EXPERIMENT_SWITCHES_R4 = {
    "structural_smoke_test": RUN_STRUCTURAL_SMOKE_TEST,
    "single_design_demo": RUN_SINGLE_DESIGN_DEMO,
    "primary_multi_schedule_study": RUN_PRIMARY_MULTI_SCHEDULE_STUDY,
    "matched_ode_baselines": RUN_MATCHED_ODE_BASELINES,
    "identifiability_geometry": RUN_IDENTIFIABILITY_GEOMETRY,
    "leave_one_schedule_out": RUN_LEAVE_ONE_SCHEDULE_OUT,
    "profile_surfaces": RUN_PROFILE_SURFACES,
    "parameter_transfer_test": RUN_PARAMETER_TRANSFER_TEST,
    "plotting_only": RUN_PLOTTING_ONLY,
}

active_experiments_r4 = [
    name
    for name, enabled in EXPERIMENT_SWITCHES_R4.items()
    if enabled
]

print("Current experiment:", CURRENT_EXPERIMENT_R4)
print("Active experiments:", active_experiments_r4)

assert RUN_IDENTIFIABILITY_GEOMETRY is True
assert RUN_PRIMARY_MULTI_SCHEDULE_STUDY is False
assert active_experiments_r4 == ["identifiability_geometry"]

print("\nIdentifiability-geometry configuration is active.")
print("Run Cell 22 now.")

# In[76]:


# ROUND 4 — CELL 22
# FINAL PUBLICATION FIGURES:
# Sensitivity SVD, local conditioning, parameter correlation,
# and weakest identifiable parameter direction
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# This cell performs NO PINN training.
#
# It uses SENSITIVITY_SUMMARY_R4 generated by the preceding Round 4
# identifiability-geometry computation and produces two final figures:
#
# FIGURE 22A
# ----------
# A. Sensitivity spectrum:
#       largest singular value, sigma_max
#       smallest singular value, sigma_min
#
# B. Sensitivity conditioning:
#       kappa = sigma_max / sigma_min
#
# C. Local parameter correlation:
#       |rho(log10(mu), c_R)|
#
# FIGURE 22B
# ----------
# Absolute components of the right singular vector associated with sigma_min.
# This vector describes the locally weakest identifiable parameter direction.
#
# IMPORTANT SCIENTIFIC POINT
# --------------------------
# A singular vector has arbitrary global sign:
#
#       v  and  -v
#
# describe exactly the same direction.
#
# Therefore the heatmap reports ABSOLUTE component magnitudes:
#
#       |v_log10(mu)|   and   |v_cR|
#
# rather than signed components.
#
# PUBLICATION REFINEMENTS
# -----------------------
# 1. Only M1-M5 appear on the x-axis.
# 2. Font sizes are increased for manuscript-width readability.
# 3. "Parameter confounding" is replaced with "Sensitivity conditioning".
# 4. sigma_max and sigma_min remain on a logarithmic axis.
# 5. kappa is shown on a linear axis because its observed range is narrow.
# 6. Correlation remains on the physically interpretable 0-1 scale.
# 7. Weak-direction magnitudes use a sequential color scale from 0 to 1.
# 8. Exact plotted values are exported to CSV for reproducibility.
#
# NORMALIZATION NOTE
# ------------------
# The singular values shown here use the same normalized sensitivity matrix
# constructed by the preceding identifiability-analysis computation.
# The exact normalization should be defined once in the Methods/caption.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


# =============================================================================
# 2. EXECUTION GUARD
# =============================================================================

if RUN_IDENTIFIABILITY_GEOMETRY or RUN_PLOTTING_ONLY:

    # =========================================================================
    # 3. SAFETY CHECK
    # =========================================================================

    if (
        "SENSITIVITY_SUMMARY_R4" not in globals()
        or not isinstance(SENSITIVITY_SUMMARY_R4, pd.DataFrame)
        or SENSITIVITY_SUMMARY_R4.empty
    ):

        print(
            "Sensitivity summary is unavailable. "
            "Run the Round 4 identifiability-geometry computation first."
        )

    else:

        # =====================================================================
        # 4. FALLBACK HELPERS
        # =====================================================================

        if "DESIGN_ORDER_R4" not in globals():
            DESIGN_ORDER_R4 = [
                "M1",
                "M2",
                "M3",
                "M4",
                "M5",
            ]


        if "set_panel_frame_r4" not in globals():

            def set_panel_frame_r4(axis):
                """
                Apply a consistent black frame around the plotting region.
                """

                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1.0)


        if "save_figure_r4" not in globals():

            def save_figure_r4(fig, stem):
                """
                Save publication figures in high-resolution PNG and vector PDF.
                """

                FIGURE_DIR_R4.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                fig.savefig(
                    FIGURE_DIR_R4 / f"{stem}.png",
                    dpi=600,
                    bbox_inches="tight",
                    facecolor="white",
                )

                fig.savefig(
                    FIGURE_DIR_R4 / f"{stem}.pdf",
                    bbox_inches="tight",
                    facecolor="white",
                )

                fig.savefig(
                    FIGURE_DIR_R4 / f"{stem}.svg",
                    bbox_inches="tight",
                    facecolor="white",
                )


        # =====================================================================
        # 5. COPY AND VALIDATE THE SENSITIVITY SUMMARY
        # =====================================================================

        summary = SENSITIVITY_SUMMARY_R4.copy()


        required_columns_r4 = [
            "design_id",
            "largest_singular_value",
            "smallest_singular_value",
            "condition_number",
            "parameter_correlation",
            "weak_direction_log10_mu",
            "weak_direction_cost",
        ]


        missing_columns_r4 = [
            column
            for column in required_columns_r4
            if column not in summary.columns
        ]


        if missing_columns_r4:

            raise KeyError(
                "SENSITIVITY_SUMMARY_R4 is missing required columns: "
                f"{missing_columns_r4}"
            )


        # ---------------------------------------------------------------------
        # Force design labels to strings before categorical ordering.
        # ---------------------------------------------------------------------

        summary["design_id"] = (
            summary["design_id"]
            .astype(str)
        )


        unexpected_designs_r4 = sorted(
            set(summary["design_id"])
            - set(DESIGN_ORDER_R4)
        )


        if unexpected_designs_r4:

            raise ValueError(
                "Unexpected design IDs were found: "
                f"{unexpected_designs_r4}"
            )


        # ---------------------------------------------------------------------
        # Enforce manuscript design order M1 -> M5.
        # ---------------------------------------------------------------------

        summary["design_id"] = pd.Categorical(
            summary["design_id"],
            categories=DESIGN_ORDER_R4,
            ordered=True,
        )


        summary = (
            summary
            .sort_values("design_id")
            .reset_index(drop=True)
        )


        if len(summary) != len(DESIGN_ORDER_R4):

            raise ValueError(
                "Expected exactly one sensitivity-summary row for each "
                f"design M1-M5, but found {len(summary)} rows."
            )


        summary["design_id"] = (
            summary["design_id"]
            .astype(str)
        )


        # ---------------------------------------------------------------------
        # Convert every plotted numerical column explicitly.
        # ---------------------------------------------------------------------

        numeric_columns_r4 = [
            "largest_singular_value",
            "smallest_singular_value",
            "condition_number",
            "parameter_correlation",
            "weak_direction_log10_mu",
            "weak_direction_cost",
        ]


        for column in numeric_columns_r4:

            summary[column] = pd.to_numeric(
                summary[column],
                errors="raise",
            )


        # ---------------------------------------------------------------------
        # Mathematical integrity checks.
        # ---------------------------------------------------------------------

        if (
            summary["largest_singular_value"] <= 0
        ).any():

            raise ValueError(
                "largest_singular_value must be strictly positive."
            )


        if (
            summary["smallest_singular_value"] <= 0
        ).any():

            raise ValueError(
                "smallest_singular_value must be strictly positive."
            )


        if (
            summary["condition_number"] <= 0
        ).any():

            raise ValueError(
                "condition_number must be strictly positive."
            )


        # =====================================================================
        # 6. PREPARE EXACT PLOTTING VALUES
        # =====================================================================

        design_labels_r4 = (
            summary["design_id"]
            .tolist()
        )


        x_r4 = np.arange(
            len(summary),
            dtype=float,
        )


        largest_sv_r4 = (
            summary[
                "largest_singular_value"
            ]
            .to_numpy(float)
        )


        smallest_sv_r4 = (
            summary[
                "smallest_singular_value"
            ]
            .to_numpy(float)
        )


        condition_number_r4 = (
            summary[
                "condition_number"
            ]
            .to_numpy(float)
        )


        abs_correlation_r4 = np.abs(
            summary[
                "parameter_correlation"
            ]
            .to_numpy(float)
        )


        abs_weak_mu_r4 = np.abs(
            summary[
                "weak_direction_log10_mu"
            ]
            .to_numpy(float)
        )


        abs_weak_cost_r4 = np.abs(
            summary[
                "weak_direction_cost"
            ]
            .to_numpy(float)
        )


        # =====================================================================
        # 7. EXPORT EXACT VALUES USED IN THE FIGURES
        # =====================================================================

        plot_values_r4 = pd.DataFrame(
            {
                "design_id":
                    design_labels_r4,

                "largest_singular_value":
                    largest_sv_r4,

                "smallest_singular_value":
                    smallest_sv_r4,

                "condition_number":
                    condition_number_r4,

                "absolute_parameter_correlation":
                    abs_correlation_r4,

                "absolute_weak_direction_log10_mu":
                    abs_weak_mu_r4,

                "absolute_weak_direction_cost":
                    abs_weak_cost_r4,
            }
        )


        sensitivity_plot_values_path_r4 = (
            TABLE_DIR_R4
            / "round4_identifiability_geometry_plot_values.csv"
        )


        plot_values_r4.to_csv(
            sensitivity_plot_values_path_r4,
            index=False,
        )


        # =====================================================================
        # 8. FINAL PUBLICATION STYLE
        # =====================================================================

        CELL22_PUBLICATION_RC_R4 = {

            "font.family":
                "sans-serif",

            "font.size":
                11.0,

            "axes.titlesize":
                12.3,

            "axes.labelsize":
                11.3,

            "xtick.labelsize":
                10.0,

            "ytick.labelsize":
                10.0,

            "legend.fontsize":
                9.4,

            "axes.linewidth":
                1.0,

            "axes.edgecolor":
                "black",

            "axes.facecolor":
                "white",

            "figure.facecolor":
                "white",

            "savefig.facecolor":
                "white",
        }


        def style_identifiability_axis_r4(axis):
            """
            Apply a consistent, restrained journal-style appearance.
            """

            axis.grid(
                which="major",
                alpha=0.15,
                linestyle="--",
                linewidth=0.65,
            )

            axis.set_axisbelow(
                True
            )

            set_panel_frame_r4(
                axis
            )


        # =====================================================================
        # 9. FIGURE A
        # IDENTIFIABILITY GEOMETRY OF NESTED SCHEDULE DESIGNS
        # =====================================================================

        with plt.rc_context(
            CELL22_PUBLICATION_RC_R4
        ):

            fig_geometry_r4, axes_r4 = plt.subplots(
                1,
                3,
                figsize=(13.8, 4.9),
            )


            # =================================================================
            # PANEL A — SENSITIVITY SPECTRUM
            # =================================================================

            axes_r4[0].plot(
                x_r4,
                largest_sv_r4,
                marker="o",
                linewidth=2.15,
                markersize=6.2,
                label=r"$\sigma_{\max}$",
            )


            axes_r4[0].plot(
                x_r4,
                smallest_sv_r4,
                marker="s",
                linewidth=2.15,
                markersize=6.2,
                label=r"$\sigma_{\min}$",
            )


            axes_r4[0].set_yscale(
                "log"
            )


            axes_r4[0].set_ylabel(
                "Normalized sensitivity singular value"
            )


            axes_r4[0].set_title(
                "Sensitivity spectrum",
                loc="left",
                fontweight="semibold",
                pad=8,
            )


            axes_r4[0].legend(
                loc="upper left",
                frameon=False,
                handlelength=2.2,
                handletextpad=0.55,
            )


            # =================================================================
            # PANEL B — SENSITIVITY CONDITIONING
            # =================================================================
            # A linear axis is deliberately used because kappa occupies a
            # relatively narrow numerical range. This is easier to read than
            # displaying values around 5-9 on a logarithmic axis.
            # =================================================================

            axes_r4[1].plot(
                x_r4,
                condition_number_r4,
                marker="D",
                linewidth=2.15,
                markersize=6.0,
            )


            axes_r4[1].set_ylabel(
                r"Condition number, $\kappa$"
            )


            axes_r4[1].set_title(
                "Sensitivity conditioning",
                loc="left",
                fontweight="semibold",
                pad=8,
            )


            axes_r4[1].yaxis.set_major_locator(
                MaxNLocator(
                    nbins=5
                )
            )


            # Add modest vertical padding without exaggerating the range.
            condition_min_r4 = float(
                np.min(
                    condition_number_r4
                )
            )


            condition_max_r4 = float(
                np.max(
                    condition_number_r4
                )
            )


            condition_padding_r4 = max(
                0.20,
                0.08
                * (
                    condition_max_r4
                    - condition_min_r4
                ),
            )


            axes_r4[1].set_ylim(
                max(
                    0.0,
                    condition_min_r4
                    - condition_padding_r4,
                ),
                condition_max_r4
                + condition_padding_r4,
            )


            # =================================================================
            # PANEL C — LOCAL PARAMETER CORRELATION
            # =================================================================

            axes_r4[2].plot(
                x_r4,
                abs_correlation_r4,
                marker="^",
                linewidth=2.15,
                markersize=6.5,
            )


            # Keep the complete correlation range visible.
            # This avoids visually exaggerating the relatively small decline.
            axes_r4[2].set_ylim(
                0.0,
                1.02,
            )


            axes_r4[2].set_ylabel(
                r"Local correlation, $|\rho(\log_{10}\mu,\,c_R)|$"
            )


            axes_r4[2].set_title(
                "Local parameter coupling",
                loc="left",
                fontweight="semibold",
                pad=8,
            )


            axes_r4[2].yaxis.set_major_locator(
                MaxNLocator(
                    nbins=6
                )
            )


            # =================================================================
            # COMMON PANEL FORMATTING
            # =================================================================

            for panel_label_r4, axis_r4 in zip(
                [
                    "A",
                    "B",
                    "C",
                ],
                axes_r4,
            ):

                axis_r4.set_xticks(
                    x_r4
                )


                axis_r4.set_xticklabels(
                    design_labels_r4
                )


                # M1--M5 already define the x-axis categories; a repeated
                # "Design" label is unnecessary at manuscript scale.
                axis_r4.set_xlabel("")


                style_identifiability_axis_r4(
                    axis_r4
                )


                axis_r4.text(
                    -0.12,
                    1.06,
                    panel_label_r4,
                    transform=axis_r4.transAxes,
                    fontsize=12.8,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )


            # =================================================================
            # FINAL MANUSCRIPT LAYOUT
            # =================================================================
            # No large figure-level title is used. The manuscript caption carries
            # the overall interpretation, leaving more space for the three panels.

            fig_geometry_r4.subplots_adjust(
                left=0.075,
                right=0.985,
                bottom=0.14,
                top=0.91,
                wspace=0.34,
            )


            save_figure_r4(
                fig_geometry_r4,
                "round4_identifiability_geometry_final",
            )


            plt.show()


        # =====================================================================
        # 10. FIGURE B
        # COMPOSITION OF THE WEAKEST IDENTIFIABLE DIRECTION
        # =====================================================================

        weak_vectors_abs_r4 = pd.DataFrame(
            {
                r"$|v_{\log_{10}\mu}|$":
                    abs_weak_mu_r4,

                r"$|v_{c_R}|$":
                    abs_weak_cost_r4,
            },
            index=design_labels_r4,
        )


        with plt.rc_context(
            CELL22_PUBLICATION_RC_R4
        ):

            fig_weak_r4, ax_weak_r4 = plt.subplots(
                figsize=(8.6, 5.0),
            )


            # -----------------------------------------------------------------
            # Sequential color scale:
            #
            # 0 = weak contribution
            # 1 = dominant contribution
            # -----------------------------------------------------------------

            image_r4 = ax_weak_r4.imshow(
                weak_vectors_abs_r4.to_numpy(
                    float
                ),
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )


            # =================================================================
            # AXIS LABELS
            # =================================================================

            ax_weak_r4.set_xticks(
                np.arange(
                    weak_vectors_abs_r4.shape[1]
                )
            )


            ax_weak_r4.set_xticklabels(
                weak_vectors_abs_r4.columns.tolist(),
                fontsize=11.0,
            )


            ax_weak_r4.set_yticks(
                np.arange(
                    weak_vectors_abs_r4.shape[0]
                )
            )


            ax_weak_r4.set_yticklabels(
                weak_vectors_abs_r4.index.tolist(),
                fontsize=10.5,
            )


            # The two parameter-component labels are self-explanatory.
            ax_weak_r4.set_xlabel("")


            ax_weak_r4.set_ylabel(
                "Design"
            )


            # =================================================================
            # NUMERICAL CELL ANNOTATIONS
            # =================================================================

            for row_index_r4 in range(
                weak_vectors_abs_r4.shape[0]
            ):

                for column_index_r4 in range(
                    weak_vectors_abs_r4.shape[1]
                ):

                    value_r4 = float(
                        weak_vectors_abs_r4.iloc[
                            row_index_r4,
                            column_index_r4,
                        ]
                    )


                    ax_weak_r4.text(
                        column_index_r4,
                        row_index_r4,
                        f"{value_r4:.3f}",
                        ha="center",
                        va="center",
                        fontsize=11.3,
                        fontweight="semibold",
                        color=(
                            "white"
                            if value_r4 >= 0.60
                            else "black"
                        ),
                    )


            # =================================================================
            # COLORBAR
            # =================================================================

            colorbar_r4 = fig_weak_r4.colorbar(
                image_r4,
                ax=ax_weak_r4,
                pad=0.025,
                fraction=0.055,
            )


            colorbar_r4.set_label(
                r"Absolute weak-direction component, $|v|$",
                fontsize=11.0,
            )


            colorbar_r4.ax.tick_params(
                labelsize=9.5
            )


            # =================================================================
            # TITLE AND FRAME
            # =================================================================

            ax_weak_r4.set_title(
                "Weakest identifiable parameter direction",
                fontsize=13.6,
                fontweight="semibold",
                pad=10,
            )


            set_panel_frame_r4(
                ax_weak_r4
            )


            fig_weak_r4.tight_layout()


            save_figure_r4(
                fig_weak_r4,
                "round4_weak_parameter_direction_final",
            )


            plt.show()


        # =====================================================================
        # 11. FINAL CONSOLE REPORT
        # =====================================================================

        print(
            "\n"
            + "=" * 82
        )


        print(
            "ROUND 4 CELL 22 FINAL FIGURES GENERATED SUCCESSFULLY"
        )


        print(
            "=" * 82
        )


        print(
            "\nSaved Figure A:"
        )


        print(
            " ",
            FIGURE_DIR_R4
            / "round4_identifiability_geometry_final",
        )


        print(
            "\nSaved Figure B:"
        )


        print(
            " ",
            FIGURE_DIR_R4
            / "round4_weak_parameter_direction_final",
        )


        print(
            "\nExact values used for plotting:"
        )


        print(
            " ",
            sensitivity_plot_values_path_r4,
        )


        print(
            "\nScientific interpretation checks:"
        )


        print(
            "  1. Larger sigma_min means stronger sensitivity in the "
            "weakest local parameter direction."
        )


        print(
            "  2. Smaller kappa means improved local sensitivity conditioning."
        )


        print(
            "  3. Parameter correlation is displayed as an absolute magnitude."
        )


        print(
            "  4. Weak-direction entries are absolute magnitudes because "
            "the global sign of an SVD singular vector is arbitrary."
        )


        print(
            "  5. Singular-value normalization follows the normalized "
            "sensitivity-matrix construction used in the preceding "
            "identifiability-analysis computation."
        )


        # =====================================================================
        # 12. OPTIONAL NUMERICAL DISPLAY FOR MANUSCRIPT CHECKING
        # =====================================================================

        display(
            plot_values_r4
        )


else:

    print(
        "Set RUN_IDENTIFIABILITY_GEOMETRY=True to generate the "
        "final sensitivity-geometry figures."
    )

# In[32]:


# =============================================================================
# ROUND 4 — TEMPORARY SWITCH FOR CELL 23
# Leave-one-schedule-out contribution experiment only
# =============================================================================
# PURPOSE
# -------
# Activate only the leave-one-schedule-out experiment.
#
# The already completed:
#   - 25-model primary experiment,
#   - Cell 18 summaries,
#   - Cell 19 figures,
#   - Cell 20 trajectory analysis,
#   - Cell 22 identifiability geometry
#
# are NOT rerun.
# =============================================================================

RUN_STRUCTURAL_SMOKE_TEST = False
RUN_SINGLE_DESIGN_DEMO = False
RUN_PRIMARY_MULTI_SCHEDULE_STUDY = False
RUN_MATCHED_ODE_BASELINES = False
RUN_IDENTIFIABILITY_GEOMETRY = False

RUN_LEAVE_ONE_SCHEDULE_OUT = True

RUN_PROFILE_SURFACES = False
RUN_PARAMETER_TRANSFER_TEST = False
RUN_PLOTTING_ONLY = False


CURRENT_EXPERIMENT_R4 = "leave_one_schedule_out"


EXPERIMENT_SWITCHES_R4 = {
    "structural_smoke_test":
        RUN_STRUCTURAL_SMOKE_TEST,

    "single_design_demo":
        RUN_SINGLE_DESIGN_DEMO,

    "primary_multi_schedule_study":
        RUN_PRIMARY_MULTI_SCHEDULE_STUDY,

    "matched_ode_baselines":
        RUN_MATCHED_ODE_BASELINES,

    "identifiability_geometry":
        RUN_IDENTIFIABILITY_GEOMETRY,

    "leave_one_schedule_out":
        RUN_LEAVE_ONE_SCHEDULE_OUT,

    "profile_surfaces":
        RUN_PROFILE_SURFACES,

    "parameter_transfer_test":
        RUN_PARAMETER_TRANSFER_TEST,

    "plotting_only":
        RUN_PLOTTING_ONLY,
}


active_experiments_r4 = [
    name
    for name, enabled in EXPERIMENT_SWITCHES_R4.items()
    if enabled
]


print("=" * 72)
print("ROUND 4 — CELL 23 CONFIGURATION")
print("=" * 72)

print(
    "Current experiment:",
    CURRENT_EXPERIMENT_R4,
)

print(
    "Active experiments:",
    active_experiments_r4,
)


# -----------------------------------------------------------------------------
# Safety assertions
# -----------------------------------------------------------------------------

assert RUN_LEAVE_ONE_SCHEDULE_OUT is True

assert RUN_PRIMARY_MULTI_SCHEDULE_STUDY is False

assert RUN_IDENTIFIABILITY_GEOMETRY is False

assert RUN_MATCHED_ODE_BASELINES is False

assert active_experiments_r4 == [
    "leave_one_schedule_out"
]


print()
print(
    "Leave-one-schedule-out configuration is active."
)

print(
    "The completed 25-model primary experiment will NOT be rerun."
)

print(
    "You may now run Cell 23."
)

# In[80]:


# ROUND 4 — CELL 23
# Leave-one-schedule-out shared-parameter inference
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# M4 is refitted after omitting each schedule in turn. A large deterioration when
# one schedule is removed indicates that the omitted environment supplies unique
# information rather than merely more observations.
#
# This is an optional expensive suite. It is resume-safe and should normally use
# three or five paired seeds after the primary experiment is complete.

LOO_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_metrics.csv"
)

PAIRED_CHANGES_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_changes.csv"
)

PAIRED_SUMMARY_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_summary.csv"
)

LOO_CONFIGURATIONS_R4 = {
    "FULL": (
        "untreated",
        "early_continuous",
        "late_continuous",
        "pulsed",
    ),
    "NO_UNTREATED": (
        "early_continuous",
        "late_continuous",
        "pulsed",
    ),
    "NO_EARLY": (
        "untreated",
        "late_continuous",
        "pulsed",
    ),
    "NO_LATE": (
        "untreated",
        "early_continuous",
        "pulsed",
    ),
    "NO_PULSED": (
        "untreated",
        "early_continuous",
        "late_continuous",
    ),
}

OMITTED_SCHEDULE_LABELS_R4 = {
    "NO_UNTREATED": "Untreated",
    "NO_EARLY": "Early continuous",
    "NO_LATE": "Late continuous",
    "NO_PULSED": "Pulsed",
}

OMITTED_ORDER_R4 = [
    "Untreated",
    "Early continuous",
    "Late continuous",
    "Pulsed",
]

LOO_ROWS_R4: List[Dict[str, Any]] = []

if RUN_LEAVE_ONE_SCHEDULE_OUT:
    if LOO_PATH_R4.exists():
        LOO_ROWS_R4 = pd.read_csv(
            LOO_PATH_R4
        ).to_dict("records")

    completed = {
        (
            str(row["configuration"]),
            str(row["seed_id"]),
        )
        for row in LOO_ROWS_R4
    }

    loo_seeds = (
        PRIMARY_SEED_IDS_R4
        if QUICK_MODE
        else [1, 2, 3]
    )

    for seed_number in loo_seeds:
        seed_id = f"seed{seed_number}"
        dataset_seed = (
            SEED
            + 10_000 * seed_number
        )
        training_seed = (
            SEED
            + 500_000
            + 10_000 * seed_number
        )

        full_bundle = build_design_bundle_r4(
            "M4",
            seed=dataset_seed,
            noise_level=0.0,
        )

        for configuration, schedule_names in (
            LOO_CONFIGURATIONS_R4.items()
        ):
            if (
                configuration,
                seed_id,
            ) in completed:
                continue

            bundle = {
                name: full_bundle[name]
                for name in schedule_names
            }

            label = (
                f"round4_LOO_{configuration}_{seed_id}"
            )

            model = AMRInverseModel(
                {
                    name: SCHEDULES_R4[name]
                    for name in bundle
                },
                model_kind="hard",
                width=NETWORK_WIDTH_R4,
                depth=NETWORK_DEPTH_R4,
                parameter_initialization=(
                    3.0e-4,
                    0.15,
                ),
            )

            result = train_shared_parameter_model_r4(
                model,
                bundle,
                design_id="M4",
                label=label,
                seed=training_seed,
            )

            metrics, _ = evaluate_design_r4(
                result,
                bundle,
            )

            metrics.update({
                "configuration": configuration,
                "seed_id": seed_id,
                "seed_number": seed_number,
                "included_schedules": ", ".join(
                    schedule_names
                ),
                "mu_error_percent": (
                    100.0
                    * metrics["relative_error_mu"]
                ),
                "cost_error_percent": (
                    100.0
                    * metrics["relative_error_cost"]
                ),
            })

            LOO_ROWS_R4.append(metrics)
            completed.add(
                (
                    configuration,
                    seed_id,
                )
            )

            pd.DataFrame(
                LOO_ROWS_R4
            ).to_csv(
                LOO_PATH_R4,
                index=False,
            )

            result.model.to("cpu")
            del model, bundle
            clear_gpu_memory()

    LOO_RESULTS_R4 = pd.DataFrame(
        LOO_ROWS_R4
    )
    display(LOO_RESULTS_R4)

else:
    if LOO_PATH_R4.exists():
        LOO_RESULTS_R4 = pd.read_csv(
            LOO_PATH_R4
        )
    else:
        LOO_RESULTS_R4 = pd.DataFrame()

    print(
        "Set RUN_LEAVE_ONE_SCHEDULE_OUT=True to run the optional "
        "schedule-contribution experiment."
    )

# =============================================================================
# PAIRED ANALYSIS AGAINST FULL M4 (same seed)
# =============================================================================

if not LOO_RESULTS_R4.empty:
    loo_metrics = [
        "rmse_resistant",
        "rmse_fraction",
        "mu_error_percent",
        "cost_error_percent",
    ]

    required_columns = [
        "configuration",
        "seed_id",
        *loo_metrics,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in LOO_RESULTS_R4.columns
    ]
    if missing_columns:
        raise KeyError(
            "LOO_RESULTS_R4 is missing required columns: "
            + ", ".join(missing_columns)
        )

    full_rows = (
        LOO_RESULTS_R4[
            LOO_RESULTS_R4["configuration"] == "FULL"
        ][["seed_id", *loo_metrics]]
        .rename(
            columns={
                metric: f"{metric}_full"
                for metric in loo_metrics
            }
        )
        .copy()
    )

    paired_rows = (
        LOO_RESULTS_R4[
            LOO_RESULTS_R4["configuration"] != "FULL"
        ]
        .merge(
            full_rows,
            on="seed_id",
            how="inner",
            validate="many_to_one",
        )
        .copy()
    )

    paired_rows["omitted_schedule"] = (
        paired_rows["configuration"]
        .map(OMITTED_SCHEDULE_LABELS_R4)
    )

    paired_change_columns = []
    for metric in loo_metrics:
        change_column = f"{metric}_change_percent"
        paired_rows[change_column] = (
            100.0
            * (
                paired_rows[metric]
                - paired_rows[f"{metric}_full"]
            )
            / paired_rows[f"{metric}_full"]
        )
        paired_change_columns.append(
            change_column
        )

    PAIRED_CHANGES_R4 = paired_rows[
        [
            "configuration",
            "omitted_schedule",
            "seed_id",
            "seed_number",
            *loo_metrics,
            *[f"{metric}_full" for metric in loo_metrics],
            *paired_change_columns,
        ]
    ].copy()

    PAIRED_CHANGES_R4.to_csv(
        PAIRED_CHANGES_PATH_R4,
        index=False,
    )

    PAIRED_SUMMARY_R4 = (
        PAIRED_CHANGES_R4
        .groupby(
            "omitted_schedule",
            observed=True,
        )[paired_change_columns]
        .mean()
        .reindex(OMITTED_ORDER_R4)
    )

    PAIRED_SUMMARY_R4.to_csv(
        PAIRED_SUMMARY_PATH_R4,
        index=True,
    )

    print("\nPaired leave-one-schedule-out changes relative to FULL M4:")
    display(PAIRED_CHANGES_R4)

    print("\nMean paired percentage change summary:")
    display(PAIRED_SUMMARY_R4)

    # =============================================================================
    # PUBLICATION SUMMARY FIGURE
    # =============================================================================

    plot_matrix = PAIRED_SUMMARY_R4.rename(
        columns={
            "rmse_resistant_change_percent": r"RMSE $R$",
            "rmse_fraction_change_percent": r"RMSE $q_R$",
            "mu_error_percent_change_percent": r"Error in $\mu$",
            "cost_error_percent_change_percent": r"Error in $c_R$",
        }
    )

    fig, ax = plt.subplots(
        figsize=(10.2, 5.8),
    )

    color_limit = 200.0

    image = ax.imshow(
        plot_matrix.to_numpy(float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
    )

    ax.set_xticks(
        np.arange(plot_matrix.shape[1])
    )
    ax.set_xticklabels(
        plot_matrix.columns,
        fontsize=12.5,
    )

    ax.set_yticks(
        np.arange(plot_matrix.shape[0])
    )
    ax.set_yticklabels(
        plot_matrix.index,
        fontsize=12.5,
    )

    ax.set_ylabel(
        "Omitted schedule",
        fontsize=13.5,
    )

    # No large in-figure title is used in the final manuscript version.
    # The overall interpretation belongs in the figure caption.

    # Thin white grid between heatmap cells
    ax.set_xticks(
        np.arange(-0.5, plot_matrix.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, plot_matrix.shape[0], 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.2,
        alpha=0.45,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Cell annotations
    for row_index in range(
        plot_matrix.shape[0]
    ):
        for column_index in range(
            plot_matrix.shape[1]
        ):
            value = plot_matrix.iloc[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if abs(value) >= 95.0
                else "black"
            )

            ax.text(
                column_index,
                row_index,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.025,
        ticks=np.arange(-200, 201, 50),
    )
    colorbar.set_label(
        "Change after schedule omission vs. full M4 (%)",
        fontsize=13,
    )
    colorbar.ax.tick_params(
        labelsize=11.5,
    )

    # Sign interpretation is intentionally moved to the manuscript caption:
    # positive values = worse inference after omission;
    # negative values = improvement relative to full M4.

    fig.subplots_adjust(
        left=0.18,
        right=0.92,
        top=0.965,
        bottom=0.12,
    )

    save_figure_r4(
        fig,
        "round4_leave_one_schedule_out_contribution_final",
    )
    plt.show()

    print("\nSaved seed-level table:")
    print(" ", PAIRED_CHANGES_PATH_R4)

    print("\nSaved mean paired-summary table:")
    print(" ", PAIRED_SUMMARY_PATH_R4)

    print("\nSaved publication figure:")
    print(
        " ",
        FIGURE_DIR_R4
        / "round4_leave_one_schedule_out_contribution_final",
    )

else:
    print(
        "No leave-one-schedule-out results are available yet. "
        "Run the suite or load round4_leave_one_schedule_out_metrics.csv first."
    )

# In[35]:


# =============================================================================
# ROUND 4 — CELL 23B
# Final publication heatmap for leave-one-schedule-out analysis
# NO TRAINING — uses completed Cell 23 CSV only
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -----------------------------------------------------------------------------
# 1. Load completed Cell 23 results
# -----------------------------------------------------------------------------
LOO_PATH_R4 = TABLE_DIR_R4 / "round4_leave_one_schedule_out_metrics.csv"

if LOO_PATH_R4.exists():
    LOO_RESULTS_R4 = pd.read_csv(LOO_PATH_R4)
else:
    raise FileNotFoundError(
        f"Completed Cell 23 results were not found:\n{LOO_PATH_R4}"
    )

print("Loaded completed Cell 23 results:", LOO_PATH_R4)

# -----------------------------------------------------------------------------
# 2. Validate expected configurations, seeds, and metrics
# -----------------------------------------------------------------------------
expected_configurations_r4 = [
    "FULL",
    "NO_UNTREATED",
    "NO_EARLY",
    "NO_LATE",
    "NO_PULSED",
]

expected_seeds_r4 = [
    "seed1",
    "seed2",
    "seed3",
]

required_metrics_r4 = [
    "rmse_resistant",
    "rmse_fraction",
    "mu_error_percent",
    "cost_error_percent",
]

required_columns_r4 = [
    "configuration",
    "seed_id",
] + required_metrics_r4

missing_columns_r4 = [
    column
    for column in required_columns_r4
    if column not in LOO_RESULTS_R4.columns
]

if missing_columns_r4:
    raise KeyError(
        f"Cell 23 CSV is missing required columns: {missing_columns_r4}"
    )

completed_pairs_r4 = set(
    zip(
        LOO_RESULTS_R4["configuration"].astype(str),
        LOO_RESULTS_R4["seed_id"].astype(str),
    )
)

expected_pairs_r4 = {
    (configuration, seed_id)
    for configuration in expected_configurations_r4
    for seed_id in expected_seeds_r4
}

missing_pairs_r4 = sorted(
    expected_pairs_r4 - completed_pairs_r4
)

if missing_pairs_r4:
    raise RuntimeError(
        "Cell 23 is incomplete. Missing pairs:\n"
        + "\n".join(map(str, missing_pairs_r4))
    )

print("Validated all 15/15 completed leave-one-schedule-out fits.")

# -----------------------------------------------------------------------------
# 3. Convert plotting metrics to numeric
# -----------------------------------------------------------------------------
loo_frame_r4 = LOO_RESULTS_R4.copy()

for metric in required_metrics_r4:
    loo_frame_r4[metric] = pd.to_numeric(
        loo_frame_r4[metric],
        errors="raise",
    )

# -----------------------------------------------------------------------------
# 4. Build paired FULL reference by seed
# -----------------------------------------------------------------------------
full_reference_r4 = (
    loo_frame_r4.loc[
        loo_frame_r4["configuration"] == "FULL",
        ["seed_id"] + required_metrics_r4,
    ]
    .drop_duplicates(subset=["seed_id"])
    .set_index("seed_id")
)

if set(full_reference_r4.index.astype(str)) != set(expected_seeds_r4):
    raise RuntimeError(
        "A unique FULL reference was not found for each of seed1, seed2, seed3."
    )

# -----------------------------------------------------------------------------
# 5. Calculate paired percentage changes relative to FULL
#
# Delta E (%) = 100 * (E_LOO - E_FULL) / E_FULL
#
# Positive -> omission worsened performance.
# Negative -> omission improved that metric.
# Zero     -> no change.
# -----------------------------------------------------------------------------
omission_order_r4 = [
    "NO_UNTREATED",
    "NO_EARLY",
    "NO_LATE",
    "NO_PULSED",
]

omission_display_r4 = {
    "NO_UNTREATED": "Omit untreated",
    "NO_EARLY": "Omit early continuous",
    "NO_LATE": "Omit late continuous",
    "NO_PULSED": "Omit pulsed",
}

metric_display_r4 = {
    "rmse_resistant": r"RMSE $R$",
    "rmse_fraction": r"RMSE $q_R$",
    "mu_error_percent": r"Error in $\mu$",
    "cost_error_percent": r"Error in $c_R$",
}

paired_rows_r4 = []

for configuration in omission_order_r4:

    configuration_frame = loo_frame_r4.loc[
        loo_frame_r4["configuration"] == configuration
    ].copy()

    for _, row in configuration_frame.iterrows():

        seed_id = str(
            row["seed_id"]
        )

        if seed_id not in full_reference_r4.index:
            raise RuntimeError(
                f"No paired FULL reference was found for {seed_id}."
            )

        for metric in required_metrics_r4:

            full_value = float(
                full_reference_r4.loc[
                    seed_id,
                    metric,
                ]
            )

            loo_value = float(
                row[metric]
            )

            if not np.isfinite(full_value) or full_value <= 0:
                raise ValueError(
                    f"Invalid FULL value for {seed_id}, {metric}: {full_value}"
                )

            paired_change_percent = (
                100.0
                * (loo_value - full_value)
                / full_value
            )

            paired_rows_r4.append(
                {
                    "configuration": configuration,
                    "seed_id": seed_id,
                    "metric": metric,
                    "full_value": full_value,
                    "loo_value": loo_value,
                    "paired_change_percent": paired_change_percent,
                }
            )

PAIRED_LOO_CHANGES_R4 = pd.DataFrame(
    paired_rows_r4
)

# -----------------------------------------------------------------------------
# 6. Mean ± sample SD across the three paired seeds
# -----------------------------------------------------------------------------
PAIRED_LOO_SUMMARY_R4 = (
    PAIRED_LOO_CHANGES_R4
    .groupby(
        [
            "configuration",
            "metric",
        ],
        observed=True,
    )[
        "paired_change_percent"
    ]
    .agg(
        mean_change_percent="mean",
        sd_change_percent="std",
        median_change_percent="median",
        n_pairs="count",
    )
    .reset_index()
)

# -----------------------------------------------------------------------------
# 7. Create mean and SD matrices in fixed manuscript order
# -----------------------------------------------------------------------------
mean_matrix_r4 = np.full(
    (
        len(omission_order_r4),
        len(required_metrics_r4),
    ),
    np.nan,
    dtype=float,
)

sd_matrix_r4 = np.full_like(
    mean_matrix_r4,
    np.nan,
)

for i, configuration in enumerate(
    omission_order_r4
):

    for j, metric in enumerate(
        required_metrics_r4
    ):

        subset = PAIRED_LOO_SUMMARY_R4.loc[
            (
                PAIRED_LOO_SUMMARY_R4["configuration"]
                == configuration
            )
            &
            (
                PAIRED_LOO_SUMMARY_R4["metric"]
                == metric
            )
        ]

        if len(subset) != 1:
            raise RuntimeError(
                f"Expected one summary row for {configuration}, {metric}; "
                f"found {len(subset)}."
            )

        mean_matrix_r4[i, j] = float(
            subset.iloc[0]["mean_change_percent"]
        )

        sd_matrix_r4[i, j] = float(
            subset.iloc[0]["sd_change_percent"]
        )

# -----------------------------------------------------------------------------
# 8. Save paired statistics
# -----------------------------------------------------------------------------
paired_change_path_r4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_changes.csv"
)

paired_summary_path_r4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_summary.csv"
)

PAIRED_LOO_CHANGES_R4.to_csv(
    paired_change_path_r4,
    index=False,
)

PAIRED_LOO_SUMMARY_R4.to_csv(
    paired_summary_path_r4,
    index=False,
)

# -----------------------------------------------------------------------------
# 9. Fallback figure saver
# -----------------------------------------------------------------------------
if "save_figure_r4" not in globals():

    def save_figure_r4(fig, stem):

        FIGURE_DIR_R4.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            FIGURE_DIR_R4 / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

# -----------------------------------------------------------------------------
# 10. Publication style
# -----------------------------------------------------------------------------
CELL23B_RC_R4 = {
    "font.family": "sans-serif",
    "font.size": 11.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

# -----------------------------------------------------------------------------
# 11. Symmetric diverging color scale centered at zero
# -----------------------------------------------------------------------------
maximum_absolute_change_r4 = float(
    np.nanmax(
        np.abs(
            mean_matrix_r4
        )
    )
)

color_limit_r4 = max(
    25.0,
    np.ceil(
        maximum_absolute_change_r4
        / 10.0
    ) * 10.0,
)

color_norm_r4 = TwoSlopeNorm(
    vmin=-color_limit_r4,
    vcenter=0.0,
    vmax=color_limit_r4,
)

# -----------------------------------------------------------------------------
# 12. Final publication heatmap
# -----------------------------------------------------------------------------
with plt.rc_context(
    CELL23B_RC_R4
):

    fig_r4, ax_r4 = plt.subplots(
        figsize=(10.6, 5.8)
    )

    image_r4 = ax_r4.imshow(
        mean_matrix_r4,
        aspect="auto",
        cmap="RdBu_r",
        norm=color_norm_r4,
    )

    ax_r4.set_xticks(
        np.arange(
            len(required_metrics_r4)
        )
    )

    ax_r4.set_xticklabels(
        [
            metric_display_r4[metric]
            for metric in required_metrics_r4
        ]
    )

    ax_r4.set_yticks(
        np.arange(
            len(omission_order_r4)
        )
    )

    ax_r4.set_yticklabels(
        [
            omission_display_r4[configuration]
            for configuration in omission_order_r4
        ]
    )

    # -------------------------------------------------------------------------
    # Mean ± sample SD annotations
    # -------------------------------------------------------------------------
    for i in range(
        mean_matrix_r4.shape[0]
    ):

        for j in range(
            mean_matrix_r4.shape[1]
        ):

            mean_value = float(
                mean_matrix_r4[i, j]
            )

            sd_value = float(
                sd_matrix_r4[i, j]
            )

            text_color = (
                "white"
                if abs(mean_value)
                >= 0.50 * color_limit_r4
                else "black"
            )

            ax_r4.text(
                j,
                i,
                (
                    f"{mean_value:+.1f}%\n"
                    f"± {sd_value:.1f}%"
                ),
                ha="center",
                va="center",
                fontsize=10.3,
                fontweight="semibold",
                color=text_color,
                linespacing=1.15,
            )

    colorbar_r4 = fig_r4.colorbar(
        image_r4,
        ax=ax_r4,
        pad=0.025,
        fraction=0.055,
    )

    colorbar_r4.set_label(
        "Paired change relative to full M4 (%)",
        fontsize=11.0,
    )

    ax_r4.set_title(
        "Contribution of individual treatment schedules to M4 inference",
        fontsize=14.5,
        fontweight="bold",
        pad=13,
    )

    ax_r4.set_xlabel(
        "Evaluation quantity"
    )

    ax_r4.set_ylabel(
        "Removed schedule"
    )

    for spine in ax_r4.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig_r4.tight_layout()

    save_figure_r4(
        fig_r4,
        "round4_leave_one_schedule_out_paired_heatmap_final",
    )

    plt.show()

# -----------------------------------------------------------------------------
# 13. Compact manuscript-check table
# -----------------------------------------------------------------------------
publication_rows_r4 = []

for i, configuration in enumerate(
    omission_order_r4
):

    output_row = {
        "Removed schedule":
            omission_display_r4[
                configuration
            ].replace(
                "Omit ",
                "",
            )
    }

    for j, metric in enumerate(
        required_metrics_r4
    ):

        output_row[
            metric_display_r4[
                metric
            ]
        ] = (
            f"{mean_matrix_r4[i, j]:+.1f} ± "
            f"{sd_matrix_r4[i, j]:.1f}%"
        )

    publication_rows_r4.append(
        output_row
    )

LOO_PUBLICATION_TABLE_R4 = pd.DataFrame(
    publication_rows_r4
)

print(
    "\nPaired leave-one-schedule-out effects "
    "(mean ± sample SD across three seeds):"
)

display(
    LOO_PUBLICATION_TABLE_R4
)

# -----------------------------------------------------------------------------
# 14. Final confirmation
# -----------------------------------------------------------------------------
print(
    "\n" + "=" * 84
)

print(
    "ROUND 4 CELL 23B COMPLETED SUCCESSFULLY"
)

print(
    "=" * 84
)

print(
    "\nNO MODELS WERE RETRAINED."
)

print(
    "\nSource results:"
)

print(
    " ",
    LOO_PATH_R4,
)

print(
    "\nSaved final figure:"
)

print(
    " ",
    FIGURE_DIR_R4
    / "round4_leave_one_schedule_out_paired_heatmap_final",
)

print(
    "\nSaved paired seed-level changes:"
)

print(
    " ",
    paired_change_path_r4,
)

print(
    "\nSaved paired mean ± SD summary:"
)

print(
    " ",
    paired_summary_path_r4,
)

print(
    "\nInterpretation:"
)

print(
    "  Positive percentage -> omission worsened performance."
)

print(
    "  Negative percentage -> omission improved that metric."
)

print(
    "  Zero percentage -> little or no change relative to paired FULL M4."
)

# In[78]:


# ROUND 4 — CELL 23C
# FINAL PUBLICATION REFINEMENT OF LEAVE-ONE-SCHEDULE-OUT ANALYSIS
# =============================================================================
# WHAT THIS CELL DOES
# -------------------
# NO TRAINING.
#
# This cell uses the paired seed-level changes already calculated by Cell 23B.
#
# It creates:
#
# FIGURE A
# --------
# A clean diverging heatmap of the MEAN paired percentage change relative to
# the full M4 design.
#
# The heatmap intentionally does NOT place SD values inside the cells because
# doing so makes the figure visually crowded.
#
# FIGURE B
# --------
# A four-panel paired-seed variability figure showing:
#
#     seed1
#     seed2
#     seed3
#     mean ± sample SD
#
# for every leave-one-schedule-out experiment.
#
# This second figure is especially important because mutation-probability
# recovery exhibits large between-seed variability.
#
# INTERPRETATION
# --------------
# Positive percentage:
#     omission increased error -> removed schedule contained useful in# ROUND 4 — CELL 23
# Leave-one-schedule-out shared-parameter inference
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# M4 is refitted after omitting each schedule in turn. A large deterioration when
# one schedule is removed indicates that the omitted environment supplies unique
# information rather than merely more observations.
#
# This is an optional expensive suite. It is resume-safe and should normally use
# three or five paired seeds after the primary experiment is complete.

LOO_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_metrics.csv"
)

PAIRED_CHANGES_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_changes.csv"
)

PAIRED_SUMMARY_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_summary.csv"
)

LOO_CONFIGURATIONS_R4 = {
    "FULL": (
        "untreated",
        "early_continuous",
        "late_continuous",
        "pulsed",
    ),
    "NO_UNTREATED": (
        "early_continuous",
        "late_continuous",
        "pulsed",
    ),
    "NO_EARLY": (
        "untreated",
        "late_continuous",
        "pulsed",
    ),
    "NO_LATE": (
        "untreated",
        "early_continuous",
        "pulsed",
    ),
    "NO_PULSED": (
        "untreated",
        "early_continuous",
        "late_continuous",
    ),
}

OMITTED_SCHEDULE_LABELS_R4 = {
    "NO_UNTREATED": "Untreated",
    "NO_EARLY": "Early continuous",
    "NO_LATE": "Late continuous",
    "NO_PULSED": "Pulsed",
}

OMITTED_ORDER_R4 = [
    "Untreated",
    "Early continuous",
    "Late continuous",
    "Pulsed",
]

LOO_ROWS_R4: List[Dict[str, Any]] = []

if RUN_LEAVE_ONE_SCHEDULE_OUT:
    if LOO_PATH_R4.exists():
        LOO_ROWS_R4 = pd.read_csv(
            LOO_PATH_R4
        ).to_dict("records")

    completed = {
        (
            str(row["configuration"]),
            str(row["seed_id"]),
        )
        for row in LOO_ROWS_R4
    }

    loo_seeds = (
        PRIMARY_SEED_IDS_R4
        if QUICK_MODE
        else [1, 2, 3]
    )

    for seed_number in loo_seeds:
        seed_id = f"seed{seed_number}"
        dataset_seed = (
            SEED
            + 10_000 * seed_number
        )
        training_seed = (
            SEED
            + 500_000
            + 10_000 * seed_number
        )

        full_bundle = build_design_bundle_r4(
            "M4",
            seed=dataset_seed,
            noise_level=0.0,
        )

        for configuration, schedule_names in (
            LOO_CONFIGURATIONS_R4.items()
        ):
            if (
                configuration,
                seed_id,
            ) in completed:
                continue

            bundle = {
                name: full_bundle[name]
                for name in schedule_names
            }

            label = (
                f"round4_LOO_{configuration}_{seed_id}"
            )

            model = AMRInverseModel(
                {
                    name: SCHEDULES_R4[name]
                    for name in bundle
                },
                model_kind="hard",
                width=NETWORK_WIDTH_R4,
                depth=NETWORK_DEPTH_R4,
                parameter_initialization=(
                    3.0e-4,
                    0.15,
                ),
            )

            result = train_shared_parameter_model_r4(
                model,
                bundle,
                design_id="M4",
                label=label,
                seed=training_seed,
            )

            metrics, _ = evaluate_design_r4(
                result,
                bundle,
            )

            metrics.update({
                "configuration": configuration,
                "seed_id": seed_id,
                "seed_number": seed_number,
                "included_schedules": ", ".join(
                    schedule_names
                ),
                "mu_error_percent": (
                    100.0
                    * metrics["relative_error_mu"]
                ),
                "cost_error_percent": (
                    100.0
                    * metrics["relative_error_cost"]
                ),
            })

            LOO_ROWS_R4.append(metrics)
            completed.add(
                (
                    configuration,
                    seed_id,
                )
            )

            pd.DataFrame(
                LOO_ROWS_R4
            ).to_csv(
                LOO_PATH_R4,
                index=False,
            )

            result.model.to("cpu")
            del model, bundle
            clear_gpu_memory()

    LOO_RESULTS_R4 = pd.DataFrame(
        LOO_ROWS_R4
    )
    display(LOO_RESULTS_R4)

else:
    if LOO_PATH_R4.exists():
        LOO_RESULTS_R4 = pd.read_csv(
            LOO_PATH_R4
        )
    else:
        LOO_RESULTS_R4 = pd.DataFrame()

    print(
        "Set RUN_LEAVE_ONE_SCHEDULE_OUT=True to run the optional "
        "schedule-contribution experiment."
    )

# =============================================================================
# PAIRED ANALYSIS AGAINST FULL M4 (same seed)
# =============================================================================

if not LOO_RESULTS_R4.empty:
    loo_metrics = [
        "rmse_resistant",
        "rmse_fraction",
        "mu_error_percent",
        "cost_error_percent",
    ]

    required_columns = [
        "configuration",
        "seed_id",
        *loo_metrics,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in LOO_RESULTS_R4.columns
    ]
    if missing_columns:
        raise KeyError(
            "LOO_RESULTS_R4 is missing required columns: "
            + ", ".join(missing_columns)
        )

    full_rows = (
        LOO_RESULTS_R4[
            LOO_RESULTS_R4["configuration"] == "FULL"
        ][["seed_id", *loo_metrics]]
        .rename(
            columns={
                metric: f"{metric}_full"
                for metric in loo_metrics
            }
        )
        .copy()
    )

    paired_rows = (
        LOO_RESULTS_R4[
            LOO_RESULTS_R4["configuration"] != "FULL"
        ]
        .merge(
            full_rows,
            on="seed_id",
            how="inner",
            validate="many_to_one",
        )
        .copy()
    )

    paired_rows["omitted_schedule"] = (
        paired_rows["configuration"]
        .map(OMITTED_SCHEDULE_LABELS_R4)
    )

    paired_change_columns = []
    for metric in loo_metrics:
        change_column = f"{metric}_change_percent"
        paired_rows[change_column] = (
            100.0
            * (
                paired_rows[metric]
                - paired_rows[f"{metric}_full"]
            )
            / paired_rows[f"{metric}_full"]
        )
        paired_change_columns.append(
            change_column
        )

    PAIRED_CHANGES_R4 = paired_rows[
        [
            "configuration",
            "omitted_schedule",
            "seed_id",
            "seed_number",
            *loo_metrics,
            *[f"{metric}_full" for metric in loo_metrics],
            *paired_change_columns,
        ]
    ].copy()

    PAIRED_CHANGES_R4.to_csv(
        PAIRED_CHANGES_PATH_R4,
        index=False,
    )

    PAIRED_SUMMARY_R4 = (
        PAIRED_CHANGES_R4
        .groupby(
            "omitted_schedule",
            observed=True,
        )[paired_change_columns]
        .mean()
        .reindex(OMITTED_ORDER_R4)
    )

    PAIRED_SUMMARY_R4.to_csv(
        PAIRED_SUMMARY_PATH_R4,
        index=True,
    )

    print("\nPaired leave-one-schedule-out changes relative to FULL M4:")
    display(PAIRED_CHANGES_R4)

    print("\nMean paired percentage change summary:")
    display(PAIRED_SUMMARY_R4)

    # =============================================================================
    # PUBLICATION SUMMARY FIGURE
    # =============================================================================

    plot_matrix = PAIRED_SUMMARY_R4.rename(
        columns={
            "rmse_resistant_change_percent": r"RMSE $R$",
            "rmse_fraction_change_percent": r"RMSE $q_R$",
            "mu_error_percent_change_percent": r"Error in $\mu$",
            "cost_error_percent_change_percent": r"Error in $c_R$",
        }
    )

    fig, ax = plt.subplots(
        figsize=(10.2, 5.8),
    )

    color_limit = 200.0

    image = ax.imshow(
        plot_matrix.to_numpy(float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
    )

    ax.set_xticks(
        np.arange(plot_matrix.shape[1])
    )
    ax.set_xticklabels(
        plot_matrix.columns,
        fontsize=12.5,
    )

    ax.set_yticks(
        np.arange(plot_matrix.shape[0])
    )
    ax.set_yticklabels(
        plot_matrix.index,
        fontsize=12.5,
    )

    ax.set_ylabel(
        "Omitted schedule",
        fontsize=13.5,
    )

    ax.set_title(
        "Schedule-specific contribution to multi-schedule inference",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )

    # Thin white grid between heatmap cells
    ax.set_xticks(
        np.arange(-0.5, plot_matrix.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, plot_matrix.shape[0], 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.2,
        alpha=0.45,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Cell annotations
    for row_index in range(
        plot_matrix.shape[0]
    ):
        for column_index in range(
            plot_matrix.shape[1]
        ):
            value = plot_matrix.iloc[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if abs(value) >= 95.0
                else "black"
            )

            ax.text(
                column_index,
                row_index,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.025,
        ticks=np.arange(-200, 201, 50),
    )
    colorbar.set_label(
        "Change after schedule omission vs. full M4 (%)",
        fontsize=13,
    )
    colorbar.ax.tick_params(
        labelsize=11.5,
    )

    fig.text(
        0.5,
        0.035,
        (
            "Positive values indicate worse inference after omission; "
            "negative values indicate improvement relative to the full M4 design."
        ),
        ha="center",
        va="center",
        fontsize=11.2,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.92,
        top=0.87,
        bottom=0.18,
    )

    save_figure_r4(
        fig,
        "round4_leave_one_schedule_out_contribution_final",
    )
    plt.show()

    print("\nSaved seed-level table:")
    print(" ", PAIRED_CHANGES_PATH_R4)

    print("\nSaved mean paired-summary table:")
    print(" ", PAIRED_SUMMARY_PATH_R4)

    print("\nSaved publication figure:")
    print(
        " ",
        FIGURE_DIR_R4
        / "round4_leave_one_schedule_out_contribution_final",
    )

else:
    print(
        "No leave-one-schedule-out results are available yet. "
        "Run the suite or load round4_leave_one_schedule_out_metrics.csv first."
    )

# ---------------------------------------------------------------------------
# [Packaging note: this line was originally merged with the closing
#  parenthesis above in the source notebook, which broke Python parsing.
#  It has been restored here as a standalone comment. No code logic was
#  changed -- see CHANGELOG.md.]
# ...formation.
#
# Negative percentage:
#     omission reduced that particular error metric.
#
# Zero:
#     little change relative to the paired full-M4 fit.
#
# IMPORTANT
# ---------
# Negative effects should not automatically be interpreted as evidence that a
# schedule is biologically detrimental. They may reflect redundancy or a
# parameter-specific optimization/identifiability trade-off.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 2. LOAD THE PAIRED CHANGES GENERATED BY CELL 23B
# =============================================================================

paired_change_path_r4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_changes.csv"
)


if paired_change_path_r4.exists():

    PAIRED_LOO_CHANGES_R4 = pd.read_csv(
        paired_change_path_r4
    )

    print(
        "Loaded paired Cell 23 results from:",
        paired_change_path_r4,
    )

else:

    raise FileNotFoundError(
        "The paired Cell 23B results were not found:\n"
        f"{paired_change_path_r4}\n\n"
        "Run Cell 23B first. No retraining is required."
    )


# =============================================================================
# 3. VALIDATE THE PAIRED RESULTS
# =============================================================================

required_columns_r4 = [
    "configuration",
    "seed_id",
    "metric",
    "paired_change_percent",
]


missing_columns_r4 = [
    column
    for column in required_columns_r4
    if column not in PAIRED_LOO_CHANGES_R4.columns
]


if missing_columns_r4:

    raise KeyError(
        "Paired LOO table is missing columns: "
        f"{missing_columns_r4}"
    )


PAIRED_LOO_CHANGES_R4[
    "paired_change_percent"
] = pd.to_numeric(
    PAIRED_LOO_CHANGES_R4[
        "paired_change_percent"
    ],
    errors="raise",
)


# =============================================================================
# 4. FIXED MANUSCRIPT ORDER
# =============================================================================

omission_order_r4 = [
    "NO_UNTREATED",
    "NO_EARLY",
    "NO_LATE",
    "NO_PULSED",
]


omission_display_r4 = {
    "NO_UNTREATED":
        "Untreated",

    "NO_EARLY":
        "Early continuous",

    "NO_LATE":
        "Late continuous",

    "NO_PULSED":
        "Pulsed",
}


metric_order_r4 = [
    "rmse_resistant",
    "rmse_fraction",
    "mu_error_percent",
    "cost_error_percent",
]


metric_heatmap_labels_r4 = {
    "rmse_resistant":
        r"RMSE $R$",

    "rmse_fraction":
        r"RMSE $q_R$",

    "mu_error_percent":
        r"Error in $\mu$",

    "cost_error_percent":
        r"Error in $c_R$",
}


metric_panel_titles_r4 = {
    "rmse_resistant":
        r"Resistant-state reconstruction, $R(t)$",

    "rmse_fraction":
        r"Phenotype-composition reconstruction, $q_R(t)$",

    "mu_error_percent":
        r"Mutation-probability identification, $\mu$",

    "cost_error_percent":
        r"Resistance-fitness-cost identification, $c_R$",
}


# =============================================================================
# 5. COMPUTE MEAN ± SAMPLE SD
# =============================================================================

LOO_FINAL_SUMMARY_R4 = (
    PAIRED_LOO_CHANGES_R4
    .groupby(
        [
            "configuration",
            "metric",
        ],
        observed=True,
    )[
        "paired_change_percent"
    ]
    .agg(
        mean_change_percent="mean",
        sd_change_percent="std",
        median_change_percent="median",
        n_pairs="count",
    )
    .reset_index()
)


# =============================================================================
# 6. BUILD MATRICES FOR THE HEATMAP
# =============================================================================

mean_matrix_r4 = np.full(
    (
        len(omission_order_r4),
        len(metric_order_r4),
    ),
    np.nan,
    dtype=float,
)


sd_matrix_r4 = np.full_like(
    mean_matrix_r4,
    np.nan,
)


for row_index, configuration in enumerate(
    omission_order_r4
):

    for column_index, metric in enumerate(
        metric_order_r4
    ):

        subset = LOO_FINAL_SUMMARY_R4.loc[
            (
                LOO_FINAL_SUMMARY_R4[
                    "configuration"
                ] == configuration
            )
            &
            (
                LOO_FINAL_SUMMARY_R4[
                    "metric"
                ] == metric
            )
        ]


        if len(subset) != 1:

            raise RuntimeError(
                "Expected exactly one summary result for "
                f"{configuration}, {metric}."
            )


        mean_matrix_r4[
            row_index,
            column_index,
        ] = float(
            subset.iloc[0][
                "mean_change_percent"
            ]
        )


        sd_matrix_r4[
            row_index,
            column_index,
        ] = float(
            subset.iloc[0][
                "sd_change_percent"
            ]
        )


# =============================================================================
# 7. FIGURE SAVING FALLBACK
# =============================================================================

if "save_figure_r4" not in globals():

    def save_figure_r4(
        figure,
        stem,
    ):

        FIGURE_DIR_R4.mkdir(
            parents=True,
            exist_ok=True,
        )


        figure.savefig(
            FIGURE_DIR_R4
            / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )


        figure.savefig(
            FIGURE_DIR_R4
            / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )


# =============================================================================
# 8. PUBLICATION STYLE
# =============================================================================

CELL23C_RC_R4 = {

    "font.family":
        "sans-serif",

    "font.size":
        11.0,

    "axes.titlesize":
        12.0,

    "axes.labelsize":
        11.0,

    "xtick.labelsize":
        10.0,

    "ytick.labelsize":
        10.0,

    "legend.fontsize":
        9.0,

    "axes.linewidth":
        1.0,

    "axes.edgecolor":
        "black",

    "axes.facecolor":
        "white",

    "figure.facecolor":
        "white",

    "savefig.facecolor":
        "white",
}


# =============================================================================
# 9. FIGURE A — FINAL CLEAN CONTRIBUTION HEATMAP
# =============================================================================
# The color scale is symmetric around zero:
#
#     blue  -> omission reduced the metric
#     white -> little/no effect
#     red   -> omission increased the metric
# =============================================================================

maximum_effect_r4 = float(
    np.nanmax(
        np.abs(
            mean_matrix_r4
        )
    )
)


# Round upward to a visually clean 25% interval.
color_limit_r4 = max(
    50.0,
    np.ceil(
        maximum_effect_r4
        / 25.0
    )
    * 25.0,
)


heatmap_norm_r4 = TwoSlopeNorm(
    vmin=-color_limit_r4,
    vcenter=0.0,
    vmax=color_limit_r4,
)


with plt.rc_context(
    CELL23C_RC_R4
):

    fig_heat_r4, ax_heat_r4 = plt.subplots(
        figsize=(9.7, 5.3),
    )


    image_r4 = ax_heat_r4.imshow(
        mean_matrix_r4,
        aspect="auto",
        cmap="RdBu_r",
        norm=heatmap_norm_r4,
    )


    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------

    ax_heat_r4.set_xticks(
        np.arange(
            len(metric_order_r4)
        )
    )


    ax_heat_r4.set_xticklabels(
        [
            metric_heatmap_labels_r4[
                metric
            ]
            for metric in metric_order_r4
        ]
    )


    ax_heat_r4.set_yticks(
        np.arange(
            len(omission_order_r4)
        )
    )


    ax_heat_r4.set_yticklabels(
        [
            omission_display_r4[
                configuration
            ]
            for configuration in omission_order_r4
        ]
    )


    # -------------------------------------------------------------------------
    # Add only the MEAN effect to each cell.
    #
    # SD is deliberately moved to Figure B.
    # -------------------------------------------------------------------------

    for row_index in range(
        mean_matrix_r4.shape[0]
    ):

        for column_index in range(
            mean_matrix_r4.shape[1]
        ):

            value = float(
                mean_matrix_r4[
                    row_index,
                    column_index,
                ]
            )


            text_color = (
                "white"
                if abs(value)
                >= 0.48 * color_limit_r4
                else "black"
            )


            # -------------------------------------------------------------
            # Publication-quality annotation formatting
            # -------------------------------------------------------------
            # Preserve one decimal place for small effects so values such as
            # -0.5% are never displayed as the misleading "-0%".
            # Larger effects remain integer percentages for a clean heatmap.
            if abs(value) < 10.0:
                value_label = f"{value:+.1f}%"
            else:
                value_label = f"{value:+.0f}%"

            ax_heat_r4.text(
                column_index,
                row_index,
                value_label,
                ha="center",
                va="center",
                fontsize=11.7,
                fontweight="bold",
                color=text_color,
            )


    # -------------------------------------------------------------------------
    # Add thin boundaries between cells for cleaner separation.
    # -------------------------------------------------------------------------

    ax_heat_r4.set_xticks(
        np.arange(
            -0.5,
            len(metric_order_r4),
            1.0,
        ),
        minor=True,
    )


    ax_heat_r4.set_yticks(
        np.arange(
            -0.5,
            len(omission_order_r4),
            1.0,
        ),
        minor=True,
    )


    ax_heat_r4.grid(
        which="minor",
        linewidth=1.0,
        alpha=0.45,
    )


    ax_heat_r4.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    # -------------------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------------------

    colorbar_r4 = fig_heat_r4.colorbar(
        image_r4,
        ax=ax_heat_r4,
        pad=0.025,
        fraction=0.055,
    )


    colorbar_r4.set_label(
        "Paired change relative to full M4 (%)"
    )


    # -------------------------------------------------------------------------
    # Labels and title
    # -------------------------------------------------------------------------

    ax_heat_r4.set_ylabel(
        "Omitted schedule"
    )


    ax_heat_r4.set_xlabel(
        "Evaluation quantity"
    )


    ax_heat_r4.set_title(
        "Schedule-specific contribution to multi-schedule inference",
        fontsize=14.5,
        fontweight="bold",
        pad=12,
    )


    for spine in ax_heat_r4.spines.values():

        spine.set_visible(
            True
        )

        spine.set_color(
            "black"
        )

        spine.set_linewidth(
            1.0
        )


    fig_heat_r4.tight_layout()


    save_figure_r4(
        fig_heat_r4,
        "round4_leave_one_schedule_out_heatmap_final",
    )


    plt.show()


# =============================================================================
# 10. FIGURE B — SEED-LEVEL VARIABILITY
# =============================================================================
# Why this figure matters:
#
# The mutation-probability effects have very large SD values. Showing the
# individual paired seeds prevents a reviewer from mistaking a large mean for
# a stable effect.
# =============================================================================

seed_order_r4 = [
    "seed1",
    "seed2",
    "seed3",
]


seed_markers_r4 = {
    "seed1":
        "o",

    "seed2":
        "s",

    "seed3":
        "^",
}


# Small fixed offsets prevent seed markers from sitting directly on top of one
# another without changing the actual schedule category.
seed_offsets_r4 = {
    "seed1":
        -0.10,

    "seed2":
        0.00,

    "seed3":
        0.10,
}


panel_labels_r4 = [
    "A",
    "B",
    "C",
    "D",
]


with plt.rc_context(
    CELL23C_RC_R4
):

    fig_seed_r4, axes_seed_r4 = plt.subplots(
        2,
        2,
        figsize=(11.7, 8.1),
    )


    axes_seed_r4 = axes_seed_r4.flatten()


    x_r4 = np.arange(
        len(omission_order_r4),
        dtype=float,
    )


    for panel_label, axis, metric in zip(
        panel_labels_r4,
        axes_seed_r4,
        metric_order_r4,
    ):

        metric_frame = (
            PAIRED_LOO_CHANGES_R4[
                PAIRED_LOO_CHANGES_R4[
                    "metric"
                ] == metric
            ]
            .copy()
        )


        # ---------------------------------------------------------------------
        # Individual paired seeds
        # ---------------------------------------------------------------------

        for seed_id in seed_order_r4:

            seed_values = []


            for configuration in omission_order_r4:

                subset = metric_frame.loc[
                    (
                        metric_frame[
                            "configuration"
                        ] == configuration
                    )
                    &
                    (
                        metric_frame[
                            "seed_id"
                        ].astype(str)
                        == seed_id
                    ),
                    "paired_change_percent",
                ]


                if len(subset) != 1:

                    raise RuntimeError(
                        "Expected one seed-level value for "
                        f"{metric}, {configuration}, {seed_id}."
                    )


                seed_values.append(
                    float(
                        subset.iloc[0]
                    )
                )


            axis.scatter(
                x_r4
                + seed_offsets_r4[
                    seed_id
                ],
                seed_values,
                marker=seed_markers_r4[
                    seed_id
                ],
                s=48,
                alpha=0.72,
                label=seed_id,
                zorder=3,
            )


        # ---------------------------------------------------------------------
        # Mean ± sample SD
        # ---------------------------------------------------------------------

        metric_means_r4 = []


        metric_sds_r4 = []


        for configuration in omission_order_r4:

            subset = LOO_FINAL_SUMMARY_R4.loc[
                (
                    LOO_FINAL_SUMMARY_R4[
                        "configuration"
                    ] == configuration
                )
                &
                (
                    LOO_FINAL_SUMMARY_R4[
                        "metric"
                    ] == metric
                )
            ]


            metric_means_r4.append(
                float(
                    subset.iloc[0][
                        "mean_change_percent"
                    ]
                )
            )


            metric_sds_r4.append(
                float(
                    subset.iloc[0][
                        "sd_change_percent"
                    ]
                )
            )


        axis.errorbar(
            x_r4,
            metric_means_r4,
            yerr=metric_sds_r4,
            fmt="D",
            markersize=6.5,
            linewidth=1.7,
            capsize=4.0,
            color="black",
            label="Mean ± SD",
            zorder=5,
        )


        # ---------------------------------------------------------------------
        # Neutral line
        # ---------------------------------------------------------------------

        axis.axhline(
            0.0,
            linestyle="--",
            linewidth=1.1,
            color="black",
            alpha=0.65,
            zorder=1,
        )


        # ---------------------------------------------------------------------
        # Formatting
        # ---------------------------------------------------------------------

        axis.set_xticks(
            x_r4
        )


        axis.set_xticklabels(
            [
                omission_display_r4[
                    configuration
                ]
                for configuration
                in omission_order_r4
            ],
            rotation=0,
        )


        axis.set_title(
            metric_panel_titles_r4[
                metric
            ],
            loc="left",
            fontweight="semibold",
            pad=8,
        )


        axis.set_ylabel(
            "Paired change relative to full M4 (%)"
        )


        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.20,
        )


        axis.set_axisbelow(
            True
        )


        axis.yaxis.set_major_locator(
            MaxNLocator(
                nbins=6
            )
        )


        for spine in axis.spines.values():

            spine.set_visible(
                True
            )

            spine.set_color(
                "black"
            )

            spine.set_linewidth(
                1.0
            )


        axis.text(
            -0.10,
            1.07,
            panel_label,
            transform=axis.transAxes,
            fontsize=12.5,
            fontweight="bold",
            ha="left",
            va="top",
        )


    # -------------------------------------------------------------------------
    # One shared legend outside the plotting panels.
    # -------------------------------------------------------------------------

    legend_handles_r4 = [
        Line2D(
            [0],
            [0],
            marker=seed_markers_r4[
                seed_id
            ],
            linestyle="None",
            markersize=6.5,
            label=seed_id,
        )
        for seed_id in seed_order_r4
    ]


    legend_handles_r4.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            linestyle="-",
            markersize=6.5,
            label="Mean ± SD",
        )
    )


    fig_seed_r4.legend(
        handles=legend_handles_r4,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.94,
        ),
        ncol=4,
        frameon=False,
    )


    fig_seed_r4.suptitle(
        "Paired-seed stability of schedule-specific contributions",
        fontsize=15.0,
        fontweight="bold",
        y=0.99,
    )


    fig_seed_r4.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.86,
        wspace=0.27,
        hspace=0.32,
    )


    save_figure_r4(
        fig_seed_r4,
        "round4_leave_one_schedule_out_seed_stability_final",
    )


    plt.show()


# =============================================================================
# 11. FINAL NUMERICAL TABLE
# =============================================================================

final_table_rows_r4 = []


for row_index, configuration in enumerate(
    omission_order_r4
):

    output_row = {
        "Omitted schedule":
            omission_display_r4[
                configuration
            ]
    }


    for column_index, metric in enumerate(
        metric_order_r4
    ):

        output_row[
            metric_heatmap_labels_r4[
                metric
            ]
        ] = (
            f"{mean_matrix_r4[row_index, column_index]:+.1f} "
            f"± {sd_matrix_r4[row_index, column_index]:.1f}%"
        )


    final_table_rows_r4.append(
        output_row
    )


LOO_FINAL_PUBLICATION_TABLE_R4 = pd.DataFrame(
    final_table_rows_r4
)


print(
    "\nFinal paired leave-one-schedule-out summary:"
)


display(
    LOO_FINAL_PUBLICATION_TABLE_R4
)


# =============================================================================
# 12. SAVE FINAL TABLE
# =============================================================================

final_table_path_r4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_final_publication_summary.csv"
)


LOO_FINAL_PUBLICATION_TABLE_R4.to_csv(
    final_table_path_r4,
    index=False,
)


# =============================================================================
# 13. FINAL REPORT
# =============================================================================

print(
    "\n" + "=" * 86
)


print(
    "ROUND 4 CELL 23C COMPLETED SUCCESSFULLY"
)


print(
    "=" * 86
)


print(
    "\nNO TRAINING WAS PERFORMED."
)


print(
    "\nFinal heatmap:"
)


print(
    " ",
    FIGURE_DIR_R4
    / "round4_leave_one_schedule_out_heatmap_final",
)


print(
    "\nSeed-stability figure:"
)


print(
    " ",
    FIGURE_DIR_R4
    / "round4_leave_one_schedule_out_seed_stability_final",
)


print(
    "\nFinal publication summary:"
)


print(
    " ",
    final_table_path_r4,
)

# In[39]:




# In[77]:


# ROUND 4 — CELL 24
# FINAL PUBLICATION VERSION
# Parameter-objective geometry for M1 versus M4
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# This experiment visualizes practical parameter identifiability directly in
# the mechanistic ODE model, independently of PINN optimization.
#
# For each candidate parameter pair
#
#       theta = (log10(mu), c_R),
#
# the mechanistic observation residual is evaluated.
#
# The objective geometry is compared between:
#
#       M1 : early-continuous schedule only
#       M4 : untreated + early-continuous + late-continuous + pulsed
#
# SCIENTIFIC PURPOSE
# ------------------
# A long narrow low-objective valley indicates compensation between mu and c_R:
# many parameter combinations can explain the observations similarly well.
#
# A more localized objective region indicates improved practical
# identifiability.
#
# IMPORTANT INTERPRETATION
# ------------------------
# The objective surface is NOT expected to become perfectly isotropic under M4.
# Round 4 sensitivity analysis already showed that strong coupling between
# log10(mu) and c_R persists. Therefore, this figure is intended to determine
# whether multi-schedule observations sharpen the objective geometry rather
# than claim that parameter confounding disappears completely.
#
# FINAL PUBLICATION REFINEMENTS
# -----------------------------
# 1. Existing 41 x 41 surfaces are reused unless explicit recomputation is
#    requested.
#
# 2. Raw SSE is converted to residual-count-normalized MSE:
#
#       MSE = SSE / N_residuals
#
#    because M1 and M4 contain different numbers of schedules/observations.
#
# 3. The excess objective is referenced to the exact true parameter pair:
#
#       Delta MSE(theta)
#           = max[MSE(theta) - MSE(theta_true), 0].
#
# 4. M1 and M4 use the SAME contour levels and SAME color normalization.
#    Therefore colors are directly comparable between panels.
#
# 5. The true parameter pair and finite-grid minimum are shown separately.
#
# 6. The colorbar occupies a dedicated axis and cannot overlap Panel B.
#
# 7. A shared legend is placed outside the data panels.
#
# 8. Thin contour overlays emphasize the compensation-valley geometry.
#
# 9. Exact numerical grid-minimum information is exported to CSV.
#
# NO PINN TRAINING IS PERFORMED IN THIS CELL.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 2. FILE PATHS
# =============================================================================

PROFILE_SURFACE_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_profile_surfaces_M1_M4.csv"
)

PROFILE_SURFACE_SUMMARY_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_profile_surfaces_M1_M4_summary.csv"
)


# =============================================================================
# 3. EXECUTION POLICY
# =============================================================================
# You have already completed the expensive 41 x 41 profile calculation.
#
# KEEP THIS FALSE.
#
# False:
#     reuse the existing profile-surface CSV.
#
# True:
#     recompute the full ODE profile surfaces.
# =============================================================================

RECOMPUTE_PROFILE_SURFACES_R4 = False


PROFILE_EPSILON_R4 = 1.0e-14


PROFILE_DESIGNS_R4 = [
    "M1",
    "M4",
]


TRUE_LOG10_MU_R4 = math.log10(
    TRUE_PARAMETERS.mutation_probability
)


TRUE_COST_R4 = float(
    TRUE_PARAMETERS.resistance_fitness_cost
)


# =============================================================================
# 4. FALLBACK PLOTTING HELPERS
# =============================================================================

if "set_panel_frame_r4" not in globals():

    def set_panel_frame_r4(axis):
        """
        Draw a consistent black frame around a publication panel.
        """

        for spine in axis.spines.values():

            spine.set_visible(
                True
            )

            spine.set_color(
                "black"
            )

            spine.set_linewidth(
                1.0
            )


if "save_figure_r4" not in globals():

    def save_figure_r4(
        figure,
        stem,
    ):
        """
        Save high-resolution raster and vector versions.
        """

        FIGURE_DIR_R4.mkdir(
            parents=True,
            exist_ok=True,
        )


        figure.savefig(
            FIGURE_DIR_R4
            / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )


        figure.savefig(
            FIGURE_DIR_R4
            / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

        figure.savefig(
            FIGURE_DIR_R4
            / f"{stem}.svg",
            bbox_inches="tight",
            facecolor="white",
        )


# =============================================================================
# 5. DETERMINE WHETHER RECOMPUTATION IS REQUIRED
# =============================================================================

need_recompute_r4 = (
    RUN_PROFILE_SURFACES
    and (
        RECOMPUTE_PROFILE_SURFACES_R4
        or not PROFILE_SURFACE_PATH_R4.exists()
    )
)


# =============================================================================
# 6. OPTIONAL FULL PROFILE-SURFACE COMPUTATION
# =============================================================================

if need_recompute_r4:

    grid_size_r4 = (
        13
        if QUICK_MODE
        else 41
    )


    log_mu_grid_r4 = np.linspace(
        -5.5,
        -2.5,
        grid_size_r4,
    )


    cost_grid_r4 = np.linspace(
        0.0,
        0.30,
        grid_size_r4,
    )


    profile_rows_r4 = []


    total_profile_points_r4 = (
        len(PROFILE_DESIGNS_R4)
        * grid_size_r4
        * grid_size_r4
    )


    completed_profile_points_r4 = 0


    print(
        f"Computing {total_profile_points_r4} profile points "
        f"({grid_size_r4} x {grid_size_r4} per design)."
    )


    for design_id_r4 in PROFILE_DESIGNS_R4:

        bundle_r4 = build_design_bundle_r4(
            design_id_r4,
            seed=SEED,
            noise_level=0.0,
        )


        print(
            f"\nComputing objective surface for {design_id_r4}..."
        )


        for row_number_r4, log_mu_r4 in enumerate(
            log_mu_grid_r4,
            start=1,
        ):

            for cost_r4 in cost_grid_r4:

                residual_r4 = np.asarray(
                    ode_residual_vector(
                        np.array(
                            [
                                log_mu_r4,
                                cost_r4,
                            ],
                            dtype=float,
                        ),
                        bundle_r4,
                    ),
                    dtype=float,
                ).reshape(-1)


                profile_rows_r4.append(
                    {
                        "design_id":
                            design_id_r4,

                        "log10_mu":
                            float(
                                log_mu_r4
                            ),

                        "cost":
                            float(
                                cost_r4
                            ),

                        "n_residuals":
                            int(
                                residual_r4.size
                            ),

                        "sse":
                            float(
                                np.sum(
                                    residual_r4 ** 2
                                )
                            ),

                        "mse":
                            float(
                                np.mean(
                                    residual_r4 ** 2
                                )
                            ),
                    }
                )


                completed_profile_points_r4 += 1


            print(
                f"  {design_id_r4}: "
                f"{row_number_r4}/{grid_size_r4} rows completed "
                f"({completed_profile_points_r4}/"
                f"{total_profile_points_r4} total points)"
            )


    PROFILE_SURFACE_R4 = pd.DataFrame(
        profile_rows_r4
    )


    PROFILE_SURFACE_R4.to_csv(
        PROFILE_SURFACE_PATH_R4,
        index=False,
    )


    print(
        "\nSaved newly calculated profile grid:"
    )

    print(
        PROFILE_SURFACE_PATH_R4
    )


# =============================================================================
# 7. LOAD EXISTING PROFILE SURFACE
# =============================================================================

elif PROFILE_SURFACE_PATH_R4.exists():

    PROFILE_SURFACE_R4 = pd.read_csv(
        PROFILE_SURFACE_PATH_R4
    )


    print(
        "Loaded existing profile grid:"
    )

    print(
        PROFILE_SURFACE_PATH_R4
    )


else:

    raise FileNotFoundError(
        "No completed profile-surface CSV was found.\n"
        "Set RUN_PROFILE_SURFACES=True and rerun Cell 24."
    )


# =============================================================================
# 8. VALIDATE PROFILE-SURFACE TABLE
# =============================================================================

required_profile_columns_r4 = [
    "design_id",
    "log10_mu",
    "cost",
    "sse",
]


missing_profile_columns_r4 = [
    column
    for column in required_profile_columns_r4
    if column not in PROFILE_SURFACE_R4.columns
]


if missing_profile_columns_r4:

    raise KeyError(
        "PROFILE_SURFACE_R4 is missing required columns: "
        f"{missing_profile_columns_r4}"
    )


PROFILE_SURFACE_R4 = (
    PROFILE_SURFACE_R4
    .copy()
)


PROFILE_SURFACE_R4[
    "design_id"
] = (
    PROFILE_SURFACE_R4[
        "design_id"
    ]
    .astype(str)
)


for column_r4 in [
    "log10_mu",
    "cost",
    "sse",
]:

    PROFILE_SURFACE_R4[
        column_r4
    ] = pd.to_numeric(
        PROFILE_SURFACE_R4[
            column_r4
        ],
        errors="raise",
    )


# =============================================================================
# 9. COMPUTE EXACT TRUE-PAIR REFERENCE AND RESIDUAL COUNTS
# =============================================================================
# M1 and M4 contain different numbers of observations.
#
# Comparing raw SSE would therefore partly reflect residual count.
#
# We instead use:
#
#       MSE = SSE / N_residuals.
#
# Only one exact ODE residual evaluation per design is needed here.
# =============================================================================

residual_count_r4 = {}

true_reference_mse_r4 = {}


for design_id_r4 in PROFILE_DESIGNS_R4:

    bundle_r4 = build_design_bundle_r4(
        design_id_r4,
        seed=SEED,
        noise_level=0.0,
    )


    true_residual_r4 = np.asarray(
        ode_residual_vector(
            np.array(
                [
                    TRUE_LOG10_MU_R4,
                    TRUE_COST_R4,
                ],
                dtype=float,
            ),
            bundle_r4,
        ),
        dtype=float,
    ).reshape(-1)


    residual_count_r4[
        design_id_r4
    ] = int(
        true_residual_r4.size
    )


    true_reference_mse_r4[
        design_id_r4
    ] = float(
        np.mean(
            true_residual_r4 ** 2
        )
    )


# =============================================================================
# 10. CONVERT RAW SSE TO RESIDUAL-NORMALIZED MSE
# =============================================================================

PROFILE_SURFACE_R4[
    "n_residuals"
] = (
    PROFILE_SURFACE_R4[
        "design_id"
    ]
    .map(
        residual_count_r4
    )
    .astype(int)
)


PROFILE_SURFACE_R4[
    "mse"
] = (
    PROFILE_SURFACE_R4[
        "sse"
    ]
    / PROFILE_SURFACE_R4[
        "n_residuals"
    ]
)


PROFILE_SURFACE_R4[
    "true_reference_mse"
] = (
    PROFILE_SURFACE_R4[
        "design_id"
    ]
    .map(
        true_reference_mse_r4
    )
    .astype(float)
)


# =============================================================================
# 11. EXCESS OBJECTIVE RELATIVE TO THE EXACT TRUE PARAMETER PAIR
# =============================================================================
# The synthetic observations were produced using the matched ODE model.
#
# Therefore the exact true pair provides the natural mechanistic reference.
# =============================================================================

PROFILE_SURFACE_R4[
    "delta_mse"
] = np.maximum(
    PROFILE_SURFACE_R4[
        "mse"
    ]
    - PROFILE_SURFACE_R4[
        "true_reference_mse"
    ],
    0.0,
)


PROFILE_SURFACE_R4[
    "log10_delta_mse"
] = np.log10(
    PROFILE_SURFACE_R4[
        "delta_mse"
    ]
    + PROFILE_EPSILON_R4
)


# Save the enriched table.
PROFILE_SURFACE_R4.to_csv(
    PROFILE_SURFACE_PATH_R4,
    index=False,
)


# =============================================================================
# 12. BUILD PIVOT MATRICES
# =============================================================================

profile_pivots_r4 = {}

profile_z_values_r4 = {}

all_profile_z_r4 = []


for design_id_r4 in PROFILE_DESIGNS_R4:

    frame_r4 = PROFILE_SURFACE_R4.loc[
        PROFILE_SURFACE_R4[
            "design_id"
        ] == design_id_r4
    ].copy()


    pivot_r4 = (
        frame_r4
        .pivot(
            index="cost",
            columns="log10_mu",
            values="log10_delta_mse",
        )
        .sort_index(
            axis=0
        )
        .sort_index(
            axis=1
        )
    )


    profile_pivots_r4[
        design_id_r4
    ] = pivot_r4


    z_values_r4 = pivot_r4.to_numpy(
        dtype=float
    )


    profile_z_values_r4[
        design_id_r4
    ] = z_values_r4


    finite_z_r4 = z_values_r4[
        np.isfinite(
            z_values_r4
        )
    ]


    all_profile_z_r4.extend(
        finite_z_r4.ravel().tolist()
    )


all_profile_z_r4 = np.asarray(
    all_profile_z_r4,
    dtype=float,
)


# =============================================================================
# 13. DEFINE ONE SHARED COLOR SCALE
# =============================================================================
# This is essential for valid visual comparison.
#
# M1 and M4 must not independently choose their contour ranges.
#
# The 2nd percentile is used as the lower display bound so isolated
# near-zero cells do not consume almost the entire dynamic range.
# =============================================================================

shared_vmin_r4 = float(
    np.nanpercentile(
        all_profile_z_r4,
        2.0,
    )
)


shared_vmax_r4 = float(
    np.nanmax(
        all_profile_z_r4
    )
)


if shared_vmax_r4 <= shared_vmin_r4:

    shared_vmax_r4 = (
        shared_vmin_r4
        + 1.0
    )


shared_contour_levels_r4 = np.linspace(
    shared_vmin_r4,
    shared_vmax_r4,
    30,
)


# Sparse contour-line levels for visual valley structure.
shared_line_levels_r4 = (
    shared_contour_levels_r4[
        3::5
    ]
)


# =============================================================================
# 14. NUMERICAL PROFILE-SURFACE SUMMARY
# =============================================================================

profile_summary_rows_r4 = []


for design_id_r4 in PROFILE_DESIGNS_R4:

    frame_r4 = PROFILE_SURFACE_R4.loc[
        PROFILE_SURFACE_R4[
            "design_id"
        ] == design_id_r4
    ].copy()


    minimum_row_r4 = frame_r4.loc[
        frame_r4[
            "mse"
        ].idxmin()
    ]


    profile_summary_rows_r4.append(
        {
            "design_id":
                design_id_r4,

            "schedule_count":
                (
                    1
                    if design_id_r4 == "M1"
                    else 4
                ),

            "grid_min_log10_mu":
                float(
                    minimum_row_r4[
                        "log10_mu"
                    ]
                ),

            "grid_min_mu":
                float(
                    10.0
                    ** float(
                        minimum_row_r4[
                            "log10_mu"
                        ]
                    )
                ),

            "grid_min_cost":
                float(
                    minimum_row_r4[
                        "cost"
                    ]
                ),

            "grid_min_mse":
                float(
                    minimum_row_r4[
                        "mse"
                    ]
                ),

            "true_reference_mse":
                float(
                    true_reference_mse_r4[
                        design_id_r4
                    ]
                ),

            "abs_grid_error_log10_mu":
                abs(
                    float(
                        minimum_row_r4[
                            "log10_mu"
                        ]
                    )
                    - TRUE_LOG10_MU_R4
                ),

            "abs_grid_error_cost":
                abs(
                    float(
                        minimum_row_r4[
                            "cost"
                        ]
                    )
                    - TRUE_COST_R4
                ),

            "n_residuals":
                int(
                    residual_count_r4[
                        design_id_r4
                    ]
                ),
        }
    )


PROFILE_SURFACE_SUMMARY_R4 = pd.DataFrame(
    profile_summary_rows_r4
)


PROFILE_SURFACE_SUMMARY_R4.to_csv(
    PROFILE_SURFACE_SUMMARY_PATH_R4,
    index=False,
)


# =============================================================================
# 15. FINAL PUBLICATION STYLE
# =============================================================================

CELL24_PUBLICATION_RC_R4 = {

    "font.family":
        "sans-serif",

    "font.size":
        11.0,

    "axes.titlesize":
        12.8,

    "axes.labelsize":
        11.5,

    "xtick.labelsize":
        10.0,

    "ytick.labelsize":
        10.0,

    "legend.fontsize":
        9.5,

    "axes.linewidth":
        1.0,

    "axes.edgecolor":
        "black",

    "axes.facecolor":
        "white",

    "figure.facecolor":
        "white",

    "savefig.facecolor":
        "white",
}


# =============================================================================
# 16. CREATE FINAL FIGURE
# =============================================================================

with plt.rc_context(
    CELL24_PUBLICATION_RC_R4
):

    fig_r4 = plt.figure(
        figsize=(
            12.4,
            5.15,
        )
    )


    # Dedicated third column prevents colorbar overlap.
    grid_spec_r4 = GridSpec(
        1,
        3,
        figure=fig_r4,
        width_ratios=[
            1.0,
            1.0,
            0.045,
        ],
        wspace=0.22,
    )


    ax_m1_r4 = fig_r4.add_subplot(
        grid_spec_r4[
            0,
            0,
        ]
    )


    ax_m4_r4 = fig_r4.add_subplot(
        grid_spec_r4[
            0,
            1,
        ],
        sharex=ax_m1_r4,
        sharey=ax_m1_r4,
    )


    colorbar_axis_r4 = fig_r4.add_subplot(
        grid_spec_r4[
            0,
            2,
        ]
    )


    axes_r4 = [
        ax_m1_r4,
        ax_m4_r4,
    ]


    panel_titles_r4 = {

        "M1":
            "M1 — single schedule",

        "M4":
            "M4 — four schedules",
    }


    contour_reference_r4 = None


    # =========================================================================
    # 17. DRAW M1 AND M4 USING IDENTICAL COLOR NORMALIZATION
    # =========================================================================

    for (
        axis_r4,
        design_id_r4,
        panel_label_r4,
    ) in zip(
        axes_r4,
        PROFILE_DESIGNS_R4,
        [
            "A",
            "B",
        ],
    ):

        frame_r4 = PROFILE_SURFACE_R4.loc[
            PROFILE_SURFACE_R4[
                "design_id"
            ] == design_id_r4
        ].copy()


        pivot_r4 = profile_pivots_r4[
            design_id_r4
        ]


        x_values_r4 = pivot_r4.columns.to_numpy(
            dtype=float
        )


        y_values_r4 = pivot_r4.index.to_numpy(
            dtype=float
        )


        z_values_r4 = profile_z_values_r4[
            design_id_r4
        ]


        # ---------------------------------------------------------------------
        # Filled objective contours
        # ---------------------------------------------------------------------

        contour_r4 = axis_r4.contourf(
            x_values_r4,
            y_values_r4,
            z_values_r4,
            levels=shared_contour_levels_r4,
            cmap="viridis",
            extend="both",
        )


        contour_reference_r4 = (
            contour_r4
        )


        # ---------------------------------------------------------------------
        # Thin contour overlays
        # ---------------------------------------------------------------------

        axis_r4.contour(
            x_values_r4,
            y_values_r4,
            z_values_r4,
            levels=shared_line_levels_r4,
            colors="black",
            linewidths=0.35,
            alpha=0.18,
        )


        # ---------------------------------------------------------------------
        # True-parameter reference lines
        # ---------------------------------------------------------------------

        axis_r4.axvline(
            TRUE_LOG10_MU_R4,
            linestyle=":",
            linewidth=1.0,
            color="black",
            alpha=0.48,
            zorder=3,
        )


        axis_r4.axhline(
            TRUE_COST_R4,
            linestyle=":",
            linewidth=1.0,
            color="black",
            alpha=0.48,
            zorder=3,
        )


        # ---------------------------------------------------------------------
        # Exact true parameter pair
        # ---------------------------------------------------------------------

        axis_r4.plot(
            TRUE_LOG10_MU_R4,
            TRUE_COST_R4,
            marker="*",
            markersize=13,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            linestyle="None",
            zorder=6,
        )


        # ---------------------------------------------------------------------
        # Best point available on finite 41 x 41 grid
        # ---------------------------------------------------------------------

        minimum_row_r4 = frame_r4.loc[
            frame_r4[
                "mse"
            ].idxmin()
        ]


        axis_r4.plot(
            float(
                minimum_row_r4[
                    "log10_mu"
                ]
            ),
            float(
                minimum_row_r4[
                    "cost"
                ]
            ),
            marker="X",
            markersize=8.5,
            color="tab:orange",
            linestyle="None",
            zorder=7,
        )


        # ---------------------------------------------------------------------
        # Titles and axis labels
        # ---------------------------------------------------------------------

        axis_r4.set_title(
            panel_titles_r4[
                design_id_r4
            ],
            loc="left",
            fontweight="semibold",
            pad=8,
        )


        axis_r4.set_xlabel(
            r"$\log_{10}(\mu)$"
        )


        axis_r4.set_xlim(
            -5.5,
            -2.5,
        )


        axis_r4.set_ylim(
            0.0,
            0.30,
        )


        axis_r4.xaxis.set_major_locator(
            MaxNLocator(
                nbins=7
            )
        )


        axis_r4.yaxis.set_major_locator(
            MaxNLocator(
                nbins=7
            )
        )


        axis_r4.grid(
            alpha=0.10,
            linewidth=0.6,
        )


        axis_r4.set_axisbelow(
            True
        )


        set_panel_frame_r4(
            axis_r4
        )


        # ---------------------------------------------------------------------
        # Panel labels
        # ---------------------------------------------------------------------

        axis_r4.text(
            -0.105,
            1.045,
            panel_label_r4,
            transform=axis_r4.transAxes,
            fontsize=12.8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


# =============================================================================
# 18. SHARED Y-AXIS FORMATTING
# =============================================================================

    ax_m1_r4.set_ylabel(
        r"Resistance fitness cost, $c_R$"
    )


    # Avoid duplicated labels because axes share the same y scale.
    ax_m4_r4.tick_params(
        axis="y",
        labelleft=False,
    )


# =============================================================================
# 19. SHARED COLORBAR
# =============================================================================
# Both panels use this SAME colorbar and SAME normalization.
# This should also be stated explicitly in the manuscript figure caption.
# =============================================================================

    colorbar_r4 = fig_r4.colorbar(
        contour_reference_r4,
        cax=colorbar_axis_r4,
    )


    colorbar_r4.set_label(
        r"$\log_{10}\!\left(\Delta\mathrm{MSE}+10^{-14}\right)$"
    )


    colorbar_r4.ax.tick_params(
        labelsize=9.5
    )


# =============================================================================
# 20. SHARED LEGEND OUTSIDE DATA REGION
# =============================================================================

    legend_handles_r4 = [

        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=11,
            markerfacecolor="white",
            markeredgecolor="black",
            label="True parameter pair",
        ),

        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markersize=8,
            color="tab:orange",
            label="Finite-grid minimum",
        ),
    ]


    fig_r4.legend(
        handles=legend_handles_r4,
        loc="upper center",
        bbox_to_anchor=(
            0.50,
            0.972,
        ),
        ncol=2,
        frameon=False,
        fontsize=9.0,
        columnspacing=1.6,
        handletextpad=0.55,
    )


# =============================================================================
# 21. FINAL MANUSCRIPT TITLE
# =============================================================================
# No large in-figure title is used. The manuscript / Supplementary caption
# carries the overall interpretation, leaving more space for the data panels.


# =============================================================================
# 22. FINAL LAYOUT
# =============================================================================

    fig_r4.subplots_adjust(
        left=0.075,
        right=0.94,
        bottom=0.145,
        top=0.855,
    )


# =============================================================================
# 23. SAVE AND DISPLAY
# =============================================================================

    save_figure_r4(
        fig_r4,
        "round4_profile_surfaces_M1_vs_M4_final",
    )


    plt.show()


# =============================================================================
# 24. NUMERICAL MANUSCRIPT CHECK
# =============================================================================

print(
    "\nProfile-surface summary:"
)


display(
    PROFILE_SURFACE_SUMMARY_R4
)


# =============================================================================
# 25. FINAL REPORT
# =============================================================================

print(
    "\n"
    + "=" * 86
)


print(
    "ROUND 4 CELL 24 COMPLETED SUCCESSFULLY"
)


print(
    "=" * 86
)


if need_recompute_r4:

    print(
        "\nThe profile surfaces were recomputed."
    )

else:

    print(
        "\nThe existing profile grid was reused; "
        "the expensive ODE surfaces were NOT recomputed."
    )


print(
    "\nObjective comparison:"
)


print(
    "  M1 and M4 use residual-count-normalized MSE."
)


print(
    "  Both panels use exactly the same contour levels and color scale."
)


print(
    "  Delta MSE is referenced to the exact true parameter pair."
)


print(
    "\nScientific interpretation:"
)


print(
    "  Multi-schedule observations may sharpen the objective geometry, "
    "but a diagonal mu-c_R compensation direction can remain."
)


print(
    "  The figure should therefore be interpreted as evidence of improved, "
    "not necessarily complete, practical identifiability."
)


print(
    "\nFinal figure:"
)


print(
    " ",
    FIGURE_DIR_R4
    / "round4_profile_surfaces_M1_vs_M4_final",
)


print(
    "\nEnriched profile table:"
)


print(
    " ",
    PROFILE_SURFACE_PATH_R4,
)


print(
    "\nProfile summary table:"
)


print(
    " ",
    PROFILE_SURFACE_SUMMARY_PATH_R4,
)

# In[81]:


# CELL 25 — FINAL HEATMAP COLOR FIX
# EXACT VISUAL MATCH TO CELL 23C
# =============================================================================
# WHAT THIS BLOCK FIXES
# ---------------------
# Cell 23C uses:
#
#       cmap = "RdBu_r"
#
# with a symmetric signed scale around zero. However, the actual Cell 23
# observations occupy only part of that full color scale:
#
#       most negative observed effect  ~ -76%
#       most positive observed effect  ~ +187%
#       symmetric color limit          ~ ±200%
#
# Therefore, the visible Cell 23 palette does NOT extend to the darkest navy
# blue. Cell 25 previously normalized values directly from 0 to 1 over the
# entire RdBu_r palette, which incorrectly produced very dark blue cells.
#
# This block reconstructs the EXACT EFFECTIVE COLOR RANGE used by Cell 23C
# directly from the saved Cell 23 paired-change data.
#
# IMPORTANT
# ---------
# - Heatmap values remain unchanged.
# - Raw RMSE mean ± SD values remain unchanged.
# - Resistant-fraction RMSE is displayed in compact scientific notation.
# - No training is performed.
# - No transfer calculations are repeated.
# =============================================================================


# =============================================================================
# 1. EXTRA IMPORTS
# =============================================================================

from matplotlib.colors import (
    Normalize,
    TwoSlopeNorm,
    ListedColormap,
)


# =============================================================================
# 2. LOAD THE SAME CELL 23C DATA USED TO DEFINE ITS HEATMAP COLORS
# =============================================================================

CELL23_PAIRED_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_leave_one_schedule_out_paired_changes.csv"
)


# Defaults are provided only as a safety fallback.
cell23_low_color_position_r4 = 0.31
cell23_high_color_position_r4 = 0.97
cell23_color_limit_r4 = 200.0
cell23_min_effect_r4 = -76.0
cell23_max_effect_r4 = 187.0


if CELL23_PAIRED_PATH_R4.exists():

    cell23_paired_r4 = pd.read_csv(
        CELL23_PAIRED_PATH_R4
    )


    required_cell23_columns_r4 = [
        "configuration",
        "metric",
        "paired_change_percent",
    ]


    missing_cell23_columns_r4 = [
        column
        for column in required_cell23_columns_r4
        if column not in cell23_paired_r4.columns
    ]


    if not missing_cell23_columns_r4:

        cell23_metric_order_r4 = [
            "rmse_resistant",
            "rmse_fraction",
            "mu_error_percent",
            "cost_error_percent",
        ]


        cell23_configuration_order_r4 = [
            "NO_UNTREATED",
            "NO_EARLY",
            "NO_LATE",
            "NO_PULSED",
        ]


        # ---------------------------------------------------------------------
        # Reproduce the mean matrix from Cell 23C.
        # ---------------------------------------------------------------------

        cell23_mean_summary_r4 = (
            cell23_paired_r4
            .loc[
                cell23_paired_r4[
                    "configuration"
                ].isin(
                    cell23_configuration_order_r4
                )
                &
                cell23_paired_r4[
                    "metric"
                ].isin(
                    cell23_metric_order_r4
                )
            ]
            .groupby(
                [
                    "configuration",
                    "metric",
                ],
                observed=True,
            )[
                "paired_change_percent"
            ]
            .mean()
            .reset_index()
        )


        cell23_mean_matrix_r4 = np.full(
            (
                len(
                    cell23_configuration_order_r4
                ),
                len(
                    cell23_metric_order_r4
                ),
            ),
            np.nan,
            dtype=float,
        )


        for row_index_r4, configuration_r4 in enumerate(
            cell23_configuration_order_r4
        ):

            for column_index_r4, metric_r4 in enumerate(
                cell23_metric_order_r4
            ):

                subset_r4 = (
                    cell23_mean_summary_r4.loc[
                        (
                            cell23_mean_summary_r4[
                                "configuration"
                            ] == configuration_r4
                        )
                        &
                        (
                            cell23_mean_summary_r4[
                                "metric"
                            ] == metric_r4
                        ),
                        "paired_change_percent",
                    ]
                )


                if len(
                    subset_r4
                ) == 1:

                    cell23_mean_matrix_r4[
                        row_index_r4,
                        column_index_r4,
                    ] = float(
                        subset_r4.iloc[
                            0
                        ]
                    )


        # ---------------------------------------------------------------------
        # Reproduce Cell 23C's symmetric color limit.
        # ---------------------------------------------------------------------

        maximum_effect_r4 = float(
            np.nanmax(
                np.abs(
                    cell23_mean_matrix_r4
                )
            )
        )


        cell23_color_limit_r4 = max(
            50.0,
            np.ceil(
                maximum_effect_r4
                / 25.0
            )
            * 25.0,
        )


        cell23_min_effect_r4 = float(
            np.nanmin(
                cell23_mean_matrix_r4
            )
        )


        cell23_max_effect_r4 = float(
            np.nanmax(
                cell23_mean_matrix_r4
            )
        )


        # ---------------------------------------------------------------------
        # Reproduce the exact Cell 23C TwoSlopeNorm.
        # ---------------------------------------------------------------------

        cell23_norm_r4 = TwoSlopeNorm(
            vmin=-cell23_color_limit_r4,
            vcenter=0.0,
            vmax=cell23_color_limit_r4,
        )


        # ---------------------------------------------------------------------
        # Identify the ACTUAL part of RdBu_r visible in Cell 23C.
        #
        # For your current results this should be approximately:
        #
        #       low  ~ 0.31
        #       high ~ 0.97
        #
        # rather than the complete [0, 1] palette.
        # ---------------------------------------------------------------------

        cell23_low_color_position_r4 = float(
            cell23_norm_r4(
                cell23_min_effect_r4
            )
        )


        cell23_high_color_position_r4 = float(
            cell23_norm_r4(
                cell23_max_effect_r4
            )
        )


        print(
            "Cell 23C color range reproduced successfully."
        )


        print(
            "  Cell 23 symmetric color limit:",
            cell23_color_limit_r4,
        )


        print(
            "  Most negative Cell 23 mean:",
            f"{cell23_min_effect_r4:.1f}%",
        )


        print(
            "  Most positive Cell 23 mean:",
            f"{cell23_max_effect_r4:.1f}%",
        )


        print(
            "  Effective RdBu_r range:",
            f"{cell23_low_color_position_r4:.3f}",
            "to",
            f"{cell23_high_color_position_r4:.3f}",
        )


else:

    print(
        "Cell 23 paired-change CSV was not found; "
        "using the calibrated Cell 23C fallback palette."
    )


# =============================================================================
# 3. BUILD THE CELL-23-MATCHED COLORMAP
# =============================================================================
# IMPORTANT:
#
# Do NOT construct this with only four manually selected colors.
#
# Instead, densely sample the ORIGINAL RdBu_r colormap over exactly the
# portion that was visible in Cell 23C.
#
# This preserves the same smooth color progression as Cell 23.
# =============================================================================

base_cell23_cmap_r4 = plt.get_cmap(
    "RdBu_r"
)


cell23_color_samples_r4 = np.linspace(
    cell23_low_color_position_r4,
    cell23_high_color_position_r4,
    256,
)


CELL23_MATCHED_CMAP_R4 = ListedColormap(
    base_cell23_cmap_r4(
        cell23_color_samples_r4
    ),
    name="Cell23MatchedRdBu",
)


# =============================================================================
# 4. FINAL CELL 25 COLOR NORMALIZATION
# =============================================================================
# Cell 25 values are still:
#
#       0 = lowest relative error in that metric column
#       1 = largest relative error in that metric column
#
# But those values are now rendered using ONLY the color range actually visible
# in Cell 23C.
# =============================================================================

cell25_heatmap_norm_r4 = Normalize(
    vmin=0.0,
    vmax=1.0,
)


# =============================================================================
# 5. FINAL PUBLICATION FIGURE
# =============================================================================

with plt.rc_context(
    CELL25_PUBLICATION_RC_R4
):

    fig_r4, ax_r4 = plt.subplots(
        figsize=(
            11.2,
            5.35,
        )
    )


    # -------------------------------------------------------------------------
    # Heatmap
    # -------------------------------------------------------------------------

    image_r4 = ax_r4.imshow(
        normalized_matrix_r4,
        aspect="auto",
        cmap=CELL23_MATCHED_CMAP_R4,
        norm=cell25_heatmap_norm_r4,
    )


    # -------------------------------------------------------------------------
    # X-axis
    # -------------------------------------------------------------------------

    ax_r4.set_xticks(
        np.arange(
            len(
                transfer_metrics_r4
            )
        )
    )


    # Compact mathematical labels are easier to read after journal scaling.
    transfer_metric_labels_publication_r4 = {
        "rmse_sensitive":
            r"$\mathrm{RMSE}_{S}$",

        "rmse_resistant":
            r"$\mathrm{RMSE}_{R}$",

        "rmse_fraction":
            r"$\mathrm{RMSE}_{q_R}$",
    }

    ax_r4.set_xticklabels(
        [
            transfer_metric_labels_publication_r4.get(
                metric_r4,
                metric_display_r4[
                    metric_r4
                ],
            )
            for metric_r4
            in transfer_metrics_r4
        ]
    )


    # -------------------------------------------------------------------------
    # Y-axis
    # -------------------------------------------------------------------------

    ax_r4.set_yticks(
        np.arange(
            len(
                heldout_order_r4
            )
        )
    )


    ax_r4.set_yticklabels(
        [
            heldout_display_r4[
                schedule_r4
            ]
            for schedule_r4
            in heldout_order_r4
        ]
    )


    # -------------------------------------------------------------------------
    # Publication annotation formatter
    # -------------------------------------------------------------------------
    # State RMSE values retain the existing formatter. Resistant-fraction RMSE
    # is shown as (mean ± SD) x 10^k so very small values remain immediately
    # readable without long strings of leading zeros.
    def format_transfer_annotation_publication_r4(
        mean_value_r4,
        sd_value_r4,
        metric_r4,
    ):
        mean_value_r4 = float(
            mean_value_r4
        )

        sd_value_r4 = float(
            sd_value_r4
        )

        if (
            metric_r4 == "rmse_fraction"
            and mean_value_r4 != 0.0
        ):
            exponent_r4 = int(
                np.floor(
                    np.log10(
                        abs(
                            mean_value_r4
                        )
                    )
                )
            )

            scale_r4 = (
                10.0
                ** exponent_r4
            )

            mean_scaled_r4 = (
                mean_value_r4
                / scale_r4
            )

            sd_scaled_r4 = (
                sd_value_r4
                / scale_r4
            )

            return (
                rf"$({mean_scaled_r4:.3g}"
                rf"\,\pm\,{sd_scaled_r4:.3g})"
                rf"\times10^{{{exponent_r4}}}$"
            )

        return (
            f"{format_transfer_value_r4(mean_value_r4)} "
            f"± "
            f"{format_transfer_value_r4(sd_value_r4)}"
        )


    # -------------------------------------------------------------------------
    # Numerical annotations
    # -------------------------------------------------------------------------
    # Match Cell 23C:
    #
    # - NO background boxes.
    # - bold text directly on the heatmap.
    # - black on lighter colors.
    # - white only on sufficiently dark red regions.
    #
    # The synthetic visual effect below is ONLY used to reproduce Cell 23C's
    # text-color rule. It does not change any scientific value.
    # -------------------------------------------------------------------------

    for row_index_r4 in range(
        mean_matrix_r4.shape[
            0
        ]
    ):

        for column_index_r4 in range(
            mean_matrix_r4.shape[
                1
            ]
        ):

            mean_value_r4 = float(
                mean_matrix_r4[
                    row_index_r4,
                    column_index_r4,
                ]
            )


            sd_value_r4 = float(
                sd_matrix_r4[
                    row_index_r4,
                    column_index_r4,
                ]
            )


            normalized_value_r4 = float(
                normalized_matrix_r4[
                    row_index_r4,
                    column_index_r4,
                ]
            )


            # -----------------------------------------------------------------
            # Map Cell 25 normalized error onto the EFFECT RANGE actually seen
            # in Cell 23C. This is used only to choose black/white text.
            # -----------------------------------------------------------------

            cell23_equivalent_effect_r4 = (
                cell23_min_effect_r4
                +
                normalized_value_r4
                * (
                    cell23_max_effect_r4
                    - cell23_min_effect_r4
                )
            )


            # Same logic used by Cell 23C.
            text_color_r4 = (
                "white"
                if abs(
                    cell23_equivalent_effect_r4
                )
                >= (
                    0.48
                    * cell23_color_limit_r4
                )
                else "black"
            )


            metric_r4 = (
                transfer_metrics_r4[
                    column_index_r4
                ]
            )

            annotation_r4 = (
                format_transfer_annotation_publication_r4(
                    mean_value_r4,
                    sd_value_r4,
                    metric_r4,
                )
            )


            ax_r4.text(
                column_index_r4,
                row_index_r4,
                annotation_r4,

                ha="center",
                va="center",

                fontsize=10.5,
                fontweight="bold",

                color=text_color_r4,
            )


    # -------------------------------------------------------------------------
    # Thin cell boundaries — same style as Cell 23C
    # -------------------------------------------------------------------------

    ax_r4.set_xticks(
        np.arange(
            -0.5,
            len(
                transfer_metrics_r4
            ),
            1.0,
        ),
        minor=True,
    )


    ax_r4.set_yticks(
        np.arange(
            -0.5,
            len(
                heldout_order_r4
            ),
            1.0,
        ),
        minor=True,
    )


    ax_r4.grid(
        which="minor",
        linewidth=1.0,
        alpha=0.45,
    )


    ax_r4.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    # -------------------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------------------

    colorbar_r4 = fig_r4.colorbar(
        image_r4,
        ax=ax_r4,
        pad=0.025,
        fraction=0.055,
    )


    colorbar_r4.set_label(
        "Within-metric normalized error"
    )


    colorbar_r4.set_ticks(
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]
    )


    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    ax_r4.set_ylabel(
        "Held-out treatment environment"
    )


    # The three metric labels already identify the evaluation quantities.
    ax_r4.set_xlabel("")


    # No large in-figure title is used in the final manuscript version.
    # The overall transfer interpretation belongs in the figure caption.


    # -------------------------------------------------------------------------
    # Frame — identical style to Cell 23C
    # -------------------------------------------------------------------------

    for spine_r4 in ax_r4.spines.values():

        spine_r4.set_visible(
            True
        )


        spine_r4.set_color(
            "black"
        )


        spine_r4.set_linewidth(
            1.0
        )


    # -------------------------------------------------------------------------
    # Final layout
    # -------------------------------------------------------------------------

    fig_r4.subplots_adjust(
        left=0.17,
        right=0.91,
        bottom=0.13,
        top=0.97,
    )


    # -------------------------------------------------------------------------
    # Save final publication figure
    # -------------------------------------------------------------------------

    save_figure_r4(
        fig_r4,
        "round4_heldout_schedule_parameter_transfer_final",
    )


    plt.show()

# In[54]:


# ROUND 4 — CELL 26
# FINAL PUBLICATION VERSION
# Optimization histories, convergence diagnostics, and CUDA-memory usage
# =============================================================================
# WHAT THIS CELL IS ABOUT
# -----------------------
# This cell summarizes the optimization behavior of one matched completed run
# per nested schedule design (M1-M5).
#
# The three panels show:
#
#   A. Validation convergence
#   B. Physics-residual convergence
#   C. CUDA reserved-memory history
#
# IMPORTANT SCIENTIFIC/PLOTTING DETAILS
# -------------------------------------
# 1. Histories are loaded from disk, so this cell can be executed after a
#    kernel restart.
#
# 2. The lowest-numbered paired seed is used for every design. This provides
#    a matched-seed convergence comparison rather than choosing the visually
#    best run.
#
# 3. Zero/non-positive physics values are NOT clipped to 1e-16.
#    They are replaced with NaN instead. Therefore, periods during which the
#    physics residual was not evaluated do not appear as artificially perfect
#    physics satisfaction.
#
# 4. Training-phase transitions are detected from:
#
#       stage / phase / training_stage / optimizer_stage
#
#    when available. If no explicit phase column exists, epoch resets are used
#    as a fallback.
#
# 5. The GPU panel reports CUDA RESERVED memory recorded at logging points.
#    It should not be interpreted as total GPU capacity or instantaneous peak
#    device utilization.
#
# 6. The legend is placed below the main title and outside all data panels.
#
# NO MODEL TRAINING IS PERFORMED IN THIS CELL.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


# =============================================================================
# 2. LOAD PRIMARY RESULTS IF THE KERNEL WAS RESTARTED
# =============================================================================

PRIMARY_RESULTS_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_all_seeds_metrics.csv"
)


if (
    "ROUND4_PRIMARY_ALL_RUNS" not in globals()
    or ROUND4_PRIMARY_ALL_RUNS is None
    or ROUND4_PRIMARY_ALL_RUNS.empty
):

    if PRIMARY_RESULTS_PATH_R4.exists():

        ROUND4_PRIMARY_ALL_RUNS = pd.read_csv(
            PRIMARY_RESULTS_PATH_R4
        )

        print(
            "Loaded completed Round 4 primary results:"
        )

        print(
            " ",
            PRIMARY_RESULTS_PATH_R4,
        )

    else:

        raise FileNotFoundError(
            "The completed primary Round 4 results were not found:\n"
            f"{PRIMARY_RESULTS_PATH_R4}\n\n"
            "Complete the primary experiment before generating "
            "optimization diagnostics."
        )


# =============================================================================
# 3. VALIDATE PRIMARY RESULT COLUMNS
# =============================================================================

required_primary_columns_r4 = [
    "design_id",
    "label",
]


missing_primary_columns_r4 = [
    column
    for column in required_primary_columns_r4
    if column not in ROUND4_PRIMARY_ALL_RUNS.columns
]


if missing_primary_columns_r4:

    raise KeyError(
        "Primary results are missing required columns: "
        f"{missing_primary_columns_r4}"
    )


# =============================================================================
# 4. DESIGN ORDER
# =============================================================================

if "DESIGN_ORDER_R4" not in globals():

    DESIGN_ORDER_R4 = [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]


# =============================================================================
# 5. CONSISTENT DESIGN COLORS
# =============================================================================
# The same design uses the same color in Panels A-C.
# =============================================================================

default_colors_r4 = plt.rcParams[
    "axes.prop_cycle"
].by_key()[
    "color"
]


DESIGN_COLORS_R4 = {
    design_id_r4:
        default_colors_r4[
            index_r4
            % len(
                default_colors_r4
            )
        ]
    for index_r4, design_id_r4
    in enumerate(
        DESIGN_ORDER_R4
    )
}


# =============================================================================
# 6. LOAD ONE MATCHED HISTORY PER DESIGN
# =============================================================================
# We use the lowest-numbered seed for every design.
#
# This is preferable to selecting the run with the best validation score,
# because it avoids visual cherry-picking of convergence behavior.
# =============================================================================

history_frames_r4 = []

selected_history_rows_r4 = []


for design_id_r4 in DESIGN_ORDER_R4:

    matching_rows_r4 = (
        ROUND4_PRIMARY_ALL_RUNS.loc[
            ROUND4_PRIMARY_ALL_RUNS[
                "design_id"
            ].astype(str)
            == str(
                design_id_r4
            )
        ]
        .copy()
    )


    if matching_rows_r4.empty:

        print(
            f"Warning: no primary run found for {design_id_r4}."
        )

        continue


    # -------------------------------------------------------------------------
    # Select the same seed ordering rule for every design.
    # -------------------------------------------------------------------------

    if "seed_number" in matching_rows_r4.columns:

        matching_rows_r4 = (
            matching_rows_r4
            .sort_values(
                "seed_number"
            )
        )

    elif "seed_id" in matching_rows_r4.columns:

        matching_rows_r4 = (
            matching_rows_r4
            .sort_values(
                "seed_id"
            )
        )


    selected_row_r4 = (
        matching_rows_r4
        .iloc[0]
    )


    selected_label_r4 = str(
        selected_row_r4[
            "label"
        ]
    )


    history_path_r4 = (
        HISTORY_DIR_R4
        / f"{selected_label_r4}_history.csv"
    )


    if not history_path_r4.exists():

        print(
            f"Warning: history file not found for {design_id_r4}:"
        )

        print(
            " ",
            history_path_r4,
        )

        continue


    history_r4 = pd.read_csv(
        history_path_r4
    )


    history_r4[
        "design_id"
    ] = design_id_r4


    history_r4[
        "selected_label"
    ] = selected_label_r4


    history_r4[
        "history_index"
    ] = np.arange(
        len(
            history_r4
        ),
        dtype=int,
    )


    history_frames_r4.append(
        history_r4
    )


    selected_history_rows_r4.append(
        {
            "design_id":
                design_id_r4,

            "label":
                selected_label_r4,

            "history_file":
                str(
                    history_path_r4
                ),

            "n_logged_points":
                len(
                    history_r4
                ),

            "seed_number":
                (
                    selected_row_r4[
                        "seed_number"
                    ]
                    if "seed_number"
                    in selected_row_r4.index
                    else np.nan
                ),

            "seed_id":
                (
                    selected_row_r4[
                        "seed_id"
                    ]
                    if "seed_id"
                    in selected_row_r4.index
                    else ""
                ),
        }
    )


# =============================================================================
# 7. STOP IF NO HISTORIES WERE FOUND
# =============================================================================

if not history_frames_r4:

    raise FileNotFoundError(
        "No primary history files were found in:\n"
        f"{HISTORY_DIR_R4}"
    )


histories_r4 = pd.concat(
    history_frames_r4,
    ignore_index=True,
)


SELECTED_HISTORY_RUNS_R4 = pd.DataFrame(
    selected_history_rows_r4
)


print(
    "\nMatched convergence histories:"
)


display(
    SELECTED_HISTORY_RUNS_R4
)


# =============================================================================
# 8. VALIDATE HISTORY METRICS
# =============================================================================

required_history_columns_r4 = [
    "validation_score",
    "physics",
    "gpu_reserved_gb",
]


missing_history_columns_r4 = [
    column
    for column in required_history_columns_r4
    if column not in histories_r4.columns
]


if missing_history_columns_r4:

    raise KeyError(
        "History files are missing required diagnostic columns: "
        f"{missing_history_columns_r4}"
    )


# Convert required quantities safely to numeric.
for column_r4 in required_history_columns_r4:

    histories_r4[
        column_r4
    ] = pd.to_numeric(
        histories_r4[
            column_r4
        ],
        errors="coerce",
    )


# =============================================================================
# 9. CLEAN VALIDATION AND PHYSICS HISTORIES
# =============================================================================
# CRITICAL FIX:
#
# Previously:
#
#     physics.clip(lower=1e-16)
#
# converted zero or unavailable physics values into an apparently perfect
# residual of 1e-16.
#
# That is misleading on a logarithmic axis.
#
# Instead:
#
#     physics <= 0  -> NaN
#
# Matplotlib then leaves those portions blank.
# =============================================================================

histories_r4[
    "validation_plot"
] = histories_r4[
    "validation_score"
].where(
    np.isfinite(
        histories_r4[
            "validation_score"
        ]
    )
    &
    (
        histories_r4[
            "validation_score"
        ]
        > 0
    ),
    np.nan,
)


histories_r4[
    "physics_plot"
] = histories_r4[
    "physics"
].where(
    np.isfinite(
        histories_r4[
            "physics"
        ]
    )
    &
    (
        histories_r4[
            "physics"
        ]
        > 0
    ),
    np.nan,
)


histories_r4[
    "gpu_reserved_plot"
] = histories_r4[
    "gpu_reserved_gb"
].where(
    np.isfinite(
        histories_r4[
            "gpu_reserved_gb"
        ]
    ),
    np.nan,
)


# =============================================================================
# 10. DETECT TRAINING-PHASE BOUNDARIES
# =============================================================================
# Priority:
#
#   A. Explicit stage/phase column
#   B. Epoch reset locations
#   C. No phase markers if neither is available
#
# We derive the displayed boundaries from the first available matched history.
# =============================================================================

PHASE_COLUMN_CANDIDATES_R4 = [
    "stage",
    "phase",
    "training_stage",
    "optimizer_stage",
]


phase_column_r4 = next(
    (
        column
        for column in PHASE_COLUMN_CANDIDATES_R4
        if column in histories_r4.columns
    ),
    None,
)


phase_boundaries_r4 = []

phase_names_r4 = []


# Use M1 if available, otherwise the first design found.
reference_design_r4 = next(
    (
        design_id
        for design_id in DESIGN_ORDER_R4
        if design_id
        in histories_r4[
            "design_id"
        ].unique()
    ),
    histories_r4[
        "design_id"
    ].iloc[0],
)


reference_history_r4 = (
    histories_r4.loc[
        histories_r4[
            "design_id"
        ]
        == reference_design_r4
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# -----------------------------------------------------------------------------
# Option A: explicit phase/stage names
# -----------------------------------------------------------------------------

if phase_column_r4 is not None:

    phases_r4 = (
        reference_history_r4[
            phase_column_r4
        ]
        .astype(str)
        .fillna(
            ""
        )
        .tolist()
    )


    change_indices_r4 = [
        index_r4
        for index_r4 in range(
            1,
            len(
                phases_r4
            ),
        )
        if phases_r4[
            index_r4
        ]
        != phases_r4[
            index_r4 - 1
        ]
    ]


    phase_boundaries_r4 = (
        change_indices_r4
    )


    if len(
        phases_r4
    ) > 0:

        phase_names_r4 = [
            phases_r4[
                0
            ]
        ]


        for change_index_r4 in change_indices_r4:

            phase_names_r4.append(
                phases_r4[
                    change_index_r4
                ]
            )


# -----------------------------------------------------------------------------
# Option B: detect epoch resets
# -----------------------------------------------------------------------------

elif (
    "epoch"
    in reference_history_r4.columns
):

    epoch_values_r4 = pd.to_numeric(
        reference_history_r4[
            "epoch"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    valid_epoch_mask_r4 = np.isfinite(
        epoch_values_r4
    )


    if np.sum(
        valid_epoch_mask_r4
    ) >= 2:

        epoch_difference_r4 = np.diff(
            epoch_values_r4
        )


        phase_boundaries_r4 = (
            np.where(
                epoch_difference_r4
                < 0
            )[0]
            + 1
        ).tolist()


        if len(
            phase_boundaries_r4
        ) == 2:

            phase_names_r4 = [
                "Pretraining",
                "Physics warm-up",
                "Joint optimization",
            ]

        elif len(
            phase_boundaries_r4
        ) > 0:

            phase_names_r4 = [
                f"Phase {index_r4 + 1}"
                for index_r4 in range(
                    len(
                        phase_boundaries_r4
                    )
                    + 1
                )
            ]


# =============================================================================
# 11. CLEAN PHASE NAMES FOR DISPLAY
# =============================================================================

def clean_phase_name_r4(name):
    """
    Convert internal stage identifiers into publication-friendly labels.
    """

    text = str(
        name
    ).strip()


    text = text.replace(
        "_",
        " "
    )


    lower_text = text.lower()


    if "pretrain" in lower_text:

        return "Pretraining"


    if (
        "warm" in lower_text
        and "phys" in lower_text
    ):

        return "Physics warm-up"


    if (
        "joint" in lower_text
        or "adam" in lower_text
    ):

        return "Joint optimization"


    if "lbfgs" in lower_text:

        return "L-BFGS"


    return text.title()


phase_names_r4 = [
    clean_phase_name_r4(
        name
    )
    for name in phase_names_r4
]


# =============================================================================
# 12. PUBLICATION STYLE
# =============================================================================

CELL26_PUBLICATION_RC_R4 = {

    "font.family":
        "sans-serif",

    "font.size":
        10.5,

    "axes.titlesize":
        12.5,

    "axes.labelsize":
        11.0,

    "xtick.labelsize":
        9.5,

    "ytick.labelsize":
        9.5,

    "legend.fontsize":
        9.5,

    "axes.linewidth":
        1.0,

    "axes.edgecolor":
        "black",

    "axes.facecolor":
        "white",

    "figure.facecolor":
        "white",

    "savefig.facecolor":
        "white",
}


# =============================================================================
# 13. CREATE FINAL THREE-PANEL FIGURE
# =============================================================================

with plt.rc_context(
    CELL26_PUBLICATION_RC_R4
):

    fig_r4, axes_r4 = plt.subplots(
        1,
        3,
        figsize=(
            14.2,
            5.2,
        ),
    )


    # =========================================================================
    # 14. PLOT EACH DESIGN
    # =========================================================================

    for design_id_r4 in DESIGN_ORDER_R4:

        frame_r4 = (
            histories_r4.loc[
                histories_r4[
                    "design_id"
                ]
                == design_id_r4
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        if frame_r4.empty:
            continue


        x_r4 = np.arange(
            len(
                frame_r4
            ),
            dtype=float,
        )


        color_r4 = DESIGN_COLORS_R4[
            design_id_r4
        ]


        # ---------------------------------------------------------------------
        # Panel A — validation convergence
        # ---------------------------------------------------------------------

        axes_r4[
            0
        ].plot(
            x_r4,
            frame_r4[
                "validation_plot"
            ],

            color=color_r4,

            marker="o",
            markersize=2.8,

            linewidth=1.6,

            label=design_id_r4,
        )


        # ---------------------------------------------------------------------
        # Panel B — physics residual
        # ---------------------------------------------------------------------

        axes_r4[
            1
        ].plot(
            x_r4,
            frame_r4[
                "physics_plot"
            ],

            color=color_r4,

            marker="o",
            markersize=2.8,

            linewidth=1.6,

            label=design_id_r4,
        )


        # ---------------------------------------------------------------------
        # Panel C — CUDA reserved memory
        # ---------------------------------------------------------------------

        axes_r4[
            2
        ].plot(
            x_r4,
            frame_r4[
                "gpu_reserved_plot"
            ],

            color=color_r4,

            marker="o",
            markersize=2.8,

            linewidth=1.6,

            label=design_id_r4,
        )


    # =========================================================================
    # 15. LOGARITHMIC Y-SCALES
    # =========================================================================

    axes_r4[
        0
    ].set_yscale(
        "log"
    )


    axes_r4[
        1
    ].set_yscale(
        "log"
    )


    # Minor log ticks improve readability without excessive labels.
    for axis_r4 in axes_r4[
        :2
    ]:

        axis_r4.yaxis.set_minor_locator(
            LogLocator(
                base=10.0,
                subs=np.arange(
                    2,
                    10,
                )
                * 0.1,
            )
        )


        axis_r4.yaxis.set_minor_formatter(
            NullFormatter()
        )


    # =========================================================================
    # 16. TRAINING-PHASE BOUNDARIES
    # =========================================================================
    # Draw phase changes on Panels A and B only.
    #
    # The GPU panel remains visually simpler because it is a resource diagnostic.
    # =========================================================================

    for boundary_r4 in phase_boundaries_r4:

        for axis_r4 in axes_r4[
            :2
        ]:

            axis_r4.axvline(
                boundary_r4
                - 0.5,

                color="0.35",

                linestyle="--",

                linewidth=0.85,

                alpha=0.55,

                zorder=0,
            )



    # =========================================================================
    # 17. PANEL TITLES AND LABELS
    # =========================================================================

    axes_r4[
        0
    ].set_title(
        "Validation convergence",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
        pad=8,
    )


    axes_r4[
        0
    ].set_ylabel(
        "Validation score"
    )


    axes_r4[
        1
    ].set_title(
        "Physics-residual convergence",
        loc="left",
        fontweight="semibold",
        pad=8,
    )


    axes_r4[
        1
    ].set_ylabel(
        "Physics-residual loss"
    )


    axes_r4[
        2
    ].set_title(
        "CUDA memory reservation",
        loc="left",
        fontweight="semibold",
        pad=8,
    )


    axes_r4[
        2
    ].set_ylabel(
        "Reserved CUDA memory (GB)"
    )


    axes_r4[
        2
    ].set_ylim(
        bottom=0.0
    )


    # =========================================================================
    # 18. COMMON AXIS FORMATTING
    # =========================================================================

    for (
        panel_label_r4,
        axis_r4,
    ) in zip(
        [
            "A",
            "B",
            "C",
        ],
        axes_r4,
    ):

        axis_r4.set_xlabel(
            "Logged optimization step"
        )


        # Light publication-style grid.
        axis_r4.grid(
            which="major",
            alpha=0.16,
            linewidth=0.7,
        )


        if axis_r4 in axes_r4[
            :2
        ]:

            axis_r4.grid(
                which="minor",
                alpha=0.055,
                linewidth=0.45,
            )


        axis_r4.set_axisbelow(
            True
        )


        set_panel_frame_r4(
            axis_r4
        )


        axis_r4.text(
            -0.13,
            1.055,
            panel_label_r4,

            transform=axis_r4.transAxes,

            fontsize=12.5,
            fontweight="bold",

            ha="left",
            va="bottom",
        )


    # =========================================================================
    # 19. CLEAN SHARED LEGEND
    # =========================================================================

    handles_r4, labels_r4 = (
        axes_r4[
            0
        ]
        .get_legend_handles_labels()
    )


    # Remove accidental duplicates while preserving order.
    legend_mapping_r4 = {}


    for handle_r4, label_r4 in zip(
        handles_r4,
        labels_r4,
    ):

        if label_r4 not in legend_mapping_r4:

            legend_mapping_r4[
                label_r4
            ] = handle_r4


    fig_r4.legend(
        list(
            legend_mapping_r4.values()
        ),

        list(
            legend_mapping_r4.keys()
        ),

        loc="upper center",

        bbox_to_anchor=(
            0.50,
            0.915,
        ),

        ncol=len(
            legend_mapping_r4
        ),

        frameon=False,

        handlelength=2.0,

        columnspacing=1.5,
    )


    # =========================================================================
    # 20. TRAINING-PHASE KEY
    # =========================================================================
    # Phase names are intentionally kept OUTSIDE Panel A. The vertical dashed
    # lines in Panels A and B still show the transition locations, while this
    # compact key explains the phase sequence without obscuring the data.
    # =========================================================================

    if phase_names_r4:

        phase_key_r4 = (
            "Training phases:   "
            + "   |   ".join(
                phase_names_r4
            )
        )

        fig_r4.text(
            0.50,
            0.835,
            phase_key_r4,
            ha="center",
            va="center",
            fontsize=8.6,
            color="0.30",
        )


    # =========================================================================
    # 21. MAIN TITLE
    # =========================================================================

    fig_r4.suptitle(
        "Optimization diagnostics across nested schedule designs",
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
    )


    # =========================================================================
    # 22. FINAL LAYOUT
    # =========================================================================
    # Large top margin separates:
    #
    #       title
    #       legend
    #       panel titles
    #
    # and prevents the overlap present in the original figure.
    # =========================================================================

    fig_r4.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.15,
        top=0.70,
        wspace=0.30,
    )


    # =========================================================================
    # 23. SAVE AND DISPLAY
    # =========================================================================

    save_figure_r4(
        fig_r4,
        "round4_convergence_and_gpu_memory_final",
    )


    plt.show()


# =============================================================================
# 24. CREATE NUMERICAL DIAGNOSTIC SUMMARY
# =============================================================================

diagnostic_summary_rows_r4 = []


for design_id_r4 in DESIGN_ORDER_R4:

    frame_r4 = (
        histories_r4.loc[
            histories_r4[
                "design_id"
            ]
            == design_id_r4
        ]
        .copy()
    )


    if frame_r4.empty:
        continue


    valid_validation_r4 = frame_r4[
        "validation_plot"
    ].dropna()


    valid_physics_r4 = frame_r4[
        "physics_plot"
    ].dropna()


    valid_memory_r4 = frame_r4[
        "gpu_reserved_plot"
    ].dropna()


    diagnostic_summary_rows_r4.append(
        {
            "design_id":
                design_id_r4,

            "logged_points":
                int(
                    len(
                        frame_r4
                    )
                ),

            "minimum_validation_score":
                (
                    float(
                        valid_validation_r4.min()
                    )
                    if not valid_validation_r4.empty
                    else np.nan
                ),

            "final_validation_score":
                (
                    float(
                        valid_validation_r4.iloc[
                            -1
                        ]
                    )
                    if not valid_validation_r4.empty
                    else np.nan
                ),

            "minimum_physics_residual":
                (
                    float(
                        valid_physics_r4.min()
                    )
                    if not valid_physics_r4.empty
                    else np.nan
                ),

            "final_physics_residual":
                (
                    float(
                        valid_physics_r4.iloc[
                            -1
                        ]
                    )
                    if not valid_physics_r4.empty
                    else np.nan
                ),

            "maximum_logged_cuda_reserved_gb":
                (
                    float(
                        valid_memory_r4.max()
                    )
                    if not valid_memory_r4.empty
                    else np.nan
                ),
        }
    )


ROUND4_OPTIMIZATION_DIAGNOSTICS_R4 = pd.DataFrame(
    diagnostic_summary_rows_r4
)


OPTIMIZATION_DIAGNOSTICS_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_optimization_diagnostics_summary.csv"
)


ROUND4_OPTIMIZATION_DIAGNOSTICS_R4.to_csv(
    OPTIMIZATION_DIAGNOSTICS_PATH_R4,
    index=False,
)


print(
    "\nOptimization-diagnostic summary:"
)


display(
    ROUND4_OPTIMIZATION_DIAGNOSTICS_R4
)


# =============================================================================
# 25. FINAL REPORT
# =============================================================================

print(
    "\n"
    + "=" * 86
)


print(
    "ROUND 4 CELL 26 COMPLETED SUCCESSFULLY"
)


print(
    "=" * 86
)


print(
    "\nNO MODEL TRAINING WAS PERFORMED."
)


print(
    "\nMatched designs plotted:"
)


print(
    " ",
    [
        design_id_r4
        for design_id_r4
        in DESIGN_ORDER_R4
        if design_id_r4
        in histories_r4[
            "design_id"
        ].unique()
    ],
)


if phase_boundaries_r4:

    print(
        "\nDetected optimization phase boundaries:"
    )


    print(
        " ",
        phase_boundaries_r4,
    )


    if phase_names_r4:

        print(
            "Phase labels:"
        )

        print(
            " ",
            phase_names_r4,
        )

else:

    print(
        "\nNo explicit optimization phase boundaries were available "
        "in the saved history files."
    )


print(
    "\nPhysics-history correction:"
)


print(
    "  Non-positive/unavailable physics values were omitted rather than "
    "being clipped to 1e-16."
)


print(
    "\nGPU-memory interpretation:"
)


print(
    "  The figure reports CUDA reserved memory recorded at history log points; "
    "it is not a measure of total GPU capacity or instantaneous peak usage."
)


print(
    "\nSaved final figure:"
)


print(
    " ",
    FIGURE_DIR_R4
    / "round4_convergence_and_gpu_memory_final",
)


print(
    "\nSaved diagnostic summary:"
)


print(
    " ",
    OPTIMIZATION_DIAGNOSTICS_PATH_R4,
)

# In[56]:


# =============================================================================
# ROUND 4 — CELL 27
# FINAL MANUSCRIPT REPRODUCIBILITY AND HARD-CONSTRAINT AUDIT
# =============================================================================
# PURPOSE
# -------
# This is the final integrity/reproducibility check for Round 4.
#
# It does NOT train or optimize any model.
#
# It:
#   1. Reloads the completed 25-run primary table if necessary.
#   2. Identifies the exact primary M1-M5 runs.
#   3. Verifies that every expected all-schedule prediction file exists.
#   4. Audits every saved primary trajectory for:
#
#          S(t) >= 0
#         chedule prediction file exists.
#   4. Audits every saved primary trajectory for:
#
#          R(t) >= 0
#          S(t) + R(t) <= K
#
#      and, when t = T_START is present:
#
#          S(T_START) = INITIAL_S
#          R(T_START) = INITIAL_R
#
#   5. Saves:
#
#          round4_constraint_audit.csv
#          round4_manifest.json
#          round4_file_inventory.txt
#
# IMPORTANT
# ---------
# This final version intentionally does NOT use:
#
#     ROUND4_SMOKE_RESULT
#     ROUND4_DEMO_RESULT
#
# because those objects do not exist when the smoke/demo cells are skipped
# during the manuscript-scale final run.
#
# NO MODEL TRAINING IS PERFORMED.
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
from dataclasses import asdict, is_dataclass
import json
import math
import platform

import numpy as np
import pandas as pd


# =============================================================================
# 2. FINAL OUTPUT PATHS
# =============================================================================

PRIMARY_RESULTS_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_primary_multischedule_all_seeds_metrics.csv"
)

CONSTRAINT_AUDIT_PATH_R4 = (
    TABLE_DIR_R4
    / "round4_constraint_audit.csv"
)

MANIFEST_PATH_R4 = (
    REPORT_DIR_R4
    / "round4_manifest.json"
)

INVENTORY_PATH_R4 = (
    REPORT_DIR_R4
    / "round4_file_inventory.txt"
)


# =============================================================================
# 3. ENSURE OUTPUT DIRECTORIES EXIST
# =============================================================================

for directory_r4 in [
    TABLE_DIR_R4,
    REPORT_DIR_R4,
]:
    directory_r4.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# 4. JSON-SAFE CONVERSION HELPER
# =============================================================================

def json_safe_r4(value):
    """
    Recursively convert common Python, NumPy, Path and dataclass objects
    to JSON-serializable forms.
    """

    if is_dataclass(value):
        return json_safe_r4(
            asdict(value)
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): json_safe_r4(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe_r4(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


# =============================================================================
# 5. LOAD THE COMPLETED PRIMARY RESULTS TABLE
# =============================================================================
# This makes Cell 27 safe after a kernel restart.
# =============================================================================

if (
    "ROUND4_PRIMARY_ALL_RUNS" not in globals()
    or ROUND4_PRIMARY_ALL_RUNS is None
    or ROUND4_PRIMARY_ALL_RUNS.empty
):

    if not PRIMARY_RESULTS_PATH_R4.exists():

        raise FileNotFoundError(
            "The completed Round 4 primary-results table was not found:\n"
            f"{PRIMARY_RESULTS_PATH_R4}\n\n"
            "Cell 27 requires the completed primary experiment."
        )

    ROUND4_PRIMARY_ALL_RUNS = pd.read_csv(
        PRIMARY_RESULTS_PATH_R4
    )

    print(
        "Loaded completed primary Round 4 results:"
    )
    print(
        " ",
        PRIMARY_RESULTS_PATH_R4
    )


# =============================================================================
# 6. IDENTIFY ONLY THE FINAL PRIMARY M1-M5 RUNS
# =============================================================================

if "DESIGN_ORDER_R4" not in globals():
    DESIGN_ORDER_R4 = [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]


required_primary_columns_r4 = [
    "label",
    "design_id",
]


missing_primary_columns_r4 = [
    column_r4
    for column_r4 in required_primary_columns_r4
    if column_r4 not in ROUND4_PRIMARY_ALL_RUNS.columns
]


if missing_primary_columns_r4:

    raise KeyError(
        "Primary results are missing required columns: "
        f"{missing_primary_columns_r4}"
    )


PRIMARY_AUDIT_RUNS_R4 = (
    ROUND4_PRIMARY_ALL_RUNS.loc[
        ROUND4_PRIMARY_ALL_RUNS[
            "design_id"
        ].astype(str).isin(
            DESIGN_ORDER_R4
        )
    ]
    .copy()
)


# Keep only the design-level rows if that column exists.
if "row_type" in PRIMARY_AUDIT_RUNS_R4.columns:

    design_rows_r4 = (
        PRIMARY_AUDIT_RUNS_R4[
            "row_type"
        ].astype(str)
        == "design"
    )

    if design_rows_r4.any():
        PRIMARY_AUDIT_RUNS_R4 = (
            PRIMARY_AUDIT_RUNS_R4.loc[
                design_rows_r4
            ]
            .copy()
        )


# Keep only ALL-schedule summary rows if schedule exists.
if "schedule" in PRIMARY_AUDIT_RUNS_R4.columns:

    all_rows_r4 = (
        PRIMARY_AUDIT_RUNS_R4[
            "schedule"
        ].astype(str)
        == "ALL"
    )

    if all_rows_r4.any():
        PRIMARY_AUDIT_RUNS_R4 = (
            PRIMARY_AUDIT_RUNS_R4.loc[
                all_rows_r4
            ]
            .copy()
        )


# Remove accidental duplicates.
PRIMARY_AUDIT_RUNS_R4 = (
    PRIMARY_AUDIT_RUNS_R4
    .drop_duplicates(
        subset=[
            "label"
        ]
    )
)


# Sort cleanly.
sort_columns_r4 = [
    column_r4
    for column_r4 in [
        "design_id",
        "seed_number",
        "seed_id",
    ]
    if column_r4
    in PRIMARY_AUDIT_RUNS_R4.columns
]


if sort_columns_r4:

    PRIMARY_AUDIT_RUNS_R4 = (
        PRIMARY_AUDIT_RUNS_R4
        .sort_values(
            sort_columns_r4
        )
        .reset_index(
            drop=True
        )
    )


EXPECTED_PRIMARY_LABELS_R4 = (
    PRIMARY_AUDIT_RUNS_R4[
        "label"
    ]
    .astype(str)
    .tolist()
)


print(
    "\nPrimary manuscript runs identified:",
    len(EXPECTED_PRIMARY_LABELS_R4),
)


# For your final experiment this should be 25.
if not QUICK_MODE:

    if len(EXPECTED_PRIMARY_LABELS_R4) != 25:

        print(
            "WARNING:"
        )
        print(
            "  Expected 25 final primary runs "
            "(5 designs x 5 seeds), but found",
            len(EXPECTED_PRIMARY_LABELS_R4),
        )


# =============================================================================
# 7. BUILD THE EXPECTED PRIMARY PREDICTION FILE LIST
# =============================================================================
# IMPORTANT:
#
# We construct each path directly from the primary-run labels.
# Therefore optional LOO/demo/smoke prediction files can never enter this audit.
# =============================================================================

expected_prediction_paths_r4 = {
    label_r4: (
        PREDICTION_DIR_R4
        / f"{label_r4}_all_schedule_predictions.csv"
    )
    for label_r4
    in EXPECTED_PRIMARY_LABELS_R4
}


missing_prediction_labels_r4 = [
    label_r4
    for label_r4, path_r4
    in expected_prediction_paths_r4.items()
    if not path_r4.exists()
]


found_prediction_labels_r4 = [
    label_r4
    for label_r4, path_r4
    in expected_prediction_paths_r4.items()
    if path_r4.exists()
]


print(
    "Saved primary prediction sets found:",
    len(found_prediction_labels_r4),
)


print(
    "Missing primary prediction sets:",
    len(missing_prediction_labels_r4),
)


if missing_prediction_labels_r4:

    print(
        "Missing labels:"
    )

    for label_r4 in missing_prediction_labels_r4:
        print(
            " ",
            label_r4,
        )


# =============================================================================
# 8. HARD-CONSTRAINT DEFINITIONS
# =============================================================================

K_R4 = float(
    TRUE_PARAMETERS.carrying_capacity
)


INITIAL_S_R4 = float(
    INITIAL_S
)


INITIAL_R_R4 = float(
    INITIAL_R
)


INITIAL_S_NORMALIZED_R4 = (
    INITIAL_S_R4
    / K_R4
)


INITIAL_R_NORMALIZED_R4 = (
    INITIAL_R_R4
    / K_R4
)


# Numerical tolerances after writing/reading CSV files.
CONSTRAINT_TOL_R4 = 1.0e-8
INITIAL_TOL_R4 = 1.0e-8


# =============================================================================
# 9. AUDIT EVERY SAVED PRIMARY TRAJECTORY
# =============================================================================

constraint_rows_r4 = []


required_prediction_columns_r4 = [
    "time",
    "predicted_sensitive",
    "predicted_resistant",
]


for label_r4 in EXPECTED_PRIMARY_LABELS_R4:

    prediction_path_r4 = (
        expected_prediction_paths_r4[
            label_r4
        ]
    )


    if not prediction_path_r4.exists():
        continue


    prediction_frame_r4 = pd.read_csv(
        prediction_path_r4
    )


    missing_columns_r4 = [
        column_r4
        for column_r4
        in required_prediction_columns_r4
        if column_r4
        not in prediction_frame_r4.columns
    ]


    if missing_columns_r4:

        constraint_rows_r4.append(
            {
                "label":
                    label_r4,

                "schedule":
                    "FILE_ERROR",

                "n_points":
                    0,

                "minimum_sensitive_normalized":
                    np.nan,

                "minimum_resistant_normalized":
                    np.nan,

                "maximum_total_normalized":
                    np.nan,

                "initial_state_error_normalized":
                    np.nan,

                "initial_state_checked":
                    False,

                "positivity_passed":
                    False,

                "capacity_passed":
                    False,

                "initial_condition_passed":
                    False,

                "all_constraints_passed":
                    False,

                "audit_note":
                    (
                        "Missing required columns: "
                        + ", ".join(
                            missing_columns_r4
                        )
                    ),
            }
        )

        continue


    # -------------------------------------------------------------------------
    # Separate schedules.
    # -------------------------------------------------------------------------

    if "schedule" in prediction_frame_r4.columns:

        schedule_groups_r4 = (
            prediction_frame_r4
            .groupby(
                "schedule",
                observed=True,
                sort=False,
            )
        )

    else:

        schedule_groups_r4 = [
            (
                "ALL",
                prediction_frame_r4,
            )
        ]


    for (
        schedule_name_r4,
        schedule_frame_r4,
    ) in schedule_groups_r4:

        schedule_frame_r4 = (
            schedule_frame_r4
            .copy()
            .sort_values(
                "time"
            )
            .reset_index(
                drop=True
            )
        )


        time_values_r4 = pd.to_numeric(
            schedule_frame_r4[
                "time"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        sensitive_values_r4 = pd.to_numeric(
            schedule_frame_r4[
                "predicted_sensitive"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        resistant_values_r4 = pd.to_numeric(
            schedule_frame_r4[
                "predicted_resistant"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        finite_mask_r4 = (
            np.isfinite(
                time_values_r4
            )
            &
            np.isfinite(
                sensitive_values_r4
            )
            &
            np.isfinite(
                resistant_values_r4
            )
        )


        n_finite_r4 = int(
            finite_mask_r4.sum()
        )


        if n_finite_r4 == 0:

            constraint_rows_r4.append(
                {
                    "label":
                        label_r4,

                    "schedule":
                        str(
                            schedule_name_r4
                        ),

                    "n_points":
                        0,

                    "minimum_sensitive_normalized":
                        np.nan,

                    "minimum_resistant_normalized":
                        np.nan,

                    "maximum_total_normalized":
                        np.nan,

                    "initial_state_error_normalized":
                        np.nan,

                    "initial_state_checked":
                        False,

                    "positivity_passed":
                        False,

                    "capacity_passed":
                        False,

                    "initial_condition_passed":
                        False,

                    "all_constraints_passed":
                        False,

                    "audit_note":
                        "No finite prediction points.",
                }
            )

            continue


        time_valid_r4 = (
            time_values_r4[
                finite_mask_r4
            ]
        )


        sensitive_valid_r4 = (
            sensitive_values_r4[
                finite_mask_r4
            ]
        )


        resistant_valid_r4 = (
            resistant_values_r4[
                finite_mask_r4
            ]
        )


        # ---------------------------------------------------------------------
        # Normalize by carrying capacity.
        # ---------------------------------------------------------------------

        s_norm_r4 = (
            sensitive_valid_r4
            / K_R4
        )


        r_norm_r4 = (
            resistant_valid_r4
            / K_R4
        )


        total_norm_r4 = (
            s_norm_r4
            +
            r_norm_r4
        )


        minimum_sensitive_r4 = float(
            np.min(
                s_norm_r4
            )
        )


        minimum_resistant_r4 = float(
            np.min(
                r_norm_r4
            )
        )


        maximum_total_r4 = float(
            np.max(
                total_norm_r4
            )
        )


        # ---------------------------------------------------------------------
        # Positivity constraints.
        # ---------------------------------------------------------------------

        positivity_passed_r4 = bool(
            minimum_sensitive_r4
            >= -CONSTRAINT_TOL_R4

            and

            minimum_resistant_r4
            >= -CONSTRAINT_TOL_R4
        )


        # ---------------------------------------------------------------------
        # Carrying-capacity constraint.
        # ---------------------------------------------------------------------

        capacity_passed_r4 = bool(
            maximum_total_r4
            <= (
                1.0
                + CONSTRAINT_TOL_R4
            )
        )


        # ---------------------------------------------------------------------
        # Initial-condition audit.
        # ---------------------------------------------------------------------

        initial_state_checked_r4 = False
        initial_state_error_r4 = np.nan
        initial_condition_passed_r4 = True


        initial_candidates_r4 = np.where(
            np.isclose(
                time_valid_r4,
                float(
                    T_START
                ),
                atol=1.0e-10,
                rtol=0.0,
            )
        )[0]


        if len(
            initial_candidates_r4
        ) > 0:

            initial_index_r4 = int(
                initial_candidates_r4[
                    0
                ]
            )


            predicted_initial_s_norm_r4 = float(
                sensitive_valid_r4[
                    initial_index_r4
                ]
                / K_R4
            )


            predicted_initial_r_norm_r4 = float(
                resistant_valid_r4[
                    initial_index_r4
                ]
                / K_R4
            )


            initial_state_error_r4 = float(
                max(
                    abs(
                        predicted_initial_s_norm_r4
                        -
                        INITIAL_S_NORMALIZED_R4
                    ),
                    abs(
                        predicted_initial_r_norm_r4
                        -
                        INITIAL_R_NORMALIZED_R4
                    ),
                )
            )


            initial_state_checked_r4 = True


            initial_condition_passed_r4 = bool(
                initial_state_error_r4
                <= INITIAL_TOL_R4
            )


        # ---------------------------------------------------------------------
        # Final decision.
        # ---------------------------------------------------------------------

        all_constraints_passed_r4 = bool(
            positivity_passed_r4
            and
            capacity_passed_r4
            and
            initial_condition_passed_r4
        )


        constraint_rows_r4.append(
            {
                "label":
                    label_r4,

                "schedule":
                    str(
                        schedule_name_r4
                    ),

                "n_points":
                    n_finite_r4,

                "minimum_sensitive_normalized":
                    minimum_sensitive_r4,

                "minimum_resistant_normalized":
                    minimum_resistant_r4,

                "maximum_total_normalized":
                    maximum_total_r4,

                "initial_state_error_normalized":
                    initial_state_error_r4,

                "initial_state_checked":
                    initial_state_checked_r4,

                "positivity_passed":
                    positivity_passed_r4,

                "capacity_passed":
                    capacity_passed_r4,

                "initial_condition_passed":
                    initial_condition_passed_r4,

                "all_constraints_passed":
                    all_constraints_passed_r4,

                "audit_note":
                    "OK",
            }
        )


# =============================================================================
# 10. CREATE THE FINAL CONSTRAINT-AUDIT TABLE
# =============================================================================

ROUND4_CONSTRAINT_AUDIT = pd.DataFrame(
    constraint_rows_r4
)


if ROUND4_CONSTRAINT_AUDIT.empty:

    raise RuntimeError(
        "No primary prediction trajectory could be audited."
    )


ROUND4_CONSTRAINT_AUDIT.to_csv(
    CONSTRAINT_AUDIT_PATH_R4,
    index=False,
)


constraint_total_count_r4 = int(
    len(
        ROUND4_CONSTRAINT_AUDIT
    )
)


constraint_pass_count_r4 = int(
    ROUND4_CONSTRAINT_AUDIT[
        "all_constraints_passed"
    ]
    .astype(bool)
    .sum()
)


constraint_fail_count_r4 = (
    constraint_total_count_r4
    -
    constraint_pass_count_r4
)


all_constraints_passed_r4 = bool(
    ROUND4_CONSTRAINT_AUDIT[
        "all_constraints_passed"
    ]
    .astype(bool)
    .all()
)


# =============================================================================
# 11. INITIAL-CONDITION COVERAGE
# =============================================================================

initial_checked_count_r4 = int(
    ROUND4_CONSTRAINT_AUDIT[
        "initial_state_checked"
    ]
    .astype(bool)
    .sum()
)


# =============================================================================
# 12. MANUSCRIPT-SCALE COMPLETENESS CHECK
# =============================================================================

primary_prediction_complete_r4 = bool(
    len(
        missing_prediction_labels_r4
    )
    == 0
)


expected_primary_run_count_r4 = int(
    len(
        EXPECTED_PRIMARY_LABELS_R4
    )
)


found_primary_run_count_r4 = int(
    len(
        found_prediction_labels_r4
    )
)


# =============================================================================
# 13. RUNTIME SWITCHES
# =============================================================================

RUNTIME_SWITCHES_R4 = {

    "structural_smoke_test":
        bool(
            globals().get(
                "RUN_STRUCTURAL_SMOKE_TEST",
                False,
            )
        ),

    "single_design_demo":
        bool(
            globals().get(
                "RUN_SINGLE_DESIGN_DEMO",
                False,
            )
        ),

    "primary_multi_schedule_study":
        bool(
            globals().get(
                "RUN_PRIMARY_MULTI_SCHEDULE_STUDY",
                False,
            )
        ),

    "matched_ode_baselines":
        bool(
            globals().get(
                "RUN_MATCHED_ODE_BASELINES",
                False,
            )
        ),

    "identifiability_geometry":
        bool(
            globals().get(
                "RUN_IDENTIFIABILITY_GEOMETRY",
                False,
            )
        ),

    "leave_one_schedule_out":
        bool(
            globals().get(
                "RUN_LEAVE_ONE_SCHEDULE_OUT",
                False,
            )
        ),

    "profile_surfaces":
        bool(
            globals().get(
                "RUN_PROFILE_SURFACES",
                False,
            )
        ),

    "parameter_transfer_test":
        bool(
            globals().get(
                "RUN_PARAMETER_TRANSFER_TEST",
                False,
            )
        ),

    "plotting_only":
        bool(
            globals().get(
                "RUN_PLOTTING_ONLY",
                False,
            )
        ),
}


# =============================================================================
# 14. DESIGN METADATA
# =============================================================================

design_manifest_r4 = {}


if "ROUND4_DESIGNS" in globals():

    for design_id_r4 in DESIGN_ORDER_R4:

        if design_id_r4 not in ROUND4_DESIGNS:
            continue


        design_object_r4 = (
            ROUND4_DESIGNS[
                design_id_r4
            ]
        )


        if is_dataclass(
            design_object_r4
        ):

            design_manifest_r4[
                design_id_r4
            ] = asdict(
                design_object_r4
            )

        else:

            design_manifest_r4[
                design_id_r4
            ] = json_safe_r4(
                design_object_r4
            )


# =============================================================================
# 15. BUILD BASE REPRODUCIBILITY MANIFEST
# =============================================================================

ROUND4_AUDIT = {

    "round":
        4,

    "quick_mode":
        bool(
            QUICK_MODE
        ),

    "run_label":
        str(
            RUN_LABEL
        ),

    "project_root":
        str(
            PROJECT_ROOT.resolve()
        ),

    "output_root":
        str(
            ROUND4_ROOT.resolve()
        ),

    "python":
        platform.python_version(),

    "platform":
        platform.platform(),

    "seed":
        int(
            SEED
        ),

    "device":
        str(
            globals().get(
                "DEVICE",
                "unknown",
            )
        ),

    "dtype":
        str(
            globals().get(
                "DTYPE",
                "unknown",
            )
        ),

    "training_budgets": {

        "pretrain_epochs":
            int(
                PRETRAIN_EPOCHS_R4
            ),

        "physics_warmup_epochs":
            int(
                PHYSICS_WARMUP_EPOCHS_R4
            ),

        "adam_epochs":
            int(
                ADAM_EPOCHS_R4
            ),

        "lbfgs_max_iter":
            int(
                LBFGS_MAX_ITER_R4
            ),

        "collocation_points_per_interval":
            int(
                COLLOCATION_POINTS_PER_INTERVAL_R4
            ),

        "validation_points_per_interval":
            int(
                VALIDATION_POINTS_PER_INTERVAL_R4
            ),

        "lbfgs_points_per_interval":
            int(
                globals().get(
                    "LBFGS_POINTS_PER_INTERVAL_R4",
                    0,
                )
            ),

        "early_stopping_patience":
            int(
                globals().get(
                    "EARLY_STOPPING_PATIENCE_R4",
                    0,
                )
            ),

        "log_every":
            int(
                globals().get(
                    "LOG_EVERY_R4",
                    0,
                )
            ),

        "resample_every":
            int(
                globals().get(
                    "RESAMPLE_EVERY_R4",
                    0,
                )
            ),
    },

    "network": {

        "width":
            int(
                NETWORK_WIDTH_R4
            ),

        "depth":
            int(
                NETWORK_DEPTH_R4
            ),

        "state_learning_rate":
            float(
                STATE_LEARNING_RATE_R4
            ),

        "parameter_learning_rate":
            float(
                PARAMETER_LEARNING_RATE_R4
            ),

        "gradient_clip_norm":
            float(
                GRADIENT_CLIP_NORM_R4
            ),
    },

    "primary_seed_ids":
        json_safe_r4(
            PRIMARY_SEED_IDS_R4
        ),

    "primary_designs":
        json_safe_r4(
            design_manifest_r4
        ),

    "initial_experiment_switches":
        json_safe_r4(
            globals().get(
                "EXPERIMENT_SWITCHES_R4",
                {},
            )
        ),

    "runtime_switches_at_final_audit":
        json_safe_r4(
            RUNTIME_SWITCHES_R4
        ),

    "repository_provenance":
        json_safe_r4(
            globals().get(
                "REPOSITORY_PROVENANCE_R4",
                {},
            )
        ),

    "primary_prediction_completeness": {

        "expected_primary_runs":
            expected_primary_run_count_r4,

        "found_primary_prediction_sets":
            found_primary_run_count_r4,

        "missing_prediction_sets":
            len(
                missing_prediction_labels_r4
            ),

        "missing_labels":
            missing_prediction_labels_r4,

        "complete":
            primary_prediction_complete_r4,
    },

    "hard_constraint_audit": {

        "carrying_capacity":
            K_R4,

        "constraint_tolerance_normalized":
            CONSTRAINT_TOL_R4,

        "initial_condition_tolerance_normalized":
            INITIAL_TOL_R4,

        "audited_trajectory_records":
            constraint_total_count_r4,

        "passed_trajectory_records":
            constraint_pass_count_r4,

        "failed_trajectory_records":
            constraint_fail_count_r4,

        "initial_conditions_checked":
            initial_checked_count_r4,

        "all_constraints_passed":
            all_constraints_passed_r4,

        "constraint_table":
            str(
                CONSTRAINT_AUDIT_PATH_R4.resolve()
            ),
    },
}


# =============================================================================
# 16. FILE-COUNT HELPER
# =============================================================================

def count_files_r4(directory_r4):
    """
    Count files recursively inside a directory.
    """

    return len(
        [
            path_r4
            for path_r4
            in directory_r4.rglob("*")
            if path_r4.is_file()
        ]
    )


# =============================================================================
# 17. CREATE PLACEHOLDER REPORT FILES BEFORE FINAL INVENTORY
# =============================================================================
# This ensures that both report files themselves appear in the inventory/count.
# =============================================================================

MANIFEST_PATH_R4.touch(
    exist_ok=True
)


INVENTORY_PATH_R4.touch(
    exist_ok=True
)


# =============================================================================
# 18. FINAL FILE COUNTS
# =============================================================================

FINAL_FILE_COUNTS_R4 = {

    "model_files":
        count_files_r4(
            MODEL_DIR_R4
        ),

    "prediction_files":
        count_files_r4(
            PREDICTION_DIR_R4
        ),

    "history_files":
        count_files_r4(
            HISTORY_DIR_R4
        ),

    "figure_files":
        count_files_r4(
            FIGURE_DIR_R4
        ),

    "table_files":
        count_files_r4(
            TABLE_DIR_R4
        ),

    "report_files":
        count_files_r4(
            REPORT_DIR_R4
        ),
}


ROUND4_AUDIT[
    "file_counts"
] = FINAL_FILE_COUNTS_R4


# =============================================================================
# 19. CREATE FINAL FILE INVENTORY
# =============================================================================

inventory_r4 = sorted(
    str(
        path_r4.relative_to(
            PROJECT_ROOT
        )
    )
    for path_r4
    in ROUND4_ROOT.rglob("*")
    if path_r4.is_file()
)


FINAL_FILE_COUNTS_R4[
    "total_files"
] = len(
    inventory_r4
)


ROUND4_AUDIT[
    "file_counts"
][
    "total_files"
] = len(
    inventory_r4
)


INVENTORY_PATH_R4.write_text(
    "\n".join(
        inventory_r4
    ),
    encoding="utf-8",
)


# =============================================================================
# 20. WRITE FINAL JSON MANIFEST
# =============================================================================

with open(
    MANIFEST_PATH_R4,
    "w",
    encoding="utf-8",
) as file_r4:

    json.dump(
        json_safe_r4(
            ROUND4_AUDIT
        ),
        file_r4,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 21. DISPLAY THE CONSTRAINT-AUDIT SUMMARY
# =============================================================================

constraint_preview_columns_r4 = [
    "label",
    "schedule",
    "minimum_sensitive_normalized",
    "minimum_resistant_normalized",
    "maximum_total_normalized",
    "initial_state_error_normalized",
    "all_constraints_passed",
]


print(
    "\nConstraint-audit preview:"
)


display(
    ROUND4_CONSTRAINT_AUDIT[
        constraint_preview_columns_r4
    ].head(
        15
    )
)


# =============================================================================
# 22. OPTIONAL PER-DESIGN CONSTRAINT SUMMARY
# =============================================================================

audit_with_design_r4 = (
    ROUND4_CONSTRAINT_AUDIT
    .copy()
)


audit_with_design_r4[
    "design_id"
] = (
    audit_with_design_r4[
        "label"
    ]
    .str.extract(
        r"round4_(M\d)_"
    )[
        0
    ]
)


ROUND4_CONSTRAINT_SUMMARY_BY_DESIGN = (
    audit_with_design_r4
    .groupby(
        "design_id",
        observed=True,
    )
    .agg(
        audited_trajectories=(
            "all_constraints_passed",
            "size",
        ),

        passed_trajectories=(
            "all_constraints_passed",
            "sum",
        ),

        minimum_S_over_K=(
            "minimum_sensitive_normalized",
            "min",
        ),

        minimum_R_over_K=(
            "minimum_resistant_normalized",
            "min",
        ),

        maximum_total_over_K=(
            "maximum_total_normalized",
            "max",
        ),

        maximum_initial_error=(
            "initial_state_error_normalized",
            "max",
        ),
    )
    .reset_index()
)


print(
    "\nConstraint summary by design:"
)


display(
    ROUND4_CONSTRAINT_SUMMARY_BY_DESIGN
)


# =============================================================================
# 23. FINAL IMPLEMENTATION AUDIT REPORT
# =============================================================================

print(
    "\n"
    + "=" * 86
)


print(
    "ROUND 4 FINAL IMPLEMENTATION AUDIT"
)


print(
    "=" * 86
)


print(
    "\nRun configuration:"
)


print(
    "  QUICK_MODE:",
    QUICK_MODE,
)


print(
    "  Run label:",
    RUN_LABEL,
)


print(
    "  Output root:",
    ROUND4_ROOT.resolve(),
)


print(
    "\nPrimary prediction completeness:"
)


print(
    "  Expected primary runs:",
    expected_primary_run_count_r4,
)


print(
    "  Found primary prediction sets:",
    found_primary_run_count_r4,
)


print(
    "  Missing prediction sets:",
    len(
        missing_prediction_labels_r4
    ),
)


print(
    "  Complete:",
    primary_prediction_complete_r4,
)


print(
    "\nHard-constraint audit:"
)


print(
    "  Audited trajectory records:",
    constraint_total_count_r4,
)


print(
    "  Passed:",
    constraint_pass_count_r4,
)


print(
    "  Failed:",
    constraint_fail_count_r4,
)


print(
    "  Initial conditions checked:",
    initial_checked_count_r4,
)


print(
    "  All hard constraints passed:",
    all_constraints_passed_r4,
)


print(
    "\nRound 4 output inventory:"
)


print(
    "  Model files:",
    FINAL_FILE_COUNTS_R4[
        "model_files"
    ],
)


print(
    "  Prediction files:",
    FINAL_FILE_COUNTS_R4[
        "prediction_files"
    ],
)


print(
    "  History files:",
    FINAL_FILE_COUNTS_R4[
        "history_files"
    ],
)


print(
    "  Figure files:",
    FINAL_FILE_COUNTS_R4[
        "figure_files"
    ],
)


print(
    "  Table files:",
    FINAL_FILE_COUNTS_R4[
        "table_files"
    ],
)


print(
    "  Report files:",
    FINAL_FILE_COUNTS_R4[
        "report_files"
    ],
)


print(
    "  Total files:",
    FINAL_FILE_COUNTS_R4[
        "total_files"
    ],
)


print(
    "\nSaved constraint audit:"
)


print(
    " ",
    CONSTRAINT_AUDIT_PATH_R4.resolve(),
)


print(
    "\nSaved reproducibility manifest:"
)


print(
    " ",
    MANIFEST_PATH_R4.resolve(),
)


print(
    "\nSaved complete file inventory:"
)


print(
    " ",
    INVENTORY_PATH_R4.resolve(),
)


print(
    "\nNO MODEL TRAINING WAS PERFORMED."
)


# =============================================================================
# 24. FINAL MANUSCRIPT-SCALE STATUS
# =============================================================================

if QUICK_MODE:

    print(
        "\nQuick structural configuration detected."
    )

    print(
        "For manuscript-scale experiments, use QUICK_MODE=False."
    )

else:

    print(
        "\nManuscript-scale Round 4 configuration completed."
    )


# =============================================================================
# 25. FINAL ASSERTIONS
# =============================================================================
# These are intentionally placed AFTER writing the audit files.
#
# Therefore, even if an integrity check fails, the audit CSV and manifest
# remain available for diagnosis.
# =============================================================================

if missing_prediction_labels_r4:

    raise AssertionError(
        "Round 4 primary prediction set is incomplete. "
        f"{len(missing_prediction_labels_r4)} expected prediction "
        "file(s) are missing."
    )


if not all_constraints_passed_r4:

    failed_rows_r4 = (
        ROUND4_CONSTRAINT_AUDIT.loc[
            ~ROUND4_CONSTRAINT_AUDIT[
                "all_constraints_passed"
            ]
            .astype(bool)
        ]
    )

    print(
        "\nFAILED CONSTRAINT-AUDIT ROWS:"
    )

    display(
        failed_rows_r4
    )

    raise AssertionError(
        "At least one saved primary trajectory violated the "
        "hard feasibility constraints beyond numerical tolerance."
    )


# =============================================================================
# 26. SUCCESS MESSAGE
# =============================================================================

print(
    "\n"
    + "=" * 86
)


print(
    "ROUND 4 CELL 27 COMPLETED SUCCESSFULLY"
)


print(
    "=" * 86
)


print(
    "\nFinal audit status:"
)


print(
    "  Primary prediction completeness : PASS"
)


print(
    "  Positivity constraint           : PASS"
)


print(
    "  Carrying-capacity constraint    : PASS"
)


print(
    "  Initial-condition preservation  : PASS"
)


print(
    "  Reproducibility manifest        : SAVED"
)


print(
    "  Round 4 file inventory          : SAVED"
)

# In[ ]:



