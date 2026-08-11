# Simplex- and Capacity-Preserving Eco-Evolutionary PINNs for Sparse-Data Inference of Antimicrobial Resistance Dynamics
Code and executed notebook for **"Simplex- and Capacity-Preserving
Eco-Evolutionary PINNs for Sparse-Data Inference of Antimicrobial Resistance
Dynamics"** (Quaid Iqbal, 2026).

The paper develops a hard-constrained, physics-informed neural network
(PINN) that reconstructs latent sensitive/resistant bacterial phenotype
trajectories from sparse, partially aggregated observations, while jointly
inferring mutation probability ($\mu$) and resistance fitness cost ($c_R$).
This repository contains the full pipeline used to produce every number,
table, and figure in the manuscript: the deterministic eco-evolutionary
model, a stochastic (Gillespie) reference simulator, the architecture
ablation (data-only / vanilla / soft-constrained / positive-output /
simplex-only / hard PINN), the O1–O4 observation-richness study, the
nested M1–M5 multi-schedule identifiability study, and all identifiability
diagnostics (sensitivity SVD, objective surfaces, leave-one-schedule-out,
held-out-schedule transfer).

## Repository structure

```
.
├── notebooks/
│   └── AMR_PINN_Rounds1-4_FINAL.ipynb   # full executed notebook (Rounds 1-4), outputs stripped
├── round1_deterministic_model.py                       # Round 1 — deterministic ODE model, forward analysis
├── round2_stochastic_gillespie.py                      # Round 2 — Gillespie stochastic reference simulator
├── round3_architecture_ablation_observation_regimes.py # Round 3 — architecture ablation + O1-O4
├── round4_multischedule_identifiability.py             # Round 4 — M1-M5 + identifiability analysis
├── figures/
│   ├── round1/ ... round4/     # 63 figure previews extracted from the executed notebook
├── outputs/                    # populated when you (re)run the scripts — see "Output layout" below
├── requirements.txt
├── CHANGELOG.md                 # documents the one whitespace-level fix applied when packaging
├── LICENSE
└── README.md
```

`notebooks/AMR_PINN_Rounds1-4_FINAL.ipynb` is the canonical, single source
of truth: it is the exact working notebook used for the paper (Rounds 1–4
in one file, separated by markdown section headers), with cell outputs
stripped to keep the repository lightweight. The four `round*.py` files
are a plain-script export of the same code (via `jupyter nbconvert`), split
at the original round boundaries, provided so the code can be read, diffed,
or imported without a notebook viewer. **Both forms contain identical
logic** — see `CHANGELOG.md` for the one cosmetic fix applied during
packaging (a single lost line break that broke whole-script parsing of one
cell; no computation changed).

## What each round does

| Round | Script | Content | Maps to paper |
|---|---|---|---|
| 1 | `round1_deterministic_model.py` | Deterministic sensitive/resistant ODE model, high-accuracy solver, treatment schedules, sensitivity sweeps, publication figures for the forward model | §2.2–2.3 (model + schedules) |
| 2 | `round2_stochastic_gillespie.py` | Exact Gillespie stochastic simulator, ensembles, extinction/rescue probabilities, demographic-noise figures | §2.17 (stochastic reference generation) |
| 3 | `round3_architecture_ablation_observation_regimes.py` | Seven-way architecture ablation (data-only → hard PINN) under O3; five-seed O1–O4 observation-richness study | §2.12, §3.1–3.2 (Tables 2–3, Figs. 2–3) |
| 4 | `round4_multischedule_identifiability.py` | Nested five-seed M1–M5 multi-schedule study; sensitivity SVD & objective-surface identifiability geometry; leave-one-schedule-out; held-out-schedule parameter transfer; final feasibility audit | §2.13–2.18, §3.3–3.7 (Tables 4, Figs. 4–7) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is **not required** — every script auto-selects
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — but the
manuscript-scale Round 3/Round 4 runs (9000 Adam epochs × 25 primary fits)
are slow on CPU. The paper's runs used PyTorch 2.7.1+cu118 on an NVIDIA
RTX 3070 (see `requirements.txt` for the exact captured environment).

## Reproducing the results

Run the notebook top-to-bottom (`notebooks/AMR_PINN_Rounds1-4_FINAL.ipynb`),
or run the four scripts in order from the repository root:

```bash
python round1_deterministic_model.py
python round2_stochastic_gillespie.py
python round3_architecture_ablation_observation_regimes.py
python round4_multischedule_identifiability.py
```

Each script detects its working directory and writes outputs relative to
the project root (it walks up one level automatically if launched from
`notebooks/`), so both entry points are safe.

### Quick smoke test vs. manuscript-scale run

Rounds 3 and 4 are controlled by a `QUICK_MODE` flag near the top of each
script (`QUICK_MODE: bool = False` by default):

- `QUICK_MODE = True` — a fast structural smoke test (a handful of epochs,
  e.g. 15 pretraining / 30 Adam epochs) to confirm the pipeline runs
  end-to-end on your machine before committing to a full run. Outputs are
  written under `round3/quick/` and `round4/quick/` and are explicitly
  flagged as preliminary.
- `QUICK_MODE = False` (default) — the manuscript-scale run reported in the
  paper: 1500 pretraining epochs, 9000 joint-Adam epochs, L-BFGS refinement,
  5 paired seeds per design (25 primary Round 4 fits). Outputs are written
  under `round3/final/` and `round4/final/`, matching the numbers in
  Tables 2–4 of the paper. Both scripts assert `RUN_LABEL == "final"`
  before running the primary manuscript experiment, as a guard against
  accidentally reporting a quick-mode run as final.

### Output layout

Running the pipeline (re)creates, at the project root:

```
data/{dense,sparse,noisy}/                       # Round 1 deterministic reference trajectories + synthetic observations
data/stochastic/{ensembles,summaries,pinn_exports}/  # Round 2 Gillespie ensembles and exports
outputs/{figures,tables,reports}/                # Round 1 & Round 2 figures/tables
round3/final/{figures,tables,reports,cache}/     # Round 3 architecture ablation + O1-O4
round4/final/{models,predictions,histories,figures,tables,reports,cache}/  # Round 4 M1-M5 + identifiability
```

(`round3/quick/` and `round4/quick/` appear instead if `QUICK_MODE = True`.)
Empty placeholders for this tree are already committed via `.gitkeep` files;
the generated contents themselves are git-ignored (see `.gitignore`).

This is exactly the layout the LaTeX source of the paper expects
(`\graphicspath{{round3/final/figures/}{round4/final/figures/}}`), so
figures generated by Round 3/4 can be dropped straight into the paper's
`figures/` folder without renaming.

`figures/round1` … `figures/round4` in *this* repository are a static,
pre-generated copy of the 63 figures produced by the original executed
run (extracted from the notebook's saved outputs), so results are
browsable on GitHub without re-running anything.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{iqbal2026simplex,
  title   = {Simplex- and Capacity-Preserving Eco-Evolutionary {PINN}s for
             Sparse-Data Inference of Antimicrobial Resistance Dynamics},
  author  = {Iqbal, Quaid},
  year    = {2026},
  note    = {Preprint / repository: see LICENSE for usage terms}
}
```

## License

MIT — see `LICENSE`.
