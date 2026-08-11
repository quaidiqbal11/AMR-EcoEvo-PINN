# Changelog

## Packaging pass (this repository)

The original working notebook (`AMR_PINN_Round1_Round2_FINAL_EXECUTED.ipynb`,
103 cells, Rounds 1-4) was repackaged for public release. Every code and
markdown cell was checked with Python's `compile()` to confirm it parses.

**One fix was applied, and it is purely cosmetic / whitespace-level:**

- In the Round 4 "leave-one-schedule-out contribution" cell (originally
  "ROUND 4 — CELL 23C"), a line break between a `print(...)` statement and
  the following comment block had been lost, merging `)` with the start of
  a comment ("...formation.") into `)formation.`. This produced a
  `SyntaxError` and prevented that one cell from executing top-to-bottom as
  a plain script (it still ran fine inside Jupyter cell-by-cell in the
  original session, since the corruption only affects whole-script parsing
  boundaries in the same cell). The line break was restored and the
  orphaned comment fragment kept as-is, immediately after the corrected
  `print(...)` call. **No executable logic, computation, or numeric output
  was changed.**

This is the only content difference between this repository and the
original notebook. Everything else (model definitions, training loops,
data generation, metrics, and figure/table generation) is unmodified.

## What else changed (packaging only, not content)

- Cell outputs (embedded PNG figure previews, large stdout logs) were
  stripped from the notebook shipped in `notebooks/` to keep the repository
  a reasonable size for GitHub. The 63 previously embedded figure previews
  were extracted as standalone PNGs into `figures/round1` … `figures/round4`
  so they remain browsable without re-running anything.
- The single notebook was also exported, unmodified in content, into four
  plain `.py` scripts split at the original "Round 2" / "Round 3" / "Round 4"
  markdown section markers, so the code can be read, diffed, and imported
  without a notebook viewer:
  - `round1_deterministic_model.py`
  - `round2_stochastic_gillespie.py`
  - `round3_architecture_ablation_observation_regimes.py`
  - `round4_multischedule_identifiability.py`
