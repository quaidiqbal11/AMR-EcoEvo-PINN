# Figure scripts

The scripts in this folder redraw the article figures at their final PLOS Computational Biology size. Every figure is at most 7.5 × 8.75 in, uses Arial text of 8-11 pt (Liberation Sans if Arial is not installed), and is saved as a vector PDF plus an RGB TIFF with LZW compression. Fig 1 is saved at 300 dpi and Figs 2-10 at 600 dpi.

```bash
python figures/make_all_figures.py      # Fig 1-10
python figures/Fig2_representative_M4_trajectories.py
```

Output goes to `figures/output/`. Input paths are set in `plos_style.py` (SECTION 1) relative to the repository root. If your results are stored somewhere else, set the environment variable `AMR_REPO_ROOT` to that folder.

| Script | Figure |
|---|---|
| `Fig1_framework_overview.py` | Proposed simplex- and capacity-preserving PINN framework |
| `Fig2_representative_M4_trajectories.py` | Representative state reconstruction, four-environment design |
| `Fig3_identifiability_geometry.py` | Local identifiability geometry, designs M1-M5 |
| `Fig4_real_data_reconstruction.py` | Experimental-data reconstruction |
| `Fig5_generalization_design_sensitivity.py` | Experimental-data generalization and dose-design sensitivity |
| `Fig6_observation_regimes.py` | Effect of phenotype-observation richness |
| `Fig7_profile_surfaces.py` | Observation-objective geometry in the (μ, c_R) plane |
| `Fig8_parameter_recovery.py` | Recovery of shared evolutionary parameters |
| `Fig9_independent_86h_validation.py` | Independent validation at the 86-h endpoint |
| `Fig10_leave_one_schedule_out.py` | Leave-one-schedule-out refits |
