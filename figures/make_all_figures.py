"""
Re-draw Fig 1 - Fig 10 for PLOS Computational Biology.

    python figures/make_all_figures.py

Outputs (figures/output/):
  FigN.pdf  vector version used in the LaTeX manuscript
  FigN.tif  PLOS submission file (RGB, LZW)
Fig 1 needs no data. Figs 2-10 read the CSV tables written by the notebooks.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRIPTS = [
    "Fig1_framework_overview.py",
    "Fig2_representative_M4_trajectories.py",
    "Fig3_identifiability_geometry.py",
    "Fig4_real_data_reconstruction.py",
    "Fig5_generalization_design_sensitivity.py",
    "Fig6_observation_regimes.py",
    "Fig7_profile_surfaces.py",
    "Fig8_parameter_recovery.py",
    "Fig9_leave_one_schedule_out.py",
    "Fig10_independent_86h_validation.py",
]

failed = []
for script in SCRIPTS:
    try:
        runpy.run_path(str(HERE / script), run_name="__main__")
    except Exception as error:  # keep going so one missing file does not stop the rest
        failed.append(script)
        print(f"FAILED {script}: {error}")

print("\nDone." if not failed else f"\nFinished with {len(failed)} failure(s): {failed}")
