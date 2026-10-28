"""
Copy the results that already exist on your computer into this repository's layout,
so the figures can be redrawn and the CSV tables committed without retraining.

    python tools/collect_results.py
    python tools/collect_results.py --synthetic-root "C:/Users/me" --real-root "D:/Real dataset"

What is copied (CSV / JSON / TXT / TEX only; model weights are skipped)
    <synthetic-root>/round3/final/...           -> results/synthetic/round3/final/...
    <synthetic-root>/round4/final/...           -> results/synthetic/round4/final/...
    <synthetic-root>/data/..., outputs/...      -> results/synthetic/data/..., outputs/...
    <real-root>/Ramsayer_*.csv                  -> data/real/
    <real-root>/Real_PINN_Results/...           -> results/real_data/...
Existing files in the repository are overwritten. Files larger than 50 MB are
skipped (GitHub rejects files over 100 MB).
"""
import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEEP = {".csv", ".json", ".txt", ".tex"}
MAX_BYTES = 50 * 1024 ** 2


def copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        print(f"  (not found) {src}")
        return 0
    n = 0
    for f in src.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in KEEP:
            continue
        if f.stat().st_size > MAX_BYTES:
            print(f"  skipped (>50 MB): {f}")
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    print(f"  {n:5d} files  {src}  ->  {dst.relative_to(REPO)}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-root", type=Path, default=Path.home(),
                        help="folder that contains round3/, round4/, data/, outputs/ (default: home folder)")
    parser.add_argument("--real-root", type=Path, default=Path.home() / "Desktop" / "Real dataset",
                        help="folder that contains the Ramsayer CSVs and Real_PINN_Results/")
    args = parser.parse_args()

    syn, real = args.synthetic_root, args.real_root
    out_syn, out_real = REPO / "results" / "synthetic", REPO / "results" / "real_data"

    print("Synthetic benchmarks")
    for sub in ["round3/final", "round4/final", "data", "outputs"]:
        copy_tree(syn / sub, out_syn / sub)

    print("Experimental data")
    (REPO / "data" / "real").mkdir(parents=True, exist_ok=True)
    for name in ["Ramsayer_PRIMARY_PINN_data.csv", "Ramsayer_qR_48h_EXACT_linked.csv",
                 "Ramsayer_qR_86h_ENDPOINT_unpaired.csv"]:
        src = real / name
        if src.exists():
            shutil.copy2(src, REPO / "data" / "real" / name)
            print(f"  copied {name}")
        else:
            print(f"  (not found) {src}")
    copy_tree(real / "Real_PINN_Results", out_real)
    print("\nDone. Check the size of the repository before committing:  git count-objects -vH")


if __name__ == "__main__":
    main()
