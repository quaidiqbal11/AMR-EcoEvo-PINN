# Results

The notebooks write everything here. The CSV tables are small and are committed so that the figures can be redrawn without retraining. Trained model weights (`*.pt`) are not committed (see `.gitignore`).

```
results/
├── synthetic/                    notebooks/01_synthetic_benchmarks.ipynb
│   ├── data/                     dense, sparse, noisy and stochastic benchmark data (Rounds 1-2)
│   ├── outputs/                  Round 1-2 figures, tables, reports
│   ├── round3/final/             tables/, predictions/, histories/, figures/, reports/
│   └── round4/final/             tables/, predictions/, histories/, figures/, reports/
└── real_data/                    notebooks/02_real_data_Ramsayer.ipynb
    ├── tables/                   final_R37 ... final_R310 result tables
    ├── figures/publication_ready/source_data/
    └── reports/
```
