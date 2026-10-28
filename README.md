# Differentiable Divergences Between Time Series: small-scale replication

A CPU-only Python implementation of the soft-DTW divergence and its "sharp" variant from

> Blondel, Mensch, Vert. *Differentiable Divergences Between Time Series.* AISTATS 2021 (arXiv:2010.08354).

evaluated on a subset of the UCR time series archive. It covers the paper's main experiments at a scale that fits on a laptop or Google Colab:

| Paper section | What is reproduced | Command |
|---|---|---|
| Prop. 2-4 | Numerical checks: soft-DTW can be negative and is not minimised at X=Y; the divergences are zero there; gradients are exact | `sanity` |
| Sec. 4.1-4.2, Fig. 2-4 | Denoising effect, time-series averaging (barycenters), interpolation | `demo` |
| Sec. 4.3, Tables 2-3 | 1-nearest-neighbour and nearest-centroid classification, win-rate tables | `classify` |

Compared measures: Euclidean, DTW, soft-DTW (SDTW), SDTW divergence, sharp soft-DTW, sharp divergence. The mean-cost divergence is omitted, since the paper itself does not recommend it.

## Repository layout

```
softdtw_div/
├── softdtw_div.py        # core library (numba): DTW, soft-DTW, sharp, divergences, gradients, barycenters
├── run_experiments.py    # pipeline: data loading, sanity / demo / classify commands
├── requirements.txt
├── README.md
├── data/UCR/             # (not in git) datasets, created by you or auto-downloaded
└── results/              # (not in git) CSVs and figures, created at run time
```

## 1. Environment setup

Tested with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, numba 0.68.0, pandas 3.0.2, matplotlib 3.10.8. Any recent Python 3.9+ with the versions in `requirements.txt` should work.

```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows (cmd / PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

The first run of any command is slow (about 10-30 s) because numba compiles the kernels. They are cached afterwards.

## 2. Getting the data (UCR archive)

The loader tries three sources in this order:

1. **Local files** in `data/UCR/<Name>/<Name>_TRAIN.tsv` and `<Name>_TEST.tsv`.
2. **tslearn** auto-download (installed via `requirements.txt`).
3. **aeon** auto-download (only if you `pip install aeon`).

Manual download (if auto-download fails or you are offline):

1. Download the UCR 2018 archive from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ (also mirrored at https://www.timeseriesclassification.com).
2. Unzip it (password: `someone`) so the layout is:
   ```
   data/UCR/GunPoint/GunPoint_TRAIN.tsv
   data/UCR/GunPoint/GunPoint_TEST.tsv
   data/UCR/ECG200/ECG200_TRAIN.tsv
   ...
   ```

Datasets used by default (all univariate, equal-length, small):
`GunPoint, ECG200, CBF, ItalyPowerDemand, Coffee, TwoLeadECG, ArrowHead, Trace, Plane, MoteStrain, SonyAIBORobotSurface1, SyntheticControl`

Notes:
- The loader uses the archive's predefined train/test split, encodes labels to 0..K-1, and z-normalises each series.
- Variable-length datasets (NaN padded) are skipped automatically with a message.
- If a name differs between sources (for example `SyntheticControl` vs `synthetic_control`), use the name your source provides.
- `--datasets synthetic` runs on a small built-in toy dataset with no download. It is only for checking the installation, not for reporting results.

## 3. Running the experiments

All commands are run from the project folder.

```bash
# (a) Verify the theory and the gradients (about 1 minute)
python run_experiments.py sanity

# (b) Figures: denoising (Fig. 2), barycenters (Fig. 3), interpolation (Fig. 4)
python run_experiments.py demo --dataset GunPoint --gamma 10

# (c) Classification: 1-NN on all default datasets
python run_experiments.py classify

# (d) Classification with nearest centroid as well (slower)
python run_experiments.py classify --centroid

# (e) A quick run on a few datasets
python run_experiments.py classify --datasets GunPoint ECG200 ItalyPowerDemand --centroid
```

### Useful options for `classify`

| Option | Default | Meaning |
|---|---|---|
| `--datasets` | 12 small UCR sets | dataset names |
| `--gammas` | `0.01 0.1 1 10` | γ grid searched by cross-validation (the paper uses 1e-4 ... 1e4) |
| `--cv-repeats` | 3 | number of random 2/3 vs 1/3 hold-out splits for γ selection (the paper uses 5) |
| `--centroid` | off | also run nearest-centroid classification |
| `--max-per-class` | 20 | series per class used to compute each barycenter |
| `--iters` | 50 | L-BFGS iterations per barycenter (the paper uses 200) |
| `--max-train`, `--max-test` | 0 (all) | subsample for speed |
| `--seed` | 0 | random seed |
| `--data-dir`, `--out` | `data/UCR`, `results` | input and output folders |

To get closer to the paper's protocol (much slower): `--gammas 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 --cv-repeats 5 --iters 200 --max-per-class 1000`.

### Running time

The cost grows roughly like (number of train × test pairs) × length². On a normal laptop the default 1-NN run over the 12 datasets takes minutes to tens of minutes, and `--centroid` adds more. If it is too slow, use fewer datasets, `--max-train 50 --max-test 100`, or fewer γ values. Avoid large datasets such as ElectricDevices or StarLightCurves.

### Google Colab

```python
from google.colab import files
files.upload()                       # upload softdtw_div.py and run_experiments.py
!pip install numba tslearn
!python run_experiments.py sanity
!python run_experiments.py classify --datasets GunPoint ECG200 --centroid
```

No GPU is needed (numba parallelises across CPU cores). To keep the results, download `results/` or copy it to Google Drive.

## 4. Outputs

Everything is written to `results/`:

| File | Content |
|---|---|
| `classification_results.csv` | accuracy of every method per dataset and task, plus the chosen γ |
| `win_table_1nn.csv`, `win_table_centroid.csv` | % of datasets where the row method is within 99% or better than the column method (Tables 2/3 style) |
| `accuracy_1nn.png`, `accuracy_centroid.png` | accuracy bar charts |
| `fig2_denoising_<dataset>.png` | argmin of soft-DTW moves away from the target; the divergence stays on it |
| `fig3_barycenters_<dataset>.png` | averages of 10 series for each measure |
| `fig4_interpolation_<dataset>.png` | interpolations between two series |

The console also prints accuracy tables, mean accuracy and mean rank per method.

## 5. Reproducibility notes

- **Seeds.** All randomness (sub-sampling, γ-selection splits, series chosen for the demo and for barycenters) comes from `numpy.random.default_rng(--seed)`. With the same seed, data, and library versions, the 1-NN results are deterministic. Each pairwise distance is computed independently, so numba's parallel execution does not change the numbers.
- **Data.** Results depend on the exact dataset files. Use the predefined UCR train/test splits (the default) and keep the same archive version (UCR 2018).
- **Library versions.** Floating-point differences between numpy/numba/scipy versions can change L-BFGS paths slightly, so barycenters and nearest-centroid accuracies may differ in the last digits. To freeze your environment for submission, run `pip freeze > requirements-lock.txt` after a successful run.
- **Differences from the paper (smaller scale).**
  - Fewer datasets (about 12 instead of 84), smaller γ grid, 3 CV repeats instead of 5, and 50 L-BFGS iterations instead of 200.
  - Barycenters use at most `--max-per-class` series per class.
  - For nearest centroid, γ is reused from the 1-NN cross-validation rather than tuned separately.
  - DTW barycenters are computed with L-BFGS on the DTW sub-gradient (as in the paper's objective), not with DBA.
  - kNN with k > 1 (paper appendix tables) is not included.
- **Expected qualitative behaviour (from the paper).** The divergences should beat their uncorrected versions, and the SDTW divergence should do best overall. With about 12 datasets the numbers are noisier than the paper's 84-dataset summary, so compare trends rather than exact percentages.
- **Sanity-check caveat.** Non-negativity of the divergences is proved in the paper only for special costs; under the squared Euclidean cost used here it is a conjecture. The `sanity` command is a numerical check, not a proof, and the sharp divergence can occasionally come out slightly negative at large γ.

## 6. Implementation summary

- Soft-DTW value and expected alignment (the gradient w.r.t. the cost matrix) follow the paper's Algorithms 1-2. The directional derivative and the Hessian-vector product (Algorithms 3-4) give the sharp soft-DTW value and gradient.
- Gradients w.r.t. the series use the squared-Euclidean Jacobian-transpose products from Appendix B.2. The correction terms use the symmetric version for C(X, X).
- The `sanity` command checks every analytic gradient against central finite differences.
- Divergences: `D(X,Y) = f(X,Y) - ½ f(X,X) - ½ f(Y,Y)` with `f` = soft-DTW or sharp soft-DTW.
- Barycenters minimise `Σ_i w_i · kind(X, Y_i)` with L-BFGS-B. Biased measures start from the Euclidean mean. Divergences start from the solution of their biased counterpart.

## Citation

```
@inproceedings{blondel2021differentiable,
  title     = {Differentiable Divergences Between Time Series},
  author    = {Blondel, Mathieu and Mensch, Arthur and Vert, Jean-Philippe},
  booktitle = {AISTATS},
  year      = {2021}
}
```
