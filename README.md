# Rice NIR classification — replication package (Manuscript v6)

Code, data and intermediate results that reproduce every number, table and data
figure in `manuscript/Manuscript_v6.tex`: *Rice variety discrimination by NIR
hyperspectral imaging: validation design, accuracy metric and variable selection*.

Everything downstream of the pre-treated profile matrix is in Python. Image
acquisition, grain segmentation, profile extraction and pre-treatment
(Savitzky-Golay smoothing and z-score normalisation of each profile) were done at
Universidad Nacional de Frontera (UNF) before the data were shared; the matrix
in `data/` is already pre-treated.

## Data

| File | Contents |
|---|---|
| `data/X.mat` | 6000 × 263 pre-treated reflectance profiles (camera bands 47–309, 999–1649 nm) |
| `data/Y.mat` | 6000 class labels, integers 11–22 (12 varieties × 500 grains) |
| `data/nBandas.mat` | wavelength (nm) of the camera's 336 bands; the analysis uses entries 47–309 |
| `data/DATA_ARROZ.xlsx` | same profiles and labels, spreadsheet format |

Sensor: Resonon Pika NIR-640. Within each variety the rows are taken to be five
images of 100 grains in acquisition order; the image is the unit of the grouped
validation. Data supplied by Wilson Castro Silupu (UNF); cite him as the data
source.

## Code

Run from the repository root, e.g. `python code/22_Figuras_datos.py`. Paths are
resolved by `code/rutas.py` (inputs from `data/`, intermediate files in
`results/`, figures in `figures/`).

| Script | Produces | In the paper |
|---|---|---|
| `10_Validacion_UNF.py` | four classifiers on the full spectrum, random vs image-held-out division (5-fold CV × 6); training-only SFS and FSCNCA vs random subsets; confusion-matrix and band figures | Table 2, Fig. 7, Table 4, Fig. 8a–b |
| `14_CAFS_v6_baseline.py` | training-only CAFS (line-by-line port of UNF's MATLAB CAFS v6) vs random subsets | Table 5, Fig. 8c |
| `21_Seleccion_todos.py` | SFS, FSCNCA and CAFS fitted once on all spectra (`sel`); those subsets vs 30 random subsets of the same size on 30 splits, four classifiers (`sub A C`) | Fig. 6, Table 3 |
| `19_Subconjuntos_UNF.py` | per-image accuracy of the four classifiers (`img A C`) | input to 20 |
| `17_Unir_resultados.py` | merges the per-job files of 10 into `resultados_UNF*.xlsx` | — |
| `12_Significancia.py` | split-level tests (paired t, Wilcoxon, Holm) for the band baselines | Tables 4–5 |
| `20_Inferencia_imagen.py` | image-level Wilcoxon tests and bootstrap intervals; Table 3 statistics | Section 3.4, Table 3 |
| `22_Figuras_datos.py` | mean profiles, PCA, bands selected on all spectra | Figs. 4–6 |

Order: 10, 14, 21 (`sel`, then `sub`) and 19 (`img`) → 17 → 12, 20 → 22.

10, 14, 19 and 21 are the heavy steps (tens of CPU-hours, mostly the
random-subset controls); they were run on the GUANE cluster (SC3-UIS) through
their command-line modes, one job per range of splits. 12, 17, 20 and 22 take
seconds to a minute on a laptop from the files already in `results/`.

## Results

`results/` holds every intermediate file the scripts read or write (`res_*`,
`resultados_*`; per-image accuracies in `results/mejoras/`), so the tables can be
checked without rerunning the heavy steps. `results/res_sel_todos.json` lists the
bands selected on all spectra.

## Manuscript

`manuscript/Manuscript_v6.tex` (Elsevier `elsarticle`) with `references2.bib` and
`Figures/`. Compile:

```
cd manuscript
pdflatex -interaction=nonstopmode Manuscript_v6 && bibtex Manuscript_v6
pdflatex -interaction=nonstopmode Manuscript_v6 && pdflatex -interaction=nonstopmode Manuscript_v6
```

## Dependencies

Python 3.14 with numpy, scipy, pandas, scikit-learn 1.9, joblib, matplotlib,
openpyxl.

## Declaration of generative AI and AI-assisted technologies

During the preparation of this work the authors used Claude Code (Anthropic,
Claude Opus 5.5 model) to write and run the analysis code and statistics and to
draft parts of the text. The authors reviewed and edited all content, verified
every reported result and reference, and take full responsibility for the
content of the published article.
