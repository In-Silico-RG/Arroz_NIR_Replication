# Rice NIR classification — replication package (Manuscript_v4)

Code, data, and intermediate results needed to reproduce the numbers in
`manuscript/Manuscript_v4.tex`: a joint re-analysis, with Wilson Castro Silupu
(Universidad Nacional de Frontera, Peru), of the NIR hyperspectral
classification of 12 Peruvian rice varieties.

This is a focused replication package, not the full research repository.
A separate improvement study (deferred as a possible second paper) is not
included here.

## Data

`data/X.mat` (6000x263 NIR absorbance spectra), `data/Y.mat` (6000x1 class
labels, integers 11-22), `data/DATA_ARROZ.xlsx` (same data, alternate
format). Supplied by Wilson Castro Silupu (UNF); cite him as the data
source. Sensor: Resonon Pika NIR-640, 1000-1700 nm, 263 bands. 12 varieties,
500 spectra each.

## Code

Scripts in `code/`, run in order from this directory (each resolves paths
relative to itself; see each script's header for its exact inputs/outputs):

| Script | Produces |
|---|---|
| `10_Validacion_UNF.py` | main validation: random vs. image-held-out, SFS/FSCNCA band selection |
| `11_CAFS_baseline.py` | CAFS baseline (own implementation) |
| `14_CAFS_v6_baseline.py` | CAFS v6 baseline (port of UNF's MATLAB code) |
| `15_Estructura_filas.py` | row-order / grid structure check |
| `16_Figura_bandas_UNF.py` | band-selection figure |
| `19_Subconjuntos_UNF.py` | UNF's own 9-band subsets vs. random 9-band subsets |
| `20_Inferencia_imagen.py` | image-level Wilcoxon + bootstrap inference |
| `17_Unir_resultados.py` | merges the above into `resultados_UNF*.xlsx` |
| `12_Significancia.py` | paired significance tests (Wilcoxon, paired t, Holm) |
| `13_Tabla3_recalculada.py` | Table 3 (overall accuracy, 4 classifiers, full + 9-band subsets) |

Full reruns of 10/11/14/19/20 took roughly 145 CPU-hours total on a compute
cluster (mostly the random-subset controls); 12/13/17 take seconds locally
given the files in `results/`.

## Results

`results/` holds every intermediate file the scripts above read or write
(`res_*`, `resultados_UNF*`), including the `_relab`/`_excl` variants used
only as sensitivity checks (relabelling 3 suspect images; excluding them) —
not the main reported numbers, which use the labels as received.

## Manuscript

`manuscript/Manuscript_v4.tex` (Elsevier `elsarticle` class) + `references2.bib`
+ `Figures/`. Compile with:

```
pdflatex -interaction=nonstopmode -halt-on-error Manuscript_v4.tex
bibtex Manuscript_v4
pdflatex -interaction=nonstopmode -halt-on-error Manuscript_v4.tex
pdflatex -interaction=nonstopmode -halt-on-error Manuscript_v4.tex
```

`Figures/Fig_prof_means.eps` and `Figures/Fig_PCA.eps` need converting to
PDF first if `pdflatex` reports them missing (`repstopdf` is not available
on every TeX install):

```
cd manuscript/Figures && for f in *.eps; do ps2pdf -dEPSCrop "$f" "${f%.eps}-eps-converted-to.pdf"; done
```

## Dependencies

```
numpy pandas scipy scikit-learn openpyxl matplotlib seaborn
```

## Declaration of generative AI and AI-assisted technologies

During the preparation of this work the authors used Claude Code (Anthropic,
Claude Opus 5.5 model) to write and run the analysis code and statistics and
to draft parts of the text. The authors reviewed and edited all content,
verified every reported result and reference, and take full responsibility
for the content of the published article.
