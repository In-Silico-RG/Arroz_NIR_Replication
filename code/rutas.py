"""Paths shared by the scripts, valid in both layouts:

  lab folder   Arroz_NIR/draft_UNF_conjunto/*.py; data at Arroz_NIR/X.mat, Y.mat,
               NUEVA_DATA_SEPT_2026/nBandas.mat; results next to the scripts;
               figures in figuras/; per-image results in mejoras/
  repository   code/*.py; data/ (X.mat, Y.mat, nBandas.mat); results/; figures/
"""
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
_REPO = (RAIZ / "data" / "X.mat").exists()

DATOS = RAIZ / "data" if _REPO else RAIZ
NBANDAS = DATOS / "nBandas.mat" if _REPO else RAIZ / "NUEVA_DATA_SEPT_2026" / "nBandas.mat"
RES = RAIZ / "results" if _REPO else AQUI
IMGDIR = RES / "mejoras" if _REPO else AQUI / "mejoras"
FIG = RAIZ / "figures" if _REPO else AQUI / "figuras"
for _d in (RES, IMGDIR, FIG):
    _d.mkdir(exist_ok=True)
