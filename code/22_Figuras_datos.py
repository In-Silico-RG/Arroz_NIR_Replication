"""
Manuscript v6 (2026-11-03): the descriptive figures, in Python (replace UNF's MATLAB
Fig_prof_means.eps, Fig_PCA.eps and fig_unf_bandas_matlab.png).

  figuras/fig_py_perfiles.png  mean profile of each variety (Fig. 2)
  figuras/fig_py_pca.png       scores on PC1-PC2, 95 % ellipse per variety (Fig. 3);
                               PCA of the column-centred matrix, as MATLAB pca(X)
  figuras/fig_py_bandas.png    bands selected once on all spectra by
                               21_Seleccion_todos.py (res_sel_todos.json) (Fig. 4)

    python 22_Figuras_datos.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.io import loadmat
from scipy.stats import chi2

AQUI = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from rutas import DATOS, NBANDAS, RES, IMGDIR, FIG  # noqa: E402
RAIZ = AQUI.parent
X = loadmat(DATOS / "X.mat")["X"].astype(float)
Y = loadmat(DATOS / "Y.mat")["Y"].ravel().astype(int)
BW = loadmat(NBANDAS)["nBandas"].ravel()[46:309]
CLASES = np.unique(Y)
NOMBRES = ["La Unión", "Bellavista", "Mallares", "La Conquista", "Pitipo", "Amazonas",
           "Capirona", "IR43", "La Esperanza", "Tinajones", "La Puntilla", "Capoteña"]
COL = plt.get_cmap("tab20")(np.r_[0:20:2, 1:5:2][:12])
MARC = ["o", "s", "D"]


def perfiles():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c, n, col in zip(CLASES, NOMBRES, COL):
        ax.plot(BW, X[Y == c].mean(0), color=col, lw=1.5, label=n)
    ax.set_xlim(BW[0], BW[-1])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised reflectance")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_py_perfiles.png", dpi=300)


def pca():
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    S = U[:, :2] * s[:2]
    ev = s ** 2 / np.sum(s ** 2) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6))
    r = np.sqrt(chi2.ppf(0.95, 2))
    for k, (c, n, col) in enumerate(zip(CLASES, NOMBRES, COL)):
        P = S[Y == c]
        ax.scatter(P[:, 0], P[:, 1], s=8, color=col, marker=MARC[k % 3], alpha=0.6,
                   label=n, linewidths=0)
        val, vec = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
        ax.add_patch(Ellipse(P.mean(0), 2 * r * np.sqrt(val[1]), 2 * r * np.sqrt(val[0]),
                             angle=ang, fill=False, color=col, lw=1.5))
    ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, markerscale=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_py_pca.png", dpi=300)
    print(f"PC1 {ev[0]:.1f} %, PC2 {ev[1]:.1f} %")


def bandas():
    f = RES / "res_sel_todos.json"
    if not f.exists():
        print("res_sel_todos.json not found; Fig. 4 skipped")
        return
    sel = json.loads(f.read_text())
    medias = [X[Y == c].mean(0) for c in CLASES]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for a, letra, met in zip(ax, "abc", ("FSCNCA", "SFS", "CAFS")):
        for m in medias:
            a.plot(BW, m, color="0.35", lw=0.8)
        for i in sel[met]["idx0"]:
            a.axvline(BW[i], color="C3", ls="--", lw=1.0)
        a.set_title(f"({letra}) {met}: {len(sel[met]['idx0'])} bands", fontsize=11)
        a.set_xlabel("Wavelength (nm)")
        a.set_xlim(BW[0], BW[-1])
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Normalised reflectance")
    fig.tight_layout()
    fig.savefig(FIG / "fig_py_bandas.png", dpi=200)


if __name__ == "__main__":
    perfiles()
    pca()
    bandas()
