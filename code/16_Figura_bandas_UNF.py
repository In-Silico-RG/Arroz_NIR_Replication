"""
Figura de las bandas seleccionadas en el analisis MATLAB de UNF (FSCNCA, SFS, CAFS),
sobre los perfiles medios de reflectancia de las 12 variedades. Sustituye a
Fig_FSCNA1/Fig_SFSS/Fig_CAFS.eps, donde las lineas rojas finas se veian negras
(revision externa, punto 15). Posiciones exactas: 13_Tabla3_recalculada.NM.

    /home/aldo/myenv/bin/python3 draft_UNF_conjunto/16_Figura_bandas_UNF.py
Guarda figuras/fig_unf_bandas_matlab.png.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
X = loadmat(RAIZ / "X.mat")["X"].astype(float)
Y = loadmat(RAIZ / "Y.mat")["Y"].ravel().astype(int)
BW = loadmat(RAIZ / "NUEVA_DATA_SEPT_2026" / "nBandas.mat")["nBandas"].ravel()[46:309]
NM = {
    "FSCNCA": [1038, 1114, 1119, 1124, 1237, 1368, 1371, 1568, 1649],
    "SFS":    [1033, 1102, 1124, 1129, 1151, 1237, 1274, 1435, 1649],
    "CAFS":   [1012, 1063, 1232, 1234, 1420, 1435, 1624, 1639, 1649],
}


def main():
    medias = [X[Y == c].mean(0) for c in np.unique(Y)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for a, (letra, (met, nm)) in zip(ax, zip("abc", NM.items())):
        for m in medias:
            a.plot(BW, m, color="0.35", lw=0.8)
        for w in nm:
            a.axvline(w, color="C3", ls="--", lw=1.0)
        a.set_title(f"({letra}) {met}: 9 bands", fontsize=11)
        a.set_xlabel("Wavelength (nm)")
        a.set_xlim(BW[0], BW[-1])
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Normalised reflectance")
    fig.tight_layout()
    fig.savefig(AQUI / "figuras" / "fig_unf_bandas_matlab.png", dpi=200)


if __name__ == "__main__":
    main()
