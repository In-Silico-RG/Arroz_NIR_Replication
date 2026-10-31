"""
Estructura de las filas de X.mat dentro de cada variedad (revision externa M1/M7).

Distancia euclidiana media entre espectros de la misma variedad separados por
`lag` filas (lag = 1..120), frente a la de pares tomados al azar dentro de la
variedad. Si las filas siguen el orden de adquisicion, la distancia crece con la
separacion y se acerca a la de pares al azar a la escala de una imagen.

    /home/aldo/myenv/bin/python3 draft_UNF_conjunto/15_Estructura_filas.py
Guarda figuras/fig_unf_filas.png y resultados_UNF_filas.xlsx.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v10", AQUI / "10_Validacion_UNF.py")
v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10)
X, Y, FIG = v10.X, v10.Y, v10.FIG
LAGS = np.arange(1, 121)


def main():
    rng = np.random.default_rng(0)
    d_lag = np.zeros(len(LAGS))
    d_azar = []
    for c in v10.CLASES:
        Xc = X[Y == c]
        n = len(Xc)
        for j, L in enumerate(LAGS):
            d_lag[j] += np.linalg.norm(Xc[L:] - Xc[:-L], axis=1).mean() / len(v10.CLASES)
        i1, i2 = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
        ok = i1 != i2
        d_azar.append(np.linalg.norm(Xc[i1[ok]] - Xc[i2[ok]], axis=1).mean())
    azar = float(np.mean(d_azar))
    pd.DataFrame({"lag": LAGS, "distancia": d_lag}).assign(azar=azar).to_excel(
        AQUI / "resultados_UNF_filas.xlsx", index=False)
    for L in (1, 9, 10, 11, 20, 50, 100):
        print(f"lag {L:3d}: {d_lag[L - 1]:.3f}")
    print(f"random pairs: {azar:.3f}")
    fig, a = plt.subplots(figsize=(7, 3.6))
    a.plot(LAGS, d_lag, "-", color="C0", lw=1.2, label="rows separated by lag (same variety)")
    a.plot(LAGS[9::10], d_lag[9::10], "o", color="C0", ms=4, label="multiples of 10")
    a.axhline(azar, color="C7", ls="--", label="random pairs (same variety)")
    for L in (10, 50, 100):
        a.axvline(L, color="C3", lw=0.6, alpha=0.6)
    a.set_xlabel("Separation in row order (rows)")
    a.set_ylabel("Mean Euclidean distance")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_unf_filas.png", dpi=200)


if __name__ == "__main__":
    main()
