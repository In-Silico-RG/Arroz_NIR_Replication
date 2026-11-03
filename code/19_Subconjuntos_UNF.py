"""
Para el manuscrito v4 (2026-10-30), etiquetas tal como se recibieron, mismas 30
particiones de 10_Validacion_UNF.py (particion(b, i)):

  sub A C   los tres subconjuntos de 9 bandas del analisis MATLAB de UNF (FSCNCA, SFS,
            CAFS; posiciones en 13_Tabla3_recalculada.NM) frente a 30 subconjuntos de 9
            bandas al azar en cada particion, con una imagen por variedad fuera (b = 100);
            cuatro clasificadores. Los subconjuntos de UNF se eligieron con todos los
            espectros, lo que los favorece: la prueba es conservadora.
  img A C   exactitud por imagen (60 imagenes) de los cuatro clasificadores con espectro
            completo, b = 1 y b = 100, para la inferencia a nivel de imagen.

Salida: res_sub9_A_C.xlsx, res_img_A_C.xlsx.
"""
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

AQUI = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from rutas import DATOS, NBANDAS, RES, IMGDIR, FIG  # noqa: E402
_spec = importlib.util.spec_from_file_location("v10", AQUI / "10_Validacion_UNF.py")
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)

def _t3():
    """v4 only: UNF's MATLAB subsets (13_Tabla3_recalculada.NM). Replaced in v6 by
    21_Seleccion_todos.py; the 'sub' mode is kept for the record, not used in the paper."""
    s = importlib.util.spec_from_file_location("t3", AQUI / "13_Tabla3_recalculada.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

MODELOS = ("SVM", "L-SVM", "LDA", "ANN")
N_AZAR = 30
IMG = v.grupos(100)


def sub_split(i):
    warnings.filterwarnings("ignore")
    t3 = _t3()
    tr, te = v.particion(100, i)
    rng = np.random.default_rng(95000 + i)
    azar = [np.sort(rng.choice(v.X.shape[1], 9, replace=False)) for _ in range(N_AZAR)]
    filas = []
    for m in MODELOS:
        for met, cols in t3.SUB.items():
            mod = t3.modelo(m, len(cols)).fit(v.X[tr][:, cols], v.Y[tr])
            filas.append({"split": i, "modelo": m, "metodo": met,
                          "acc": float(np.mean(mod.predict(v.X[te][:, cols]) == v.Y[te]))})
        for j, cols in enumerate(azar):
            mod = t3.modelo(m, len(cols)).fit(v.X[tr][:, cols], v.Y[tr])
            filas.append({"split": i, "modelo": m, "metodo": "random", "j": j,
                          "acc": float(np.mean(mod.predict(v.X[te][:, cols]) == v.Y[te]))})
    return filas


def img_split(b, i):
    warnings.filterwarnings("ignore")
    tr, te = v.particion(b, i)
    filas = []
    for m in MODELOS:
        yp = v.modelos()[m].fit(v.X[tr], v.Y[tr]).predict(v.X[te])
        for im in np.unique(IMG[te]):
            r = IMG[te] == im
            filas.append({"b": b, "split": i, "modelo": m, "imagen": int(im),
                          "n": int(r.sum()), "acc": float(np.mean(yp[r] == v.Y[te][r]))})
    return filas


def main(arg):
    t0 = time.time()
    modo, a, c = arg[0], int(arg[1]), int(arg[2])
    if modo == "sub":
        out = Parallel(n_jobs=-1)(delayed(sub_split)(i) for i in range(a, c))
        d = pd.DataFrame([f for o in out for f in o])
        d.to_excel(RES / f"res_sub9_{a}_{c}.xlsx", index=False)
        print((d.groupby(["modelo", "metodo"]).acc.mean() * 100).round(2).unstack().to_string())
    else:
        out = Parallel(n_jobs=-1)(delayed(img_split)(b, i) for b in (1, 100) for i in range(a, c))
        d = pd.DataFrame([f for o in out for f in o])
        d.to_excel(IMGDIR / f"res_img_{a}_{c}.xlsx", index=False)
        print((d.groupby(["b", "modelo"]).acc.mean() * 100).round(2).to_string())
    print(f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
