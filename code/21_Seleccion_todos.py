"""
Manuscript v6 (2026-11-03): the one-shot band selections, in Python.

Replaces UNF's MATLAB selections (fscnca, sequentialfs, CAFS run once on all the
spectra; previously read from their figures in 13_Tabla3_recalculada.NM) with the
Python implementations already used for the training-only baselines:

  SFS     10_Validacion_UNF.sfs: forward selection, LDA criterion, inner 5-fold CV
          grouped over blocks of 10 rows; first 9 bands.
  FSCNCA  10_Validacion_UNF.fscnca_pesos: feature-weighted NCA (Yang et al. 2012);
          the 9 bands with the largest weight.
  CAFS    14_CAFS_v6_baseline.cafs_v6: line-by-line port of UNF's
          cafs_matlab_classification_v6.m; returns its own subset size.

Each method is fitted ONCE on all 6000 spectra, as in UNF's analysis, so the test
grains take part in the selection (this favours the subsets; stated in the paper).

    python 21_Seleccion_todos.py sel          -> res_sel_todos.json
    python 21_Seleccion_todos.py sub A C      -> res_subtodos_A_C.xlsx
        splits A..C-1 of 10_Validacion_UNF.particion(100, i) (one image per variety
        held out), four classifiers, each selected subset against 30 random subsets
        of the same size per split.
    python 21_Seleccion_todos.py resumen      -> resultados_seleccion_todos.xlsx
"""
import importlib.util
import json
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


def _cargar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v = _cargar("v10", "10_Validacion_UNF.py")
ca = _cargar("c14", "14_CAFS_v6_baseline.py")        # cafs_v6

X, Y = v.X, v.Y
K = 9
MODELOS = ("SVM", "L-SVM", "LDA", "ANN")
N_AZAR = 30
SEL = RES / "res_sel_todos.json"


def seleccionar():
    g10 = v.grupos(10)
    out = {}
    t0 = time.time()
    out["SFS"] = [int(j) for j in v.sfs(X, Y, g10, K)]
    print(f"SFS {time.time() - t0:.0f} s: {out['SFS']}", flush=True)
    t0 = time.time()
    w = v.fscnca_pesos(X, Y, 71000)
    out["FSCNCA"] = [int(j) for j in np.argsort(-w)[:K]]
    print(f"FSCNCA {time.time() - t0:.0f} s: {out['FSCNCA']}", flush=True)
    t0 = time.time()
    final, traza = ca.cafs_v6(X, Y, g10, 81000)
    out["CAFS"] = [int(j) for j in final]
    print(f"CAFS {time.time() - t0:.0f} s: sizes {[t['m'] for t in traza]}, "
          f"final {len(final)}: {out['CAFS']}", flush=True)
    nm = ca.v10_nm
    res = {m: {"idx0": sorted(b), "nm": [int(round(x)) for x in sorted(nm(sorted(b)))]}
           for m, b in out.items()}
    res["_cafs_traza"] = [{"iter": t["iter"], "m": t["m"], "f1": float(t["f1_interno"])}
                          for t in traza]
    SEL.write_text(json.dumps(res, indent=1))
    print(json.dumps({m: r["nm"] for m, r in res.items() if not m.startswith("_")}))


def un_split(i, sub):
    warnings.filterwarnings("ignore")
    tr, te = v.particion(100, i)
    rng = np.random.default_rng(96000 + i)
    tamanos = sorted({len(c) for c in sub.values()})
    azar = {m: [np.sort(rng.choice(X.shape[1], m, replace=False)) for _ in range(N_AZAR)]
            for m in tamanos}
    filas = []
    for mod in MODELOS:
        def acc(cols):
            f = v.modelo(mod, len(cols)).fit(X[tr][:, cols], Y[tr])
            return float(np.mean(f.predict(X[te][:, cols]) == Y[te]))
        for met, cols in sub.items():
            filas.append({"split": i, "modelo": mod, "metodo": met, "m": len(cols),
                          "acc": acc(cols)})
        for m in tamanos:
            for j, cols in enumerate(azar[m]):
                filas.append({"split": i, "modelo": mod, "metodo": f"random{m}", "m": m,
                              "j": j, "acc": acc(list(cols))})
    return filas


def subconjuntos(a, c):
    sel = json.loads(SEL.read_text())
    sub = {m: r["idx0"] for m, r in sel.items() if not m.startswith("_")}
    t0 = time.time()
    out = Parallel(n_jobs=-1)(delayed(un_split)(i, sub) for i in range(a, c))
    d = pd.DataFrame([f for o in out for f in o])
    d.to_excel(RES / f"res_subtodos_{a}_{c}.xlsx", index=False)
    print((d.groupby(["modelo", "metodo"]).acc.mean() * 100).round(2).unstack().to_string())
    print(f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg[0] == "sel":
        seleccionar()
    elif arg[0] == "sub":
        subconjuntos(int(arg[1]), int(arg[2]))
