"""
Linea base de bandas al azar para CAFS v6 (codigo MATLAB de UNF, recibido 2026-10-30).

Traduccion de cafs_inner_selection_v2 / compute_selection_score / generate_CA_2_k_2
de cafs_matlab_classification_v6.m con los parametros de b_AnalizarPerfilesArroz_2.m:
  SNV por espectro antes de seleccionar; naive Bayes gaussiano (fitcnb 'normal');
  covering array binario CA(N;2,k,2) de tamano optimo (N minimo con
  C(N-1, ceil(N/2)) >= k; columnas = primeras k combinaciones de peso ceil(N/2) sobre
  las filas 2..N, la fila 1 queda vacia); cada fila no vacia es un subconjunto,
  puntuado con F1-macro medio en CV interna de 5 pliegues;
  puntaje = F1 - lambda_t * max(0, (p - 20)/k)^2, lambda_t = 0.10 (1 + 2 (t-1)/50);
  se conserva la mejor fila (empates: mayor F1, luego mayor exactitud);
  parada: p <= MinFeatures = 10, Patience = 3 iteraciones con mejora <= 1e-4,
  MaxIterations = 50; se devuelve el mejor subconjunto de todas las iteraciones.
Diferencias con el MATLAB: la CV interna es StratifiedGroupKFold(5) sobre bloques de
10 filas (no cvpartition aleatoria), para que la seleccion no use granos de la misma
adquisicion en entrenamiento y validacion, como en las demas lineas base; no hay HPO
del naive Bayes final, porque cada subconjunto se evalua con SVM y LDA.

Protocolo igual a SFS/FSCNCA en 10_Validacion_UNF.py: CV agrupada b = 100 (una imagen
por variedad en prueba), particiones particion(100, s, base=500), s = 0..9,
seleccion solo en entrenamiento; el subconjunto final y el de cada iteracion se
comparan con 30 subconjuntos al azar del mismo tamano (semillas 70000 + s).

    /home/aldo/myenv/bin/python3 draft_UNF_conjunto/14_CAFS_v6_baseline.py
Guarda resultados_UNF_cafs_v6.xlsx y figuras/fig_unf_bandas_cafs_v6.png.
"""

import importlib.util
import time
import warnings
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB

AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v10", AQUI / "10_Validacion_UNF.py")
v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10)

X, Y, FIG = v10.X, v10.Y, v10.FIG
N_SPLITS = v10.N_SPLITS_BANDAS
N_AZAR = v10.N_AZAR
OPC = dict(Patience=3, MinImprovement=1e-4, MinFeatures=10, MaxIterations=50,
           TargetFeatures=20, CompactnessPenalty=0.10, ExcessPenaltyPower=2.0,
           AdaptivePenaltyAlpha=2.0)
EPS = np.finfo(float).eps


def snv(M):
    s = M.std(1, ddof=1, keepdims=True)
    s[s == 0] = 1
    return (M - M.mean(1, keepdims=True)) / s


def ca_2_k_2(k):
    """generate_CA_2_k_2: CA(N;2,k,2) de columnas de peso fijo."""
    n = 1
    while True:
        n += 1
        w = -(-n // 2)
        if comb(n - 1, w) >= k:
            break
    ca = np.zeros((n, k), dtype=bool)
    for c, filas in zip(range(k), combinations(range(1, n), w)):
        ca[list(filas), c] = True
    return ca


def cv_nb(Xs, y, g, cols, semilla):
    """evaluate_classifier_cv_metrics con fitcnb 'normal': F1-macro y exactitud medios."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    f1, acc = [], []
    for tr, te in cv.split(Xs, y, g):
        p = GaussianNB().fit(Xs[tr][:, cols], y[tr]).predict(Xs[te][:, cols])
        f1.append(f1_score(y[te], p, average="macro", labels=v10.CLASES, zero_division=0))
        acc.append(accuracy_score(y[te], p))
    return np.mean(f1), np.mean(acc)


def puntaje(f1, p, k, it):
    lam = OPC["CompactnessPenalty"] * (1 + OPC["AdaptivePenaltyAlpha"] * (it - 1) / OPC["MaxIterations"])
    return f1 - lam * max(0.0, (p - OPC["TargetFeatures"]) / max(1, k)) ** OPC["ExcessPenaltyPower"]


def cafs_v6(Xtr, ytr, gtr, semilla):
    """cafs_inner_selection_v2. Devuelve (mejor subconjunto global, traza por iteracion)."""
    warnings.filterwarnings("ignore")
    Xs = snv(Xtr)
    vig = np.arange(Xs.shape[1])
    previo, sin_mejora = -np.inf, 0
    mejor_g = (-np.inf, -np.inf, vig)
    traza = []
    for it in range(1, OPC["MaxIterations"] + 1):
        k = len(vig)
        ca = ca_2_k_2(k)
        best = None
        for fila in ca:
            if not fila.any():
                continue
            cols = vig[fila]
            f1, acc = cv_nb(Xs, ytr, gtr, cols, semilla + it)
            sc = puntaje(f1, len(cols), k, it)
            if (best is None or sc > best[0]
                    or (abs(sc - best[0]) < EPS and f1 > best[1])
                    or (abs(sc - best[0]) < EPS and abs(f1 - best[1]) < EPS and acc > best[2])):
                best = (sc, f1, acc, cols)
        sc, f1, acc, cols = best
        mejora = sc - previo
        sin_mejora = 0 if mejora > OPC["MinImprovement"] else sin_mejora + 1
        if sc > mejor_g[0] or (abs(sc - mejor_g[0]) < EPS and f1 > mejor_g[1]):
            mejor_g = (sc, f1, cols)
        traza.append({"iter": it, "k_antes": k, "filas_CA": len(ca), "m": len(cols),
                      "f1_interno": f1, "puntaje": sc, "bandas": cols})
        vig = cols
        previo = max(previo, sc)
        if len(cols) <= OPC["MinFeatures"] or sin_mejora >= OPC["Patience"] or len(vig) <= 1:
            break
    return mejor_g[2], traza


def main():
    g10 = v10.grupos(10)
    filas, sel = [], []
    t0 = time.time()
    for s in range(N_SPLITS):
        tr, te = v10.particion(100, s, base=500)   # mismas particiones que SFS y FSCNCA
        final, traza = cafs_v6(X[tr], Y[tr], g10[tr], 80000 + 100 * s)
        print(f"split {s}: tamanos {[t['m'] for t in traza]}, final {len(final)} bandas, "
              f"{time.time() - t0:.0f} s", flush=True)
        sel.append({"split": s, "m": len(final), "indices_0": " ".join(map(str, final)),
                    "nm": " ".join(str(int(round(w))) for w in v10_nm(final))})
        rng = np.random.default_rng(70000 + s)
        casos = [("final", len(traza), final)] + [(f"iter{t['iter']}", t["iter"], t["bandas"])
                                                  for t in traza]
        for etiqueta, it, bandas in casos:
            m = len(bandas)
            azar = [np.sort(rng.choice(X.shape[1], m, replace=False)) for _ in range(N_AZAR)]
            for nombre in ("SVM", "LDA"):
                base = {"split": s, "caso": etiqueta, "iter": it, "m": m, "modelo": nombre}
                filas.append({**base, "metodo": "CAFS_v6",
                              "acc": v10.evaluar_subconjunto(list(bandas), tr, te, nombre)})
                accs = Parallel(n_jobs=-1)(
                    delayed(v10.evaluar_subconjunto)(list(c), tr, te, nombre) for c in azar)
                filas += [{**base, "metodo": "random", "acc": a} for a in accs]
        pd.DataFrame(filas).to_csv(AQUI / f"resultados_UNF_cafs_v6_parcial{v10.SUFIJO}.csv", index=False)
    d = pd.DataFrame(filas)
    r = (d.groupby(["modelo", "caso", "metodo"])
         .agg(m_medio=("m", "mean"), m_min=("m", "min"), m_max=("m", "max"),
              n=("acc", "size"), mean=("acc", "mean"), std=("acc", "std"),
              q05=("acc", lambda x: x.quantile(0.05)), q95=("acc", lambda x: x.quantile(0.95)))
         .round(4))
    print(r.to_string(), flush=True)
    with pd.ExcelWriter(AQUI / f"resultados_UNF_cafs_v6{v10.SUFIJO}.xlsx") as w:
        r.to_excel(w, sheet_name="resumen")
        d.to_excel(w, sheet_name="corridas", index=False)
        pd.DataFrame(sel).to_excel(w, sheet_name="seleccion_final", index=False)
    fig(d)
    print(f"total {time.time() - t0:.0f} s", flush=True)


def v10_nm(idx):
    from scipy.io import loadmat
    nb = loadmat(AQUI.parent / "NUEVA_DATA_SEPT_2026" / "nBandas.mat")["nBandas"].ravel()
    return nb[46:309][np.asarray(idx)]


def fig(d):
    fig_, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    it = d[d.caso.str.startswith("iter")]
    # solo iteraciones alcanzadas en al menos 9 de los 10 splits
    alc = it[it.metodo == "CAFS_v6"].groupby("iter").split.nunique()
    it = it[it.iter.isin(alc[alc >= 9].index)]
    fin = d[d.caso == "final"]
    for a, nombre in zip(ax, ("SVM", "LDA")):
        dd = it[it.modelo == nombre]
        az = dd[dd.metodo == "random"].groupby("iter")
        cf = dd[dd.metodo == "CAFS_v6"].groupby("iter")
        mm = cf.m.mean()
        a.fill_between(mm, 100 * az.acc.quantile(0.05), 100 * az.acc.quantile(0.95),
                       alpha=0.25, color="C7", label="random subsets, 5-95 %")
        a.plot(mm, 100 * az.acc.mean(), "o-", color="C7", label="random subsets, mean")
        a.errorbar(mm, 100 * cf.acc.mean(), 100 * cf.acc.std(), marker="s", color="C3",
                   capsize=3, label="CAFS v6, iteration subsets")
        ff = fin[(fin.modelo == nombre) & (fin.metodo == "CAFS_v6")]
        a.errorbar(ff.m.mean(), 100 * ff.acc.mean(), 100 * ff.acc.std(), marker="*",
                   ms=12, color="k", capsize=3, label="CAFS v6, returned subset")
        a.set_xscale("log")
        a.set_xticks([10, 20, 50, 100, 263], ["10", "20", "50", "100", "263"])
        a.xaxis.set_minor_formatter(plt.NullFormatter())
        a.set_title(f"{nombre}, one image per variety held out")
        a.set_xlabel("Number of bands (mean over splits)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Overall accuracy (%)")
    ax[0].legend(fontsize=8)
    fig_.tight_layout()
    fig_.savefig(FIG / f"fig_unf_bandas_cafs_v6{v10.SUFIJO}.png", dpi=200)


if __name__ == "__main__":
    main()
