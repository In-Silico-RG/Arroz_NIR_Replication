"""
Linea base de bandas al azar para CAFS (complemento de U3 en 10_Validacion_UNF.py).

Seleccion iterativa por covering arrays siguiendo la logica de cafspy.ICAFS
(Castro et al. 2022): en cada iteracion se genera un covering array binario de
fuerza t sobre las bandas vigentes, cada fila define un subconjunto, se puntua
con F1-macro en validacion cruzada y se conserva el mejor (>=, como ICAFS).
Diferencias con cafspy: devuelve las bandas elegidas; la validacion interna es
GroupKFold(5) sobre bloques de 10 filas; clasificador GaussianNB; parada cuando
quedan < 5 bandas o el F1 no mejora.

Protocolo igual a u3(): hold-out agrupado b = 100, mismas semillas, seleccion
solo en entrenamiento, comparacion contra 30 subconjuntos al azar del mismo
tamano evaluados con SVM y LDA.

    /home/aldo/myenv/bin/python3 draft_UNF_conjunto/11_CAFS_baseline.py
Guarda resultados_UNF_cafs.xlsx y figuras/fig_unf_bandas_cafs.png.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from testflows.combinatorics import Covering

AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v10", AQUI / "10_Validacion_UNF.py")
v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10)

X, Y, FIG = v10.X, v10.Y, v10.FIG
N_SPLITS = v10.N_SPLITS_BANDAS
N_AZAR = v10.N_AZAR
FUERZA = 2
T_MAX = 15
MIN_BANDAS = 5
N_JOBS = 4


def f1_cv(Xs, y, g, cols):
    cv = GroupKFold(n_splits=5)
    return cross_val_score(GaussianNB(), Xs[:, cols], y, groups=g, cv=cv,
                           scoring="f1_macro").mean()


def icafs(Xtr, ytr, gtr):
    """ICAFS con trazas: lista de (iteracion, filas_CA, f1, bandas)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xs = (Xtr - mu) / sd
    vigentes = list(range(Xs.shape[1]))
    traza = [(0, 0, f1_cv(Xs, ytr, gtr, vigentes), list(vigentes))]
    mejor_previo = traza[0][2]
    for it in range(1, T_MAX + 1):
        ca = Covering({str(j): [0, 1] for j in vigentes}, strength=FUERZA).array
        filas = [[j for j in vigentes if fila[str(j)] == 1] for fila in ca]
        filas = [f for f in filas if f]
        puntos = Parallel(n_jobs=N_JOBS)(delayed(f1_cv)(Xs, ytr, gtr, f) for f in filas)
        parcial, elegidas = 0.0, vigentes
        for p, f in zip(puntos, filas):
            if p >= parcial:
                parcial, elegidas = p, f
        traza.append((it, len(ca), parcial, list(elegidas)))
        vigentes = elegidas
        if parcial <= mejor_previo or len(vigentes) < MIN_BANDAS:
            break
        mejor_previo = parcial
    return traza


def main():
    g10 = v10.grupos(10)
    filas = []
    t0 = time.time()
    for s in range(N_SPLITS):
        tr, te = v10.holdout(100, 50000 + s)
        ts = time.time()
        traza = icafs(X[tr], Y[tr], g10[tr])
        print(f"split {s}: CAFS {time.time() - ts:.0f} s, tamanos "
              f"{[len(b) for *_, b in traza]}, f1 {[round(t[2], 3) for t in traza]}",
              flush=True)
        rng = np.random.default_rng(60000 + s)
        for it, n_ca, f1, bandas in traza[1:]:
            m = len(bandas)
            azar = [np.sort(rng.choice(X.shape[1], m, replace=False)) for _ in range(N_AZAR)]
            for nombre in ("SVM", "LDA"):
                filas.append({"split": s, "iter": it, "filas_CA": n_ca, "f1_interno": f1,
                              "m": m, "modelo": nombre, "metodo": "CAFS",
                              "acc": v10.evaluar_subconjunto(bandas, tr, te, nombre),
                              "bandas": " ".join(map(str, bandas))})
                accs = Parallel(n_jobs=N_JOBS)(
                    delayed(v10.evaluar_subconjunto)(c, tr, te, nombre) for c in azar)
                for a in accs:
                    filas.append({"split": s, "iter": it, "filas_CA": n_ca, "f1_interno": f1,
                                  "m": m, "modelo": nombre, "metodo": "random", "acc": a})
        print(f"  split {s} evaluado, acumulado {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(filas).to_csv(AQUI / "resultados_UNF_cafs_parcial.csv", index=False)
    d = pd.DataFrame(filas)
    r = (d.groupby(["modelo", "iter", "metodo"])
         .agg(m_medio=("m", "mean"), m_min=("m", "min"), m_max=("m", "max"),
              n=("acc", "size"), mean=("acc", "mean"), std=("acc", "std"),
              q05=("acc", lambda x: x.quantile(0.05)),
              q95=("acc", lambda x: x.quantile(0.95)))
         .round(4))
    print(r, flush=True)
    with pd.ExcelWriter(AQUI / "resultados_UNF_cafs.xlsx") as w:
        r.to_excel(w, sheet_name="resumen")
        d.to_excel(w, sheet_name="corridas", index=False)
    fig(d)
    print(f"total {time.time() - t0:.0f} s", flush=True)


def fig(d):
    fig_, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, nombre in zip(ax, ("SVM", "LDA")):
        dd = d[d.modelo == nombre]
        az = dd[dd.metodo == "random"].groupby("iter")
        cf = dd[dd.metodo == "CAFS"].groupby("iter")
        mm = cf.m.mean()
        a.fill_between(mm, 100 * az.acc.quantile(0.05), 100 * az.acc.quantile(0.95),
                       alpha=0.25, color="C7", label="random subsets, 5-95 %")
        a.plot(mm, 100 * az.acc.mean(), "o-", color="C7", label="random subsets, mean")
        a.errorbar(mm, 100 * cf.acc.mean(), 100 * cf.acc.std(), marker="s",
                   color="C3", capsize=3, label="CAFS (naive Bayes criterion)")
        a.set_xscale("log")
        a.set_title(f"{nombre}, grouped hold-out b=100")
        a.set_xlabel("Number of bands (mean over splits)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Overall accuracy (%)")
    ax[0].legend()
    fig_.tight_layout()
    fig_.savefig(FIG / "fig_unf_bandas_cafs.png", dpi=200)


if __name__ == "__main__":
    main()
