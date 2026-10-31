"""
Resultados para el manuscrito conjunto con UNF (opcion A, AC 2026-09-26).

Reproduce en Python los clasificadores del borrador de UNF (SVM lineal, SVM lineal
de templateLinear = L-SVM, LDA y red neuronal de una capa) y anade tres cosas.
Particion (revision 2026-10-30, tras la revision externa M2): validacion cruzada
agrupada de 5 pliegues repetida 6 veces (30 particiones) para todo b; en cada
pliegue se reserva el 20 % de cada variedad (con b = 100, exactamente una imagen).
El hold-out 70/30 anterior reservaba 2 de 5 bloques (40 %) con b = 100.
  U1  metrica: exactitud global y F1-macro junto a la "exactitud" por clase
      uno-contra-resto que reporta el borrador (= 1 - 2e/K)
  U2  hold-out agrupado: se reservan bloques enteros de b filas contiguas por
      variedad (b = 10, 50, 100; b = 100 corresponde a una imagen de 100 granos
      si el orden de filas sigue la adquisicion, pendiente de confirmar por UNF)
  U3  linea base de bandas al azar: seleccion secuencial hacia adelante (SFS,
      criterio LDA, ajustada solo en entrenamiento) contra subconjuntos al azar
      del mismo tamano, bajo hold-out agrupado b = 100

Ejecutar desde cualquier directorio:
    python3 draft_UNF_conjunto/10_Validacion_UNF.py
Guarda figuras/fig_unf_*.png y resultados_UNF.xlsx junto a este script.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FIG = AQUI / "figuras"
FIG.mkdir(exist_ok=True)

X = loadmat(RAIZ / "X.mat")["X"].astype(float)
Y = loadmat(RAIZ / "Y.mat")["Y"].ravel().astype(int)

# Analisis de sensibilidad (2026-10-30): tres imagenes (bloques de 100 filas) que con
# validacion por imagen se clasifican enteras como otra variedad (Bellavista img 5 ->
# La Union, La Conquista img 1 -> Mallares, Capirona img 5 -> La Puntilla). Con
# ARROZ_EXCLUIR=1 se eliminan y las salidas llevan el sufijo _excl.
# Con ARROZ_RELABEL=1 (escenario: etiquetas confirmadas como erroneas) esas tres
# imagenes pasan a la variedad que aparentan (Bellavista -> La Union, La Conquista ->
# Mallares, Capirona -> La Puntilla); sufijo _relab. Los bloques (imagenes) se definen
# siempre con la etiqueta original (Y_ORIG), de modo que cada imagen sigue entera.
import os
SOSPECHOSAS = {12: (4, 11), 14: (0, 13), 17: (4, 21)}   # clase: (bloque 0..4, clase aparente)
Y_ORIG = Y.copy()
SUFIJO = ""
if os.environ.get("ARROZ_EXCLUIR") == "1":
    SUFIJO = "_excl"
    _keep = np.ones(len(Y), bool)
    for _c, (_b, _n) in SOSPECHOSAS.items():
        _keep[np.flatnonzero(Y_ORIG == _c)[100 * _b:100 * (_b + 1)]] = False
    X, Y, Y_ORIG = X[_keep], Y[_keep], Y_ORIG[_keep]
elif os.environ.get("ARROZ_RELABEL") == "1":
    SUFIJO = "_relab"
    for _c, (_b, _n) in SOSPECHOSAS.items():
        Y[np.flatnonzero(Y_ORIG == _c)[100 * _b:100 * (_b + 1)]] = _n
CLASES = np.unique(Y)
K = len(CLASES)
N_FOLDS = 5
N_REP_CV = 6
N_REP = N_FOLDS * N_REP_CV          # 30 particiones por tamano de bloque
BLOQUES = [1, 10, 50, 100]


# ---------------------------------------------------------------- particiones
def grupos(b):
    """Etiqueta de bloque: b filas contiguas dentro de cada variedad."""
    g = np.empty(len(Y), dtype=int)
    for c in CLASES:
        idx = np.flatnonzero(Y_ORIG == c)
        g[idx] = c * 10000 + np.arange(len(idx)) // b
    return g


def particion(b, i, base=0):
    """Particion i (0..29) de la CV agrupada: repeticion i // 5, pliegue i % 5.
    En cada variedad los bloques de b filas se barajan (semilla 1000*b + base + rep)
    y se reparten en 5 pliegues iguales; el pliegue i % 5 es la prueba (20 %)."""
    rep, k = divmod(i, N_FOLDS)
    rng = np.random.default_rng(1000 * b + base + rep)
    g = grupos(b)
    prueba = np.zeros(len(Y), bool)
    for c in CLASES:
        bl = np.unique(g[Y == c])
        rng.shuffle(bl)
        prueba |= np.isin(g, np.array_split(bl, N_FOLDS)[k])
    return np.flatnonzero(~prueba), np.flatnonzero(prueba)


# ---------------------------------------------------------------- modelos
def modelos():
    p = X.shape[1]
    return {
        # fitcecoc + templateSVM('Standardize',true): ECOC uno-contra-uno, C = 1
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
        # LDA propiamente dicho (el borrador usa templateLinear; ver docs/02)
        "LDA": LinearDiscriminantAnalysis(),
        # fitcecoc + templateLinear (el "LDA" del MATLAB): SVM lineal, bisagra,
        # Lambda = 1/n (C = 1), sin estandarizar, uno-contra-uno, ajustado a convergencia
        "L-SVM": OneVsOneClassifier(LinearSVC(C=1.0, loss="hinge", dual=True, max_iter=1000)),
        # fitcnet: una capa oculta ReLU, LBFGS; tamano sqrt(p)
        "ANN": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(int(round(np.sqrt(p))),),
                                           solver="lbfgs", max_iter=1000,
                                           random_state=0)),
    }


def metricas(y, yp):
    cm = confusion_matrix(y, yp, labels=CLASES)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    ovr = np.mean((tp + tn) / cm.sum())          # metrica del borrador
    return {"acc": accuracy_score(y, yp),
            "f1_macro": f1_score(y, yp, average="macro"),
            "acc_ovr_mean": ovr, "cm": cm}


def una_corrida(b, rep, nombre):
    import warnings
    warnings.filterwarnings("ignore")
    tr, te = particion(b, rep)
    m = modelos()[nombre]
    m.fit(X[tr], Y[tr])
    return {"b": b, "rep": rep, "modelo": nombre, **metricas(Y[te], m.predict(X[te]))}


# ---------------------------------------------------------------- U1 + U2
def u1_u2(bloques=BLOQUES):
    tareas = [(b, r, n) for b in bloques for r in range(N_REP)
              for n in ("SVM", "L-SVM", "LDA", "ANN")]
    filas = Parallel(n_jobs=-1)(delayed(una_corrida)(*t) for t in tareas)
    d = pd.DataFrame(filas)
    # matrices de confusion acumuladas (filas = verdadera), por modelo y b
    cms = {(b, n): np.sum([c for c in g.cm], axis=0) for (b, n), g in d.groupby(["b", "modelo"])}
    return d.drop(columns="cm"), cms


# ---------------------------------------------------------------- U3
K_BANDAS = [5, 10, 20, 40]
N_SPLITS_BANDAS = 10
N_AZAR = 30


def cv_lda(Xs, y, g, folds=5):
    """Exactitud LDA con validacion agrupada interna (solo entrenamiento)."""
    ug = np.unique(g)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    parte = {gg: i % folds for i, gg in enumerate(ug)}
    f = np.array([parte[gg] for gg in g])
    acc = []
    for k in range(folds):
        m = LinearDiscriminantAnalysis().fit(Xs[f != k], y[f != k])
        acc.append(np.mean(m.predict(Xs[f == k]) == y[f == k]))
    return np.mean(acc)


def sfs(Xtr, ytr, gtr, kmax):
    elegidas, resto = [], list(range(Xtr.shape[1]))
    for _ in range(kmax):
        puntos = Parallel(n_jobs=-1)(
            delayed(cv_lda)(Xtr[:, elegidas + [j]], ytr, gtr) for j in resto)
        j = resto[int(np.argmax(puntos))]
        elegidas.append(j)
        resto.remove(j)
    return elegidas


def evaluar_subconjunto(cols, tr, te, nombre):
    import warnings
    warnings.filterwarnings("ignore")
    m = modelos()[nombre]
    m.fit(X[tr][:, cols], Y[tr])
    return metricas(Y[te], m.predict(X[te][:, cols]))["acc"]


def u3(splits=None):
    g100 = grupos(100)
    # SFS agrupa internamente por bloques de 10 filas, porque con b = 100 el
    # entrenamiento solo tiene 3-4 bloques por variedad
    g10 = grupos(10)
    filas = []
    for s in (splits if splits is not None else range(N_SPLITS_BANDAS)):
        tr, te = particion(100, s, base=500)
        t0 = time.time()
        orden = sfs(X[tr], Y[tr], g10[tr], max(K_BANDAS))
        print(f"  split {s}: SFS {time.time() - t0:.0f} s", flush=True)
        rng = np.random.default_rng(60000 + s)
        for k in K_BANDAS:
            for nombre in ("SVM", "LDA"):
                filas.append({"split": s, "k": k, "modelo": nombre, "metodo": "SFS",
                              "acc": evaluar_subconjunto(orden[:k], tr, te, nombre)})
                azar = [np.sort(rng.choice(X.shape[1], k, replace=False)) for _ in range(N_AZAR)]
                accs = Parallel(n_jobs=-1)(
                    delayed(evaluar_subconjunto)(c, tr, te, nombre) for c in azar)
                for a in accs:
                    filas.append({"split": s, "k": k, "modelo": nombre,
                                  "metodo": "random", "acc": a})
        filas.append({"split": s, "k": -1, "modelo": "orden_SFS", "metodo": "SFS",
                      "acc": np.nan, "bandas": " ".join(map(str, orden))})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- U3 FSCNCA
# fscnca de MATLAB: NCA con un peso por banda (Yang, Wang & Zuo 2012).
# Distancia D_ij = sum_r w_r^2 |x_ir - x_jr|, kernel exp(-D/sigma), sigma = 1,
# objetivo (1/n) sum_i p_i - lambda sum_r w_r^2, lambda = 1/n (defecto de MATLAB).
# Ascenso por gradiente (Adam) sobre un subconjunto de referencia del entrenamiento.
NCA_POR_CLASE = 100      # filas de referencia por variedad (1200 en total)
NCA_ITER = 150
NCA_LR = 0.02
NCA_LOTE = 100


def fscnca_pesos(Xtr, ytr, semilla):
    rng = np.random.default_rng(semilla)
    idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), NCA_POR_CLASE, replace=False)
                          for c in np.unique(ytr)])
    Xr = Xtr[idx]
    Xr = ((Xr - Xr.mean(0)) / Xr.std(0)).astype(np.float32)
    yr = ytr[idx]
    n, p = Xr.shape
    lam = 1.0 / n
    w = np.ones(p, np.float32)
    m = np.zeros(p, np.float32)
    v = np.zeros(p, np.float32)
    for it in range(1, NCA_ITER + 1):
        G = np.zeros(p, np.float64)
        w2 = w ** 2
        for a in range(0, n, NCA_LOTE):
            sl = slice(a, min(a + NCA_LOTE, n))
            Dabs = np.abs(Xr[sl, None, :] - Xr[None, :, :])          # (c, n, p)
            D = Dabs @ w2                                            # (c, n)
            D[np.arange(D.shape[0]), np.arange(sl.start, sl.stop)] = np.inf
            L = -D
            L -= L.max(1, keepdims=True)
            P = np.exp(L)
            P /= P.sum(1, keepdims=True)
            Ym = yr[sl, None] == yr[None, :]
            pi = (P * Ym).sum(1)
            T = pi[:, None] * P - P * Ym
            G += np.einsum("cn,cnp->p", T, Dabs, optimize=True)
        g = (2 * w * (G / n - lam)).astype(np.float32)   # gradiente de ascenso
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        w = w + NCA_LR * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return np.abs(w)


def u3_fscnca(splits=None):
    filas = []
    for s in (splits if splits is not None else range(N_SPLITS_BANDAS)):
        tr, te = particion(100, s, base=500)
        t0 = time.time()
        w = fscnca_pesos(X[tr], Y[tr], 70000 + s)
        orden = list(np.argsort(-w))
        print(f"  split {s}: FSCNCA {time.time() - t0:.0f} s", flush=True)
        rng = np.random.default_rng(60000 + s)     # mismos subconjuntos al azar que u3()
        for k in K_BANDAS:
            for nombre in ("SVM", "LDA"):
                filas.append({"split": s, "k": k, "modelo": nombre, "metodo": "FSCNCA",
                              "acc": evaluar_subconjunto(orden[:k], tr, te, nombre)})
                azar = [np.sort(rng.choice(X.shape[1], k, replace=False)) for _ in range(N_AZAR)]
                accs = Parallel(n_jobs=-1)(
                    delayed(evaluar_subconjunto)(c, tr, te, nombre) for c in azar)
                for a in accs:
                    filas.append({"split": s, "k": k, "modelo": nombre,
                                  "metodo": "random", "acc": a})
        filas.append({"split": s, "k": -1, "modelo": "orden_FSCNCA", "metodo": "FSCNCA",
                      "acc": np.nan, "bandas": " ".join(map(str, orden[:max(K_BANDAS)]))})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- figuras
NOMBRES = ["La Unión", "Bellavista", "Mallares", "La Conquista", "Pitipo", "Amazonas",
           "Capirona", "IR43", "La Esperanza", "Tinajones", "La Puntilla", "Capoteña"]


def fig_cm(cms):
    """Matrices de confusion (% por fila) del espectro completo: b = 1 y b = 100."""
    mods = ("SVM", "L-SVM", "LDA", "ANN")
    fig, ax = plt.subplots(2, 4, figsize=(16, 9.6), layout="constrained")
    for f, b in enumerate((1, 100)):
        for c, n in enumerate(mods):
            cm = cms[(b, n)].astype(float)
            pc = 100 * cm / cm.sum(1, keepdims=True)
            a = ax[f, c]
            a.imshow(pc, cmap="Blues", vmin=0, vmax=100)
            for i in range(12):
                for j in range(12):
                    if round(pc[i, j]) >= 1:
                        a.text(j, i, f"{pc[i, j]:.0f}", ha="center", va="center", fontsize=6,
                               color="white" if pc[i, j] > 50 else "black")
            acc = 100 * np.trace(cm) / cm.sum()
            a.set_title(f"{n}, {'random division' if b == 1 else 'one image held out'}: {acc:.1f}%", fontsize=10)
            a.set_xticks(range(12), NOMBRES if f == 1 else [""] * 12, rotation=90, fontsize=6)
            a.set_yticks(range(12), NOMBRES if c == 0 else [""] * 12, fontsize=6)
    fig.supxlabel("Predicted variety", fontsize=9)
    fig.supylabel("True variety", fontsize=9)
    fig.savefig(FIG / f"fig_unf_cm{SUFIJO}.png", dpi=200)


def fig_protocolo(df, suf=""):
    r = df.groupby(["modelo", "b"]).agg(["mean", "std"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for nombre, col in zip(("SVM", "L-SVM", "LDA", "ANN"), ("C0", "C3", "C1", "C2")):
        d = r.loc[nombre]
        x = np.arange(len(BLOQUES))
        ax[0].errorbar(x, 100 * d[("acc", "mean")], 100 * d[("acc", "std")],
                       marker="o", color=col, label=nombre, capsize=3)
        ax[1].plot(x, 100 * d[("acc_ovr_mean", "mean")], "o--", color=col, label=f"{nombre}, per-class OvR")
        ax[1].plot(x, 100 * d[("acc", "mean")], "o-", color=col, label=f"{nombre}, overall")
    for a in ax:
        a.set_xticks(np.arange(len(BLOQUES)), ["random" if b == 1 else f"b={b}" for b in BLOQUES])
        a.set_xlabel("Validation unit (contiguous rows kept together)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Overall accuracy (%)")
    ax[0].set_title("Grouped 5-fold CV, 6 repetitions")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Per-class one-vs-rest vs overall")
    ax[0].legend()
    ax[1].legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_unf_protocolo{suf}.png", dpi=200)


def fig_bandas(df, suf=""):
    d = df[df.k > 0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, nombre in zip(ax, ("SVM", "LDA")):
        dd = d[d.modelo == nombre]
        az = dd[dd.metodo == "random"].groupby("k").acc
        sf = dd[dd.metodo == "SFS"].groupby("k").acc
        a.fill_between(K_BANDAS, 100 * az.quantile(0.05), 100 * az.quantile(0.95),
                       alpha=0.25, color="C7", label="random subsets, 5-95 %")
        a.plot(K_BANDAS, 100 * az.mean(), "o-", color="C7", label="random subsets, mean")
        a.errorbar(K_BANDAS, 100 * sf.mean(), 100 * sf.std(), marker="s",
                   color="C3", capsize=3, label="SFS (LDA criterion)")
        a.set_title(f"{nombre}, one image per variety held out")
        a.set_xlabel("Number of bands")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Overall accuracy (%)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / f"fig_unf_bandas{suf}.png", dpi=200)


def fig_bandas_metodo(df, metodo, archivo):
    d = df[df.k > 0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, nombre in zip(ax, ("SVM", "LDA")):
        dd = d[d.modelo == nombre]
        az = dd[dd.metodo == "random"].groupby("k").acc
        sel = dd[dd.metodo == metodo].groupby("k").acc
        a.fill_between(K_BANDAS, 100 * az.quantile(0.05), 100 * az.quantile(0.95),
                       alpha=0.25, color="C7", label="random subsets, 5-95 %")
        a.plot(K_BANDAS, 100 * az.mean(), "o-", color="C7", label="random subsets, mean")
        a.errorbar(K_BANDAS, 100 * sel.mean(), 100 * sel.std(), marker="s",
                   color="C0", capsize=3, label=metodo)
        a.set_title(f"{nombre}, one image per variety held out")
        a.set_xlabel("Number of bands")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Overall accuracy (%)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / archivo, dpi=200)


def correr_fscnca():
    t0 = time.time()
    print("U3 FSCNCA ...", flush=True)
    d = u3_fscnca()
    r = (d[d.k > 0].groupby(["modelo", "k", "metodo"]).acc
         .agg(["mean", "std", lambda s: s.quantile(0.95)]).round(4))
    r.columns = ["mean", "std", "q95"]
    print(r)
    with pd.ExcelWriter(AQUI / "resultados_UNF.xlsx", mode="a", engine="openpyxl",
                        if_sheet_exists="replace") as w:
        r.to_excel(w, sheet_name="U3_fscnca_resumen")
        d.to_excel(w, sheet_name="U3_fscnca_corridas", index=False)
    fig_bandas_metodo(d, "FSCNCA", "fig_unf_bandas_fscnca.png")
    print(f"total {time.time() - t0:.0f} s")


def _modo_cluster(arg):
    """Modos para GUANE (cada trabajo escribe su propio archivo; 17_Unir_resultados.py
    los junta): u12 B | sfs A C | fscnca A C   (B = tamano de bloque; A..C-1 = splits)."""
    t0 = time.time()
    if arg[0] == "u12":
      for b in map(int, arg[1:]):
        d12, cms = u1_u2([b])
        d12.to_excel(AQUI / f"res_u12_b{b}{SUFIJO}.xlsx", index=False)
        np.savez(AQUI / f"res_cm_b{b}{SUFIJO}.npz", **{f"{n}_b{bb}": c for (bb, n), c in cms.items()})
    elif arg[0] == "sfs":
        a, c = int(arg[1]), int(arg[2])
        u3(range(a, c)).to_excel(AQUI / f"res_sfs_{a}_{c}{SUFIJO}.xlsx", index=False)
    elif arg[0] == "fscnca":
        a, c = int(arg[1]), int(arg[2])
        u3_fscnca(range(a, c)).to_excel(AQUI / f"res_fscnca_{a}_{c}{SUFIJO}.xlsx", index=False)
    print(f"total {time.time() - t0:.0f} s", flush=True)


_ARGV = __import__("sys").argv
if __name__ == "__main__" and len(_ARGV) > 2 and _ARGV[1] in ("u12", "sfs", "fscnca"):
    _modo_cluster(_ARGV[1:])
elif __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "fscnca":
    correr_fscnca()
elif __name__ == "__main__":
    t0 = time.time()
    print("U1+U2 ...", flush=True)
    d12, cms = u1_u2()
    np.savez(AQUI / "resultados_UNF_cm.npz",
             **{f"{n}_b{b}": c for (b, n), c in cms.items()}, clases=CLASES)
    fig_cm(cms)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    resumen = (d12.groupby(["modelo", "b"])[["acc", "f1_macro", "acc_ovr_mean"]]
               .agg(["mean", "std"]).round(4))
    print(resumen)
    print("U3 ...", flush=True)
    d3 = u3()
    r3 = (d3[d3.k > 0].groupby(["modelo", "k", "metodo"]).acc
          .agg(["mean", "std", lambda s: s.quantile(0.95)]).round(4))
    r3.columns = ["mean", "std", "q95"]
    print(r3)
    with pd.ExcelWriter(AQUI / "resultados_UNF.xlsx") as w:
        resumen.to_excel(w, sheet_name="U1U2_resumen")
        d12.to_excel(w, sheet_name="U1U2_corridas", index=False)
        r3.to_excel(w, sheet_name="U3_resumen")
        d3.to_excel(w, sheet_name="U3_corridas", index=False)
    fig_protocolo(d12)
    fig_bandas(d3)
    print(f"total {time.time() - t0:.0f} s")
