"""Paired significance tests on the per-split results of 10_Validacion_UNF.py and
11_CAFS_baseline.py.

Classifiers: at each block size b the four models share the same 30 splits
(grouped 5-fold CV x 6), so SVM/L-SVM/LDA/ANN are compared pairwise (Wilcoxon
signed-rank and paired t). Random vs grouped (b=1 vs b=100) uses different splits: Mann-Whitney U and
Welch t.

Band selection: on each split the selected subset and its 30 random subsets share the
split; the method's accuracy is paired with the mean random accuracy on that split.
Also reported: the fraction of random subsets the method beats on each split.

Holm correction within each family. Output: resultados_UNF_significancia.xlsx
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
import sys
SUF = sys.argv[1] if len(sys.argv) > 1 else ""      # "", "_excl" o "_relab"


def holm(p):
    p = np.asarray(p, float)
    orden = np.argsort(p)
    ajust = np.empty_like(p)
    acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, (len(p) - r) * p[i])
        ajust[i] = min(1.0, acum)
    return ajust


def pareado(a, b):
    d = np.asarray(a) - np.asarray(b)
    w = stats.wilcoxon(a, b) if np.any(d != 0) else None
    t = stats.ttest_rel(a, b)
    return {"n": len(d), "media_dif": 100 * d.mean(), "sd_dif": 100 * d.std(ddof=1),
            "p_wilcoxon": w.pvalue if w else 1.0, "p_t": t.pvalue}


def clasificadores():
    d = pd.read_excel(AQUI / f"resultados_UNF{SUF}.xlsx", sheet_name="U1U2_corridas")
    filas = []
    for b, g in d.groupby("b"):
        w = g.pivot(index="rep", columns="modelo", values="acc")
        for m1, m2 in combinations(["SVM", "ANN", "LDA", "L-SVM"], 2):
            filas.append({"b": b, "comparacion": f"{m1} - {m2}", **pareado(w[m1], w[m2])})
    r = pd.DataFrame(filas)
    r["p_wilcoxon_holm"] = holm(r.p_wilcoxon)
    r["p_t_holm"] = holm(r.p_t)
    filas = []
    for m, g in d.groupby("modelo"):
        a, c = g[g.b == 1].acc, g[g.b == 100].acc
        filas.append({"modelo": m, "media_b1": 100 * a.mean(), "media_b100": 100 * c.mean(),
                      "dif": 100 * (a.mean() - c.mean()),
                      "p_mannwhitney": stats.mannwhitneyu(a, c).pvalue,
                      "p_welch": stats.ttest_ind(a, c, equal_var=False).pvalue})
    u = pd.DataFrame(filas)
    u["p_mannwhitney_holm"] = holm(u.p_mannwhitney)
    return r, u


def bandas(d, metodo, nivel):
    filas = []
    for (nombre, k), g in d.groupby(["modelo", nivel]):
        met, azar, gana, tam = [], [], [], []
        for s, gs in g.groupby("split"):
            m = gs[gs.metodo == metodo].acc
            a = gs[gs.metodo == "random"].acc
            if len(m) != 1 or len(a) == 0:
                continue
            met.append(m.iloc[0]); azar.append(a.mean()); gana.append((m.iloc[0] > a).mean())
            if "m" in gs:
                tam.append(gs.m.iloc[0])
        if len(met) < (9 if nivel == "caso" else 2):   # CAFS: solo iteraciones alcanzadas por >= 9 splits
            continue
        fila = {"metodo": metodo, "modelo": nombre, nivel: k, **pareado(met, azar),
                "frac_azar_superados": np.mean(gana)}
        if tam:
            fila["bandas_media"] = np.mean(tam)
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla3():
    """Tabla 3 recalculada: cada subconjunto de UNF contra el perfil completo, mismo
    modelo y mismas 30 particiones (b = 1 y b = 100)."""
    t = pd.read_excel(AQUI / f"resultados_UNF_tabla3{SUF}.xlsx", sheet_name="corridas")
    c = pd.read_excel(AQUI / f"resultados_UNF{SUF}.xlsx", sheet_name="U1U2_corridas")
    c = c[c.b.isin([1, 100])].assign(seleccion="None")   # perfil completo, mismas particiones
    t = pd.concat([t, c], ignore_index=True)
    filas = []
    for (b, mo), g in t.groupby(["b", "modelo"]):
        w = g.pivot(index="rep", columns="seleccion", values="acc")
        if "FSCNCA" not in w:
            continue
        for sel in ("FSCNCA", "SFS", "CAFS"):
            filas.append({"b": b, "modelo": mo, "comparacion": f"{sel} - None",
                          **pareado(w[sel], w["None"])})
    r = pd.DataFrame(filas)
    r["p_wilcoxon_holm"] = holm(r.p_wilcoxon)
    r["p_t_holm"] = holm(r.p_t)
    return r


def main():
    r, u = clasificadores()
    t3 = tabla3()
    hoja = AQUI / f"resultados_UNF{SUF}.xlsx"
    sfs = bandas(pd.read_excel(hoja, sheet_name="U3_corridas").query("k > 0"), "SFS", "k")
    fsc = bandas(pd.read_excel(hoja, sheet_name="U3_fscnca_corridas").query("k > 0"), "FSCNCA", "k")
    # CAFS: version v6 de UNF (14_CAFS_v6_baseline.py); nivel = caso (iterN o final)
    v6 = bandas(pd.read_excel(AQUI / f"resultados_UNF_cafs_v6{SUF}.xlsx", sheet_name="corridas"),
                "CAFS_v6", "caso")
    bt = pd.concat([sfs, fsc, v6], ignore_index=True)
    bt["p_wilcoxon_holm"] = holm(bt.p_wilcoxon)
    bt["p_t_holm"] = holm(bt.p_t)
    with pd.ExcelWriter(AQUI / f"resultados_UNF_significancia{SUF}.xlsx") as w:
        r.to_excel(w, sheet_name="clasificadores_pareado", index=False)
        u.to_excel(w, sheet_name="aleatoria_vs_b100", index=False)
        bt.to_excel(w, sheet_name="bandas_vs_azar", index=False)
        t3.to_excel(w, sheet_name="tabla3_seleccion_vs_completo", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False)); print(); print(u.round(5).to_string(index=False)); print()
    print(bt.drop(columns=["p_t"]).round(4).to_string(index=False)); print()
    print(t3.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
