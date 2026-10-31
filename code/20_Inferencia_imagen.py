"""
Inferencia para el manuscrito v4 (2026-10-30), a partir de las salidas de
19_Subconjuntos_UNF.py (mejoras/res_img_*.xlsx, mejoras/res_sub9_*.xlsx).

1. Clasificadores a nivel de imagen: la exactitud de cada una de las 60 imagenes se
   promedia sobre las repeticiones en que se evaluo; los clasificadores se comparan
   en pares sobre las 60 imagenes (Wilcoxon de rangos con signo, Holm sobre 12) con
   intervalo de confianza del 95 % por bootstrap de imagenes; division aleatoria frente
   a una imagen fuera, en pares por imagen (Holm sobre 4). Evita tratar como
   independientes las 30 particiones, que reutilizan las mismas imagenes.
2. Subconjuntos de 9 bandas de UNF frente a 30 subconjuntos de 9 bandas al azar por
   particion (b = 100): diferencia media, fraccion de subconjuntos al azar superados,
   t pareada y Wilcoxon sobre las 30 particiones (Holm sobre 12; las particiones no son
   independientes, se informa como descriptivo).

Salida: resultados_UNF_v4_inferencia.xlsx
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
MOD = ["SVM", "L-SVM", "LDA", "ANN"]
B = 10000


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out = np.empty_like(p)
    acc = 0.0
    for r, i in enumerate(o):
        acc = max(acc, (len(p) - r) * p[i])
        out[i] = min(1.0, acc)
    return out


def ic_boot(x, rng):
    x = np.asarray(x)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    rng = np.random.default_rng(2026)
    d = pd.concat([pd.read_excel(f) for f in sorted((AQUI / "mejoras").glob("res_img_*.xlsx"))])
    img = d.groupby(["b", "modelo", "imagen"]).apply(
        lambda g: np.average(g.acc, weights=g.n), include_groups=False).rename("acc").reset_index()
    resumen = []
    for (b, m), g in img.groupby(["b", "modelo"]):
        lo, hi = ic_boot(g.acc, rng)
        resumen.append({"b": b, "modelo": m, "media": 100 * g.acc.mean(),
                        "ic_lo": 100 * lo, "ic_hi": 100 * hi, "n_imagenes": len(g)})
    resumen = pd.DataFrame(resumen)
    pares = []
    for b, g in img.groupby("b"):
        w = g.pivot(index="imagen", columns="modelo", values="acc")
        for m1, m2 in combinations(MOD, 2):
            dif = w[m1] - w[m2]
            lo, hi = ic_boot(dif, rng)
            p = stats.wilcoxon(w[m1], w[m2]).pvalue if np.any(dif != 0) else 1.0
            pares.append({"b": b, "comparacion": f"{m1} - {m2}", "dif": 100 * dif.mean(),
                          "ic_lo": 100 * lo, "ic_hi": 100 * hi, "p_wilcoxon": p})
    pares = pd.DataFrame(pares)
    pares["p_holm"] = holm(pares.p_wilcoxon)
    caida = []
    for m, g in img.groupby("modelo"):
        w = g.pivot(index="imagen", columns="b", values="acc")
        dif = w[1] - w[100]
        lo, hi = ic_boot(dif, rng)
        caida.append({"modelo": m, "dif": 100 * dif.mean(), "ic_lo": 100 * lo, "ic_hi": 100 * hi,
                      "p_wilcoxon": stats.wilcoxon(w[1], w[100]).pvalue})
    caida = pd.DataFrame(caida)
    caida["p_holm"] = holm(caida.p_wilcoxon)

    s = pd.concat([pd.read_excel(f) for f in sorted((AQUI / "mejoras").glob("res_sub9_*.xlsx"))])
    sub = []
    for (m, met), g in s[s.metodo != "random"].groupby(["modelo", "metodo"]):
        az = s[(s.metodo == "random") & (s.modelo == m)]
        azm = az.groupby("split").acc.mean()
        gm = g.set_index("split").acc
        frac = np.mean([(gm[i] > az[az.split == i].acc).mean() for i in gm.index])
        sub.append({"modelo": m, "metodo": met, "acc": 100 * gm.mean(), "sd": 100 * gm.std(),
                    "azar": 100 * azm.mean(), "azar_q05": 100 * az.acc.quantile(0.05),
                    "azar_q95": 100 * az.acc.quantile(0.95), "dif": 100 * (gm - azm).mean(),
                    "frac_azar_superados": frac,
                    "p_t": stats.ttest_rel(gm, azm).pvalue,
                    "p_wilcoxon": stats.wilcoxon(gm, azm).pvalue})
    sub = pd.DataFrame(sub)
    sub["p_t_holm"] = holm(sub.p_t)
    sub["p_wilcoxon_holm"] = holm(sub.p_wilcoxon)

    with pd.ExcelWriter(AQUI / "resultados_UNF_v4_inferencia.xlsx") as w:
        resumen.to_excel(w, sheet_name="por_imagen_resumen", index=False)
        pares.to_excel(w, sheet_name="clasificadores_imagen", index=False)
        caida.to_excel(w, sheet_name="aleatoria_vs_imagen", index=False)
        sub.to_excel(w, sheet_name="subconjuntos9_vs_azar", index=False)
        img.to_excel(w, sheet_name="exactitud_por_imagen", index=False)
    pd.set_option("display.width", 200)
    for t in (resumen, pares, caida, sub):
        print(t.round(4).to_string(index=False)); print()


if __name__ == "__main__":
    main()
