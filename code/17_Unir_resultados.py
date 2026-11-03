"""Junta los archivos de los trabajos de GUANE (res_u12.xlsx, res_sfs_*.xlsx,
res_fscnca.xlsx) en resultados_UNF.xlsx y rehace las figuras de bandas.

    /home/aldo/myenv/bin/python3 draft_UNF_conjunto/17_Unir_resultados.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from rutas import DATOS, NBANDAS, RES, IMGDIR, FIG  # noqa: E402
_spec = importlib.util.spec_from_file_location("v10", AQUI / "10_Validacion_UNF.py")
v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v10)


def resumen_bandas(d):
    r = (d[d.k > 0].groupby(["modelo", "k", "metodo"]).acc
         .agg(["mean", "std", lambda s: s.quantile(0.95)]).round(4))
    r.columns = ["mean", "std", "q95"]
    return r


def _archivos(patron, suf):
    """res_*{suf}.xlsx de la variante: sin sufijo = datos completos."""
    fs = sorted(RES.glob(patron))
    if suf:
        return [f for f in fs if f.stem.endswith(suf)]
    return [f for f in fs if not f.stem.endswith(("_excl", "_relab"))]


def main(suf=""):
    d12 = pd.concat([pd.read_excel(f) for f in _archivos("res_u12_b*.xlsx", suf)], ignore_index=True)
    cms = {}
    for f in _archivos("res_cm_b*.npz", suf):
        z = np.load(f)
        for k in z.files:
            n, b = k.rsplit("_b", 1)
            cms[(int(b), n)] = z[k]
    np.savez(RES / f"resultados_UNF_cm{suf}.npz", **{f"{n}_b{b}": c for (b, n), c in cms.items()})
    resumen = (d12.groupby(["modelo", "b"])[["acc", "f1_macro", "acc_ovr_mean"]]
               .agg(["mean", "std"]).round(4))
    d3 = pd.concat([pd.read_excel(f) for f in _archivos("res_sfs_*.xlsx", suf)], ignore_index=True)
    df = pd.concat([pd.read_excel(f) for f in _archivos("res_fscnca_*.xlsx", suf)], ignore_index=True)
    with pd.ExcelWriter(RES / f"resultados_UNF{suf}.xlsx") as w:
        resumen.to_excel(w, sheet_name="U1U2_resumen")
        d12.to_excel(w, sheet_name="U1U2_corridas", index=False)
        resumen_bandas(d3).to_excel(w, sheet_name="U3_resumen")
        d3.to_excel(w, sheet_name="U3_corridas", index=False)
        resumen_bandas(df).to_excel(w, sheet_name="U3_fscnca_resumen")
        df.to_excel(w, sheet_name="U3_fscnca_corridas", index=False)
    v10.SUFIJO = suf
    v10.fig_cm(cms)
    v10.fig_protocolo(d12, suf)
    v10.fig_bandas(d3, suf)
    v10.fig_bandas_metodo(df, "FSCNCA", f"fig_unf_bandas_fscnca{suf}.png")
    print(resumen); print(resumen_bandas(d3)); print(resumen_bandas(df))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
