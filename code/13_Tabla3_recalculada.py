"""Recalcula la Tabla 3 del manuscrito UNF con exactitud global (Ec. 3).

Subconjuntos que UNF selecciono con todos los datos en b_AnalizarPerfilesArroz_2.m
(recibido 2026-10-30), 9 bandas cada uno, evaluados con la misma particion que
10_Validacion_UNF.py (CV agrupada 5 pliegues x 6 repeticiones; el perfil completo
sale de ese script, mismas particiones):
  FSCNCA y SFS: leidos de las lineas rojas de Fig_FSCNA1.eps y Fig_SFSS.eps
                (calibradas con las 9 bandas de CAFS, residuo 0.00 nm)
  CAFS:         cafs_v6_selected_wavelengths_by_fold.csv, pliegue externo 1
Cuatro modelos:
  SVM    fitcecoc + templateSVM('Standardize',true): SVC lineal C=1, estandarizado
  L-SVM  fitcecoc + templateLinear (el "LDA" del borrador): SVM lineal, bisagra,
         Lambda = 1/n (C = 1), sin estandarizar, solver dual (p > 100), uno-contra-uno
  LDA    discriminante lineal propiamente dicho
  ANN    fitcnet, una capa oculta de round(sqrt(p)) unidades ReLU, estandarizado
Divisiones b = 1 (aleatoria) y b = 100 (una imagen por variedad en prueba).

Salida: resultados_UNF_tabla3.xlsx (resumen y corridas).
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v", AQUI / "10_Validacion_UNF.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

NB = loadmat(AQUI.parent / "NUEVA_DATA_SEPT_2026" / "nBandas.mat")["nBandas"].ravel()
BW = NB[46:309]                       # bandas_rango = [47, 309] en MATLAB
NM = {
    "FSCNCA": [1038, 1114, 1119, 1124, 1237, 1368, 1371, 1568, 1649],
    "SFS":    [1033, 1102, 1124, 1129, 1151, 1237, 1274, 1435, 1649],
    "CAFS":   [1012, 1063, 1232, 1234, 1420, 1435, 1624, 1639, 1649],
}
SUB = {k: [int(np.argmin(abs(BW - w))) for w in nm] for k, nm in NM.items()}


def modelo(nombre, p):
    if nombre == "SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    if nombre == "L-SVM":
        return OneVsOneClassifier(LinearSVC(C=1.0, loss="hinge", dual=True, max_iter=1000))
    if nombre == "LDA":
        return LinearDiscriminantAnalysis()
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(int(round(np.sqrt(p))),),
                                       solver="lbfgs", max_iter=1000, random_state=0))


def corrida(sub, b, rep, nombre):
    warnings.filterwarnings("ignore")
    cols = SUB[sub]
    tr, te = v.particion(b, rep)
    m = modelo(nombre, len(cols)).fit(v.X[tr][:, cols], v.Y[tr])
    return {"seleccion": sub, "b": b, "rep": rep, "modelo": nombre,
            **{k: x for k, x in v.metricas(v.Y[te], m.predict(v.X[te][:, cols])).items() if k != "cm"}}


def main():
    tareas = [(s, b, r, m) for s in SUB for b in (1, 100) for r in range(v.N_REP)
              for m in ("SVM", "L-SVM", "LDA", "ANN")]
    d = pd.DataFrame(Parallel(n_jobs=-1)(delayed(corrida)(*t) for t in tareas))
    r = (d.groupby(["b", "seleccion", "modelo"])[["acc", "f1_macro", "acc_ovr_mean"]]
         .agg(["mean", "std"]) * 100).round(2)
    with pd.ExcelWriter(AQUI / f"resultados_UNF_tabla3{v.SUFIJO}.xlsx") as w:
        r.to_excel(w, sheet_name="resumen")
        d.to_excel(w, sheet_name="corridas", index=False)
        pd.DataFrame([{"seleccion": k, "nm": ", ".join(map(str, NM[k])),
                       "indices_0": ", ".join(map(str, SUB[k]))} for k in NM]
                     ).to_excel(w, sheet_name="bandas", index=False)
    pd.set_option("display.width", 200)
    print(r.to_string())


if __name__ == "__main__":
    main()
