"""
Manuscript v7 (2026-11-06): the schematic figures, in Python, replacing the Visio
drawings of `Figuras - Articulo.vsdx` (UNF, 2026-10-07), which had wrong labels.

  figuras/model_LDA.png  Fig. 2a  LDA as fitted (pooled covariance, linear scores)
  figuras/model_SVM.png  Fig. 2b  linear soft-margin SVM, C = 1, one-vs-one voting
  figuras/model_ANN.png  Fig. 2c  one ReLU hidden layer of round(sqrt(p)) neurons
  figuras/fig_CM.png     Fig. 3   confusion-matrix structure for class i: TP, FN, FP, TN

Notation follows the manuscript: K classes, N_ij = grains of variety i (row, actual)
assigned to variety j (column, predicted), p bands. Model settings as in
10_Validacion_UNF.modelo.

    python 24_Figuras_esquemas.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rutas import FIG  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 12})


# --------------------------------------------------------------------- Fig. 2
def caja(ax, x, y, w, h, lineas, dy=0.36):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.2))
    y0 = y + h / 2 + dy * (len(lineas) - 1) / 2
    for k, t in enumerate(lineas):
        ax.text(x + w / 2, y0 - k * dy, t, ha="center", va="center", fontsize=12.5)


def flecha(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.2, color="k"))


def esquema(nombre, entrada, medio_titulo, medio, salida, leyenda, alto_caja=1.5):
    nl = len(leyenda)
    H = 2.6 + 0.33 * nl
    fig, ax = plt.subplots(figsize=(8.0, H * 0.82))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.33 * nl - 0.3, 2.6)
    ax.axis("off")
    yc = 1.05
    y0 = yc - alto_caja / 2
    caja(ax, 0.05, y0, 2.0, alto_caja, entrada)
    caja(ax, 2.6, y0, 4.2, alto_caja, medio)
    caja(ax, 7.6, y0 + 0.2, 2.2, alto_caja - 0.4, salida)
    flecha(ax, 2.05, 2.6, yc)
    flecha(ax, 6.8, 7.6, yc)
    for x, t in ((1.05, "Input"), (4.7, medio_titulo), (8.7, "Output")):
        ax.text(x, y0 + alto_caja + 0.3, t, ha="center", va="center", fontsize=12.5,
                fontweight="bold")
    ax.text(0.2, -0.05, "where", fontsize=11.5, style="italic", va="center")
    for k, (s, d) in enumerate(leyenda):
        y = -0.38 - 0.33 * k
        ax.text(0.9, y, s, fontsize=11.5, va="center")
        ax.text(2.0, y, d, fontsize=11.5, va="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / nombre, dpi=300)
    plt.close(fig)


X_IN = [r"$\mathbf{x}\in\mathbb{R}^{p}$"]
X_STD = [r"$\mathbf{x}\in\mathbb{R}^{p}$", "standardized"]
ARGMAX = [r"$\hat{y}=\mathrm{arg\,max}_{c}\;\delta_c(\mathbf{x})$"]

esquema("model_LDA.png", X_IN, "Discriminant functions",
        [r"$S_W^{-1}S_B\,\mathbf{w}=\lambda\,\mathbf{w}$",
         r"$\delta_c(\mathbf{x})=\mathbf{w}_c^{\top}\mathbf{x}+b_c,\;\;c=1,\ldots,K$"],
        ARGMAX,
        [(r"$p$", "number of predictors (bands)"),
         (r"$K$", "number of classes (varieties)"),
         (r"$S_B,\ S_W$", "between- and within-class (pooled) scatter matrices"),
         (r"$\mathbf{w},\ \lambda$", "discriminant direction and its eigenvalue"),
         (r"$\mathbf{w}_c,\ b_c$", "coefficients and intercept of class $c$"),
         (r"$\delta_c(\mathbf{x})$", "discriminant score of class $c$"),
         (r"$\hat{y}$", "predicted class")])

esquema("model_SVM.png", X_STD, "Pairwise linear SVMs",
        [r"$\min_{\mathbf{w},b,\xi}\;\frac{1}{2}\|\mathbf{w}\|^{2}+C\,\sum\,\xi_n$",
         r"s.t. $y_n(\mathbf{w}^{\top}\mathbf{x}_n+b)\geq 1-\xi_n,\;\xi_n\geq 0$",
         r"$f_{kl}(\mathbf{x})=\mathbf{w}_{kl}^{\top}\mathbf{x}+b_{kl}$"],
        ARGMAX,
        [(r"$p,\ K$", "number of predictors and of classes"),
         (r"$f_{kl}$", "binary SVM of class $k$ against class $l$; $K(K-1)/2$ of them"),
         (r"$\mathbf{w},\ b$", "weight vector (normal to the hyperplane) and intercept"),
         (r"$y_n$", r"binary label of training grain $n$, $\in\{-1,+1\}$"),
         (r"$\xi_n$", "slack of training grain $n$ (margin violation)"),
         (r"$C$", "penalty on margin violations ($C=1$)"),
         (r"$\delta_c(\mathbf{x})$", "votes for class $c$ over the pairwise SVMs"),
         (r"$\hat{y}$", "predicted class")],
        alto_caja=1.7)

esquema("model_ANN.png", X_STD, "Hidden layer  →  output layer",
        [r"$\mathbf{h}=\mathrm{ReLU}(W_1\mathbf{x}+\mathbf{b}_1)$",
         r"$\mathbf{o}=W_2\mathbf{h}+\mathbf{b}_2$",
         r"$\mathbf{P}=\mathrm{softmax}(\mathbf{o})$"],
        [r"$\hat{y}=\mathrm{arg\,max}_{c}\;P_c$"],
        [(r"$p,\ K$", "number of predictors and of classes"),
         (r"$q$", r"hidden neurons, $q=\mathrm{round}(\sqrt{p})$ (16 for 263 bands)"),
         (r"$\mathbf{h}$", r"hidden activations, $\mathbf{h}\in\mathbb{R}^{q}$"),
         (r"$W_i,\ \mathbf{b}_i$", "weights and biases of layer $i$"),
         (r"$\mathbf{o},\ \mathbf{P}$", "output scores and class probabilities ($K$ each)"),
         (r"$\hat{y}$", "predicted class")],
        alto_caja=1.7)


# --------------------------------------------------------------------- Fig. 3
COL = {"TP": "#8cbf5a", "FN": "#f2b26b", "FP": "#7fb2e0", "TN": "#e6e6e6"}
idx = ["1", None, "i", None, "K"]          # None = ellipsis row/column


def tipo(r, c):
    if r == 2 and c == 2:
        return "TP"
    if r == 2:
        return "FN"
    if c == 2:
        return "FP"
    return "TN"


fig, ax = plt.subplots(figsize=(10.5, 5.0))
ax.set_xlim(-1.3, 12.2)
ax.set_ylim(-0.6, 6.4)
ax.set_aspect("equal")
ax.axis("off")
s = 1.0
for r in range(5):
    for c in range(5):
        x, y = c * s, (4 - r) * s
        ax.add_patch(Rectangle((x, y), s, s, fc=COL[tipo(r, c)], ec="k", lw=1.0))
        ir, ic = idx[r], idx[c]
        if ir and ic:
            t = rf"$N_{{{ir},{ic}}}$"
        elif ir is None and ic is None:
            t = r"$\ddots$" if r == c else r"$\cdots$"
        elif ir is None:
            t = r"$\vdots$"
        else:
            t = r"$\cdots$"
        ax.text(x + s / 2, y + s / 2, t, ha="center", va="center", fontsize=14)
for k, v in enumerate(idx):
    lab = rf"$C_{{{v}}}$" if v else r"$\cdots$"
    ax.text(k * s + s / 2, 5.25, lab, ha="center", va="center", fontsize=14)
    lab = rf"$C_{{{v}}}$" if v else r"$\vdots$"
    ax.text(-0.35, (4 - k) * s + s / 2, lab, ha="center", va="center", fontsize=14)
ax.text(2.5, 5.95, r"Predicted class ($j$)", ha="center", va="center", fontsize=14,
        fontweight="bold")
ax.text(-1.0, 2.5, r"Actual class ($i$)", ha="center", va="center", rotation=90,
        fontsize=14, fontweight="bold")

leyenda = [
    ("TP", "True positives", r"$TP_i=N_{ii}$"),
    ("FN", "False negatives (rest of row $i$)", r"$FN_i=\sum_{k=1}^{K}N_{ik}-TP_i$"),
    ("FP", "False positives (rest of column $i$)", r"$FP_i=\sum_{k=1}^{K}N_{ki}-TP_i$"),
    ("TN", "True negatives (all other cells)",
     r"$TN_i=\sum_{j=1}^{K}\sum_{k=1}^{K}N_{jk}-(TP_i+FP_i+FN_i)$"),
]
for k, (key, nom, f) in enumerate(leyenda):
    y = 4.6 - 1.35 * k
    ax.add_patch(Rectangle((5.8, y - 0.2), 0.4, 0.4, fc=COL[key], ec="k", lw=0.8))
    ax.text(6.4, y, f"{key}$_i$  {nom}", va="center", fontsize=13, fontweight="bold")
    ax.text(6.4, y - 0.55, f, va="center", fontsize=13.5)
fig.tight_layout(pad=0.2)
fig.savefig(FIG / "fig_CM.png", dpi=300)
plt.close(fig)
print("written:", *(FIG / n for n in ("model_LDA.png", "model_SVM.png",
                                       "model_ANN.png", "fig_CM.png")), sep="\n  ")
