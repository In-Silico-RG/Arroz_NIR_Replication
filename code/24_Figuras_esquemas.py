"""
Manuscript v7 (2026-11-06): the schematic figures, in Python, replacing the Visio
drawings of `Figuras - Articulo.vsdx` (UNF, 2026-10-07).

  figuras/fig_workflow.png  Fig. 1  workflow (figure*, full text width)
  figuras/fig_models.png    Fig. 2  LDA, SVM and ANN diagrams (figure*, full text width)
  figuras/fig_CM.png        Fig. 3  confusion-matrix structure for class i (column width)

Each figure is drawn at its printed size (elsarticle 5p: text width 522 pt = 7.22 in,
column 252 pt = 3.49 in), so font sizes are the printed sizes (6-8 pt).

Fig. 1 reuses four pictures from the Visio file (camera, segmented image, grain ROI,
extracted profiles; extracted to figuras/visio/); its other insets are drawn from the
data: PCA scores and mean profiles from X.mat, and the pooled grouped-division confusion
matrix of the SVM from res_cm_b100.npz (10_Validacion_UNF.py). Sums over k, j run
from 1 to K. Notation as in the manuscript: K classes, N_ij = grains of variety i (actual) assigned to j (predicted),
p bands. Model settings as in 10_Validacion_UNF.modelo.

    python 24_Figuras_esquemas.py
"""
import sys
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.io import loadmat

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from rutas import DATOS, NBANDAS, RES, FIG  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 7})
TW, CW = 7.22, 3.49          # text and column width, inches
VISIO = FIG / "visio"        # pictures extracted from Figuras - Articulo.vsdx
DPI = 400
GRIS = "#f2f2f2"


class Lienzo:
    """Figure with one full-size axes in inches (origin bottom-left)."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")

    def caja(self, x, y, w, h, texto="", fs=7, fc="white", ec="k", lw=0.8, bold=False,
             redondo=True):
        estilo = "round,pad=0,rounding_size=0.05" if redondo else "square,pad=0"
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=estilo, fc=fc, ec=ec, lw=lw))
        if texto:
            self.ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
                         fontweight="bold" if bold else "normal", linespacing=1.25)

    def flecha(self, x0, y0, x1, y1, lw=0.9):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                          mutation_scale=8, lw=lw, color="k",
                                          shrinkA=0, shrinkB=0))

    def linea(self, xs, ys, lw=0.9):
        self.ax.plot(xs, ys, color="k", lw=lw, solid_capstyle="butt")

    def texto(self, x, y, s, fs=7, **kw):
        self.ax.text(x, y, s, fontsize=fs, **kw)

    def inset(self, x, y, w, h):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])

    def imagen(self, f, x, y, w, h):
        a = self.inset(x, y, w, h)
        im = plt.imread(f)
        fondo = (im[..., :3].min(-1) if im.ndim == 3 else im) < (0.94 if im.dtype != np.uint8 else 240)
        r, c = np.where(fondo)
        if r.size:                                   # trim the white margins
            im = im[r.min():r.max() + 1, c.min():c.max() + 1]
        a.imshow(im)
        a.set_anchor("C")
        a.axis("off")

    def guardar(self, nombre):
        self.fig.savefig(FIG / nombre, dpi=DPI)
        plt.close(self.fig)


# ------------------------------------------------------------------- Fig. 1
def visio_media():
    """Extract the four pictures used from the .vsdx (a zip) into figuras/visio/."""
    from PIL import Image
    nombres = {"image6": "camara", "image2": "imagen_segmentada", "image3": "grano_roi",
               "image1": "perfiles_extraidos"}
    if all((VISIO / f"{v}.png").exists() for v in nombres.values()):
        return
    VISIO.mkdir(exist_ok=True)
    with zipfile.ZipFile(AQUI.parent / "Figuras - Articulo.vsdx") as z:
        for n in z.namelist():
            if n.startswith("visio/media/") and Path(n).stem in nombres:
                with z.open(n) as f:
                    Image.open(f).convert("RGB").save(VISIO / f"{nombres[Path(n).stem]}.png")


def fig_workflow():
    visio_media()
    X = loadmat(DATOS / "X.mat")["X"].astype(float)
    Y = loadmat(DATOS / "Y.mat")["Y"].ravel().astype(int)
    BW = loadmat(NBANDAS)["nBandas"].ravel()[46:309]
    cols = plt.get_cmap("tab20")(np.r_[0:20:2, 1:5:2][:12])
    clases = np.unique(Y)

    L = Lienzo(TW, 3.75)
    marcos = [((0.05, 0.45, 2.55, 3.25), "a) Database generation"),
              ((2.75, 0.45, 1.35, 3.25), "b) Exploratory\nanalysis"),
              ((4.25, 0.45, 2.92, 3.25), "c) Model building and validation")]
    for (x, y, w, h), t in marcos:
        L.caja(x, y, w, h, ec="0.35", lw=0.8)
        L.texto(x + 0.08, y + h - 0.1, t, fs=8, fontweight="bold", va="top", linespacing=1.1)

    # ---- a) database generation
    xa, wa = 0.15, 2.35
    xt, wt = xa + 1.05, wa - 1.05
    L.caja(xt, 3.08, wt, 0.3, "Rice grains\n12 varieties × 500 grains", fs=6.5, fc=GRIS)
    filas = [
        (2.28, ["camara"], "Image acquisition\nResonon Pika NIR-640\n336 bands, 888–1718 nm\n"
                           "5 images × 100 grains\nper variety"),
        (1.47, ["imagen_segmentada", "grano_roi"], "Grain segmentation\n11 × 11-pixel ROI\n"
                                                   "mean profile per grain"),
        (0.66, ["perfiles_extraidos"], "Pre-treatment\nSavitzky–Golay\n(w = 11, p = 2)\n"
                                       "Z-score per profile"),
    ]
    for y, pics, t in filas:
        wp = 0.98 / len(pics)
        for k, p in enumerate(pics):
            L.imagen(VISIO / f"{p}.png", xa + k * wp, y, wp - 0.03, 0.62)
        L.caja(xt, y + 0.02, wt, 0.66, t, fs=6.2, fc=GRIS)
    xm = xt + wt / 2
    L.flecha(xm, 3.08, xm, 2.96)
    L.flecha(xm, 2.30, xm, 2.15)
    L.flecha(xm, 1.49, xm, 1.34)
    L.flecha(xm, 0.68, xm, 0.30)
    # output of a) and the bus to b) and c)
    L.caja(0.3, 0.05, 2.95, 0.25, "6000 profiles × 263 bands (999–1649 nm)", fs=7, bold=True,
           fc="#fff3c4")
    L.linea([3.25, 5.71], [0.175, 0.175])
    L.flecha(3.42, 0.175, 3.42, 0.45)
    L.flecha(5.71, 0.175, 5.71, 0.45)

    # ---- b) exploratory analysis, insets drawn from the data
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    S = U[:, :2] * s[:2]
    a = L.inset(2.97, 2.0, 1.0, 1.1)
    for c, col in zip(clases, cols):
        P = S[Y == c][::4]
        a.scatter(P[:, 0], P[:, 1], s=0.5, color=col, linewidths=0)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("PC1", fontsize=6, labelpad=1); a.set_ylabel("PC2", fontsize=6, labelpad=1)
    L.texto(3.42, 1.78, "PCA scores", fs=6.5, ha="center", va="center")
    a = L.inset(2.97, 0.85, 1.0, 0.75)
    for c, col in zip(clases, cols):
        a.plot(BW, X[Y == c].mean(0), color=col, lw=0.5)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("Wavelength", fontsize=6, labelpad=1)
    L.texto(3.42, 0.6, "Mean profiles;\nbetween/within SD per band", fs=6,
            ha="center", va="center")

    # ---- c) model building and validation
    x0, w0 = 4.35, 2.72
    xs = x0 + 1.12
    L.caja(x0, 2.98, 1.0, 0.38, "Full spectrum\n263 bands", fs=6.5, fc=GRIS)
    L.caja(xs, 2.98, x0 + w0 - xs, 0.38, "Band selection: FSCNCA, SFS, CAFS\n"
                                          "vs random subsets of equal size", fs=6.2, fc=GRIS)
    L.caja(x0, 2.55, w0, 0.24, "Classifiers: LDA, L-SVM, SVM, ANN", fs=6.5, fc=GRIS)
    L.flecha(x0 + 0.5, 2.98, x0 + 0.5, 2.79)
    L.flecha(xs + (x0 + w0 - xs) / 2, 2.98, xs + (x0 + w0 - xs) / 2, 2.79)
    L.flecha(x0 + w0 / 2, 2.55, x0 + w0 / 2, 2.38)
    # division schematic: one variety = 5 groups of 100 spectra (drawn as 5 × 5 dots)
    L.caja(x0, 1.18, w0, 1.2, ec="0.5", lw=0.6)
    L.texto(x0 + w0 / 2, 2.26, "5-fold cross-validation × 6 (30 splits)", fs=6.5,
            ha="center", va="center", fontweight="bold")
    test_r = np.random.default_rng(3).random((5, 5, 5)) < 0.2
    for k, (titulo, sub) in enumerate((("Random division", "test spectra from every group"),
                                       ("Grouped division", "one whole group held out"))):
        xb = x0 + 0.1 + k * 1.33
        L.texto(xb + 0.62, 2.08, titulo, fs=6.5, ha="center", va="center", fontweight="bold")
        for im in range(5):
            xi = xb + im * 0.255
            L.caja(xi, 1.53, 0.22, 0.42, ec="0.4", lw=0.5, redondo=False)
            for r in range(5):
                for c in range(5):
                    t = test_r[im, r, c] if k == 0 else im == 2
                    L.ax.plot(xi + 0.03 + c * 0.04, 1.6 + r * 0.068, "o", ms=1.6,
                              mec="none", color="#d62728" if t else "0.6")
        L.texto(xb + 0.62, 1.43, sub, fs=6, ha="center", va="center")
    L.texto(x0 + 0.12, 1.27, "each block: a group of 100 spectra",
            fs=5.5, va="center", color="0.3")
    L.ax.plot([x0 + 2.02], [1.27], "o", ms=2.2, mec="none", color="#d62728")
    L.texto(x0 + 2.07, 1.27, "test", fs=5.5, va="center", color="0.3")
    L.ax.plot([x0 + 2.3], [1.27], "o", ms=2.2, mec="none", color="0.6")
    L.texto(x0 + 2.35, 1.27, "training", fs=5.5, va="center", color="0.3")
    L.flecha(x0 + 0.8 + (w0 - 0.8) / 2, 1.18, x0 + 0.8 + (w0 - 0.8) / 2, 0.95)
    # evaluation
    cm = np.load(RES / "res_cm_b100.npz")["SVM_b100"].astype(float)
    cm = cm / cm.sum(1, keepdims=True)
    a = L.inset(x0 + 0.08, 0.5, 0.6, 0.6)
    a.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    a.set_xticks([]); a.set_yticks([])
    L.caja(x0 + 0.8, 0.55, w0 - 0.8, 0.4, "Performance evaluation: confusion\n"
                                          "matrix, accuracy, macro-F1,\ngroup-level tests",
           fs=6.2, fc=GRIS)
    L.guardar("fig_workflow.png")


# ------------------------------------------------------------------- Fig. 2
def fig_models():
    L = Lienzo(TW, 3.15)
    colw, gap = 2.28, 0.13
    argmax = r"$\hat{y}=\mathrm{arg\,max}_c\;\delta_c(\mathbf{x})$"
    paneles = [
        ("(a) LDA", r"$\mathbf{x}\in\mathbb{R}^{p}$",
         [r"$S_W^{-1}S_B\,\mathbf{w}=\lambda\,\mathbf{w}$",
          r"$\delta_c(\mathbf{x})=\mathbf{w}_c^{\top}\mathbf{x}+b_c$",
          r"$c=1,\ldots,K$"], argmax,
         [(r"$p,\ K$", "number of bands, of classes"),
          (r"$S_B,\ S_W$", "between-, within-class scatter"),
          (r"$\mathbf{w},\ \lambda$", "discriminant direction, eigenvalue"),
          (r"$\mathbf{w}_c,\ b_c$", "coefficients, intercept of class $c$"),
          (r"$\delta_c(\mathbf{x})$", "discriminant score of class $c$"),
          (r"$\hat{y}$", "predicted class")]),
        ("(b) SVM", r"$\mathbf{x}\in\mathbb{R}^{p}$, standardized",
         [r"$\min\ \frac{1}{2}\|\mathbf{w}\|^{2}+C\,\sum\,\xi_n$",
          r"s.t. $y_n(\mathbf{w}^{\top}\mathbf{x}_n+b)\geq 1-\xi_n$, $\xi_n\geq 0$",
          r"$f_{kl}(\mathbf{x})=\mathbf{w}_{kl}^{\top}\mathbf{x}+b_{kl}$"], argmax,
         [(r"$f_{kl}$", "SVM of class $k$ vs $l$; $K(K-1)/2$ pairs"),
          (r"$\mathbf{w},\ b$", "weight vector, intercept"),
          (r"$y_n$", r"label of training grain $n$, $\pm 1$"),
          (r"$\xi_n$", "slack (margin violation)"),
          (r"$C$", "penalty on violations, $C=1$"),
          (r"$\delta_c(\mathbf{x})$", "votes for class $c$")]),
        ("(c) ANN", r"$\mathbf{x}\in\mathbb{R}^{p}$, standardized",
         [r"$\mathbf{h}=\mathrm{ReLU}(W_1\mathbf{x}+\mathbf{b}_1)$",
          r"$\mathbf{o}=W_2\mathbf{h}+\mathbf{b}_2$",
          r"$\mathbf{P}=\mathrm{softmax}(\mathbf{o})$"],
         r"$\hat{y}=\mathrm{arg\,max}_c\;P_c$",
         [(r"$q$", r"hidden neurons, $q=\mathrm{round}(\sqrt{p})$"),
          (r"$\mathbf{h}$", r"hidden activations, $\mathbf{h}\in\mathbb{R}^{q}$"),
          (r"$W_i,\ \mathbf{b}_i$", "weights, biases of layer $i$"),
          (r"$\mathbf{o},\ \mathbf{P}$", "output scores, class probabilities"),
          (r"$\hat{y}$", "predicted class")]),
    ]
    for k, (titulo, entrada, medio, salida, ley) in enumerate(paneles):
        x = 0.04 + k * (colw + gap)
        L.texto(x + colw / 2, 3.02, titulo, fs=8, fontweight="bold", ha="center", va="center")
        L.caja(x + 0.25, 2.6, colw - 0.5, 0.26, entrada, fs=7.5)
        L.flecha(x + colw / 2, 2.6, x + colw / 2, 2.44)
        L.caja(x, 1.66, colw, 0.78, fc=GRIS)
        for i, t in enumerate(medio):
            L.texto(x + colw / 2, 2.27 - i * 0.22, t, fs=7.5, ha="center", va="center")
        L.flecha(x + colw / 2, 1.66, x + colw / 2, 1.5)
        L.caja(x + 0.25, 1.22, colw - 0.5, 0.28, salida, fs=7.5)
        for i, (sym, d) in enumerate(ley):
            y = 1.02 - i * 0.165
            L.texto(x + 0.02, y, sym, fs=6.5, va="center")
            L.texto(x + 0.5, y, d, fs=6.5, va="center")
    L.guardar("fig_models.png")


# ------------------------------------------------------------------- Fig. 3
def fig_CM():
    COL = {"TP": "#8cbf5a", "FN": "#f2b26b", "FP": "#7fb2e0", "TN": "#e6e6e6"}
    idx = ["1", None, "i", None, "K"]          # None = ellipsis row/column

    def tipo(r, c):
        return "TP" if r == c == 2 else "FN" if r == 2 else "FP" if c == 2 else "TN"

    L = Lienzo(CW, 4.1)
    s, x0, y0 = 0.36, 0.95, 1.85
    for r in range(5):
        for c in range(5):
            x, y = x0 + c * s, y0 + (4 - r) * s
            L.caja(x, y, s, s, fc=COL[tipo(r, c)], lw=0.6, redondo=False)
            ir, ic = idx[r], idx[c]
            if ir and ic:
                t = rf"$N_{{{ir},{ic}}}$"
            elif ir is None and ic is None:
                t = r"$\ddots$" if r == c else r"$\cdots$"
            else:
                t = r"$\vdots$" if ir is None else r"$\cdots$"
            L.texto(x + s / 2, y + s / 2, t, fs=7.5, ha="center", va="center")
    for k, v in enumerate(idx):
        L.texto(x0 + k * s + s / 2, y0 + 5 * s + 0.12, rf"$C_{{{v}}}$" if v else r"$\cdots$",
                fs=7.5, ha="center", va="center")
        L.texto(x0 - 0.17, y0 + (4 - k) * s + s / 2, rf"$C_{{{v}}}$" if v else r"$\vdots$",
                fs=7.5, ha="center", va="center")
    L.texto(x0 + 2.5 * s, y0 + 5 * s + 0.33, r"Predicted class ($j$)", fs=8,
            fontweight="bold", ha="center", va="center")
    L.texto(x0 - 0.45, y0 + 2.5 * s, r"Actual class ($i$)", fs=8, fontweight="bold",
            rotation=90, ha="center", va="center")
    leyenda = [
        ("TP", "True positives", r"$TP_i=N_{ii}$"),
        ("FN", "False negatives (rest of row $i$)", r"$FN_i=\Sigma_{k}\,N_{ik}-TP_i$"),
        ("FP", "False positives (rest of column $i$)", r"$FP_i=\Sigma_{k}\,N_{ki}-TP_i$"),
        ("TN", "True negatives (all other cells)",
         r"$TN_i=\Sigma_{j}\Sigma_{k}\,N_{jk}-(TP_i+FP_i+FN_i)$"),
    ]
    for k, (key, nom, f) in enumerate(leyenda):
        y = 1.6 - k * 0.37
        L.caja(0.12, y - 0.06, 0.12, 0.12, fc=COL[key], lw=0.5, redondo=False)
        L.texto(0.32, y, f"{key}$_i$: {nom}", fs=7, fontweight="bold", va="center")
        L.texto(0.32, y - 0.17, f, fs=7.5, va="center")
    L.guardar("fig_CM.png")


if __name__ == "__main__":
    fig_workflow()
    fig_models()
    fig_CM()
    print("written:", *(FIG / n for n in ("fig_workflow.png", "fig_models.png", "fig_CM.png")),
          sep="\n  ")
