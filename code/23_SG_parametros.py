"""
Recover the Savitzky-Golay parameters applied by UNF from the pre-treated matrix
(profile_totales.mat, 6000 x 336 + label), 2026-11-05.

SG smoothing is a fixed convolution: the noise that survives it carries the filter's
frequency response |H(f)|^2, including its zeros. The mean power spectrum of the
first-differenced profiles (interior bands) is compared with |H(f)|^2 for every odd
window 5..39 and order 0..7; then the predicted zeros of the best candidates are
checked against the data. Also checks the per-profile z-score (over 336 bands).

Result: w = 11, p = 2 (p = 3 gives identical smoothing coefficients), d = 0;
PSD at the four predicted zeros (0.16, 0.26, 0.36, 0.45 cycles/band) is 4-9 % of
its local median; w = 9, 13, 15 and p = 4 do not match. Rows: mean 0, SD 1 (ddof=1).

    python 23_SG_parametros.py
"""
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_coeffs

AQUI = Path(__file__).resolve().parent
cands = [AQUI.parent / "data" / "profile_totales.mat",
         AQUI.parent / "ENVIO_ALDO_2026-10-30" / "Envio Aldo" / "profile_totales.mat"]
f_mat = next((c for c in cands if c.exists()), None)
if f_mat is None:
    sys.exit("profile_totales.mat not found")
X = loadmat(f_mat)["profile_totales"][:, :-1].astype(float)

print(f"rows: max |mean| {np.abs(X.mean(1)).max():.1e}, SD (ddof=1) "
      f"{X.std(1, ddof=1).min():.6f}-{X.std(1, ddof=1).max():.6f}")

R = np.diff(X[:, 30:-30], axis=1)
R -= R.mean(1, keepdims=True)
n = R.shape[1]
f = np.fft.rfftfreq(n)
F = (np.abs(np.fft.rfft(R * np.hanning(n), axis=1)) ** 2).mean(0)
F /= np.abs(1 - np.exp(-2j * np.pi * f)) ** 2 + 1e-12


def resp(w, p):
    k = np.arange(w) - (w - 1) / 2
    return np.abs(np.exp(-2j * np.pi * np.outer(f, k)) @ savgol_coeffs(w, p))


m = f > 0.08
fits = sorted(((np.corrcoef(np.log(resp(w, p)[m] ** 2 + 1e-6), np.log(F[m]))[0, 1], w, p)
               for w in range(5, 41, 2) for p in range(0, min(w - 1, 8))), reverse=True)
print("best fits (corr of log PSD with log |H|^2):")
for r, w, p in fits[:6]:
    print(f"  w={w:2d} p={p}  r={r:.4f}")
for w, p in [(11, 2), (9, 2), (13, 2), (15, 2), (11, 4)]:
    H = resp(w, p)
    z = [i for i in range(2, len(f) - 2) if H[i] < H[i - 1] and H[i] < H[i + 1] and H[i] < 0.05]
    rat = [F[i] / np.median(F[max(i - 6, 0):i + 7]) for i in z]
    print(f"w={w} p={p}: zeros at f={np.round(f[z], 3)}, PSD/local median={np.round(rat, 3)}")
