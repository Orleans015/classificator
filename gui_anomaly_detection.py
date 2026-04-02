"""
Workflow GUI — SXR Profile Anomaly Detector  (PyQt6 edition)
=============================================================
Visualises the original profile, adjusted profile,
reconstructed profile and the time-series correlation
for a given shot PID.

Outlier-detection method
-------------------------
  Model residuals — single-pass: flag diodes where
                    (profile − AE(profile)) > mean + z·σ  (positive
                    residuals only); z-threshold is user-configurable.

Requirements:
    pip install PyQt6 matplotlib numpy scipy torch
"""

import os
import sys
import numpy as np

# This is a really bad way of doing this!!
root = "/home/IPP-HGW/orluca/devel/QXTdataAccessPython"
for dirpath, dirnames, filenames in os.walk(root):
    if any(f.endswith(".py") for f in filenames):
        if dirpath not in sys.path:
            sys.path.append(dirpath)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QFrame, QSizePolicy, QGridLayout, QDialog, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from scipy.signal import welch
from scipy.stats  import entropy as scipy_entropy

# ── colour palette ────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#4f8ef7"
ACCENT2 = "#f7a24f"
ACCENT3 = "#4ff7a2"
ACCENT4 = "#f74f4f"
FG      = "#e8eaf0"
FG_DIM  = "#6b7280"
BORDER  = "#2a2d3a"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   FG,
    "axes.titlecolor":   FG,
    "xtick.color":       FG_DIM,
    "ytick.color":       FG_DIM,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": FG,
    "text.color":        FG,
    "font.family":       "monospace",
    "font.size":         9,
})

# ── global stylesheet ─────────────────────────────────────────────────────────
STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {FG};
    font-family: "Courier New", monospace;
    font-size: 10px;
}}
QFrame#sidebar {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}
QFrame#topbar {{
    background-color: {PANEL};
    border-bottom: 1px solid {BORDER};
}}
QLabel#title {{
    color: {ACCENT};
    font-size: 14px;
    font-weight: bold;
}}
QLabel#section {{
    color: {ACCENT};
    font-size: 9px;
    font-weight: bold;
}}
QLabel#dim {{
    color: {FG_DIM};
    font-size: 9px;
}}
QLineEdit {{
    background-color: #252836;
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 9px;
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled {{
    background-color: #1a1d27;
    color: {FG_DIM};
    border: 1px solid {BORDER};
}}
QPushButton#run {{
    background-color: {ACCENT};
    color: #0f1117;
    font-weight: bold;
    font-size: 11px;
    border: none;
    border-radius: 4px;
    padding: 9px;
}}
QPushButton#run:hover {{
    background-color: #7aabff;
}}
QPushButton#run:disabled {{
    background-color: #2a3550;
    color: {FG_DIM};
}}
QPushButton#browse {{
    background-color: {BORDER};
    color: {FG};
    border: none;
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 10px;
}}
QPushButton#browse:hover {{
    background-color: #3a3d4a;
}}
QFrame#divider {{
    background-color: {BORDER};
    max-height: 1px;
}}
QLabel#corr_ok  {{ color: {SUCCESS}; font-weight: bold; }}
QLabel#corr_bad {{ color: {DANGER};  font-weight: bold; }}
QLabel#verdict_ok  {{ color: {SUCCESS}; font-size: 12px; font-weight: bold; }}
QLabel#verdict_bad {{ color: {DANGER};  font-size: 12px; font-weight: bold; }}
QLabel#status {{ color: {FG_DIM}; font-size: 9px; }}
QPushButton#method_btn {{
    background-color: {BORDER};
    color: {FG_DIM};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 9px;
}}
QPushButton#method_btn:checked {{
    background-color: #1e3a5f;
    color: {ACCENT};
    border: 1px solid {ACCENT};
}}
QPushButton#method_btn:disabled {{
    background-color: #1e3a5f;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    opacity: 0.7;
}}
QPushButton#spectral {{
    background-color: #1a2e1a;
    color: {ACCENT3};
    font-weight: bold;
    font-size: 11px;
    border: 1px solid {ACCENT3};
    border-radius: 4px;
    padding: 9px;
}}
QPushButton#spectral:hover {{
    background-color: #253525;
}}
QPushButton#spectral:disabled {{
    background-color: {PANEL};
    color: {FG_DIM};
    border: 1px solid {BORDER};
}}
QComboBox {{
    background-color: #252836;
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px 4px;
}}
QComboBox:disabled {{
    background-color: #1a1d27;
    color: {FG_DIM};
}}
QComboBox QAbstractItemView {{
    background-color: #252836;
    color: {FG};
    selection-background-color: {ACCENT};
}}
QLabel#diode_corr_header {{
    color: {ACCENT3};
    font-size: 9px;
    font-weight: bold;
}}
QLabel#diode_corr_row {{
    color: {FG_DIM};
    font-size: 8px;
    font-family: monospace;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Pure computation helpers  (stateless, importable, testable)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray, min_data: float, max_data: float) -> np.ndarray:
    """Min-max normalise to [0, 1], returned as float32."""
    return ((arr - min_data) / (max_data - min_data)).astype(np.float32)


def denormalize(arr: np.ndarray, min_data: float, max_data: float) -> np.ndarray:
    """Invert min-max normalisation."""
    return arr * (max_data - min_data) + min_data


def model_forward_single(model, profile_norm: np.ndarray) -> np.ndarray:
    """Run the autoencoder on one normalised 1-D profile (float32 numpy array)."""
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(profile_norm).unsqueeze(0)).squeeze(0).numpy()


def model_forward_batch(model, batch_norm: np.ndarray) -> np.ndarray:
    """
    Run the autoencoder on a batch shaped (n_samples, n_diodes).
    Returns (n_samples, n_diodes) float32.
    """
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(batch_norm)).numpy()


def reconstruct(model, profile: np.ndarray,
                min_data: float, max_data: float) -> np.ndarray:
    """Normalise → model → denormalise for a single 1-D profile."""
    rec_norm = model_forward_single(model, normalize(profile, min_data, max_data))
    return denormalize(rec_norm, min_data, max_data)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    return float(np.corrcoef(a, b)[0, 1])


def apply_gain(profile: np.ndarray,
               outlier_mask: np.ndarray,
               gain_ratio: float) -> np.ndarray:
    """
    Return a gain-corrected copy of *profile*.

    gain_ratio = old_gain / new_gain.

    When gain_ratio > 1 the majority of channels recorded with the higher
    old gain and look inflated — they are the ones to scale; the spatial
    outliers are the channels that did NOT change gain.
    When gain_ratio < 1 the detected outlier channels are the ones that
    changed gain and need to be rescaled.
    """
    adjusted = profile.copy()
    if gain_ratio > 1:
        adjusted[~outlier_mask] *= gain_ratio  # majority-changed channels
    else:
        adjusted[outlier_mask] *= gain_ratio   # outlier channels changed gain
    return adjusted


def fill_outlier_gaps(outlier_mask: np.ndarray,
                      profile: np.ndarray,
                      tol: float = 0.10) -> np.ndarray:
    """
    For each pair of consecutive flagged diodes (i, j) that are at most
    8 positions apart, check whether every diode m between them is within
    *tol* (relative) of the average of profile[i] and profile[j].  If so,
    flag m as well — this closes small intra-camera gaps in the outlier region.
    """
    filled          = outlier_mask.copy()
    outlier_indices = np.where(outlier_mask)[0]

    for k in range(len(outlier_indices) - 1):
        i = outlier_indices[k]
        j = outlier_indices[k + 1]

        if j - i > 8:          # only close gaps within the same camera
            continue

        ref = (profile[i] + profile[j]) / 2.0
        for m in range(i + 1, j):
            if abs(profile[m] - ref) / (abs(ref) + 1e-10) <= tol:
                filled[m] = True

    return filled


# ── Single-pass model-residuals outlier detection ────────────────────────────

def detect_outliers_residual(profile: np.ndarray,
                              reconstruction: np.ndarray,
                              z_thresh: float) -> np.ndarray:
    """
    Flag diodes where  (profile − reconstruction) > mean + z_thresh · std.

    Only positive residuals are caught: diodes whose signal sits well above
    what the autoencoder expects given the rest of the spatial profile.
    """
    residuals = profile - reconstruction
    threshold = np.mean(residuals) + z_thresh * np.std(residuals)
    return residuals > threshold


# ── Batched helpers ───────────────────────────────────────────────────────────

def compute_correlation_series(model, signals: np.ndarray,
                                indices: np.ndarray,
                                min_data: float, max_data: float,
                                batch_size: int = 256) -> np.ndarray:
    """
    Compute per-time-step Pearson ρ(signal, AE(signal)) for the given column
    indices of *signals* (shape: n_diodes × n_time), in batches.

    Returns a float32 array of length len(indices).
    """
    correlations = np.empty(len(indices), dtype=np.float32)
    col = 0

    for start in range(0, len(indices), batch_size):
        idx_batch    = indices[start : start + batch_size]
        batch_sig    = signals[:, idx_batch]                    # (n_diodes, b)
        batch_norm   = normalize(batch_sig.T, min_data, max_data)  # (b, n_diodes)
        rec_norm     = model_forward_batch(model, batch_norm)   # (b, n_diodes)
        rec_dn       = denormalize(rec_norm, min_data, max_data)
        for j in range(len(idx_batch)):
            correlations[col] = pearson(batch_sig[:, j], rec_dn[j, :])
            col += 1
        del batch_sig, batch_norm, rec_norm, rec_dn

    return correlations


def _reconstruct_only_batched(model, signals: np.ndarray,
                               min_data: float, max_data: float,
                               batch_size: int = 256) -> np.ndarray:
    """
    Pass every time step of *signals* (n_diodes × n_time) through the
    autoencoder in batches.  No gain correction is applied here — the caller
    is responsible for providing already-corrected data.

    Returns rec shaped (n_diodes, n_time).  *signals* is NOT mutated.
    """
    n_time = signals.shape[1]
    rec    = np.empty_like(signals)

    for start in range(0, n_time, batch_size):
        end   = min(start + batch_size, n_time)
        batch = normalize(signals[:, start:end].T,
                          min_data, max_data)       # (b, n_diodes)
        out   = model_forward_batch(model, batch)   # (b, n_diodes)
        rec[:, start:end] = denormalize(out.T, min_data, max_data)
        del batch, out

    return rec


# ─────────────────────────────────────────────────────────────────────────────
#  Background worker
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, pid, model_path, min_data, max_data, gain_ratio,
                 z_thresh=0.2):
        super().__init__()
        self.pid        = pid
        self.model_path = model_path
        self.min_data   = min_data
        self.max_data   = max_data
        self.gain_ratio = gain_ratio
        self.z_thresh   = z_thresh

    def run(self):
        try:
            results = run_analysis(
                self.pid, self.model_path,
                self.min_data, self.max_data, self.gain_ratio,
                z_thresh    = self.z_thresh,
                progress_cb = self.progress.emit,
            )
            self.finished.emit(results)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  Core analysis  (orchestration only — heavy lifting delegated to helpers)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(pid, model_path, min_data, max_data, gain_ratio,
                 z_thresh=0.2,
                 progress_cb=None):
    import gc
    import torch

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    # ── imports ───────────────────────────────────────────────────────────────
    try:
        from model    import AutoEncoder
        from read_h5f import load_sxr_data
    except ImportError as e:
        raise ImportError(
            f"Could not import project modules: {e}\n"
            "Make sure 'model.py' and 'read_h5f.py' are on sys.path."
        ) from e

    # ── model ─────────────────────────────────────────────────────────────────
    _p("Loading model …")
    model = AutoEncoder.load_from_checkpoint(model_path)
    model.eval()

    # ── data ──────────────────────────────────────────────────────────────────
    _p(f"Loading SXR data for PID {pid} …")
    try:
        data, fpath, data_dict = load_sxr_data(pid)
    except Exception:
        _p(f"Data for PID {pid} not found — launching data access GUI …")
        try:
            import gui_qxt_loader as gqxt
            import subprocess
            subprocess.run(["python", os.path.abspath(gqxt.__file__)])
            data, fpath, data_dict = load_sxr_data(pid)
        except ImportError as e:
            raise ImportError(
                f"Could not import data access modules: {e}\n"
                "Make sure 'gui_qxt_loader.py' is on sys.path."
            ) from e

    timedata  = data[0,  :].astype(np.float32)
    signals    = data[1:, :].astype(np.float32)
    diode_ref  = data[152,:].astype(np.float32)
    del data
    gc.collect()

    diode_keys = [k for k in data_dict.keys() if k != "timedata"]
    del data_dict
    gc.collect()

    # ── reference time instant (peak emission) ────────────────────────────────
    max_emission = int(np.argmax(diode_ref))
    time_instant = float(timedata[max_emission])
    profile      = signals[:, max_emission].copy()
    n_diodes     = len(profile)
    x            = np.arange(n_diodes)

    # ── original reconstruction (single profile snapshot) ────────────────────
    _p("Running model on original profile …")
    rec_original  = reconstruct(model, profile, min_data, max_data)
    corr_original = pearson(profile, rec_original)

    # ── model-residuals outlier detection ─────────────────────────────────────
    _p(f"Running model-residuals detection (z={z_thresh}) …")
    outlier_mask  = detect_outliers_residual(profile, rec_original, z_thresh)
    outlier_mask  = fill_outlier_gaps(outlier_mask, profile, tol=0.1)
    mask          = ~outlier_mask
    _p(f"[Residuals] {int(outlier_mask.sum())} diodes flagged.")

    # ── gain-corrected single profile and its reconstruction ──────────────────
    adjusted      = apply_gain(profile, outlier_mask, gain_ratio)
    rec_adjusted  = reconstruct(model, adjusted, min_data, max_data)
    corr_adjusted = pearson(adjusted, rec_adjusted)

    gc.collect()

    # ── downsampled time-series indices (for corr plot + spectral) ────────────
    step    = max(10, signals.shape[1] // 500)
    indices = np.arange(0, signals.shape[1], step)

    # ── correlation time-series (on original signals) ─────────────────────────
    _p("Computing correlation time-series …")
    correlations = compute_correlation_series(
        model, signals, indices, min_data, max_data
    )
    gc.collect()

    # ── snapshot for spectral analysis ────────────────────────────────────────
    signals_sampled = signals[:, indices].copy()

    # ── gain-corrected full time series (no model pass) ───────────────────────
    _p("Applying gain correction to full time-series …")
    adjusted_full = signals.copy()
    if gain_ratio > 1:
        adjusted_full[~outlier_mask, :] *= gain_ratio
    else:
        adjusted_full[outlier_mask, :] *= gain_ratio

    # ── model reconstruction of the gain-corrected full time series ───────────
    _p("Running model on adjusted full time-series …")
    rec_adjusted_full = _reconstruct_only_batched(
        model, adjusted_full, min_data, max_data
    )
    gc.collect()

    _p("Done.")
    return dict(
        pid               = pid,
        time_instant      = time_instant,
        x                 = x,
        timedata          = timedata[indices],
        timedata_full     = timedata,
        profile           = profile,
        rec_original      = rec_original,
        outlier_mask      = outlier_mask,
        mask              = mask,
        adjusted          = adjusted,
        rec_adjusted      = rec_adjusted,
        adjusted_full     = adjusted_full,
        rec_adjusted_full = rec_adjusted_full,
        corr_original     = corr_original,
        corr_adjusted     = corr_adjusted,
        correlations      = correlations,
        diode_signal      = diode_ref[indices],
        file_path         = fpath,
        diode_keys        = diode_keys,
        method            = "residuals",
        signals_sampled   = signals_sampled,
        z_thresh          = z_thresh,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Small UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lbl(text, obj_name="dim"):
    w = QLabel(text)
    w.setObjectName(obj_name)
    return w


def _divider():
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.Shape.HLine)
    return f


def _entry(default=""):
    return QLineEdit(default)


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral Analysis Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SpectralAnalysisDialog(QDialog):
    """
    Stand-alone spectral analysis window.

    Three panels (all sharing the same diode-index x-axis):
      • Top    — 2-D PSD heatmap (diode × frequency, log₁₀ scale)
      • Middle — total spectral power per diode
      • Bottom — spectral entropy per diode

    Outlier diodes from the parent analysis are highlighted with red bands
    in all three panels.
    """

    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Spectral Analysis — PID {results['pid']}")
        self.resize(1240, 860)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLESHEET)
        self._r = results
        self._build_ui()
        self._compute_and_plot()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        ctrl   = QWidget()
        ctrl.setStyleSheet(f"background:{PANEL}; border-radius:4px;"
                           f" border:1px solid {BORDER}; padding:2px;")
        ctrl_h = QHBoxLayout(ctrl)
        ctrl_h.setContentsMargins(12, 6, 12, 6)
        ctrl_h.setSpacing(14)

        def _combo(items, width=90, default_idx=0):
            c = QComboBox()
            c.addItems(items)
            c.setCurrentIndex(default_idx)
            c.setFixedWidth(width)
            c.currentIndexChanged.connect(self._replot)
            return c

        ctrl_h.addWidget(_lbl("Window:", "dim"))
        self._win_combo     = _combo(["hann", "hamming", "blackman", "boxcar"])
        ctrl_h.addWidget(self._win_combo)

        ctrl_h.addWidget(_lbl("nperseg:", "dim"))
        self._nperseg_combo = _combo(["64", "128", "256", "512"], 70, 2)
        ctrl_h.addWidget(self._nperseg_combo)

        ctrl_h.addWidget(_lbl("f-max (Hz):", "dim"))
        self._fmax_combo = _combo(
            ["All", "50", "100", "200", "500", "1000", "2000", "5000"], 80)
        ctrl_h.addWidget(self._fmax_combo)

        ctrl_h.addWidget(_lbl("Colormap:", "dim"))
        self._cmap_combo = _combo(
            ["inferno", "magma", "plasma", "viridis", "cividis", "hot"], 80)
        ctrl_h.addWidget(self._cmap_combo)

        ctrl_h.addStretch()
        self._info_lbl = _lbl("", "dim")
        ctrl_h.addWidget(self._info_lbl)
        root.addWidget(ctrl)

        self._fig  = Figure(figsize=(11, 8), dpi=100, facecolor=BG)
        canvas     = FigureCanvasQTAgg(self._fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        root.addWidget(canvas, stretch=1)

        tb_w = QWidget()
        tb_w.setStyleSheet(f"background:{PANEL};")
        tb_h = QHBoxLayout(tb_w)
        tb_h.setContentsMargins(4, 0, 4, 0)
        toolbar = NavigationToolbar2QT(canvas, tb_w)
        toolbar.setStyleSheet(f"background:{PANEL}; color:{FG};")
        tb_h.addWidget(toolbar)
        root.addWidget(tb_w)

        self._canvas = canvas

    def _compute_psd_matrix(self, nperseg: int, window: str):
        sigs      = self._r["signals_sampled"]
        timedata  = self._r["timedata"]
        dt_ms     = float(np.median(np.diff(timedata)))
        fs_est    = 1.0 / (dt_ms * 1e-3)

        n_diodes  = sigs.shape[0]
        freqs, _  = welch(sigs[0], fs=fs_est, nperseg=nperseg, window=window)
        psd_mat   = np.empty((n_diodes, len(freqs)), dtype=np.float32)
        for i in range(n_diodes):
            _, p = welch(sigs[i].astype(np.float64),
                         fs=fs_est, nperseg=nperseg, window=window)
            psd_mat[i] = p.astype(np.float32)

        return freqs, psd_mat, fs_est

    def _compute_and_plot(self):
        nperseg = int(self._nperseg_combo.currentText())
        window  = self._win_combo.currentText()
        self._freqs, self._psd, self._fs = self._compute_psd_matrix(nperseg, window)
        self._draw()

    def _replot(self):
        self._compute_and_plot()

    def _draw(self):
        r            = self._r
        freqs        = self._freqs
        psd          = self._psd
        outlier_mask = r["outlier_mask"]
        n_diodes     = psd.shape[0]
        x            = np.arange(n_diodes)
        cmap         = self._cmap_combo.currentText()

        fmax_txt  = self._fmax_combo.currentText().replace(" ", "")
        freq_mask = np.ones(len(freqs), dtype=bool) if fmax_txt == "All" \
                    else freqs <= float(fmax_txt)
        freqs_plt = freqs[freq_mask]
        psd_plt   = psd[:, freq_mask]

        total_power  = psd_plt.sum(axis=1)
        spec_entropy = np.array([
            float(scipy_entropy(psd_plt[i] / (psd_plt[i].sum() + 1e-30)))
            for i in range(n_diodes)
        ], dtype=np.float32)
        dom_freq = freqs_plt[np.argmax(psd_plt, axis=1)]

        self._fig.clf()
        gs = gridspec.GridSpec(3, 2, figure=self._fig,
                               height_ratios=[3, 1, 1], width_ratios=[30, 1],
                               hspace=0.52, wspace=0.03,
                               left=0.07, right=0.95, top=0.93, bottom=0.07)
        ax_heat  = self._fig.add_subplot(gs[0, 0])
        ax_cbar  = self._fig.add_subplot(gs[0, 1])
        ax_power = self._fig.add_subplot(gs[1, 0])
        ax_entr  = self._fig.add_subplot(gs[2, 0])

        for ax in (ax_heat, ax_power, ax_entr):
            ax.set_facecolor(PANEL)
            ax.grid(True, alpha=0.35)

        self._fig.suptitle(
            f"Spectral Analysis — PID: {r['pid']}   |   "
            f"window={self._win_combo.currentText()}  "
            f"nperseg={self._nperseg_combo.currentText()}  "
            f"fs≈{self._fs:.0f} Hz",
            color=FG, fontsize=10, y=0.985,
        )

        im = ax_heat.imshow(
            np.log10(psd_plt.T + 1e-30), aspect="auto", origin="lower",
            extent=[-0.5, n_diodes - 0.5, freqs_plt[0], freqs_plt[-1]],
            cmap=cmap, interpolation="nearest",
        )
        ax_heat.plot(x, dom_freq, color=ACCENT3, lw=1.2, ls="--",
                     alpha=0.8, label="Dominant freq / diode")
        ax_heat.set_ylabel("Frequency  (Hz)", color=FG)
        ax_heat.set_title(
            "Power Spectral Density  [log₁₀(V²/Hz)] — each column is one diode",
            color=FG, pad=6, fontsize=9)
        ax_heat.tick_params(labelbottom=False)
        ax_heat.legend(fontsize=7, loc="upper right")

        cb = self._fig.colorbar(im, cax=ax_cbar)
        cb.set_label("log₁₀ PSD", color=FG, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG_DIM)

        ax_power.fill_between(x, total_power, alpha=0.20, color=ACCENT)
        ax_power.plot(x, total_power, color=ACCENT, lw=1.5,
                      label="Total spectral power")
        ax_power.set_ylabel("Total power\n(V²/Hz)", color=FG, fontsize=8)
        ax_power.tick_params(labelbottom=False)

        ax_entr.fill_between(x, spec_entropy, alpha=0.20, color=ACCENT2)
        ax_entr.plot(x, spec_entropy, color=ACCENT2, lw=1.5,
                     label="Spectral entropy")
        ax_entr.set_ylabel("Entropy\n(nats)", color=FG, fontsize=8)
        ax_entr.set_xlabel("Diode index", color=FG)

        if outlier_mask.any():
            out_idx = np.where(outlier_mask)[0]
            for ax in (ax_heat, ax_power, ax_entr):
                for idx in out_idx:
                    ax.axvspan(idx - 0.5, idx + 0.5,
                               color=ACCENT4, alpha=0.22, linewidth=0, zorder=0)
            ax_power.scatter(out_idx, total_power[out_idx],
                             color=ACCENT4, s=50, zorder=6,
                             label=f"Outliers ({len(out_idx)})")
            ax_entr.scatter(out_idx, spec_entropy[out_idx],
                            color=ACCENT4, s=50, zorder=6,
                            label=f"Outliers ({len(out_idx)})")

        for ax in (ax_power, ax_entr):
            ax.legend(fontsize=7, loc="upper right")
        for ax in (ax_heat, ax_power, ax_entr):
            ax.set_xlim(-0.5, n_diodes - 0.5)

        n_out = int(outlier_mask.sum())
        self._info_lbl.setText(
            f"{n_diodes} diodes  |  {n_out} flagged  |  "
            f"{len(freqs_plt)} freq bins  |  fs ≈ {self._fs:.0f} Hz"
        )
        self._canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXR Profile Anomaly Detector")
        self.resize(1480, 860)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        self._thread  = None
        self._worker  = None
        self._results = None
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root   = QWidget()
        self.setCentralWidget(root)
        root_v = QVBoxLayout(root)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(52)
        tb_h = QHBoxLayout(topbar)
        tb_h.setContentsMargins(20, 0, 20, 0)
        tb_h.addWidget(_lbl("⚡  SXR Anomaly Detector", "title"))
        tb_h.addStretch()
        self._status_lbl = _lbl("Ready", "status")
        tb_h.addWidget(self._status_lbl)
        root_v.addWidget(topbar)

        body   = QWidget()
        body_h = QHBoxLayout(body)
        body_h.setContentsMargins(12, 12, 12, 12)
        body_h.setSpacing(10)
        root_v.addWidget(body, stretch=1)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(264)
        sidebar.setSizePolicy(QSizePolicy.Policy.Fixed,
                              QSizePolicy.Policy.Expanding)
        self._build_sidebar(sidebar)
        body_h.addWidget(sidebar)

        plot_w = QWidget()
        self._build_plots(plot_w)
        body_h.addWidget(plot_w, stretch=1)

    def _build_sidebar(self, parent):
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.setSpacing(4)

        def section(text):
            lay.addSpacing(10)
            lay.addWidget(_lbl(text, "section"))
            lay.addWidget(_divider())

        def field(label, default):
            lay.addWidget(_lbl(label))
            e = _entry(default)
            lay.addWidget(e)
            return e

        section("SHOT")
        self._pid_edit = field("PID", "20250401.49")

        section("MODEL")
        lay.addWidget(_lbl("Checkpoint path"))
        path_row = QWidget()
        path_h   = QHBoxLayout(path_row)
        path_h.setContentsMargins(0, 0, 0, 0)
        path_h.setSpacing(4)
        self._model_edit = _entry(
            "/home/IPP-HGW/orluca/devel/classificator/"
            "W7-X_QXT/AE360/version_0/best_model_.ckpt"
        )
        browse_btn = QPushButton("…")
        browse_btn.setObjectName("browse")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse_model)
        path_h.addWidget(self._model_edit)
        path_h.addWidget(browse_btn)
        lay.addWidget(path_row)

        section("NORMALISATION")
        self._min_edit = field("min_data", "-0.2294265627861023")
        self._max_edit = field("max_data",  "8.772955894470215")

        section("GAIN RATIO")
        gain_row = QWidget()
        gain_h   = QHBoxLayout(gain_row)
        gain_h.setContentsMargins(0, 0, 0, 0)
        gain_h.setSpacing(6)
        for label, val, attr in [("Old gain", "5", "_old_gain"),
                                  ("New gain", "2", "_new_gain")]:
            gain_h.addWidget(_lbl(label))
            e = _entry(val)
            e.setFixedWidth(46)
            setattr(self, attr, e)
            gain_h.addWidget(e)
        lay.addWidget(gain_row)

        section("OUTLIER DETECTION")

        # ── single greyed-out method indicator ────────────────────────────────
        self._method_btn = QPushButton("Model Residuals")
        self._method_btn.setObjectName("method_btn")
        self._method_btn.setCheckable(True)
        self._method_btn.setChecked(True)
        self._method_btn.setEnabled(False)   # purely decorative
        lay.addWidget(self._method_btn)

        # ── z-threshold ───────────────────────────────────────────────────────
        lay.addSpacing(4)
        lay.addWidget(_lbl("Z-threshold"))
        self._z_edit = _entry("0.2")
        lay.addWidget(self._z_edit)

        # ── action buttons ────────────────────────────────────────────────────
        lay.addStretch()

        self._run_btn = QPushButton("▶  RUN ANALYSIS")
        self._run_btn.setObjectName("run")
        self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn)

        self._save_btn = QPushButton("💾  SAVE ADJUSTED DATA")
        self._save_btn.setObjectName("run")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_adjusted_data)
        lay.addWidget(self._save_btn)

        self._save_rec_btn = QPushButton("💾  SAVE MODEL RECONSTRUCTION")
        self._save_rec_btn.setObjectName("run")
        self._save_rec_btn.setEnabled(False)
        self._save_rec_btn.clicked.connect(self._save_reconstruction_data)
        lay.addWidget(self._save_rec_btn)

        self._spectral_btn = QPushButton("📊  SPECTRAL ANALYSIS")
        self._spectral_btn.setObjectName("spectral")
        self._spectral_btn.setEnabled(False)
        self._spectral_btn.setToolTip(
            "Per-diode PSD heatmap, total power, and spectral entropy")
        self._spectral_btn.clicked.connect(self._open_spectral_analysis)
        lay.addWidget(self._spectral_btn)

        # ── correlation readout ───────────────────────────────────────────────
        lay.addSpacing(6)
        self._corr_widget = QWidget()
        self._corr_layout = QGridLayout(self._corr_widget)
        self._corr_layout.setContentsMargins(0, 0, 0, 0)
        self._corr_layout.setSpacing(3)
        lay.addWidget(self._corr_widget)

    def _build_plots(self, parent):
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._fig = Figure(figsize=(10, 7), dpi=100, facecolor=BG)
        gs = gridspec.GridSpec(2, 2, figure=self._fig,
                               hspace=0.42, wspace=0.20,
                               left=0.07, right=0.93,
                               top=0.93, bottom=0.08)
        self._ax_orig = self._fig.add_subplot(gs[0, 0])
        self._ax_adj  = self._fig.add_subplot(gs[0, 1])
        self._ax_rec  = self._fig.add_subplot(gs[1, 0])
        self._ax_corr = self._fig.add_subplot(gs[1, 1])
        self.twin_ax_corr = self._ax_corr.twinx()

        for ax, title in [
            (self._ax_orig, "Original profile  vs  Reconstruction"),
            (self._ax_adj,  "Adjusted profile  (outliers in red)"),
            (self._ax_rec,  "Adjusted profile  vs  AE reconstruction"),
            (self._ax_corr, "Correlation over time"),
        ]:
            ax.grid(True)
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=FG, pad=8)

        canvas = FigureCanvasQTAgg(self._fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        lay.addWidget(canvas, stretch=1)

        tb_container = QWidget()
        tb_container.setStyleSheet(f"background:{PANEL};")
        tb_h = QHBoxLayout(tb_container)
        tb_h.setContentsMargins(4, 0, 4, 0)
        toolbar = NavigationToolbar2QT(canvas, tb_container)
        toolbar.setStyleSheet(f"background:{PANEL}; color:{FG};")
        tb_h.addWidget(toolbar)
        lay.addWidget(tb_container)

        self._canvas = canvas

    # ── handlers ──────────────────────────────────────────────────────────────

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model checkpoint", "",
            "Checkpoint (*.ckpt);;All files (*.*)"
        )
        if path:
            self._model_edit.setText(path)

    def _run(self):
        try:
            pid        = self._pid_edit.text().strip()
            model_path = self._model_edit.text().strip()
            min_data   = float(self._min_edit.text())
            max_data   = float(self._max_edit.text())
            gain_ratio = float(self._old_gain.text()) / float(self._new_gain.text())
        except ValueError as e:
            QMessageBox.critical(self, "Parameter error", str(e))
            return

        try:
            z_thresh = float(self._z_edit.text())
        except ValueError:
            z_thresh = 0.2

        import gc
        self._results = None
        self._save_btn.setEnabled(False)
        self._save_rec_btn.setEnabled(False)
        self._spectral_btn.setEnabled(False)
        gc.collect()

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  Running …")
        self._status_lbl.setText("Starting …")

        self._thread = QThread()
        self._worker = AnalysisWorker(
            pid, model_path, min_data, max_data, gain_ratio,
            z_thresh = z_thresh,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._status_lbl.setText)
        self._worker.finished.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  RUN ANALYSIS")
        self._status_lbl.setText("Error — see dialog")
        QMessageBox.critical(self, "Analysis failed", msg)

    def _on_results(self, r):
        self._results = r
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  RUN ANALYSIS")
        self._save_btn.setEnabled(True)
        self._save_rec_btn.setEnabled(True)
        self._spectral_btn.setEnabled(True)

        self._status_lbl.setText(
            f"PID {r['pid']}  |  t = {r['time_instant']:.2f} ms"
            f"  |  ρ_orig = {r['corr_original']:.4f}"
            f"  |  ρ_adj = {r['corr_adjusted']:.4f}"
        )
        self._plot_results(r)
        self._update_corr_readout(r)

    # ── spectral analysis ─────────────────────────────────────────────────────

    def _open_spectral_analysis(self):
        if self._results is None:
            return
        dlg = SpectralAnalysisDialog(self._results, parent=self)
        dlg.show()

    # ── plotting ──────────────────────────────────────────────────────────────

    def _plot_results(self, r):
        for ax in (self._ax_orig, self._ax_adj,
                   self._ax_rec,  self._ax_corr):
            ax.cla()
            ax.grid(True)
            ax.set_facecolor(PANEL)
        self.twin_ax_corr.cla()
        self.twin_ax_corr.yaxis.set_label_position("right")
        self.twin_ax_corr.yaxis.tick_right()

        x, ti, pid = r["x"], r["time_instant"], r["pid"]

        # ── panel 1: original vs AE ───────────────────────────────────────────
        ax = self._ax_orig
        ax.plot(x, r["profile"],      color=ACCENT,  lw=1.8, label="Original")
        ax.plot(x, r["rec_original"], color=ACCENT2, lw=1.8, ls="--",
                label=f"AE recon  (ρ = {r['corr_original']:.4f})")
        ax.set_title("Original  vs  AE Reconstruction", color=FG, pad=8)
        ax.set_xlabel("Diode index")
        ax.set_ylabel("Signal [V]")
        ax.legend(fontsize=8)

        # ── panel 2: adjusted profile with outliers ───────────────────────────
        ax  = self._ax_adj
        out = r["outlier_mask"]
        ax.plot(x, r["profile"],      color=FG_DIM,  lw=1.2, ls="--",
                alpha=0.6, label="Original")
        ax.plot(x, r["adjusted"],     color=ACCENT3, lw=1.8, label="Adjusted")
        ax.plot(x, r["rec_original"], color=ACCENT2, lw=1.2, ls=":",
                alpha=0.7, label="AE reconstruction")
        if out.any():
            ax.scatter(x[out], r["profile"][out],
                       color=ACCENT4, marker="x", s=80, lw=1.8,
                       label=f"Outliers ({out.sum()})", zorder=5)
        ax.set_title(
            f"Adjusted Profile  [Model Residuals  z={r.get('z_thresh', '?')}]",
            color=FG, pad=8)
        ax.set_xlabel("Diode index")
        ax.set_ylabel("Signal [V]")
        ax.legend(fontsize=8)

        # ── panel 3: adjusted + AE reconstruction ────────────────────────────
        ax = self._ax_rec
        ax.plot(x, r["profile"],      color=FG_DIM,  lw=1.2, ls="--",
                alpha=0.6, label="Original")
        ax.plot(x, r["adjusted"],     color=ACCENT3, lw=1.8, label="Adjusted")
        ax.plot(x, r["rec_adjusted"], color=ACCENT2, lw=1.8, ls="--",
                label=f"AE recon (adj)  (ρ = {r['corr_adjusted']:.4f})")
        ax.set_title("Adjusted  vs  AE Reconstruction", color=FG, pad=8)
        ax.set_xlabel("Diode index")
        ax.set_ylabel("Signal [V]")
        ax.legend(fontsize=8)

        # ── panel 4: correlation time-series ──────────────────────────────────
        ax  = self._ax_corr
        t   = r["timedata"]
        cor = r["correlations"]
        c   = np.where(cor < 0.9, ACCENT4, ACCENT)
        ax.scatter(t, cor, c=c, s=6, zorder=3)
        ax.plot(t, cor, color=ACCENT, lw=1.0, alpha=0.4, zorder=2)
        ax.axhline(0.9, color=WARNING, lw=1.2, ls="--", label="Threshold 0.9")
        ax.axvline(ti,  color=ACCENT2, lw=1.2, ls=":",
                   label=f"t = {ti:.1f} ms")
        self.twin_ax_corr.plot(t, r["diode_signal"], color=ACCENT2,
                               lw=1.0, alpha=0.45, label="Diode #151")
        self.twin_ax_corr.set_ylabel("Diode #151 [V]", color=ACCENT2, fontsize=8)
        self.twin_ax_corr.tick_params(axis="y", labelcolor=ACCENT2, labelsize=7)
        self.twin_ax_corr.set_facecolor(PANEL)
        ax.set_title("Correlation  over  Time", color=FG, pad=8)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pearson ρ", color=ACCENT)
        ax.tick_params(axis="y", labelcolor=ACCENT)
        ax.set_ylim(-0.1, 1.05)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = self.twin_ax_corr.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="lower left")

        self._fig.suptitle(
            f"PID: {pid}   |   t_max = {ti:.2f} ms",
            color=FG, fontsize=11, y=0.99,
        )
        self._canvas.draw_idle()

    def _update_corr_readout(self, r):
        # clear previous widgets
        while self._corr_layout.count():
            item = self._corr_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        row = 0

        # ── correlation summary ───────────────────────────────────────────────
        for label, value in [("ρ original :", r["corr_original"]),
                              ("ρ adjusted :", r["corr_adjusted"])]:
            name = "corr_ok" if value >= 0.9 else "corr_bad"
            lbl  = _lbl(label)
            val  = QLabel(f"{value:.4f}")
            val.setObjectName(name)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._corr_layout.addWidget(lbl, row, 0)
            self._corr_layout.addWidget(val, row, 1)
            row += 1

        verdict = "▸  NORMAL" if r["corr_original"] >= 0.9 else "▸  ANOMALY"
        obj     = "verdict_ok" if r["corr_original"] >= 0.9 else "verdict_bad"
        v_lbl   = QLabel(verdict)
        v_lbl.setObjectName(obj)
        v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._corr_layout.addWidget(v_lbl, row, 0, 1, 2)
        row += 1

        # ── residuals detection summary ───────────────────────────────────────
        row += 1
        self._corr_layout.addWidget(_divider(), row, 0, 1, 2)
        row += 1
        hdr = QLabel("── Residuals detection ──")
        hdr.setObjectName("diode_corr_header")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._corr_layout.addWidget(hdr, row, 0, 1, 2)
        row += 1
        n_out = int(r["outlier_mask"].sum())
        for lbl_txt, val_txt in [
            ("z-threshold", f"{r.get('z_thresh', '?')}"),
            ("n flagged",   f"{n_out}"),
        ]:
            lbl_w = QLabel(lbl_txt + " :")
            lbl_w.setObjectName("diode_corr_row")
            val_w = QLabel(val_txt)
            val_w.setObjectName("diode_corr_row")
            val_w.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._corr_layout.addWidget(lbl_w, row, 0)
            self._corr_layout.addWidget(val_w, row, 1)
            row += 1

    # ── save ──────────────────────────────────────────────────────────────────

    def _save_adjusted_data(self):
        if self._results is None:
            QMessageBox.warning(self, "No data", "Run the analysis first.")
            return

        import h5py
        from read_h5f import save_tfiles, format_pid

        r             = self._results
        h5py_path     = r["file_path"]
        pid           = r["pid"]
        adjusted_full = r["adjusted_full"]   # gain-corrected, no model pass
        timedata      = r["timedata_full"]
        diode_keys    = r["diode_keys"]

        save_dir = os.path.join(os.path.dirname(h5py_path))
        os.makedirs(save_dir, exist_ok=True)

        # Save also the  t000, t090, t180 and t270 files with adjusted keyword
        save_tfiles(h5py_path)

        try:
            for i, key in enumerate(diode_keys):
                dst_path = os.path.join(
                    save_dir,
                    f"xmcts{format_pid(pid)}adjusted_x{key}.h5f"
                )
                with h5py.File(dst_path, "w") as f:
                    grp = f.create_group("XMCTSdata")
                    grp.create_dataset("timedata", data=timedata)
                    grp.create_dataset(key,         data=adjusted_full[i, :])
            self._status_lbl.setText(
                f"Saved {len(diode_keys)} files → {save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _save_reconstruction_data(self):
        if self._results is None:
            QMessageBox.warning(self, "No data", "Run the analysis first.")
            return

        import h5py
        from read_h5f import format_pid

        r                 = self._results
        h5py_path         = r["file_path"]
        pid               = r["pid"]
        rec_adjusted_full = r["rec_adjusted_full"]   # model reconstruction of gain-corrected data
        timedata          = r["timedata_full"]
        diode_keys        = r["diode_keys"]

        save_dir = os.path.join(os.path.dirname(h5py_path))
        os.makedirs(save_dir, exist_ok=True)

        try:
            for i, key in enumerate(diode_keys):
                dst_path = os.path.join(
                    save_dir,
                    f"xmcts{format_pid(pid)}reconstructed_x{key}.h5f"
                )
                with h5py.File(dst_path, "w") as f:
                    grp = f.create_group("XMCTSdata")
                    grp.create_dataset("timedata", data=timedata)
                    grp.create_dataset(key,         data=rec_adjusted_full[i, :])
            self._status_lbl.setText(
                f"Saved {len(diode_keys)} reconstruction files → {save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = WorkflowGUI()
    window.show()
    sys.exit(app.exec())