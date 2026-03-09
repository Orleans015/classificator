"""
Workflow GUI — SXR Profile Anomaly Detector  (PyQt6 edition)
=============================================================
Visualises the original profile, adjusted profile,
reconstructed profile and the time-series correlation
for a given shot PID.

Requirements:
    pip install PyQt6 matplotlib numpy scipy torch
"""

import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

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
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Background worker
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, pid, model_path, min_data, max_data, gain_ratio):
        super().__init__()
        self.pid        = pid
        self.model_path = model_path
        self.min_data   = min_data
        self.max_data   = max_data
        self.gain_ratio = gain_ratio

    def run(self):
        try:
            results = run_analysis(
                self.pid, self.model_path,
                self.min_data, self.max_data, self.gain_ratio,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  Core computation 
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(pid, model_path, min_data, max_data, gain_ratio,
                 progress_cb=None):
    import torch
    from scipy.interpolate import make_smoothing_spline

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    try:
        from model    import AutoEncoder
        from read_h5f import load_sxr_data
    except ImportError as e:
        raise ImportError(
            f"Could not import project modules: {e}\n"
            "Make sure 'model.py' and 'read_h5f.py' are on sys.path."
        ) from e

    _p("Loading model …")
    model = AutoEncoder.load_from_checkpoint(model_path)
    model.eval()

    _p(f"Loading SXR data for PID {pid} …")
    data         = load_sxr_data(pid)
    time_base    = data[0, :]
    signals      = data[1:, :]
    max_emission = int(np.argmax(data[152, :]))
    time_instant = float(time_base[max_emission])
    profile      = data[1:, max_emission].copy()
    n_diodes     = len(profile)
    x            = np.arange(n_diodes)

    # ── original reconstruction ───────────────────────────────────────────────
    _p("Running model on original profile …")
    with torch.no_grad():
        pn  = (profile - min_data) / (max_data - min_data)
        pt  = torch.tensor(pn, dtype=torch.float32).unsqueeze(0)
        rec = model(pt).squeeze(0).numpy()
    rec_original  = rec * (max_data - min_data) + min_data
    corr_original = float(np.corrcoef(profile, rec_original)[0, 1])

    # ── iterative spline fit ──────────────────────────────────────────────────
    _p("Running iterative spline fit …")
    mask        = np.ones(n_diodes, dtype=bool)
    best_spline = None
    best_corr   = -1.0

    for _ in range(20):
        spline    = make_smoothing_spline(x[mask], profile[mask])
        svals     = spline(x)
        residuals = profile - svals
        pos       = residuals > 0
        th        = (np.mean(residuals[pos]) - 0.3 * np.std(residuals[pos])
                     if np.any(pos) else 0.0)
        new_mask  = ~((residuals > th) & (profile > svals))
        if np.array_equal(mask, new_mask):
            best_spline = spline
            break
        mask = new_mask
        sn   = (svals - min_data) / (max_data - min_data)
        with torch.no_grad():
            rt = model(torch.tensor(sn, dtype=torch.float32)
                       .unsqueeze(0)).squeeze(0).numpy()
        dn = rt * (max_data - min_data) + min_data
        if mask.sum() >= 2:
            c = float(np.corrcoef(profile[mask], dn[mask])[0, 1])
            if c > best_corr:
                best_corr   = c
                best_spline = spline

    if best_spline is None:
        best_spline = spline

    spline_final = best_spline(x)
    outlier_mask = ~mask

    # ── gain-corrected adjusted profile ──────────────────────────────────────
    adjusted        = profile.copy()
    adjusted[mask] *= gain_ratio

    # ── reconstruction of adjusted profile ───────────────────────────────────
    _p("Running model on adjusted profile …")
    with torch.no_grad():
        an = (adjusted - min_data) / (max_data - min_data)
        at = torch.tensor(an, dtype=torch.float32).unsqueeze(0)
        ar = model(at).squeeze(0).numpy()
    rec_adjusted  = ar * (max_data - min_data) + min_data
    corr_adjusted = float(np.corrcoef(profile, rec_adjusted)[0, 1])

    # ── correlation time-series ───────────────────────────────────────────────
    _p("Computing correlation time-series …")
    dn = (signals - min_data) / (max_data - min_data)
    with torch.no_grad():
        rec_all = model(torch.tensor(dn.T, dtype=torch.float32)).numpy().T
    dn_all       = rec_all * (max_data - min_data) + min_data
    correlations = np.array([
        float(np.corrcoef(signals[:, i], dn_all[:, i])[0, 1])
        for i in range(data.shape[1])
    ])

    _p("Done.")
    return dict(
        pid           = pid,
        time_instant  = time_instant,
        x             = x,
        time_base     = time_base,
        profile       = profile,
        rec_original  = rec_original,
        spline_final  = spline_final,
        outlier_mask  = outlier_mask,
        adjusted      = adjusted,
        rec_adjusted  = rec_adjusted,
        corr_original = corr_original,
        corr_adjusted = corr_adjusted,
        correlations  = correlations,
        diode_signal  = data[152, :],
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
    e = QLineEdit(default)
    return e


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

        self._thread = None
        self._worker = None
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_v = QVBoxLayout(root)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        # top bar
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

        # body
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

        # Shot
        section("SHOT")
        self._pid_edit = field("PID", "20250401.49")

        # Model
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

        # Normalisation
        section("NORMALISATION")
        self._min_edit = field("min_data", "-0.2294265627861023")
        self._max_edit = field("max_data",  "8.772955894470215")

        # Gain
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

        # spacer + run button
        lay.addStretch()
        self._run_btn = QPushButton("▶  RUN ANALYSIS")
        self._run_btn.setObjectName("run")
        self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn)

        # correlation readout
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
                               hspace=0.42, wspace=0.34,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.08)
        self._ax_orig = self._fig.add_subplot(gs[0, 0])
        self._ax_adj  = self._fig.add_subplot(gs[0, 1])
        self._ax_rec  = self._fig.add_subplot(gs[1, 0])
        self._ax_corr = self._fig.add_subplot(gs[1, 1])

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

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  Running …")
        self._status_lbl.setText("Starting …")

        self._thread = QThread()
        self._worker = AnalysisWorker(pid, model_path, min_data,
                                      max_data, gain_ratio)
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
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  RUN ANALYSIS")
        self._status_lbl.setText(
            f"PID {r['pid']}  |  t = {r['time_instant']:.2f} ms  "
            f"|  ρ_orig = {r['corr_original']:.4f}  "
            f"|  ρ_adj = {r['corr_adjusted']:.4f}"
        )
        self._plot_results(r)
        self._update_corr_readout(r)

    # ── plotting ──────────────────────────────────────────────────────────────

    def _plot_results(self, r):
        for ax in (self._ax_orig, self._ax_adj,
                   self._ax_rec,  self._ax_corr):
            ax.cla()
            ax.grid(True)
            ax.set_facecolor(PANEL)

        x, ti, pid = r["x"], r["time_instant"], r["pid"]

        # panel 1 — original vs AE reconstruction
        ax = self._ax_orig
        ax.plot(x, r["profile"],      color=ACCENT,  lw=1.8,
                label="Original")
        ax.plot(x, r["rec_original"], color=ACCENT2, lw=1.8, ls="--",
                label=f"AE recon  (ρ = {r['corr_original']:.4f})")
        ax.set_title("Original  vs  AE Reconstruction", color=FG, pad=8)
        ax.set_xlabel("Diode index")
        ax.set_ylabel("Signal [V]")
        ax.legend(fontsize=8)

        # panel 2 — adjusted profile with outliers
        ax  = self._ax_adj
        out = r["outlier_mask"]
        ax.plot(x, r["profile"],      color=FG_DIM,  lw=1.2, ls="--",
                alpha=0.6, label="Original")
        ax.plot(x, r["adjusted"],     color=ACCENT3, lw=1.8,
                label="Adjusted")
        ax.plot(x, r["spline_final"], color=ACCENT2, lw=1.2, ls=":",
                alpha=0.7, label="Spline fit")
        if out.any():
            ax.scatter(x[out], r["profile"][out],
                       color=ACCENT4, marker="x", s=80, lw=1.8,
                       label=f"Outliers ({out.sum()})", zorder=5)
        ax.set_title("Adjusted Profile  (gain correction)", color=FG, pad=8)
        ax.set_xlabel("Diode index")
        ax.set_ylabel("Signal [V]")
        ax.legend(fontsize=8)

        # panel 3 — adjusted + AE reconstruction
        ax = self._ax_rec
        ax.plot(x, r["profile"],      color=FG_DIM,  lw=1.2, ls="--",
                alpha=0.6, label="Original")
        ax.plot(x, r["adjusted"],     color=ACCENT3, lw=1.8,
                label="Adjusted")
        ax.plot(x, r["rec_adjusted"], color=ACCENT2, lw=1.8, ls="--",
                label=f"AE recon (adj)  (ρ = {r['corr_adjusted']:.4f})")
        ax.set_title("Adjusted  vs  AE Reconstruction", color=FG, pad=8)
        ax.set_xlabel("Diode index")
        ax.set_ylabel("Signal [V]")
        ax.legend(fontsize=8)

        # panel 4 — correlation over time
        ax  = self._ax_corr
        t   = r["time_base"]
        cor = r["correlations"]
        c   = np.where(cor < 0.9, ACCENT4, ACCENT)
        ax.scatter(t, cor, c=c, s=6, zorder=3)
        ax.plot(t, cor, color=ACCENT, lw=1.0, alpha=0.4, zorder=2)
        ax.axhline(0.9, color=WARNING, lw=1.2, ls="--", label="Threshold 0.9")
        ax.axvline(ti,  color=ACCENT2, lw=1.2, ls=":",  label=f"t = {ti:.1f} ms")
        ax2 = ax.twinx()
        ax2.plot(t, r["diode_signal"], color=ACCENT2,
                 lw=1.0, alpha=0.45, label="Diode #151")
        ax2.set_ylabel("Diode #151 [V]", color=ACCENT2, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=ACCENT2, labelsize=7)
        ax2.set_facecolor(PANEL)
        ax.set_title("Correlation  over  Time", color=FG, pad=8)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pearson ρ", color=ACCENT)
        ax.tick_params(axis="y", labelcolor=ACCENT)
        ax.set_ylim(-0.1, 1.05)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="lower left")

        self._fig.suptitle(
            f"PID: {pid}   |   t_max = {ti:.2f} ms",
            color=FG, fontsize=11, y=0.99,
        )
        self._canvas.draw_idle()

    def _update_corr_readout(self, r):
        while self._corr_layout.count():
            item = self._corr_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for row_idx, (label, value) in enumerate([
            ("ρ original :", r["corr_original"]),
            ("ρ adjusted :", r["corr_adjusted"]),
        ]):
            name = "corr_ok" if value >= 0.9 else "corr_bad"
            lbl = _lbl(label)
            val = QLabel(f"{value:.4f}")
            val.setObjectName(name)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._corr_layout.addWidget(lbl, row_idx, 0)
            self._corr_layout.addWidget(val, row_idx, 1)

        verdict = "▸  NORMAL" if r["corr_adjusted"] >= 0.9 else "▸  ANOMALY"
        obj     = "verdict_ok" if r["corr_adjusted"] >= 0.9 else "verdict_bad"
        v_lbl   = QLabel(verdict)
        v_lbl.setObjectName(obj)
        v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._corr_layout.addWidget(v_lbl, 2, 0, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = WorkflowGUI()
    window.show()
    sys.exit(app.exec())