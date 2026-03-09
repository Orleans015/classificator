"""
Anomaly Detection for Bell-Shaped Signal Concatenations
========================================================
Detects out-of-norm points in a signal composed of concatenated Gaussian-like structures.

Strategy:
  1. Fit a local envelope (rolling statistics) to capture the "expected" signal shape.
  2. Compute residuals from the local mean.
  3. Flag points where the residual exceeds a threshold (z-score or IQR-based).
  4. Optionally: use a rolling z-score for adaptive thresholding.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d


# ── 1. Generate a synthetic signal ──────────────────────────────────────────

def make_bell_signal(n_bells=12, n_points=2000, noise_std=0.05, seed=42):
    """Concatenate bell-shaped (Gaussian) structures with added noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, n_bells, n_points)
    signal = np.zeros(n_points)

    for i in range(n_bells):
        center = i + 0.5
        amplitude = rng.uniform(0.6, 1.4)
        width = rng.uniform(0.15, 0.35)
        signal += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

    signal += rng.normal(0, noise_std, n_points)  # background noise
    return x, signal


def inject_anomalies(signal, n_high=8, n_low=8, seed=0):
    """Inject artificial anomalies (spikes up and dips down)."""
    rng = np.random.default_rng(seed)
    noisy = signal.copy()
    anomaly_idx = []

    indices = rng.choice(len(signal), n_high + n_low, replace=False)
    high_idx = indices[:n_high]
    low_idx = indices[n_high:]

    noisy[high_idx] += rng.uniform(0.8, 1.5, n_high)   # upward spikes
    noisy[low_idx]  -= rng.uniform(0.8, 1.5, n_low)    # downward dips
    anomaly_idx = list(high_idx) + list(low_idx)
    return noisy, sorted(anomaly_idx)


# ── 2. Anomaly Detection ─────────────────────────────────────────────────────

def rolling_zscore(signal, window=50):
    """
    Compute a rolling z-score for each point.
    Points with |z| > threshold are flagged as anomalies.
    """
    n = len(signal)
    roll_mean = uniform_filter1d(signal, size=window, mode='nearest')
    # Rolling std via sqrt(E[x²] - E[x]²)
    roll_mean_sq = uniform_filter1d(signal ** 2, size=window, mode='nearest')
    roll_std = np.sqrt(np.maximum(roll_mean_sq - roll_mean ** 2, 1e-10))
    z = (signal - roll_mean) / roll_std
    return z, roll_mean, roll_std


def detect_anomalies(signal, window=50, z_threshold=3.0):
    """
    Detect anomalies using a rolling z-score approach.

    Parameters
    ----------
    signal      : 1-D array of signal values
    window      : size of the rolling window (should cover ~1 bell width in samples)
    z_threshold : number of standard deviations to flag as anomaly

    Returns
    -------
    anomaly_mask : boolean array, True where anomalies are detected
    z_scores     : rolling z-score array
    roll_mean    : rolling mean (the "expected" envelope)
    roll_std     : rolling standard deviation
    """
    z, roll_mean, roll_std = rolling_zscore(signal, window=window)
    anomaly_mask = np.abs(z) > z_threshold
    return anomaly_mask, z, roll_mean, roll_std


# ── 3. Visualisation ─────────────────────────────────────────────────────────

def plot_results(x, clean, noisy, anomaly_mask, roll_mean, roll_std,
                 true_anomaly_idx=None, z_threshold=3.0):

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('#0f0f14')
    for ax in axes:
        ax.set_facecolor('#16161e')
        ax.tick_params(colors='#888', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    CYAN   = '#00d4ff'
    ORANGE = '#ff8c42'
    GREEN  = '#39ff6e'
    RED    = '#ff3366'
    GREY   = '#555566'

    # — Panel 1: clean signal vs noisy signal —
    ax = axes[0]
    ax.plot(x, clean, color=GREY,   lw=1.0, label='Clean signal',  alpha=0.6)
    ax.plot(x, noisy, color=CYAN,   lw=0.8, label='Observed signal', alpha=0.9)
    ax.set_title('Signal (clean vs observed)', color='#ccccdd', fontsize=11, pad=8)
    ax.legend(facecolor='#1e1e2a', edgecolor='#444', labelcolor='#ccc', fontsize=9)
    ax.set_ylabel('Amplitude', color='#888', fontsize=9)

    # — Panel 2: detected anomalies with confidence band —
    ax = axes[1]
    ax.fill_between(x,
                    roll_mean - z_threshold * roll_std,
                    roll_mean + z_threshold * roll_std,
                    color=CYAN, alpha=0.12, label=f'±{z_threshold}σ band')
    ax.plot(x, noisy,      color=CYAN,   lw=0.8, alpha=0.7)
    ax.plot(x, roll_mean,  color=GREEN,  lw=1.2, label='Rolling mean', linestyle='--')

    detected   = np.where(anomaly_mask)[0]
    high_mask  = noisy[detected] > roll_mean[detected]
    low_mask   = ~high_mask

    ax.scatter(x[detected[high_mask]], noisy[detected[high_mask]],
               color=RED, s=40, zorder=5, label='Anomaly ↑ (high)', marker='^')
    ax.scatter(x[detected[low_mask]],  noisy[detected[low_mask]],
               color=ORANGE, s=40, zorder=5, label='Anomaly ↓ (low)',  marker='v')

    # True anomaly circles (if available)
    if true_anomaly_idx is not None:
        ax.scatter(x[true_anomaly_idx], noisy[true_anomaly_idx],
                   facecolors='none', edgecolors='white', s=120, lw=1.2,
                   zorder=6, label='True anomaly (ground truth)')

    ax.set_title('Anomaly detection  (rolling z-score)', color='#ccccdd', fontsize=11, pad=8)
    ax.legend(facecolor='#1e1e2a', edgecolor='#444', labelcolor='#ccc', fontsize=9, ncol=3)
    ax.set_ylabel('Amplitude', color='#888', fontsize=9)

    # — Panel 3: rolling z-score —
    z, _, _ = rolling_zscore(noisy, window=50)
    ax = axes[2]
    ax.plot(x, z, color=CYAN, lw=0.8, alpha=0.9)
    ax.axhline( z_threshold, color=RED,    lw=1.2, linestyle='--', label=f'+{z_threshold}σ')
    ax.axhline(-z_threshold, color=ORANGE, lw=1.2, linestyle='--', label=f'-{z_threshold}σ')
    ax.fill_between(x, z, z_threshold,
                    where=(z >  z_threshold), color=RED,    alpha=0.3)
    ax.fill_between(x, z, -z_threshold,
                    where=(z < -z_threshold), color=ORANGE, alpha=0.3)
    ax.set_title('Rolling z-score', color='#ccccdd', fontsize=11, pad=8)
    ax.legend(facecolor='#1e1e2a', edgecolor='#444', labelcolor='#ccc', fontsize=9)
    ax.set_ylabel('z-score', color='#888', fontsize=9)
    ax.set_xlabel('x', color='#888', fontsize=9)

    fig.suptitle('Anomaly Detection — Bell-Shaped Signal',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/anomaly_detection.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print("Plot saved to anomaly_detection.png")


# ── 4. Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build synthetic signal
    x, clean = make_bell_signal(n_bells=12, n_points=2000)
    noisy, true_idx = inject_anomalies(clean, n_high=10, n_low=10)

    # Detect anomalies
    # window  ≈ samples per bell / 3  (tune this to your data's bell width)
    # z_threshold = 3.0  (classic 3-sigma rule; lower → more sensitive)
    anomaly_mask, z_scores, roll_mean, roll_std = detect_anomalies(
        noisy, window=50, z_threshold=3.0
    )

    detected_idx = np.where(anomaly_mask)[0]
    print(f"True anomalies   : {sorted(true_idx)}")
    print(f"Detected indices : {list(detected_idx)}")
    print(f"Precision        : {len(set(detected_idx) & set(true_idx)) / max(len(detected_idx),1):.2f}")
    print(f"Recall           : {len(set(detected_idx) & set(true_idx)) / len(true_idx):.2f}")

    plot_results(x, clean, noisy, anomaly_mask, roll_mean, roll_std,
                 true_anomaly_idx=true_idx, z_threshold=3.0)
