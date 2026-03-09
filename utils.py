import torch
import torch.nn.functional as F

def second_derivative_conv(x, dx=1.0):
	# x : (B, 1, L)  -- batch, channel=1, length
	# discrete kernel [1, -2, 1] / dx^2
	kernel = torch.tensor([1.0, -2.0, 1.0], device=x.device, dtype=x.dtype).view(1,1,3) / (dx*dx)
	# padding='replicate' approx by F.pad then conv
	x_pad = F.pad(x, (1,1), mode='replicate')
	d2 = F.conv1d(x_pad, kernel)	# shape (B, 1, L)
	return d2

def local_energy(d2, window_size):
	# d2 : (B,1,L)
	# window_size should be odd for symmetric window
	w = torch.ones(1,1,window_size, device=d2.device, dtype=d2.dtype)
	# pad to keep same length
	pad = (window_size//2, window_size//2)
	d2_sq = d2 * d2
	d2_pad = F.pad(d2_sq, pad, mode='replicate')
	E_local = F.conv1d(d2_pad, w)   # (B,1,L) sum of squared second deriv in window
	return E_local

def huber_like(x, delta=1e-6):
	# Charbonnier-like: sqrt(x^2 + eps^2)
	eps = delta
	return torch.sqrt(x*x + eps*eps)

def smoothness_loss(recon, input_signal=None, dx=1.0, window_size=9,
					use_weights=True, gamma=10.0, huber_eps=1e-6):
	# recon, input_signal: tensors (B, L) or (B,1,L); returns scalar loss
	if recon.dim()==2:
		recon = recon.unsqueeze(1)
	if input_signal is not None and input_signal.dim()==2:
		input_signal = input_signal.unsqueeze(1)

	d2 = second_derivative_conv(recon, dx=dx)	   # (B,1,L)
	# robustify
	robust = huber_like(d2, delta=huber_eps)		# (B,1,L)

	E_local = local_energy(robust, window_size=window_size)  # (B,1,L)

	if use_weights and input_signal is not None:
		# compute gradient magnitude on input for edge-preserving weight
		# simple central diff (no dx factor; scale by dx if needed)
		grad_kernel = torch.tensor([-0.5, 0.0, 0.5], device=recon.device, dtype=recon.dtype).view(1,1,3)
		inp_pad = F.pad(input_signal, (1,1), mode='replicate')
		grad = F.conv1d(inp_pad, grad_kernel)	   # (B,1,L)
		weight = torch.exp(-gamma * torch.abs(grad))  # small weight where grad large
	else:
		weight = torch.ones_like(E_local)

	# average (weighted)
	numerator = (weight * E_local).sum()
	denom = weight.sum().clamp_min(1.0)
	loss = numerator / denom
	return loss.squeeze()

# Example usage in training step:
# recon: (B, L), input: (B, L)
# recon_loss = F.mse_loss(recon, input)
# smooth_loss = smoothness_loss(recon, input_signal=input, dx=1.0, window_size=9, use_weights=True)
# loss = recon_loss + lambda_smooth * smooth_loss


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def analyze_plasma_profiles(r, measurement, reconstruction, pid=None):
	"""
	Performs three spatial correlation diagnostics to distinguish 
	between noise and reconstruction accuracy.
	"""
	
	# 1. Magnitude-Squared Coherence (Frequency Domain)
	# Tells us at which spatial scales (frequencies) the signals match.
	f, Cxy = signal.coherence(measurement, reconstruction, fs=1.0/np.mean(np.diff(r)))

	# 2. Residual Analysis (Autocorrelation of the error)
	# If the residual is white noise, the reconstruction captured the physics.
	residual = measurement - reconstruction
	# find the peaks in the residual to see if there are systematic deviations
	pos_peaks, _ = signal.find_peaks(residual, height=np.std(residual))
	neg_peaks, _ = signal.find_peaks(-residual, height=np.std(residual))
	peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
	# Normalize the residual
	residual_norm = (residual - np.mean(residual)) / (np.std(residual) * len(residual))
	autocorr_res = np.correlate(residual_norm, residual_norm, mode='full')
	lags = np.arange(-len(residual) + 1, len(residual))

	# 3. Standard Pearson Correlation (Global Linear Match)
	pearson_r = np.corrcoef(measurement, reconstruction)[0, 1]

	# --- Plotting ---
	fig, axs = plt.subplots(2, 2, figsize=(12, 10))
	plt.subplots_adjust(hspace=0.3, wspace=0.3)

	# Plot A: The Profiles
	axs[0, 0].plot(r, measurement, label='Measurement (Noisy)', alpha=0.6)
	axs[0, 0].plot(r, reconstruction, label='Reconstruction', linewidth=2)
	axs[0, 0].set_xlabel("Diode number")
	if pid is not None:
		axs[0, 0].set_title(f"PID: {pid} - Plasma Brightness Profiles (R={pearson_r:.3f})")
	else:
		axs[0, 0].set_title(f"Plasma Brightness Profiles (R={pearson_r:.3f})")
	axs[0, 0].legend()

	# Plot B: Coherence
	# 
	axs[0, 1].semilogx(f, Cxy, color='green')
	axs[0, 1].axhline(0.5, linestyle='--', color='red', alpha=0.5, label='50% Coherence')
	axs[0, 1].set_title("Spatial Coherence Function")
	axs[0, 1].set_xlabel("Spatial Frequency (1/r)")
	axs[0, 1].set_ylabel("Coherence")
	axs[0, 1].grid(True, which='both', alpha=0.3)
	axs[0, 1].legend()

	# Plot C: Residuals
	axs[1, 0].plot(r, residual, color='purple', alpha=0.7)
	axs[1, 0].scatter(r[peaks], residual[peaks], color='red', label='Residual Peaks')
	axs[1, 0].set_title("Residuals (Meas - Rec)")
	axs[1, 0].set_xlabel("Diode number")

	# Plot D: Residual Autocorrelation
	# 
	axs[1, 1].plot(lags, autocorr_res)
	axs[1, 1].set_title("Autocorrelation of Residuals")
	axs[1, 1].set_xlabel("Lag")
	axs[1, 1].set_xlim(-20, 20) # Zoomed in to see the "spike"
	
	plt.show()

# # --- Example Usage with Synthetic Plasma Data ---
# r = np.linspace(0, 1, 200)
# # A typical physical profile (e.g., Gaussian core)
# clean_physics = np.exp(- (r**2) / 0.3) 
# # The Reconstruction (slightly smoothed or slightly off)
# reconstruction = np.exp(- (r**2) / 0.32) 
# # The Measurement (Physics + random high-frequency noise)
# measurement = clean_physics + np.random.normal(0, 0.05, len(r))

# analyze_plasma_profiles(r, measurement, reconstruction)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

def analyze_poloidal_profiles(theta, measurement, reconstruction):
	"""
	Analyzes poloidal brightness profiles (0 to 2pi).
	Uses Circular Correlation and Fourier Modes to distinguish noise from physics.
	"""
	# Ensure data is sorted by angle
	idx = np.argsort(theta)
	theta, meas, rec = theta[idx], measurement[idx], reconstruction[idx]
	
	N = len(theta)
	delta_theta = np.mean(np.diff(theta))

	# 1. Circular Cross-Correlation
	# Standard correlation assumes signals end; circular correlation 
	# acknowledges that 360 degrees = 0 degrees.
	meas_fft = fft.fft(meas)
	rec_fft = fft.fft(rec)
	# Correlation in freq domain: Multiplied by conjugate
	circ_corr = fft.ifft(meas_fft * np.conj(rec_fft)).real
	circ_corr = np.roll(circ_corr, N // 2) # Center the plot
	lags = np.linspace(-180, 180, N)

	# 2. Poloidal Mode Coherence (m-numbers)
	# Instead of "frequency", we talk about poloidal mode number 'm'
	f, Cxy = signal.coherence(meas, rec, fs=1.0/delta_theta)
	m_numbers = f * (2 * np.pi) # Convert freq to mode number m

	# 3. Residual Power Spectral Density (PSD)
	# Helps see if the noise is "white" or has a specific poloidal structure
	residual = meas - rec
	f_res, Pxx_res = signal.welch(residual, fs=1.0/delta_theta)

	# # 4. compute the power spectral density 
	# freqs, psd_res = signal.periodogram(residual, fs=360)

	# --- Plotting ---
	fig, axs = plt.subplots(2, 2, figsize=(14, 10))
	
	# Plot A: Polar Visualization
	ax_polar = fig.add_subplot(2, 2, 1, projection='polar')
	axs[0, 0].remove() # Replace standard axis with polar
	ax_polar.plot(theta, meas, label='Measured', alpha=0.6)
	ax_polar.plot(theta, rec, label='Reconstructed', linewidth=2)
	ax_polar.set_title("Poloidal Brightness Distribution")
	ax_polar.legend(loc='lower right')

	# Plot B: Circular Correlation
	axs[0, 1].plot(lags, circ_corr / np.max(circ_corr))
	axs[0, 1].set_title("Circular Cross-Correlation")
	axs[0, 1].set_xlabel("Angular Lag (Degrees)")
	axs[0, 1].grid(True)

	# Plot C: Mode Coherence
	axs[1, 0].stem(m_numbers, Cxy)
	axs[1, 0].set_xlim(0, 10) # Usually low-m modes are the physics
	axs[1, 0].set_title("Coherence by Poloidal Mode (m)")
	axs[1, 0].set_xlabel("Mode Number (m)")
	axs[1, 0].set_ylabel("Coherence Score")

	# Plot D: Residual Power (Noise vs Physics)
	axs[1, 1].semilogy(f_res * 2 * np.pi, Pxx_res)
	axs[1, 1].set_title("Residual Energy Spectrum")
	axs[1, 1].set_xlabel("Mode Number (m)")
	axs[1, 1].set_ylabel("Power")

	# # Plot E: Residual PSD (to see if noise is white or structured)
	# axs[2, 0].semilogy(freqs, psd_res, color='crimson', lw=2, label='Residual PSD')
	# axs[2, 0].axvspan(0, 10, color='green', alpha=0.1, label='Physics Region (Low-m)')
	# axs[2, 0].axvspan(10, 180, color='gray', alpha=0.1, label='Noise Region (High-m)')
	# axs[2, 0].set_title("Residual Power Spectral Density")
	# axs[2, 0].legend()
	# axs[2, 0].set_xlabel("Frequency (1/degree)")
	# axs[2, 0].set_ylabel("PSD")
	# axs[2, 0].set_xlim(0, 180)
	# axs[2, 0].grid(True)

	# plt.tight_layout()
	# plt.show()

# # --- Example: Poloidal asymmetry (Shafranov-like shift) ---
# theta = np.linspace(0, 2*np.pi, 256, endpoint=False)
# # Physics: Base brightness + a poloidal m=1 mode (asymmetry) + m=2 (ellipticity)
# clean_physics = 10 + 3*np.cos(theta - 0.5) + 1.5*np.cos(2*theta)
# reconstruction = 10 + 2.8*np.cos(theta - 0.45) + 1.2*np.cos(2*theta) # Slightly off
# measurement = clean_physics + np.random.normal(0, 1.2, len(theta)) # High noise

# analyze_poloidal_profiles(theta, measurement, reconstruction)

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

def low_mode_reconstruction(theta, measurement, reconstruction, m_cutoff=5):
	"""
	Filters the measurement to keep only the primary physical poloidal modes
	and compares it to the reconstruction.
	"""
	N = len(measurement)
	
	# 1. Transform to Fourier Space
	meas_fft = fft.fft(measurement)
	
	# 2. Create a Filter Mask
	# We keep the first 'm_cutoff' modes and the last 'm_cutoff' (negative frequencies)
	filter_mask = np.zeros(N)
	filter_mask[:m_cutoff+1] = 1
	filter_mask[-m_cutoff:] = 1
	
	# 3. Apply filter and Transform back
	meas_filtered_fft = meas_fft * filter_mask
	meas_physical = fft.ifft(meas_filtered_fft).real
	
	# 4. Compare the "Clean" Physics
	# This correlation is much more meaningful than correlating with the raw noise
	clean_correlation = np.corrcoef(meas_physical, reconstruction)[0, 1]

	# --- Plotting ---
	plt.figure(figsize=(10, 6))
	plt.plot(theta, measurement, alpha=0.3, label='Raw Measurement (Noisy)', color='gray')
	plt.plot(theta, meas_physical, 'r-', linewidth=2, label=f'Filtered Measurement (m <= {m_cutoff})')
	plt.plot(theta, reconstruction, 'b--', linewidth=2, label='Your Reconstruction')
	
	plt.title(f"Comparison of Physical Modes (Correlation: {clean_correlation:.4f})")
	plt.xlabel("Poloidal Angle (rad)")
	plt.ylabel("Brightness")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.show()

	return meas_physical

# # Example Usage:
# # theta, measurement, and reconstruction from previous step
# m_phys = low_mode_reconstruction(theta, measurement, reconstruction, m_cutoff=5)


def add_zeros_to_hdf(pid: str)-> str:
	"""
	Adds zeros to the PID string to ensure it has the format YYYYMMDD_XXX
	"""
	# build h5py_path with 3-digit zero-padded fractional part after the dot
	head, sep, tail = pid.strip().rpartition('.')
	if sep:
		frac = tail
		if frac.isdigit() and len(frac) < 3:
			frac = frac.zfill(3)
		pid_formatted = f"{head}_{frac}"
	else:
		pid_formatted = pid.replace('.', '_')

	return pid_formatted
