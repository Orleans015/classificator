import numpy as np

def check_clips(signal, gain, clip_threshold=0.01):
	"""Check for clipping in the data. Clearly I have to pass the entire time 
	series for each channel. If the signal is divided into windows it is possible
	that in non-dynamical parts of the signal it is flagged as clipped even though 
	It is not.
	"""
	# get the maximum value of the signal
	vmax = np.max(gain)
	# Flag the positions of the clipping in a boolean array, True where clipped
	clipped_positions = np.where(signal > vmax, True, False) 
	return clipped_positions

def check_flat(signal, flat_threshold=0.01):
	"""Check for flatlining in the data. A signal is considered flat if the
	variance is below a certain threshold.
	"""
	# Compute the variance of the signal
	variance = np.var(signal)
	# Flag the positions of the flatlining in a boolean array
	
	return variance < flat_threshold

def check_jumps(signal, jump_threshold=0.1):
	"""Check for jumps in the data. A signal is considered to have a jump if the
	difference between consecutive samples exceeds a certain threshold.
	"""
	diff = np.abs(np.diff(signal))
	return np.any(diff > jump_threshold), np.where(diff > jump_threshold)[0]

def check_drifts(signal, drift_threshold=0.1):
	"""Check for drifts in the data. A signal is considered to have a drift if the
	overall trend exceeds a certain threshold.
	"""
	trend = np.polyfit(np.arange(len(signal)), signal, 1)[0]
	return np.abs(trend) > drift_threshold

if __name__ == "__main__":
	data = np.load("../_data/npz_files/20250305.79.npz", allow_pickle=True)
	items = data['20250305.79'].item()

	signal = items['vCamout']
	gain   = items['vGain']

	clips = check_clips(signal[:, 1:], gain)

	# # print where the signal is close to the maximum of the gain
	# print("Signal is close to the maximum of the gain at the following positions:")
	# print(np.where(np.isclose(signal, np.max(gain), atol=1e-2)))

	# # print which diodes at idx == 4303 are clipped
	# print("Diodes at idx == 4303 are clipped at the following positions:")
	# print(np.where(clips[4303]))

	# # print where the clipping occurs
	# print("Clipping occurs at the following positions:")
	# print(np.where(clips))

	import seaborn as sns
	import matplotlib.pyplot as plt
	sns.set_theme(style="whitegrid", 
			   context="notebook",
			   font_scale=1.5,
			   palette='muted')
	plt.plot(signal[:,0], signal[:,1:])
	plt.plot(signal[:,0], clips, label='Clipped', linestyle='--')
	plt.legend()
	plt.show()
