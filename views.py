import os
from pyexpat import model

import numpy as np
import pandas as pd
import torch
import umap

# import the AEDataModule
from dataset import AEDataModule

# import the model
from model import AutoEncoder, VariationalAutoEncoder

# These imports are for the latent space clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score

# These imports are for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Global Variables
GAIN_PIDS = ["20250416.44", "20250416.30", "20250416.24", "20250410.45", "20250410.8", "20250408.46",
			"20250408.43", "20250403.26", "20250403.25", "20250403.23", "20250403.20", "20250403.17",
			"20250403.13", "20250402.46", "20250402.18", "20250402.16", "20250402.14", "20250402.13",
			"20250401.78", "20250401.77", "20250401.70", "20250401.60", "20250401.53", "20250401.49",
			"20241128.69", "20241128.70"]


def plot_latent_space(encoder, data_loader, device, save_path=None):
	"""
	Plots the 2D latent space of the encoder.

	Args:
		encoder (torch.nn.Module): The encoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.
		save_path (str, optional): Path to save the plot. If None, the plot is shown instead.

	Returns:
		None
	"""
	encoder.eval()
	latents = []
	labels = []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = encoder(profiles)
			latents.append(z.cpu().numpy())
			labels.append(np.zeros(z.size(0)))  # Dummy labels, modify as needed

	latents = np.concatenate(latents, axis=0)
	labels = np.concatenate(labels, axis=0)

	plt.figure(figsize=(8, 6))
	scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.7)
	plt.colorbar(scatter)
	plt.title('2D Latent Space')
	plt.xlabel('Latent Dimension 1')
	plt.ylabel('Latent Dimension 2')

	if save_path:
		plt.savefig(save_path)
		print(f"Latent space plot saved to {save_path}")
	else:
		plt.show()

def plot_latent_reconstruction(encoder, decoder, data_loader, device, num_samples=5, save_path=None):
	"""
	Plots original and reconstructed samples from the autoencoder.

	Args:
		encoder (torch.nn.Module): The encoder model.
		decoder (torch.nn.Module): The decoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.
		num_samples (int): Number of samples to plot.
		save_path (str, optional): Path to save the plot. If None, the plot is shown instead.

	Returns:
		None
	"""
	encoder.eval()
	decoder.eval()

	with torch.no_grad():
		for i, batch in enumerate(data_loader):
			batch = batch.to(device)
			z = encoder(batch)
			reconstructed = decoder(z)

			if i == 0:
				originals = batch.cpu().numpy()
				reconstructions = reconstructed.cpu().numpy()
			else:
				originals = np.concatenate((originals, batch.cpu().numpy()), axis=0)
				reconstructions = np.concatenate((reconstructions, reconstructed.cpu().numpy()), axis=0)

			if originals.shape[0] >= num_samples:
				break

	# Plotting
	fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
	for i in range(num_samples):
		axes[i, 0].imshow(originals[i].transpose(1, 2, 0))
		axes[i, 0].set_title("Original")
		axes[i, 0].axis("off")

		axes[i, 1].imshow(reconstructions[i].transpose(1, 2, 0))
		axes[i, 1].set_title("Reconstructed")
		axes[i, 1].axis("off")

	if save_path:
		plt.savefig(save_path)
		print(f"Latent space plot saved to {save_path}")
	else:
		plt.show()

def interactive_latent_space(model, data_loader, device):
	"""
	Creates an interactive plot for the latent space of the encoder.

	Args:
		encoder (torch.nn.Module): The encoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.

	Returns:
		None
	"""
	model.encoder.eval()
	latents = []
	originals = []
	pids = []
	times = []
	# cameras = []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			# try:
			# 	cameras.append(batch['camera'])
			# except KeyError:
			# 	cameras.append(None)

	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)
	# if cameras is not None:
	# 	cameras = np.concatenate(cameras, axis=0)

	fig, ax = plt.subplots(figsize=(8, 6))
	scatter = ax.scatter(latents[:, 0], latents[:, 1], c='blue', alpha=0.7)
	ax.set_title("Interactive Latent Space")
	ax.set_xlabel("Latent Dimension 1")
	ax.set_ylabel("Latent Dimension 2")

	def onpick(event):
		ind = event.ind[0]
		z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
		model.decoder.eval()
		with torch.no_grad():
			recon = model.decoder(z).cpu().numpy()[0]
		orig = originals[ind]
		fig2 = plt.figure(figsize=(8, 4))
		plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
		plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
		
		plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s")

	fig.canvas.mpl_connect('pick_event', onpick)
	scatter.set_picker(True)
	plt.show()

def interactive_latent_space_wflags(model, data_loader, device):
	"""
	Creates an interactive plot for the latent space of the encoder and adds a legend
	based on the 'clipping' flag in each batch.

	Args:
		model (torch.nn.Module): The model with encoder/decoder.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.

	Returns:
		None
	"""

	model.encoder.eval()
	latents = []
	originals = []
	pids = []
	times = []
	cameras = []
	clippings = []

	model.eval()
	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encode(profiles)
			if isinstance(z, tuple):
				# If the output is a tuple, take the first element (the mean of the VAE)
				z = z[0]
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			# cameras.append(batch['camera'])
			clippings.append(batch['clipping'])

	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)
	# cameras = np.concatenate(cameras, axis=0)
	clippings = np.concatenate(clippings, axis=0)

	# If clipping has extra dimensions (e.g., per-channel), reduce to per-sample boolean
	if clippings.ndim > 1:
		try:
			clippings = np.any(clippings, axis=1)
		except Exception:
			# fallback: flatten if shapes don't align
			print("Clipping shape mismatch, flattening.")
			clippings = clippings.reshape((clippings.shape[0], -1))
			clippings = np.any(clippings, axis=1)

	# Map clipping boolean to colors and legend
	gain_list = ["20250416.44", "20250416.30", "20250416.24", "20250410.45", "20250410.8", "20250408.46", 
			"20250408.43", "20250403.26", "20250403.25", "20250403.23", "20250403.20", "20250403.17", 
			"20250403.13", "20250402.46", "20250402.18", "20250402.16", "20250402.14", "20250402.13",
			"20250401.78", "20250401.77", "20250401.70", "20250401.60", "20250401.53", "20250401.49",]

	colors = np.where(np.isin(pids, gain_list), 'green', np.where(clippings, 'red', 'blue'))  # gains override clipping/color

	fig, ax = plt.subplots(figsize=(8, 6))
	print("Latent space dimensions:", latents.shape[1])
	scatter = ax.scatter(latents[:, 0], latents[:, 2], c=colors, alpha=0.7) # here one can decide which dimensions to plot on a 2D plane
	ax.set_title("Interactive Latent Space (colored by clipping)")
	ax.set_xlabel("Latent Dimension 1")
	ax.set_ylabel("Latent Dimension 2")

	# Legend patches
	handles = [
		Patch(color='red', label='Clipped'),
		Patch(color='blue', label='Not clipped'),
		Patch(color='green', label='Gain')
	]
	ax.legend(handles=handles, title="Clipping")

	def onpick(event):
		# event.ind can be a list of indices; pick the first
		inds = getattr(event, "ind", None)
		if not inds:
			return
		ind = inds[0]
		z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
		model.eval()
		with torch.no_grad():
			recon = model.decode(z).cpu().numpy()[0]
		orig = originals[ind]
		plt.figure(figsize=(8, 4))
		plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
		plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
		plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s | Clipped: {bool(clippings[ind])}")
		plt.legend()
		plt.show()

	fig.canvas.mpl_connect('pick_event', onpick)
	scatter.set_picker(True)
	plt.show()


def violin_plot(model, data_loader, device):
	"""
	Plots the distribution of latent spaces with dim > 2 by class.

	Args:
		model (torch.nn.Module): The autoencoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.

	Returns:
		None
	"""
	model.encoder.eval()
	latents = []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())

	latents = np.concatenate(latents, axis=0)
	print(latents.shape)

	# Make a dataframe from the latents for seaborn
	df = pd.DataFrame(latents, columns=[f"Dim {i}" for i in range(latents.shape[1])])

	# Create the violin plots
	plt.figure(figsize=(12, 6))
	sns.violinplot(data=df, inner="quartile")
	plt.title("Latent Space Distribution")
	plt.xlabel(f"Latent Dimensions")
	plt.ylabel("Density")
	plt.show()

def PairwiseScatterMatrix(model, data_loader, device):
	model.eval()
	latents, originals = [], []
	pids, times, cameras, mag_confs, clippings = [], [], [], [], []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			cameras.append(batch['camera'])
			mag_confs.append(batch['mag_conf'])
			clippings.append(batch['clipping'])

	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)

	meta = {
		"camera": np.concatenate(cameras, axis=0),
		"mag_conf": np.concatenate(mag_confs, axis=0),
		"clipping": np.any(np.concatenate(clippings, axis=0), axis=1)
	}

	df = pd.DataFrame(latents, columns=[f"Dim {i}" for i in range(latents.shape[1])])
	for k, v in meta.items():
		df[k] = v

	# Only use latent dimensions for the axes
	latent_cols = [c for c in df.columns if c.startswith("Dim ")]

	# --- Create all pairplots at once ---
	for key in meta.keys():
		print(f"Generating pairplot colored by: {key}")
		grid = sns.pairplot(
			df[latent_cols + [key]],
			hue=key,
			corner=True,
			palette="tab10",
			plot_kws={'alpha': 0.6, 's': 20}
		)
		grid.figure.suptitle(f"Pairwise Scatter Matrix Colored by '{key}'", y=1.02)
		grid.figure.tight_layout()
		plt.savefig(f"pairplot_{key}.png")
		plt.show(block=False)

def PairwiseScatterMatrix(model, data_loader, device):
	model.eval()
	latents = []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())

	latents = np.concatenate(latents, axis=0)

	# DataFrame with only latent dimensions
	df = pd.DataFrame(latents, columns=[f"Dim {i}" for i in range(latents.shape[1])])

	# Simple pairwise scatter matrix without any labeling
	grid = sns.pairplot(df, corner=True, plot_kws={'alpha': 0.6, 's': 20})
	grid.figure.suptitle("Pairwise Scatter Matrix of Latent Dimensions", y=1.02)
	grid.figure.tight_layout()
	plt.show()

def LabeledPairwiseScatterMatrix(model, data_loader, device):
	model.eval()
	latents, originals = [], []
	pids, times, cameras, mag_confs, clippings = [], [], [], [], []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			cameras.append(batch['camera'])
			mag_confs.append(batch['mag_conf'])
			clippings.append(batch['clipping'])

	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)

	meta = {
		"camera": np.concatenate(cameras, axis=0),
		"mag_conf": np.concatenate(mag_confs, axis=0),
		"clipping": np.any(np.concatenate(clippings, axis=0), axis=1)
	}

	df = pd.DataFrame(latents, columns=[f"Dim {i}" for i in range(latents.shape[1])])
	for k, v in meta.items():
		df[k] = v

	# Only use latent dimensions for the axes
	latent_cols = [c for c in df.columns if c.startswith("Dim ")]

	# --- Create all pairplots at once ---
	for key in meta.keys():
		print(f"Generating pairplot colored by: {key}")
		grid = sns.pairplot(
			df[latent_cols + [key]],
			hue=key,
			corner=True,
			palette="tab10",
			plot_kws={'alpha': 0.6, 's': 20}
		)
		grid.figure.suptitle(f"Pairwise Scatter Matrix Colored by '{key}'", y=1.02)
		grid.figure.tight_layout()
		plt.savefig(f"pairplot_{key}.png")
		plt.show(block=False)

def UMAP_latent_space(model, data_loader, device, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', save_path=None):
	"""
	Applies UMAP to the latent space and plots the result.

	Args:
		model (torch.nn.Module): The autoencoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.
		n_neighbors (int): Number of neighbors for UMAP.
		min_dist (float): Minimum distance for UMAP.
		n_components (int): Number of dimensions for UMAP.
		metric (str): Metric for UMAP.
		save_path (str, optional): Path to save the plot. If None, the plot is shown instead.

	Returns:
		None
	"""
	model.encoder.eval()
	latents = []

	# Collect originals and metadata for interaction + populate latent space
	originals = []
	pids = []
	times = []
	cameras = []
	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			cameras.append(batch['camera'])
	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)
	cameras = np.concatenate(cameras, axis=0)
	reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
	embedding = reducer.fit_transform(latents)

	# Interactive UMAP embedding plot
	if n_components == 2:
		fig, ax = plt.subplots(figsize=(8, 6))
		scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c='blue', alpha=0.7)
		ax.set_title("UMAP Projection of Latent Space")
		ax.set_xlabel("UMAP Dimension 1")
		ax.set_ylabel("UMAP Dimension 2")

		def onpick(event):
			ind = event.ind[0]
			z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
			model.decoder.eval()
			with torch.no_grad():
				recon = model.decoder(z).cpu().numpy()[0]
			orig = originals[ind]
			fig2 = plt.figure(figsize=(8, 4))
			plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
			plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
			plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s | Camera: {cameras[ind]}")
			plt.legend()
			plt.show()

		fig.canvas.mpl_connect('pick_event', onpick)
		scatter.set_picker(True)
		if save_path:
			plt.savefig(save_path)
		else:
			plt.show()

	elif n_components == 3:
		"""
		3D UMAP projection, understand why it is not working
		"""
		from mpl_toolkits.mplot3d  import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
		# scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
		ax.set_title("UMAP Projection of Latent Space")
		ax.set_xlabel("UMAP Dimension 1")
		ax.set_ylabel("UMAP Dimension 2")
		ax.set_zlabel("UMAP Dimension 3")

		# def onpick(event):
		# 	ind = event.ind[0]
		# 	z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
		# 	model.decoder.eval()
		# 	with torch.no_grad():
		# 		recon = model.decoder(z).cpu().numpy()[0]
		# 	orig = originals[ind]
		# 	fig2 = plt.figure(figsize=(8, 4))
		# 	plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
		# 	plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
		# 	plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s | Camera: {cameras[ind]}")
		# 	plt.legend()
		# 	plt.show()

		# fig.canvas.mpl_connect('pick_event', onpick)
		# scatter.set_picker(True)
		if save_path:
			plt.savefig(save_path)
		else:
			plt.show()
			
	else:
		print("Number of components not implemented")

def VAEviolin_plot(model, data_loader, device):
	"""
	Plots the distribution of latent spaces with dim > 2 by class.

	Args:
		model (torch.nn.Module): The autoencoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.

	Returns:
		None
	"""
	model.eval()
	latents = []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			mu, logvar = model.encoder(profiles)
			z = model.reparametrize(mu, logvar)
			latents.append(z.cpu().numpy())

	latents = np.concatenate(latents, axis=0)
	print(latents.shape)

	# Make a dataframe from the latents for seaborn
	df = pd.DataFrame(latents, columns=[f"Dim {i}" for i in range(latents.shape[1])])

	# Create the violin plots
	plt.figure(figsize=(12, 6))
	sns.violinplot(data=df, inner="quartile")
	plt.title("Latent Space Distribution")
	plt.xlabel(f"Latent Dimensions")
	plt.ylabel("Density")
	plt.show()

def VAEPairwiseScatterMatrix(model, data_loader, device):
	"""
	Plots the pairwise scatter matrix of the latent space dimensions.

	Args:
		model (torch.nn.Module): The autoencoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.

	Returns:
		None
	"""
	model.eval()
	latents = []
	originals = []
	pids = []
	times = []
	cameras = []
	mag_confs = []
	clippings = []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			mu, logvar = model.encode(profiles)
			z = model.reparametrize(mu, logvar)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			cameras.append(batch['camera'])
			mag_confs.append(batch['mag_conf'])
			clippings.append(batch['clipping'])

	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)
	cameras = np.concatenate(cameras, axis=0)
	mag_confs = np.concatenate(mag_confs, axis=0)
	clippings = np.concatenate(clippings, axis=0)

	df = pd.DataFrame(latents, columns=[f"Dim {i}" for i in range(latents.shape[1])])

	# Plot pairwise scatter matrix and make it interactive
	axes = sns.pairplot(df)
	plt.suptitle("Pairwise Scatter Matrix of Latent Dimensions", y=1.02)
	plt.tight_layout()

	# Attach pick event to all scatter axes
	def onpick(event):
		ind = event.ind[0]
		z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
		with torch.no_grad():
			recon = model.decode(z).cpu().numpy()[0]
		orig = originals[ind]
		fig2 = plt.figure(figsize=(8, 4))
		plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
		plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
		plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s | Camera: {cameras[ind]}")
		plt.legend()
		plt.show()

	# Find all scatter plots in the pairplot and set picker
	for ax in axes.axes.flatten():
		for coll in ax.collections:
			coll.set_picker(True)

	axes.fig.canvas.mpl_connect('pick_event', onpick)
	plt.show()

def VAEUMAP_latent_space(model, data_loader, device, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', save_path=None):
	"""
	Applies UMAP to the latent space and plots the result.

	Args:
		model (torch.nn.Module): The autoencoder model.
		data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
		device (torch.device): Device to run the model on.
		n_neighbors (int): Number of neighbors for UMAP.
		min_dist (float): Minimum distance for UMAP.
		n_components (int): Number of dimensions for UMAP.
		metric (str): Metric for UMAP.
		save_path (str, optional): Path to save the plot. If None, the plot is shown instead.

	Returns:
		None
	"""
	model.eval()
	latents = []

	# Collect originals and metadata for interaction + populate latent space
	originals = []
	pids = []
	times = []
	cameras = []
	with torch.no_grad():
		for batch in data_loader:
			profiles = batch['profile'].to(device)
			mu, logvar = model.encode(profiles)
			z = model.reparametrize(mu, logvar)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])
			cameras.append(batch['camera'])
	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)
	cameras = np.concatenate(cameras, axis=0)
	reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
	embedding = reducer.fit_transform(latents)

	# Interactive UMAP embedding plot
	if n_components == 2:
		fig, ax = plt.subplots(figsize=(8, 6))
		scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c='blue', alpha=0.7)
		ax.set_title("UMAP Projection of Latent Space")
		ax.set_xlabel("UMAP Dimension 1")
		ax.set_ylabel("UMAP Dimension 2")

		def onpick(event):
			ind = event.ind[0]
			z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
			with torch.no_grad():
				recon = model.decode(z).cpu().numpy()[0]
			orig = originals[ind]
			fig2 = plt.figure(figsize=(8, 4))
			plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
			plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
			plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s | Camera: {cameras[ind]}")
			plt.legend()
			plt.show()

		fig.canvas.mpl_connect('pick_event', onpick)
		scatter.set_picker(True)
		if save_path:
			plt.savefig(save_path)
		else:
			plt.show()

	elif n_components == 3:
		"""
		3D UMAP projection, understand why it is not working
		"""
		from mpl_toolkits.mplot3d  import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
		# scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
		ax.set_title("UMAP Projection of Latent Space")
		ax.set_xlabel("UMAP Dimension 1")
		ax.set_ylabel("UMAP Dimension 2")
		ax.set_zlabel("UMAP Dimension 3")

		# def onpick(event):
		# 	ind = event.ind[0]
		# 	z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
		# 	with torch.no_grad():
		# 		recon = model.decode(z).cpu().numpy()[0]
		# 	orig = originals[ind]
		# 	fig2 = plt.figure(figsize=(8, 4))
		# 	plt.plot(np.arange(len(orig)), orig, marker='o', label="Original")
		# 	plt.plot(np.arange(len(recon)), recon, marker='x', label="Reconstructed")
		# 	plt.title(f"PID: {pids[ind]} @ t = {times[ind]} s | Camera: {cameras[ind]}")
		# 	plt.legend()
		# 	plt.show()

		# fig.canvas.mpl_connect('pick_event', onpick)
		# scatter.set_picker(True)
		if save_path:
			plt.savefig(save_path)
		else:
			plt.show()
			
	else:
		print("Number of components not implemented")


def plot_tensorboard_csv(directory, train_file: str ="train.csv", val_file: str ="val.csv", metric="Value", step_col="Step"):
	"""
	Searches for train.csv and val.csv in a directory and plots them together.

	Args:
		directory (str): Path to the folder containing CSV files.
		metric (str): Column name for the metric to plot (e.g., 'Value').
		step_col (str): Column name for the x-axis (usually 'Step').

	Example:
		plot_tensorboard_csvs("./logs")
	"""
	# Define expected filenames
	csv_files = {
		"Train": os.path.join(directory, train_file),
		"Validation": os.path.join(directory, val_file)
	}

	plt.figure(figsize=(8, 5))

	found_any = False

	for label, file_path in csv_files.items():
		if os.path.exists(file_path):
			df = pd.read_csv(file_path)
			if step_col in df.columns and metric in df.columns:
				plt.plot(df[step_col], df[metric], label=label)
				found_any = True
			else:
				print(f"Skipping {file_path}: missing '{step_col}' or '{metric}' column.")
		else:
			print(f"{file_path} not found.")

	if found_any:
		plt.title(f"{metric} vs {step_col}")
		plt.xlabel(step_col)
		plt.ylabel(metric)
		plt.legend()
		plt.grid(True, linestyle="--", alpha=0.6)
		plt.tight_layout()
		plt.show()
	else:
		print("No valid CSV files found to plot.")

def compute_distance_embedding(model, dataloader, device):
	"""
	Computes pairwise distances between embeddings.

	Params:
		model: The model used to encode the input data.
		dataloader: The dataloader providing the input data.
		device: The device to perform computations on (CPU or GPU).

	Returns:
		A tensor containing the pairwise distances between embeddings.
	"""

	# collect embeddings then compute full pairwise distances
	embeds = []
	model.eval()
	with torch.no_grad():
		for batch in dataloader:
			x = batch['profile'].to(device)
			z = model.encode(x)
			embeds.append(z)			# keep on device for cdist efficiency
	zs = torch.cat(embeds, dim=0)	 # shape (N, D)
	dists = torch.cdist(zs, zs, p=2)  # shape (N, N), on device
	return dists.cpu()

def compute_distance_measure_reconstruction(model, dataloader, device):
	"""
	Computes the distance between original and reconstructed embeddings.

	Params:
		model: The model used to encode the input data.
		dataloader: The dataloader providing the input data.
		device: The device to perform computations on (CPU or GPU).

	Returns:
		A tensor containing the distances between original and reconstructed embeddings.
	"""
	# collect original and reconstructed embeddings
	model.eval()
	original = []
	reconstructed = []
	pids = []
	with torch.no_grad():
		for batch in dataloader:
			x = batch['profile'].to(device)
			recon = model(x)
			reconstructed.append(recon)
			original.append(x)
			pids.append(batch['pid'])
	dists = torch.norm(torch.stack(original) - torch.stack(reconstructed), dim=-1)  # shape (N,), on device
	pids = np.array(pids)
	# Unravel both dists and pids
	dists = dists.flatten()
	pids = pids.flatten()
	# save the data in a npz file
	np.savez_compressed("reconstruction_distances.npz", dists=dists, pids=pids)
	return dists.cpu(), pids

def clustering_LatentSpace(model, dataloader, device):
	# Assuming you have a trained autoencoder model and an encoder submodule
	# Example for PyTorch:
	# latent_vectors = encoder(torch.tensor(X).float()).detach().numpy()

	# Example for TensorFlow:
	# latent_vectors = encoder.predict(X)

	# Step 1: Get latent features
	model.eval()
	with torch.no_grad():
		xs = []
		for batch in dataloader:
			profiles = batch['profile'].to(device)
			z = model.encode(profiles)
			xs.append(z.cpu().numpy())
		xs = np.concatenate(xs, axis=0)

	# Step 2: (Optional) Dimensionality reduction for visualization or denoising
	pca = PCA(n_components=min(20, xs.shape[1]))
	latent_pca = pca.fit_transform(xs)

	# Step 3: Apply clustering
	kmeans = KMeans(n_clusters=5, random_state=42)
	cluster_labels = kmeans.fit_predict(latent_pca)

	# Optionally, try DBSCAN or GMM
	# db = DBSCAN(eps=0.5, min_samples=5).fit(latent_pca)
	# gmm = GaussianMixture(n_components=5).fit(latent_pca)
	# cluster_labels = gmm.predict(latent_pca)

	# Step 4: Evaluate clustering performance
	sil_score = silhouette_score(latent_pca, cluster_labels)
	print(f"Silhouette Score: {sil_score:.3f}")

	# If ground-truth fault labels are known:
	if 'y_true' in locals():
		ari = adjusted_rand_score(y_true, cluster_labels)
		print(f"Adjusted Rand Index (vs true faults): {ari:.3f}")

	# Step 5: Visualize in 2D using t-SNE
	tsne = TSNE(n_components=2, random_state=42)
	latent_2d = tsne.fit_transform(latent_pca)

	plt.figure(figsize=(8,6))
	plt.scatter(latent_2d[:,0], latent_2d[:,1], c=cluster_labels, cmap='tab10', s=15)
	plt.title("t-SNE of Encoded Data with Clusters")
	plt.colorbar(label='Cluster')
	plt.show()

	# Optional: visualize with true fault labels if available
	if 'y_true' in locals():
		plt.figure(figsize=(8,6))
		plt.scatter(latent_2d[:,0], latent_2d[:,1], c=y_true, cmap='tab10', s=15)
		plt.title("t-SNE of Encoded Data Colored by True Fault Labels")
		plt.colorbar(label='True Fault Category')
		plt.show()

def interactive_latent_space_clusters(
	model,
	data_loader,
	device,
	n_clusters=5,
	gain_pid='20250403.17'
):
	model.encoder.eval()
	latents, originals, pids, times, clippings = [], [], [], [], []

	with torch.no_grad():
		for batch in data_loader:
			profiles = batch["profile"].to(device)
			z = model.encoder(profiles)
			latents.append(z.cpu().numpy())
			originals.append(batch["profile"].cpu().numpy())
			pids.append(batch["pid"])
			times.append(batch["time"])
			clippings.append(batch["clipping"])

	# Concatenate everything
	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)
	clippings = np.concatenate(clippings, axis=0)

	if clippings.ndim > 1:
		clippings = np.any(clippings, axis=1)

	# PCA + clustering
	pca = PCA(n_components=min(20, latents.shape[1]))
	latent_pca = pca.fit_transform(latents)
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	cluster_labels = kmeans.fit_predict(latent_pca)

	# 2D visualization
	tsne = TSNE(n_components=2, random_state=42, perplexity=30)
	latent_2d = tsne.fit_transform(latent_pca)

	cmap = plt.get_cmap("tab10")
	colors = np.array([cmap(i % 10) for i in cluster_labels])
	is_gain = (pids == gain_pid)

	fig, ax = plt.subplots(figsize=(9, 7))

	# Plot 4 combinations efficiently
	mask_normal = ~clippings & ~is_gain
	mask_gain = ~clippings & is_gain
	mask_clip = clippings & ~is_gain
	mask_clip_gain = clippings & is_gain

	ax.scatter(latent_2d[mask_normal, 0], latent_2d[mask_normal, 1],
			   c=colors[mask_normal], marker='o', s=60, alpha=0.7, edgecolors='none', label='Normal')
	ax.scatter(latent_2d[mask_gain, 0], latent_2d[mask_gain, 1],
			   c=colors[mask_gain], marker='o', s=60, alpha=0.7, edgecolors='green', linewidths=1.5, label='Gain')
	ax.scatter(latent_2d[mask_clip, 0], latent_2d[mask_clip, 1],
			   c=colors[mask_clip], marker='x', s=60, alpha=0.7, label='Clipped')
	ax.scatter(latent_2d[mask_clip_gain, 0], latent_2d[mask_clip_gain, 1],
			   c=colors[mask_clip_gain], marker='x', s=60, alpha=0.7, edgecolors='green', linewidths=1.5, label='Clipped + Gain')

	ax.set_title("Latent Space Clusters (Color=Cluster, Marker=Clipping, Edge=Gain)")
	ax.set_xlabel("t-SNE Dim 1")
	ax.set_ylabel("t-SNE Dim 2")

	# Legend
	cluster_handles = [Patch(color=cmap(i % 10), label=f"Cluster {i}") for i in range(n_clusters)]
	extra_handles = [
		Patch(facecolor='gray', label='Normal (o)'),
		Patch(facecolor='gray', label='Clipped (x)'),
		Patch(edgecolor='green', facecolor='none', label='Gain')
	]
	ax.legend(handles=cluster_handles + extra_handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

	# Prepare scatter for interactive picking
	scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c='none', alpha=0)
	scatter.set_picker(True)

	def onpick(event):
		inds = getattr(event, "ind", None)
		if not inds:
			return
		ind = inds[0]
		z = torch.tensor(latents[ind:ind+1], dtype=torch.float32).to(device)
		model.decoder.eval()
		with torch.no_grad():
			recon = model.decoder(z).cpu().numpy()[0]
		orig = originals[ind]
		plt.figure(figsize=(8, 4))
		plt.plot(orig, label="Original", marker="o")
		plt.plot(recon, label="Reconstructed", marker="x")
		plt.title(
			f"PID: {pids[ind]} | Time: {times[ind]} | "
			f"Cluster: {cluster_labels[ind]} | Clipped: {bool(clippings[ind])} | Gain: {is_gain[ind]}"
		)
		plt.legend()
		plt.show()

	fig.canvas.mpl_connect("pick_event", onpick)
	plt.tight_layout()
	plt.show()

	return cluster_labels, latent_2d, latents

def plot_profiles(model, dataloader, device, num_samples=5):
	model.eval()
	latents, originals = [], []
	pids, times = [], []

	with torch.no_grad():
		for batch in dataloader:
			x = batch['profile'].to(device)
			z = model.encoder(x)
			latents.append(z.cpu().numpy())
			originals.append(batch['profile'].cpu().numpy())
			pids.append(batch['pid'])
			times.append(batch['time'])

	latents = np.concatenate(latents, axis=0)
	originals = np.concatenate(originals, axis=0)
	pids = np.concatenate(pids, axis=0)
	times = np.concatenate(times, axis=0)

	for i in range(num_samples):
		plt.figure(figsize=(8, 4))
		plt.plot(originals[i], label="Original", marker="o")
		recon = model.decoder(torch.tensor(latents[i:i+1], dtype=torch.float32).to(device)).detach().cpu().numpy()[0]
		plt.plot(recon, label="Reconstructed", marker="x")
		plt.title(f"PID: {pids[i]} | Time: {times[i]}")
		plt.legend()
		plt.savefig(f"images/profile_{i}.png")
		plt.close()

def compute_std_dev_latent(model, dataloader, device):
	'''
	Compute the standard deviation of the latent space representations.
	Then plot them on a graph where the x is the latent dimension and the y the 
	associated standard deviation.
	'''
	model.eval()
	with torch.no_grad():
		latents = []
		for batch in dataloader:
			x = batch['profile'].to(device)
			z = model.encoder(x)
			latents.append(z.cpu().numpy())
	latents = np.concatenate(latents, axis=0)

	# Plot the standard deviation of the latent space representations
	std_dev = np.std(latents, axis=0)
	plt.figure(figsize=(8, 4))
	plt.plot(std_dev, label="Standard Deviation", marker="o")
	plt.title("Latent Space Standard Deviation")
	plt.xlabel("Latent Dimension")
	plt.ylabel("Standard Deviation")
	plt.legend()
	# plt.show()
	plt.savefig("latent_std_dev.png")

	return std_dev

def testAE():
	# For some old versions it may be necessary to specify the input_dim and geometry
	# RAVEN
	# model  = AutoEncoder.load_from_checkpoint(r"\\share\mp\E5-Praktikanten\Orlandi_Luca\NNsrc\W7-X_QXT\Raven\AE17\version_8\best_model_.ckpt")
	# LOCALE
	# # In dim 17
	# model  = AutoEncoder.load_from_checkpoint(r"\\share\mp\E5-Praktikanten\Orlandi_Luca\NNsrc\W7-X_QXT\AE17\version_14\best_model_.ckpt")
	# # In dim 360
	model  = AutoEncoder.load_from_checkpoint("/home/IPP-HGW/orluca/devel/classificator/W7-X_QXT/AE360/version_0/best_model_.ckpt")
	# model  = AutoEncoder.load_from_checkpoint(r"\\share\mp\E5-Praktikanten\Orlandi_Luca\NNsrc\W7-X_QXT\Raven\AE360\version_1\best_model_.ckpt")


	# # Divided by camera
	# dm	 = AEDataModule(data_dir="data", file_name='db_by_camera.npz', nprofiles=10000)
	# All profile
	dm	 = AEDataModule(data_dir="../data", file_name='20251028_all_data.npz', nprofiles=None, normalization_strategy='minmax')
	dm.setup()
	dm.get_pids(GAIN_PIDS)
	# print(f"maximum value used for normalization: {dm.max_val}")
	# print(f"minimum value used for normalization: {dm.min_val}")
	
	# # This is to find particular cases, this routine is not working properly!!
	# train_loader = dm.train_dataloader()
	# # see if in train_loader are cases with pid == '20250312.83'
	# for batch in train_loader:
	# 	if (batch['pid'] == '20250312.83'):
	# 		print("Found matching PID in train_loader")
	# 	else:
	# 		print("Did not find matching PID in train_loader")
	# assert False

	# test_loader = dm.test_dataloader()
	test_loader = dm.full_dataloader() # i am using the full dataloader because these are profiles never seen by the model
	
	device = "cpu"
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# # print the model and do one iteration over the test set
	# print(model)
	# for batch in test_loader:
	# 	x = batch['profile'].to(device)
	# 	with torch.no_grad():
	# 		reconstructed = model(x)
	# 	break

	# # # ------------------ Distance computations ------------------ # # #
	# # check if a file for distances exists otherwise compute them
	# if os.path.exists("reconstruction_distances.npz"):
	# 	data = np.load("reconstruction_distances.npz")
	# 	dists = data["dists"]
	# 	pids = data["pids"]
	# else:
	# 	# First of all compute the distances
	# 	print("Computing distances...")
	# 	dists, pids = compute_distance_measure_reconstruction(model, test_loader, device)

	# # Once we have the distances and PIDs, we can use them for further analysis
	# # See which PID has the largest distance besides "20250312.83"
	# # print(np.where(pids == '20250312.83', False, True))
	# max_dist_idx = np.argmax(dists[np.where(pids == '20250312.83', False, True)])
	# print(f"PID with largest distance: {pids[max_dist_idx]}, Distance: {dists[max_dist_idx]}")

	# # Try with flags
	# interactive_latent_space_wflags(model, test_loader, device)

	# # w/o
	# interactive_latent_space(model, test_loader, device)
	# violin_plot(model, test_loader, device)
	# PairwiseScatterMatrix(model, test_loader, device)
	# UMAP_latent_space(model, test_loader, device, n_components=2)

	# # Try clustering
	# clustering_LatentSpace(model, test_loader, device)
	# cluster_labels, latent_2d, latents = interactive_latent_space_clusters(model, test_loader, device)

	# # Compute std_dev
	# compute_std_dev_latent(model, test_loader, device)

	# plot some profiles
	plot_profiles(model, test_loader, device, num_samples=10)

def testVAE():
	from model import VariationalAutoEncoder
	# model  = VariationalAutoEncoder.load_from_checkpoint(r"\\share\mp\E5-Praktikanten\Orlandi_Luca\NNsrc\W7-X_QXT\Raven\VAE360\version_9\best_model_.ckpt")
	model  = VariationalAutoEncoder.load_from_checkpoint(r"\\share\mp\E5-Praktikanten\Orlandi_Luca\QXT_NN\NNsrc\W7-X_QXT\Raven\VAE360\version_9\best_model_.ckpt")

	dm	 = AEDataModule(data_dir="data", file_name='20251028_all_data.npz', nprofiles=None)
	dm.setup()
	test_loader = dm.test_dataloader()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	# VAEviolin_plot(model, test_loader, device)
	# VAEPairwiseScatterMatrix(model, test_loader, device)
	# VAEUMAP_latent_space(model, test_loader, device, n_components=2)

	# Interactive latent space:
	interactive_latent_space_wflags(model, test_loader, device)


if __name__ == "__main__":

	# # Use these functions to understand how the latent space looks like
	testAE()
	# testVAE()

	# # Use this function to plot losses
	# plot_tensorboard_csv(r'\\share\mp\E5-Praktikanten\Orlandi_Luca\NNsrc\W7-X_QXT\Raven\AE17\version_2')
