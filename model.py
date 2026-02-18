import torch
import torch.nn as nn
import lightning as L

from utils import *

class AutoEncoder(L.LightningModule):
	def __init__(self, input_dim, geometry, learning_rate=1e-3, activation=torch.nn.ReLU()):
		super().__init__()
		self.input_dim = input_dim
		self.geometry = geometry
		self.learning_rate = learning_rate
		self.activation = activation
		self.save_hyperparameters(ignore=['activation'])

		assert len(geometry) >= 2, "Geometry must have at least two elements: hidden and latent sizes"

		# --- Encoder ---
		self.encoder = nn.Sequential()
		in_dim = input_dim
		for i, out_dim in enumerate(geometry[:-1]):
			self.encoder.add_module(f"encoder_layer_{i}", nn.Linear(in_dim, out_dim))
			# self.encoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm1d(out_dim)) # serve per l'ampiezza (batch)
			# self.encoder.add_module(f"encoder_regularization_{i}", nn.L1Loss())
			# self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(0.2))
			self.encoder.add_module(f"encoder_activation_{i}", activation)
			in_dim = out_dim
		self.encoder.add_module("encoder_output_layer", nn.Linear(in_dim, geometry[-1]))

		# --- Decoder (mirror of encoder) ---
		self.decoder = nn.Sequential()
		reversed_geometry = list(reversed(geometry))
		in_dim = reversed_geometry[0]
		for i, out_dim in enumerate(reversed_geometry[1:]):
			self.decoder.add_module(f"decoder_layer_{i}", nn.Linear(in_dim, out_dim))
			self.decoder.add_module(f"decoder_activation_{i}", activation)
			in_dim = out_dim
		self.decoder.add_module("decoder_output_layer", nn.Linear(in_dim, input_dim))

	def encode(self, x):
		return self.encoder(x)

	def decode(self, z):
		return self.decoder(z)

	def forward(self, x):
		return self.decode(self.encode(x))

	def reconstruct(self, x):
		return self.forward(x)

	def _common_step(self, batch, step="train"):
		x = batch['profile']
		reconstructed = self.forward(x)
		loss = self.loss_function(reconstructed, x)
		self.log_dict({f"{step}/loss": loss,}, batch_size=x.size(0))
		return loss

	def training_step(self, batch, batch_idx):
		return self._common_step(batch, step="train")

	def validation_step(self, batch, batch_idx):
		return self._common_step(batch, step="val")

	def test_step(self, batch, batch_idx):
		return self._common_step(batch, step="test")

	def loss_function(self, reconstructed, original):
		return torch.nn.functional.mse_loss(reconstructed, original)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class VariationalAutoEncoder(L.LightningModule):
	def __init__(self, input_dim, geometry, beta=4, gamma=1e-4, learning_rate=1e-3, activation=torch.nn.ReLU()):
		"""
		Args:
			geometry (list[int]): List of layer sizes, e.g. [784, 256, 64, 16]
			beta (float): Weighting factor for the KL divergence loss
			learning_rate (float): Learning rate for the optimizer
			activation (nn.Module): Activation function to use between layers
		"""
		super().__init__()
		self.input_dim = input_dim
		self.geometry = geometry
		self.set_beta(beta)
		self.set_gamma(gamma)
		self.learning_rate = learning_rate
		self.activation = activation
		self.save_hyperparameters()

		assert len(self.geometry) >= 2, "Geometry must have at least input and latent size"
		
		# Build encoder
		self.encoder = nn.Sequential() 
		self.encoder.add_module("encoder_input_layer", nn.Linear(self.input_dim, self.geometry[0]))
		self.encoder.add_module("encoder_input_activation", self.activation)
		for i, (in_dim, out_dim) in enumerate(zip(self.geometry[:-1], self.geometry[1:])):
			self.encoder.add_module(f"encoder_layer_{i}", nn.Linear(in_dim, out_dim))
			self.encoder.add_module(f"encoder_activation_{i}", self.activation)

		self.fc_mu = nn.Linear(self.geometry[-1], self.geometry[-1]//2)
		self.fc_logvar = nn.Linear(self.geometry[-1], self.geometry[-1]//2)

		# Build decoder (mirror of encoder)
		self.decoder_input = nn.Linear(self.geometry[-1]//2, self.geometry[-1])
		self.decoder = nn.Sequential()
		for i, (in_dim, out_dim) in enumerate(zip(reversed(self.geometry[1:]), reversed(self.geometry[:-1]))):
			self.decoder.add_module(f"decoder_layer_{len(self.geometry) - i - 1}", nn.Linear(in_dim, out_dim))
			self.decoder.add_module(f"decoder_activation_{len(self.geometry) - i - 1}", self.activation)
		self.decoder.add_module("decoder_output_layer", nn.Linear(self.geometry[0], self.input_dim))

	def encode(self, x):
		h = self.encoder(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		return mu, logvar

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z):
		h = self.decoder_input(z)
		return self.decoder(h)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparametrize(mu, logvar)
		y = self.decode(z)
		return y, mu, logvar

	def reconstruct(self, x):
		mu, _ = self.encode(x)
		return self.decode(mu)

	def set_beta(self, beta):
		self.beta = beta

	def set_gamma(self, gamma):
		self.gamma = gamma

	def _common_step(self, batch, step="train"):
		x = batch['profile']
		reconstructed, mu, logvar = self.forward(x)
		loss, mse, kl, sml = self.loss_function(reconstructed, x, mu, logvar)
		self.log_dict({f"{step}/loss": loss, f"{step}/mse": mse, f"{step}/kl": kl, f"{step}/sml": sml})
		return loss

	def training_step(self, batch, batch_idx):
		return self._common_step(batch, step="train")

	def validation_step(self, batch, batch_idx):
		return self._common_step(batch, step="val")

	def test_step(self, batch, batch_idx):
		return self._common_step(batch, step="test")

	def loss_function(self, reconstructed, original, mu, logvar):
		# Reconstruction loss (MSE)
		mse = torch.nn.functional.mse_loss(reconstructed, original)
		# KL divergence loss
		kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		# Smoothness loss
		sml = smoothness_loss(reconstructed, original)
		return mse + self.beta * kl + self.gamma * sml, mse, kl, sml

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LatentSpaceClassifier(L.LightningModule):
	"""
	A simple feedforward neural network classifier operating on latent representations.

	Args:
		input_dim (int): Dimension of the input latent space.
		hidden_geometry (list): Dimensions of the hidden layer.
		output_dim (int): Number of output classes.
		learning_rate (float): Learning rate for the optimizer.
		activation (nn.Module): Activation function to use between layers.
	"""
	def __init__(self, input_dim, hidden_geometry, output_dim, learning_rate=1e-3, activation=torch.nn.ReLU()):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_geometry = hidden_geometry
		self.output_dim = output_dim
		self.learning_rate = learning_rate
		self.activation = activation
		self.save_hyperparameters(ignore=['activation'])

		self.model = nn.Sequential()
		for i, out_dim in enumerate(hidden_geometry):
			in_dim = input_dim if i == 0 else hidden_geometry[i - 1]
			self.model.add_module(f"layer_{i}", nn.Linear(in_dim, out_dim))
			self.model.add_module(f"activation_{i}", activation)
		self.model.add_module("output_layer", nn.Linear(hidden_geometry[-1], output_dim))		
		
	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		return self._common_step(batch, step="train")

	def validation_step(self, batch, batch_idx):
		return self._common_step(batch, step="val")

	def test_step(self, batch, batch_idx):
		return self._common_step(batch, step="test")

	def _common_step(self, batch, step="train"):
		x = batch['latent']
		y = batch['label']
		logits = self.forward(x)
		loss = torch.nn.functional.cross_entropy(logits, y)
		self.log(f"{step}/loss", loss)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class AutoEncoderWithClassifier(L.LightningModule):
	"""
	Combines an AutoEncoder with a classifier operating on the latent space.
	
	Args:
		autoencoder (AutoEncoder): The autoencoder model.
		classifier (LatentSpaceClassifier): The classifier model.
	"""
	def __init__(self, autoencoder: AutoEncoder, classifier: LatentSpaceClassifier):
		super().__init__()
		self.autoencoder = autoencoder
		self.classifier = classifier

	def forward(self, x):
		latent = self.autoencoder.encode(x)
		reconstructed = self.autoencoder.decode(latent)
		logits = self.classifier(latent)
		return reconstructed, logits
	
	def training_step(self, batch, batch_idx):
		return self._common_step(batch, step="train")

	def validation_step(self, batch, batch_idx):
		return self._common_step(batch, step="val")
	
	def test_step(self, batch, batch_idx):
		return self._common_step(batch, step="test")
	
	def _common_step(self, batch, step="train"):
		x = batch['profile']
		y = batch['label']
		reconstructed, logits = self.forward(x)
		reconstruction_loss = self.autoencoder.loss_function(reconstructed, x)
		classification_loss = torch.nn.functional.cross_entropy(logits, y)
		loss = reconstruction_loss + classification_loss
		self.log_dict({f"{step}/loss": loss, 
				 f"{step}/reconstruction_loss": reconstruction_loss, 
				 f"{step}/classification_loss": classification_loss},
				 batch_size=x.size(0))
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.autoencoder.learning_rate)

if __name__ == "__main__":
	autoencoder = AutoEncoder(360, [10, 5, 2])
	print(autoencoder)
	# print("Number of parameters:", sum(p.numel() for p in autoencoder.parameters() if p.requires_grad))
	# print("List of parameters:", [p for p in autoencoder.parameters()])
	# print("Encoder structure:", autoencoder.encoder)
	# print("Decoder structure:", autoencoder.decoder)