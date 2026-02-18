import torch
import lightning as L
from dataset import AEDataModule
from model import AutoEncoder, VariationalAutoEncoder
from callbacks import SaveBest, SaveEveryNEpochs, BetaWarmUp
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from views import GAIN_PIDS as gain_list

def train_autoencoder(data_dir, file_name, input_dim=360, geometry=[64, 32, 16, 8], 
					  beta=1, gamma=1, batch_size=32, max_epochs=100, 
					  normalization_strategy='minmax', learning_rate=3e-4, 
					  activation=torch.nn.ReLU, nprofiles=None, model_kind='AE'):
	# Initialize the data module
	data_module = AEDataModule(data_dir, file_name, batch_size, normalization_strategy, nprofiles=nprofiles)
	data_module.prepare_data()
	data_module.setup()
	data_module.exclude_pids(gain_list)

	# Initialize the model
	if model_kind == 'AE':
		model = AutoEncoder(input_dim=input_dim, geometry=geometry, learning_rate=learning_rate, activation=activation)
		logger_name = f"AE{input_dim}"
	elif model_kind == 'VAE':
		model = VariationalAutoEncoder(input_dim=input_dim, geometry=geometry, beta=beta, gamma=gamma, learning_rate=learning_rate, activation=activation)
		logger_name = f"VAE{input_dim}"

	# Initialize a logger
	logger = TensorBoardLogger("W7-X_QXT", name=logger_name)

	# Initialize the trainer
	trainer = L.Trainer(
		logger=logger,
		max_epochs=max_epochs,
		accelerator='cpu',
		callbacks=[
			SaveBest(monitor="val/loss", logger=logger),
			SaveEveryNEpochs(10, logger=logger),
			EarlyStopping(monitor="val/loss", patience=10, mode="min"),
			# BetaWarmUp(start_epoch=50, initial_beta=0, final_beta=0.1, warmup_epochs=100),
			],
		devices=1,)


	# Train the model
	trainer.fit(model, data_module)

	# Evaluate the model on the validation set
	if data_module.val_data is not None:
		trainer.validate(model, datamodule=data_module)
	# Evaluate the model on the test set
	if data_module.test_data is not None:
		trainer.test(model, datamodule=data_module)

	return model

if __name__ == "__main__":
	data_dir = "../data"
	file_name = "20251028_all_data.npz"
	input_dim = 360
	geometry = [64, 32, 32]
	beta = 1e-7
	gamma = 1e-4
	batch_size = 32
	max_epochs = 100
	normalization_strategy = 'minmax'  # Options: 'minmax', 'zscore', 'robust', 'none'
	learning_rate = 3e-4
	nprofiles = 10000
	activation = torch.nn.ReLU()
	model_kind = 'AE'

	# Train the autoencoder
	train_autoencoder(data_dir, file_name, input_dim, geometry, beta, gamma, batch_size, 
				   max_epochs, normalization_strategy, learning_rate, activation,
				   nprofiles, model_kind=model_kind)
