import optuna
import lightning as L
import torch 

# ------------- Seed everything -------------
L.seed_everything(42)  # Lightning helper for deterministic runs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from model import AutoEncoder
from dataset import AEDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer

class Objective:
	def __init__(self, data_module: AEDataModule, input_dim: int):
		self.data_module = data_module
		self.input_dim = input_dim

	def __call__(self, trial):
		# Suggest hyperparameters
		learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
		hidden_size_0 = trial.suggest_categorical('hidden_size_0', [64, 128, 256, 512, 1024])
		hidden_size_1 = trial.suggest_categorical('hidden_size_1', [64, 128, 256, 512, 1024])
		hidden_size_2 = trial.suggest_categorical('hidden_size_2', [64, 128, 256, 512, 1024])
		hidden_size_3 = trial.suggest_categorical('hidden_size_3', [64, 128, 256, 512, 1024])
		latent_size = trial.suggest_categorical('latent_size', [4, 8, 16, 32, 64, 128, 256])
		geometry = [hidden_size_0, hidden_size_1, hidden_size_2, latent_size]

		# Initialize model
		model = AutoEncoder(input_dim=self.input_dim, geometry=geometry, learning_rate=learning_rate)

		# Logger
		logger = TensorBoardLogger("OOptuna", name=f"4layers/trial_{trial.number}")

		# Callbacks
		checkpoint_callback = ModelCheckpoint(
			monitor='val/loss',
			dirpath='checkpoints',
			filename=f'4layers/trial_{trial.number}' + '-{epoch:02d}-{val/loss:.4f}',
			save_top_k=1,
			mode='min',
		)
		early_stopping_callback = EarlyStopping(
			monitor='val/loss',
			patience=5,
			mode='min'
		)

		# Trainer
		trainer = Trainer(
			max_epochs=50,
			logger=logger,
			callbacks=[checkpoint_callback, early_stopping_callback],
			accelerator='auto',
			devices='auto'
		)

		# Train the model
		trainer.fit(model, self.data_module)

		# Return the best validation loss
		return checkpoint_callback.best_model_score.item()

if __name__ == "__main__":
	# Data module
	data_module = AEDataModule(data_dir='data', file_name='profs_w_magclip.npz', 
							batch_size=64, normalization_strategy='minmax', nprofiles=100_000)
	data_module.prepare_data()
	data_module.setup()

	# Input dimension
	sample_batch = next(iter(data_module.train_dataloader()))
	input_dim = sample_batch['profile'].shape[1]

	# Optuna study
	study = optuna.create_study(direction='minimize')
	study.optimize(Objective(data_module, input_dim), n_trials=200)

	print("Best trial:")
	trial = study.best_trial
	print(f"  Value: {trial.value}")
	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")