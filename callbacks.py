import torch 
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

class SaveBest(Callback):
	def __init__(self, monitor: str, logger: TensorBoardLogger) -> None:
		super().__init__()
		self.monitor = monitor
		self.logger = logger
		self.best_loss = float('inf')

	def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		loss = trainer.callback_metrics[self.monitor]
		if loss < self.best_loss:
			self.best_loss = loss
			trainer.save_checkpoint(f"{self.logger.log_dir}/best_model_.ckpt")
		return super().on_validation_end(trainer, pl_module)
	
	def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		print(f"Best {self.monitor} loss: {self.best_loss}")
		return super().on_train_end(trainer, pl_module)

class SaveEveryNEpochs(Callback):
	def __init__(self, n: int, logger: TensorBoardLogger) -> None:
		super().__init__()
		self.n = n
		self.logger = logger

	def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		if (trainer.current_epoch + 1) % self.n == 0:
			trainer.save_checkpoint(f"{self.logger.log_dir}/epoch_{trainer.current_epoch + 1}.ckpt")
		return super().on_train_epoch_end(trainer, pl_module)
	
class PrintLearningRate(Callback):
	def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		optimizer = trainer.optimizers[0]
		lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {trainer.current_epoch + 1}: Learning Rate = {lr}")
		return super().on_train_epoch_end(trainer, pl_module)

class BetaWarmUp(Callback):
	def __init__(self, start_epoch: int, initial_beta: float, final_beta: float, warmup_epochs: int) -> None:
		super().__init__()
		self.start_epoch = start_epoch
		self.initial_beta = initial_beta
		self.final_beta = final_beta
		self.warmup_epochs = warmup_epochs

	def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
		if self.start_epoch <= trainer.current_epoch < self.start_epoch + self.warmup_epochs:
			epoch_in_warmup = trainer.current_epoch - self.start_epoch
			beta = self.initial_beta + (self.final_beta - self.initial_beta) * (epoch_in_warmup / self.warmup_epochs)
		elif trainer.current_epoch >= self.start_epoch + self.warmup_epochs:
			beta = self.final_beta
		else:
			beta = self.initial_beta

		if hasattr(pl_module, "set_beta"):
			pl_module.set_beta(beta)
			print(f"Epoch {trainer.current_epoch + 1}: Beta = {beta}")
		else:
			print("Warning: Model does not have a set_beta method.")
		return super().on_train_epoch_start(trainer, pl_module)