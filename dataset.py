import torch
import lightning as L
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader

import os


class AEDataset(Dataset):
	def __init__(self, profiles: torch.Tensor, pids: np.ndarray, times: np.ndarray, 
			  cameras: np.ndarray, mag_confs: np.ndarray, clippings: np.ndarray):
		self.profiles = profiles
		self.pids = pids
		self.times = times
		self.cameras = cameras
		self.mag_confs = mag_confs
		self.clippings = clippings

	def __len__(self):
		return len(self.profiles)

	def __getitem__(self, idx: int):
		result = {'profile': self.profiles[idx]}
		if self.pids is not None:
			result['pid'] = self.pids[idx]
		if self.times is not None:
			result['time'] = self.times[idx]
		if self.cameras is not None:
			result['camera'] = self.cameras[idx]
		if self.mag_confs is not None:
			result['mag_conf'] = self.mag_confs[idx]
		if self.clippings is not None:
			result['clipping'] = self.clippings[idx]
		return result

class AEDataModule(L.LightningDataModule):
	def __init__(self, data_dir: str = '.', file_name: str = 'compressed_profiles.npz', batch_size: int = 32, normalization_strategy: str = 'minmax', nprofiles: int = None):
		super().__init__()
		self.data_dir = data_dir
		self.file_name = file_name
		self.batch_size = batch_size
		self.normalization = False
		self.normalization_strategy = normalization_strategy
		self.nprofiles = nprofiles
		self.data = None

	def prepare_data(self):
		# Check if the file in data_dir exists otherwise create it calling the create_db method
		path = os.path.join(self.data_dir, self.file_name)
		if not os.path.exists(path):
			print(f"The file {self.file_name} does not exist in path {self.data_dir}")
			print("It can be created by calling the right functions in create_db.py.")
			print("See the documentation there for more details")
			# from create_db import create_database_time_instants, divide_by_camera
			# divide_by_camera(data_dir=r"\\share\mp\E5-Praktikanten\Orlandi_Luca\_data\npz_files")
			# create_database_time_instants(os.path.join(self.data_dir, self.file_name))

	def setup(self, stage : str = None):
		# Initialize a seed for all the randomic operations
		generator = torch.Generator().manual_seed(42) # Seed for reproducibility

		if self.data is None:
			# Load the dataset from the specified directory and file
			self.data = np.load(f"{self.data_dir}/{self.file_name}")

		if self.nprofiles is None:
			# Here the data is loaded in the correct format from the specified file
			self.profiles = torch.tensor(self.data['profiles'], dtype=torch.float32)
			self.pids = self.data['pids']
			self.times = self.data['times']
			self.cameras = self.data['cameras'] if 'cameras' in self.data else None
			self.mag_confs = self.data['mag_confs'] if 'mag_confs' in self.data else None
			self.clippings = self.data['clippings'] if 'clippings' in self.data else None

		else:
			indices = torch.randperm(len(self.data['profiles']), generator=generator)[:self.nprofiles]
			self.profiles = torch.tensor(self.data['profiles'][indices], dtype=torch.float32)
			self.pids = self.data['pids'][indices]
			self.times = self.data['times'][indices]
			self.cameras = self.data['cameras'][indices] if 'cameras' in self.data else None
			self.mag_confs = self.data['mag_confs'][indices] if 'mag_confs' in self.data else None
			self.clippings = self.data['clippings'][indices] if 'clippings' in self.data else None

		# Normalize the data
		self.profiles = self.normalize(self.profiles)

		# Initialize the Dataset
		dataset = AEDataset(self.profiles, self.pids, self.times, self.cameras, 
					  self.mag_confs, self.clippings)

		# Create the dataset train, validation, and test splits
		self.train_data, self.val_data, self.test_data = random_split(
			dataset, [0.8, 0.1, 0.1], generator=generator,)
		
		# create also a dataset for the whole data, useful for the evaluation of the model on all the data
		self.full_data = dataset

	def train_dataloader(self):
		return DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True)

	def val_dataloader(self):
		return DataLoader(self.val_data, batch_size=self.batch_size, drop_last=True)

	def test_dataloader(self):
		return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True)

	def full_dataloader(self):
		return DataLoader(self.full_data, batch_size=self.batch_size, drop_last=True)

	def normalize(self, data: torch.Tensor) -> torch.Tensor:
		if self.normalization:
			print("Data is already normalized, returning original data")
			return data
		self.normalization = True
		if self.normalization_strategy == 'minmax':
			self.min_val = data.min()
			self.max_val = data.max()
			return (data - self.min_val) / (self.max_val - self.min_val)
		elif self.normalization_strategy == 'zscore':
			self.mean = data.mean()
			self.std = data.std()
			return (data - self.mean) / self.std
		elif self.normalization_strategy == 'robust':
			self.median = data.median()
			self.iqr = data.quantile(0.75) - data.quantile(0.25)
			return (data - self.median) / self.iqr
		elif self.normalization_strategy == 'none':
			return data
		else:
			raise ValueError(f"Unknown normalization strategy: {self.normalization_strategy}, accepted strategies are: minmax, zscore, robust, none")
		
	def denormalize(self, data: torch.Tensor) -> torch.Tensor:
		self.normalization = False
		if not self.normalization:
			print("Data was not normalized, returning original data")
			return data
		if self.normalization_strategy == 'minmax':
			return data * (self.max_val - self.min_val) + self.min_val
		elif self.normalization_strategy == 'zscore':
			return data * self.std + self.mean
		elif self.normalization_strategy == 'robust':
			return data * self.iqr + self.median
		elif self.normalization_strategy == 'none':
			return data
		else:
			raise ValueError(f"Unknown normalization strategy: {self.normalization_strategy}, accepted strategies are: minmax, zscore, robust, none")

	def divide_by_camera(self, camera_list: list = None) -> dict:
		"""
		Divides the input data by the camera.

		Args:
			camera_list (list, optional): The list of cameras to divide the data in.

		Returns:
			(dict): The modified data dictionary.
		"""
		# Load the raw data from the directory and filename
		data = np.load(f"{self.data_dir}/{self.file_name}")

		# Create the lists to hold the divided data
		profiles = []
		pids = []
		times = []
		camera_ids = []

		# Define camera ranges
		camera_ranges = {
			'1A': (0, 17), '1B': (18, 35), '1C': (36, 53), '1D': (54, 71), '1E': (72, 89),
			'2A': (90, 107), '2B': (108, 125), '2C': (126, 143), '2D': (144, 161), '2E': (162, 179),
			'3A': (180, 197), '3B': (198, 215), '3C': (216, 233), '3D': (234, 251), '3E': (252, 269),
			'4A': (270, 287), '4B': (288, 305), '4C': (306, 323), '4D': (324, 341), '4E': (342, 359)
		}

		#Redefine the profiles arrangements
		if camera_list is None:
			camera_list = list(camera_ranges.keys())
		for profile in data["profiles"]:
			for camera in camera_list:
				if camera not in camera_ranges: # Check if camera is in the defined ranges and skip if not
					continue
				start, end = camera_ranges[camera]
				profiles.append(profile[start:end]) # Append the sliced data for the current camera
				pids.append(data["pids"])
				times.append(data["times"])
				camera_ids.append(camera)

		self.profiles = torch.tensor(profiles, dtype=torch.float32)
		self.pids = np.array(pids)
		self.times = np.array(times)
		self.cameras = np.array(camera_ids)

		# Combine the lists into a dictionary
		self.data = {
			'profiles': np.array(profiles),
			'pids': np.array(pids),
			'times': np.array(times),
			'camera_ids': np.array(camera_ids)
		}

		# save the file to an npz
		print("Saving divided data by camera to npz file...")
		np.savez_compressed(f"{self.data_dir}/db_by_camera.npz", **self.data)
		print("Data saved successfully.")

	def get_pids(self, pid_list: list = None, filename="db_by_pid.npz"):
		"""
		Divides the input data by the pid.

		Args:
			pid_list (list, optional): The list of pids to divide the data in.

		Returns:
			None.
		"""
		if os.path.exists(f"{self.data_dir}/{filename}"):
			print(f"The file {filename} already exists in path {self.data_dir}, loading it...")
			self.data = np.load(f"{self.data_dir}/{filename}")
			print("Data loaded successfully.")
			return
		# Load the raw data from the directory and filename
		data = self.data if self.data is not None else np.load(f"{self.data_dir}/{self.file_name}")
		# Create the lists to hold the divided data
		profiles = []
		pids = []
		times = []
		
		# Define pids
		if pid_list is None:
			pid_list = np.unique(data["pids"])
		for profile, pid, time in zip(data["profiles"], data["pids"], data["times"]):
			if pid in pid_list: # Check if pid is in the defined list and skip if not
				profiles.append(profile) # Append the sliced data for the current pid
				pids.append(pid)
				times.append(time)

		self.profiles = torch.tensor(profiles, dtype=torch.float32)
		self.pids = np.array(pids)
		self.times = np.array(times)
		
		# Combine the lists into a dictionary
		self.data = {
			'profiles': np.array(profiles),
			'pids': np.array(pids),
			'times': np.array(times)
		}

		# save the file to an npz
		print("Saving divided data by pid to npz file...")
		np.savez_compressed(f"{self.data_dir}/{filename}", **self.data)
		print("Data saved successfully.")

	def exclude_pids(self, pid_list: list, filename="db_excluded_pid.npz"):
		"""
		Excludes the specified pids from the input data.

		Args:
			pid_list (list): The list of pids to exclude from the data.

		Returns:
			None
		"""
		if os.path.exists(f"{self.data_dir}/{filename}"):
			print(f"The file {filename} already exists in path {self.data_dir}, loading it...")
			self.data = np.load(f"{self.data_dir}/{filename}")
			print("Data loaded successfully.")
			return
		# Load the raw data from the directory and filename
		data = self.data if self.data is not None else np.load(f"{self.data_dir}/{self.file_name}")
		# Create the lists to hold the divided data
		profiles = []
		pids = []
		times = []
		
		for profile, pid, time in zip(data["profiles"], data["pids"], data["times"]):
			if pid not in pid_list: # Check if pid is in the defined list and skip if it is
				profiles.append(profile) # Append the sliced data for the current pid
				pids.append(pid)
				times.append(time)
		
		self.profiles = torch.tensor(profiles, dtype=torch.float32)
		self.pids = np.array(pids)
		self.times = np.array(times)

		# Combine the lists into a dictionary
		self.data = {
			'profiles': np.array(profiles),
			'pids': np.array(pids),
			'times': np.array(times)
		}

		# save the file to an npz
		print("Saving excluded data by pid to npz file...")
		np.savez_compressed(f"{self.data_dir}/{filename}", **self.data)
		print("Data saved successfully.")

	def cut_time_intervals(self, time_intervals: list, filename="db_cut_time_intervals.npz"):
		"""
		Gets the specified time intervals from the input data.

		Args:
			time_intervals (tuple): this are the start and end time in which the data should be found (start_time, end_time).
		Returns:
			None
		"""
		if os.path.exists(f"{self.data_dir}/{filename}"):
			print(f"The file {filename} already exists in path {self.data_dir}, loading it...")
			self.data = np.load(f"{self.data_dir}/{filename}")
			print("Data loaded successfully.")
			return
		# Load the raw data from the directory and filename
		data = self.data if self.data is not None else np.load(f"{self.data_dir}/{self.file_name}")
		# Create the lists to hold the divided data
		profiles = []
		pids = []
		times = []
		
		for profile, pid, time in zip(data["profiles"], data["pids"], data["times"]):
			if any(start <= time <= end for start, end in time_intervals): # Check if time is in the defined intervals and skip if it is not
				profiles.append(profile) # Append the sliced data for the current time
				pids.append(pid)
				times.append(time)
		# Combine the lists into a dictionary
		self.data = {
			'profiles': np.array(profiles),
			'pids': np.array(pids),
			'times': np.array(times)
		}

		# save the file to an npz
		print("Saving cut data by time intervals to npz file...")
		np.savez_compressed(f"{self.data_dir}/{filename}", **self.data)
		print("Data saved successfully.")