import os 
import torch
import numpy as np
import pandas as pd

from dataset import AEDataModule

from model import AutoEncoder

import seaborn as sns
import matplotlib.pyplot as plt

from views import GAIN_PIDS as gain_list

# set the theme for the plots
sns.set_theme(context="notebook", style="whitegrid",
              palette="muted", font_scale=1.2)

def plot_profiles(profiles, reconstructions, pids, times, latents, num_samples=1024):
    # Plot the original profiles and their latent representations
    for i in range(num_samples, num_samples+50):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(profiles[i], label='Original Profile')
        plt.plot(reconstructions[i], label='Reconstructed Profile', linestyle='--')
        plt.title(f'Profile for PID {pids[i].item()} at time {times[i].item()}')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.bar(range(len(latents[i])), latents[i], label='Latent Representation')
        plt.title(f'Latent Representation for PID {pids[i].item()} at time {times[i].item()}')
        plt.xlabel('Latent Dimension Index')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"images/profile_{i}.png")
        plt.close()

def get_model_and_data(data_dir, file_name, model_path):
    # Initialize the data module
    data_module = AEDataModule(data_dir, file_name, batch_size=2048, normalization_strategy='minmax')
    data_module.prepare_data()
    data_module.setup()
    data_module.exclude_pids(gain_list)
    data_module.cut_time_intervals((2, 10)) # only keep profiles from 2 to 10 seconds of the campaign

    # Load the model
    model = AutoEncoder.load_from_checkpoint(model_path)
    model.eval()

    return model, data_module

def get_profiles_and_latents(model, data_module):
    # Get the full dataloader
    dataloader = data_module.full_dataloader()

    # Get a batch of data
    batch = next(iter(dataloader))
    profiles = batch['profile']
    pids = batch['pid']
    times = batch['time']

    # Move the profiles to the device and get the latent representations
    device = "cpu"
    profiles = profiles.to(device)
    latents = model.encoder(profiles).detach().cpu().numpy()
    reconstructions = model(profiles).detach().cpu().numpy()

    # Convert to numpy arrays for easier indexing
    profiles = profiles.cpu().numpy()
    return profiles, reconstructions, pids, times, latents
 
if __name__ == "__main__":
    data_dir = "../data"
    file_name = "20251028_all_data.npz"
    model_path = "/home/IPP-HGW/orluca/devel/classificator/W7-X_QXT/AE360/version_2/best_model_.ckpt"

    model, data_module = get_model_and_data(data_dir, file_name, model_path)
    profiles, reconstructions, pids, times, latents = get_profiles_and_latents(model, data_module)
    plot_profiles(profiles, reconstructions, pids, times, latents, num_samples=50)