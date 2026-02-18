import os
import numpy as np
from tqdm import tqdm 

def load_sxr_data(directory):
    """
    Load SXR data from .npz files in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing .npz files.

    Returns
    -------
    data : dict
        Dictionary containing the loaded SXR data.
    """
    data = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".npz"):
            file_data = np.load(os.path.join(directory, filename), allow_pickle=True)
            for k in file_data.files:
                data[k] = file_data[k]
    # save the loaded data
    np.savez_compressed(os.path.join(directory, "short_data.npz"), **data)
    return np.load(os.path.join(directory, "short_data.npz"), allow_pickle=True)


def organize_data(data):
	"""
	Organize SXR data into profiles and labels.
	
	Parameters
	----------
	data : dict
		Dictionary containing the loaded SXR data.
	
	Returns
	-------
	profiles : np.ndarray
		Array of SXR profiles.
	labels : np.ndarray
		Array of corresponding labels for the profiles.
	"""
	profiles = data['vCamout'][:, 1:]
	times = data['vCamout'][:, 0]
	return profiles, times

if __name__ == "__main__":
	data = load_sxr_data(r"S:\E5-Praktikanten\Orlandi_Luca\_data\npz_files")
	profiles, times = organize_data(data)