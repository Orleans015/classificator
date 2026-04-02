import os
import h5py
import numpy as np


def format_pid(pid: str) -> str:
    """
    Format a program ID (e.g. '20250305.50') into a zero-padded directory name
    (e.g. '20250305_050').
    """
    head, sep, tail = pid.strip().rpartition('.')
    if sep:
        frac = tail
        if frac.isdigit() and len(frac) < 3:
            frac = frac.zfill(3)
        return f"{head}_{frac}"
    return pid.replace('.', '_')


def read_h5f_directory(h5py_path: str, decimate: int = 100) -> dict:
    """
    Read all .h5f files in the given directory and return a dictionary of datasets.

    - Files ending in _t000, _t090, _t180, _t270 are treated as reference files:
      only the time base (dataset ending in '0') is extracted from them.
    - All other .h5f files have their full XMCTSdata content loaded (excluding 'timedata').

    Parameters
    ----------
    h5py_path : str
        Path to the directory containing the .h5f files.
    decimate  : int
        Takes data every n = decimate entries

    Returns
    -------
    dict
        Dictionary mapping dataset names to numpy arrays.
        Includes a 'timedata' key from the reference files.
    """
    REFERENCE_SUFFIXES = ("_t000.h5f", "_t090.h5f", "_t180.h5f", "_t270.h5f")
    data_dict = {}
    length = np.inf

    # Check if the data has consistent length, otherwise cut it to the smallest length
    for f in os.listdir(h5py_path):
        # if there is adjusted in the name skip it
        if f.__contains__('adjusted'): continue 
        h5f_file = os.path.join(h5py_path, f)
        if f.endswith(".h5f"):
            with h5py.File(h5f_file, "r") as h5f:
                if "XMCTSdata" in h5f:
                    for dataset in h5f["XMCTSdata"].keys():
                        if dataset != "timedata":
                            length_tmp = h5f["XMCTSdata"][dataset][:].shape[0]
                            if length_tmp < length:
                                length = length_tmp

    for f in sorted(os.listdir(h5py_path)):
        h5f_file = os.path.join(h5py_path, f)
        # if there is adjusted in the name skip it
        if f.__contains__('adjusted'): continue

        # Reference files: extract time base only
        if any(f.endswith(s) for s in REFERENCE_SUFFIXES):
            with h5py.File(h5f_file, "r") as h5f:
                if "XMCTSdata" in h5f:
                    for dataset in h5f["XMCTSdata"].keys():
                        if dataset.endswith("000"):
                            data_dict["timedata"] = h5f["XMCTSdata"][dataset][:length:decimate]
            continue

        # Data files: extract all datasets except 'timedata'
        if f.endswith(".h5f"):
            with h5py.File(h5f_file, "r") as h5f:
                if "XMCTSdata" in h5f:
                    for dataset in h5f["XMCTSdata"].keys():
                        if dataset != "timedata":
                            data_dict[dataset] = h5f["XMCTSdata"][dataset][:length:decimate]

    return data_dict


def save_tfiles(h5py_path: str, keyword: str = 'adjusted'):
    """
    Duplicate specific HDF5 files in a directory with a modified naming convention.

    Iterates through a directory to find files ending in specific suffixes  
    (0, 90, 180, 270). For each match, it creates a copy where the middle 
    section of the filename is replaced by the provided keyword.

    Parameters
    ----------
    h5py_path : str
        The system path to the directory containing the .h5f files.
    keyword : str, optional
        The string to insert into the new filename (default is 'adjusted').
        This replaces the variable middle segment of the original filename.

    Returns
    -------
    None
        Copies are created directly on the filesystem.
    """
    import shutil
    import os

    REFERENCE_SUFFIXES = ("_t000.h5f", "_t090.h5f", "_t180.h5f", "_t270.h5f")
    for f in sorted(os.listdir(h5py_path)):
        if f.endswith(REFERENCE_SUFFIXES):
            h5f_file = os.path.join(h5py_path, f)
            
            # Split and rebuild filename logic
            parts = f.split('_')
            exp_num = parts[1][:3]
            new_f = f"{parts[0]}_{exp_num}{keyword}_{parts[-1]}"
            
            new_h5f_file = os.path.join(h5py_path, new_f)
            shutil.copy2(h5f_file, new_h5f_file)


def build_data_array(data_dict: dict) -> np.ndarray:
    """
    Stack all arrays in the data dictionary into a 2D numpy array.

    The resulting shape is (n_channels, n_timepoints), where the first row
    corresponds to 'timedata' and the remaining rows are the diode channels.

    Parameters
    ----------
    data_dict : dict
        Dictionary as returned by read_h5f_directory().

    Returns
    -------
    np.ndarray
        2D array of shape (n_channels, n_timepoints).
    """
    return np.array(list(data_dict.values()))


def sort_sxr_data(data: np.ndarray) -> np.ndarray:
    """
    Organize the stacked arrays so that the time base is sorted.
    
    Parameters
    ----------
    data : np.ndarray
        array returned by build_data_array()
        
    Returns
    -------
    np.ndarray
        Sorted 2D array of shape (n_channels, n_timepoints).
    """
    sorted_indices = np.argsort(data[:, 0])
    return data[sorted_indices, :]


def get_sxr_data():
    raise NotImplementedError


def load_sxr_data(pid: str, base_path: str = "/home/IPP-HGW/orluca/devel/data/HDF/_data/OP2",
                  decimate: int = 1) -> np.ndarray:
    """
    Full pipeline: given a program ID, load all SXR diode data and return
    a 2D numpy array of shape (n_channels, n_timepoints).

    Parameters
    ----------
    pid : str
        Program ID, e.g. '20250305.50'.
    base_path : str
        Root directory where the HDF data folders are stored.

    Returns
    -------
    np.ndarray
        2D array of shape (n_channels, n_timepoints).
    """
    pid_formatted = format_pid(pid)
    h5py_path = os.path.join(base_path, pid_formatted, "")
    data_dict = read_h5f_directory(h5py_path, decimate)
    data = build_data_array(data_dict)
    return data, h5py_path, data_dict


if __name__ == "__main__":
    pid = "20221019.25"
    data, h5py_path, data_dict = load_sxr_data(pid)
    print(f"Loaded data array with shape: {data.shape}")
    print(f"H5F path: {h5py_path}")
    print("First time instant across all channels:")
    for d in data[:, 0]:
        print(d)
