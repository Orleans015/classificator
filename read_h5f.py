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


def read_h5f_directory(h5py_path: str) -> dict:
    """
    Read all .h5f files in the given directory and return a dictionary of datasets.

    - Files ending in _t000, _t090, _t180, _t270 are treated as reference files:
      only the time base (dataset ending in '0') is extracted from them.
    - All other .h5f files have their full XMCTSdata content loaded (excluding 'timedata').

    Parameters
    ----------
    h5py_path : str
        Path to the directory containing the .h5f files.

    Returns
    -------
    dict
        Dictionary mapping dataset names to numpy arrays.
        Includes a 'time_base' key from the reference files.
    """
    REFERENCE_SUFFIXES = ("_t000.h5f", "_t090.h5f", "_t180.h5f", "_t270.h5f")
    data_dict = {}

    for f in os.listdir(h5py_path):
        h5f_file = os.path.join(h5py_path, f)

        # Reference files: extract time base only
        if any(f.endswith(s) for s in REFERENCE_SUFFIXES):
            with h5py.File(h5f_file, "r") as h5f:
                if "XMCTSdata" in h5f:
                    for dataset in h5f["XMCTSdata"].keys():
                        if dataset.endswith("0"):
                            data_dict["time_base"] = h5f["XMCTSdata"][dataset][:]
            continue

        # Data files: extract all datasets except 'timedata'
        if f.endswith(".h5f"):
            with h5py.File(h5f_file, "r") as h5f:
                if "XMCTSdata" in h5f:
                    for dataset in h5f["XMCTSdata"].keys():
                        if dataset != "timedata":
                            data_dict[dataset] = h5f["XMCTSdata"][dataset][:]

    return data_dict


def build_data_array(data_dict: dict) -> np.ndarray:
    """
    Stack all arrays in the data dictionary into a 2D numpy array.

    The resulting shape is (n_channels, n_timepoints), where the first row
    corresponds to 'time_base' and the remaining rows are the diode channels.

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


def load_sxr_data(pid: str, base_path: str = "/home/IPP-HGW/orluca/devel/data/HDF/_data") -> np.ndarray:
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
    data_dict = read_h5f_directory(h5py_path)
    data = build_data_array(data_dict)
    return data


if __name__ == "__main__":
    pid = "20250305.50"
    data = load_sxr_data(pid)
    print(f"Loaded data array with shape: {data.shape}")
    print("First time instant across all channels:")
    for d in data[:, 0]:
        print(d)
