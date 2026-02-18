#!/usr/bin/env python3
"""
Script to download SXR diagnostic data from multiple pulses.
Downloads 50 time steps around the pulse maximum with vCamout and gain settings.

Author: Generated for SXR diagnostic analysis
Date: 2026-02-05
"""

import numpy as np
import sys
import os

# Add the uploaded libraries to path
sys.path.insert(0, '/mnt/user-data/uploads')
import archive_access as a7x
import qxt_archive_lib as qxt


def find_pulse_maximum_window(signal, n_timesteps=50, exclude_fraction=0.15):
    """
    Find the window around the pulse maximum, excluding start/end phases.
    
    Parameters
    ----------
    signal : ndarray
        1D signal array
    n_timesteps : int
        Number of time steps to extract
    exclude_fraction : float
        Fraction of signal to exclude from start and end (default: 0.15 = 15%)
    
    Returns
    -------
    indices : ndarray
        Indices of selected time steps
    max_idx : int
        Index of the maximum value
    """
    n_total = len(signal)
    
    # Exclude start and end regions
    exclude_n = int(n_total * exclude_fraction)
    start_idx = exclude_n
    end_idx = n_total - exclude_n
    
    if end_idx <= start_idx:
        # Signal too short, just use middle section
        start_idx = 0
        end_idx = n_total
    
    # Find maximum in the valid region
    valid_signal = signal[start_idx:end_idx]
    max_in_valid = np.argmax(valid_signal)
    max_idx = start_idx + max_in_valid
    
    # Create window around maximum
    half_window = n_timesteps // 2
    window_start = max(0, max_idx - half_window)
    window_end = min(n_total, max_idx + half_window)
    
    # Adjust if we hit boundaries
    if window_end - window_start < n_timesteps:
        if window_start == 0:
            window_end = min(n_total, n_timesteps)
        else:
            window_start = max(0, n_total - n_timesteps)
    
    # Generate indices uniformly in the window
    indices = np.linspace(window_start, window_end-1, n_timesteps, dtype=int)
    
    return indices, max_idx


def download_sxr_pulses(pulse_ids, diodes, n_timesteps=50, 
                        output_file='sxr_data.npz', 
                        exclude_fraction=0.15,
                        parallel=1, quiet=False, debug=False):
    """
    Download SXR data from multiple pulses, focusing on the pulse maximum.
    
    Parameters
    ----------
    pulse_ids : list of str
        List of pulse IDs (e.g., ['20240101.001', '20240101.002'])
    diodes : array-like
        List of diode numbers to download
    n_timesteps : int
        Number of time steps to extract from each pulse (default: 50)
    output_file : str
        Output filename for .npz file
    exclude_fraction : float
        Fraction of pulse to exclude from start/end (default: 0.15 = 15%)
    parallel : int
        Parallelization level (0=serial with call_tget_1, 1=serial with call_tget_c, 
        >1=parallel processing)
    quiet : bool
        Suppress output messages
    debug : bool
        Enable debug output
    
    Returns
    -------
    dict : Dictionary containing all downloaded data
    """
    
    n_pulses = len(pulse_ids)
    n_diodes = len(diodes)
    
    print(f"\n{'='*70}")
    print(f"Downloading SXR data:")
    print(f"  Number of pulses: {n_pulses}")
    print(f"  Number of diodes: {n_diodes}")
    print(f"  Time steps per pulse: {n_timesteps}")
    print(f"  Excluding {exclude_fraction*100:.0f}% from start/end")
    print(f"  Parallel mode: {parallel}")
    print(f"{'='*70}\n")
    
    # Initialize storage dictionaries
    all_data = {
        'pulse_ids': [],
        'diodes': np.array(diodes),
        'n_timesteps': n_timesteps,
        'vCamout': {},      # Will store vCamout for each pulse
        'vTstart': {},      # Start times for each pulse
        'vTend': {},        # End times for each pulse
        'vGain': {},        # Gain settings for each pulse
        'max_indices': {},  # Index of maximum for each pulse
        'metadata': {
            'fromT': {},
            'uptoT': {},
            'refT1': {}
        }
    }
    
    # Process each pulse
    for i, pid in enumerate(pulse_ids):
        print(f"\n[{i+1}/{n_pulses}] Processing pulse {pid}...")
        
        # Create PulseData object using timeID
        pData = a7x.timeID()
        try:
            pData.init_pid(pid)
        except Exception as e:
            print(f"  ✗ Error initializing pulse {pid}: {str(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            continue
        
        # Get program info to set time ranges
        try:
            pData.set_from_upto()  # This sets fromT, uptoT, refT1 from archive
        except Exception as e:
            print(f"  ✗ Error getting time info for {pid}: {str(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            continue
        
        # Check if we got valid time ranges
        if pData.uptoT == 0 or pData.fromT == 0:
            print(f"  ✗ Invalid time ranges for {pid}")
            continue
        
        print(f"  Time range: {(pData.fromT - pData.refT1)/1e9:.3f} to " +
              f"{(pData.uptoT - pData.refT1)/1e9:.3f} s")
        
        try:
            # Download full data for all diodes
            vCamout, vTstart, vTend, vGain = qxt.tget_qxt_diodes_multi(
                pData=pData,
                diodes=diodes,
                average=0,
                nreduce=None,
                save_hdf=False,
                fpath_hdf='',
                version_hdf='',
                lstep=0,
                parallel=parallel,
                quiet=quiet,
                debug=debug
            )
            
            # vCamout has shape (n_samples, n_diodes+1) where first column is time
            n_samples = vCamout.shape[0]
            
            if n_samples < n_timesteps:
                print(f"  Warning: Only {n_samples} samples available, " +
                      f"requested {n_timesteps}")
                indices = np.arange(n_samples)
                max_idx = np.argmax(np.sum(vCamout[:, 1:], axis=1))
            else:
                # Find window around maximum (use sum of all channels)
                total_signal = np.sum(vCamout[:, 1:], axis=1)
                indices, max_idx = find_pulse_maximum_window(
                    total_signal, 
                    n_timesteps=n_timesteps,
                    exclude_fraction=exclude_fraction
                )
            
            vCamout_sampled = vCamout[indices, :]
            
            # Store data
            all_data['pulse_ids'].append(pid)
            all_data['vCamout'][pid] = vCamout_sampled
            all_data['vTstart'][pid] = vTstart
            all_data['vTend'][pid] = vTend
            all_data['vGain'][pid] = vGain
            all_data['max_indices'][pid] = max_idx
            all_data['metadata']['fromT'][pid] = pData.fromT
            all_data['metadata']['uptoT'][pid] = pData.uptoT
            all_data['metadata']['refT1'][pid] = pData.refT1
            
            print(f"  ✓ Downloaded {len(indices)} time steps")
            print(f"  ✓ Maximum at t = {vCamout[max_idx, 0]:.6f} s")
            print(f"  ✓ Sampled window: {vCamout_sampled[0,0]:.6f} to " +
                  f"{vCamout_sampled[-1,0]:.6f} s")
            print(f"  ✓ Gain settings: {vGain}")
            
        except Exception as e:
            print(f"  ✗ Error downloading pulse {pid}: {str(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            continue
    
    # Save to npz file
    if len(all_data['pulse_ids']) == 0:
        print(f"\n{'='*70}")
        print("ERROR: No data was successfully downloaded!")
        print("Please check:")
        print("  - Pulse IDs are correct and exist in archive")
        print("  - Archive connection is working")
        print("  - Time ranges are valid")
        print(f"{'='*70}\n")
        return all_data
    
    print(f"\n{'='*70}")
    print(f"Saving data to {output_file}...")
    print(f"Successfully downloaded {len(all_data['pulse_ids'])} out of {n_pulses} pulses")
    
    # Prepare data for saving (npz doesn't handle nested dicts well)
    save_dict = {
        'pulse_ids': np.array(all_data['pulse_ids'], dtype=str),
        'diodes': all_data['diodes'],
        'n_timesteps': all_data['n_timesteps'],
    }
    
    # Add vCamout, vGain, etc. for each pulse
    for pid in all_data['pulse_ids']:
        safe_pid = pid.replace('.', '_')  # npz keys can't have dots
        save_dict[f'vCamout_{safe_pid}'] = all_data['vCamout'][pid]
        save_dict[f'vTstart_{safe_pid}'] = all_data['vTstart'][pid]
        save_dict[f'vTend_{safe_pid}'] = all_data['vTend'][pid]
        save_dict[f'vGain_{safe_pid}'] = all_data['vGain'][pid]
        save_dict[f'max_idx_{safe_pid}'] = all_data['max_indices'][pid]
        save_dict[f'fromT_{safe_pid}'] = all_data['metadata']['fromT'][pid]
        save_dict[f'uptoT_{safe_pid}'] = all_data['metadata']['uptoT'][pid]
        save_dict[f'refT1_{safe_pid}'] = all_data['metadata']['refT1'][pid]
    
    np.savez_compressed(output_file, **save_dict)
    
    print(f"✓ Data saved successfully!")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")
    
    return all_data


def load_sxr_data(filename='sxr_data.npz'):
    """
    Load previously downloaded SXR data.
    
    Parameters
    ----------
    filename : str
        Path to .npz file
    
    Returns
    -------
    dict : Dictionary containing the loaded data
    """
    data = np.load(filename, allow_pickle=True)
    
    pulse_ids = data['pulse_ids']
    diodes = data['diodes']
    n_timesteps = int(data['n_timesteps'])
    
    # Reconstruct the data structure
    loaded_data = {
        'pulse_ids': pulse_ids,
        'diodes': diodes,
        'n_timesteps': n_timesteps,
        'vCamout': {},
        'vTstart': {},
        'vTend': {},
        'vGain': {},
        'max_indices': {},
        'metadata': {
            'fromT': {},
            'uptoT': {},
            'refT1': {}
        }
    }
    
    for pid in pulse_ids:
        safe_pid = pid.replace('.', '_')
        loaded_data['vCamout'][pid] = data[f'vCamout_{safe_pid}']
        loaded_data['vTstart'][pid] = data[f'vTstart_{safe_pid}']
        loaded_data['vTend'][pid] = data[f'vTend_{safe_pid}']
        loaded_data['vGain'][pid] = data[f'vGain_{safe_pid}']
        loaded_data['max_indices'][pid] = int(data[f'max_idx_{safe_pid}'])
        loaded_data['metadata']['fromT'][pid] = int(data[f'fromT_{safe_pid}'])
        loaded_data['metadata']['uptoT'][pid] = int(data[f'uptoT_{safe_pid}'])
        loaded_data['metadata']['refT1'][pid] = int(data[f'refT1_{safe_pid}'])
    
    return loaded_data


# Example usage
if __name__ == "__main__":
    
    print("SXR Data Download Tool")
    print("="*70)
    print("\nThis script downloads SXR data from W7-X archive, focusing on")
    print("the pulse maximum and excluding the start/end phases.\n")
    
    print("Example usage:")
    print("-"*70)
    print("\n# Simple: just provide pulse IDs (automatic time range detection)")
    print("pulse_ids = ['20240101.001', '20240101.002', '20240101.003']")
    print("diodes = np.arange(0, 360, dtype=np.int32)  # All cameras")
    print("")
    print("data = download_sxr_pulses(")
    print("    pulse_ids=pulse_ids,")
    print("    diodes=diodes,")
    print("    n_timesteps=50,         # Number of time steps around maximum")
    print("    exclude_fraction=0.15,  # Exclude 15% from start/end")
    print("    output_file='sxr_data.npz',")
    print("    parallel=1              # Serial processing")
    print(")")
    print("")
    print("# Load saved data")
    print("data = load_sxr_data('sxr_data.npz')")
    print("")
    print("# Access data")
    print("pid = '20240101.001'")
    print("vCamout = data['vCamout'][pid]  # shape: (50, n_diodes+1)")
    print("time = vCamout[:, 0]            # Time vector")
    print("signals = vCamout[:, 1:]        # Signal for all diodes")
    print("gains = data['vGain'][pid]      # Gain settings for each diode")
    print("max_idx = data['max_indices'][pid]  # Where maximum occurred")
    print("")
    print("-"*70)
    print("\nNote: The script automatically:")
    print("  - Fetches time ranges from archive using pulse ID")
    print("  - Finds the pulse maximum (sum of all channels)")
    print("  - Extracts 50 time steps centered on the maximum")
    print("  - Excludes 15% from start and end to avoid ramp-up/down phases")
