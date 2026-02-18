#!/usr/bin/env python3
"""
Neural Network Anomaly Detector for SXR Diagnostics

This module implements an autoencoder-based anomaly detection system that:
1. Trains on "good" (fault-free) SXR data to learn normal patterns
2. Detects faults by measuring reconstruction error
3. Classifies signals as clean or dirty based on learned baseline

The autoencoder learns to compress and reconstruct normal signals. When it
encounters faulty data (wrong gains, broken channels, etc.), it produces high
reconstruction error, flagging the anomaly.

Author: Generated for SXR diagnostic analysis
Date: 2026-02-09
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")


class SXRAutoencoder:
    """
    Autoencoder-based anomaly detector for SXR diagnostics.
    
    Architecture:
    - Encoder: Compresses input to latent representation
    - Decoder: Reconstructs input from latent space
    - Training: Only on good data
    - Detection: High reconstruction error = anomaly
    """
    
    def __init__(self, n_channels=360, n_timesteps=50, latent_dim=32):
        """
        Initialize the autoencoder.
        
        Parameters
        ----------
        n_channels : int
            Number of SXR channels (diodes)
        n_timesteps : int
            Number of time steps in each sample
        latent_dim : int
            Dimension of the latent (compressed) representation
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow")
        
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.latent_dim = latent_dim
        self.input_shape = (n_timesteps, n_channels)
        
        self.scaler = StandardScaler()
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.history = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the autoencoder architecture."""
        
        # Input
        input_layer = layers.Input(shape=self.input_shape)
        
        # Encoder
        x = layers.Flatten()(input_layer)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        encoded = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        # Decoder
        x = layers.Dense(128, activation='relu')(encoded)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(self.n_timesteps * self.n_channels, activation='linear')(x)
        decoded = layers.Reshape(self.input_shape)(x)
        
        # Full autoencoder
        self.model = models.Model(input_layer, decoded, name='autoencoder')
        self.encoder = models.Model(input_layer, encoded, name='encoder')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"\nAutoencoder Architecture:")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Total parameters: {self.model.count_params():,}")
    
    def prepare_data(self, data_dict, normalize_per_channel=True):
        """
        Prepare training data from downloaded SXR data.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary from load_sxr_data() or download_sxr_pulses()
        normalize_per_channel : bool
            Normalize each channel independently (recommended)
        
        Returns
        -------
        X : ndarray
            Prepared data array (n_samples, n_timesteps, n_channels)
        """
        samples = []
        
        for pid in data_dict['pulse_ids']:
            vCamout = data_dict['vCamout'][pid]
            # Extract signals (skip time column)
            signals = vCamout[:, 1:]  # Shape: (n_timesteps, n_channels)
            
            if normalize_per_channel:
                # Normalize each channel to [0, 1]
                signals_norm = np.zeros_like(signals)
                for i in range(signals.shape[1]):
                    sig = signals[:, i]
                    sig_min, sig_max = sig.min(), sig.max()
                    if sig_max - sig_min > 1e-10:
                        signals_norm[:, i] = (sig - sig_min) / (sig_max - sig_min)
                signals = signals_norm
            
            samples.append(signals)
        
        X = np.array(samples)
        return X
    
    def train(self, X_train, validation_split=0.2, epochs=100, batch_size=32,
              early_stopping_patience=15, verbose=1):
        """
        Train the autoencoder on good (fault-free) data.
        
        Parameters
        ----------
        X_train : ndarray
            Training data (n_samples, n_timesteps, n_channels)
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        early_stopping_patience : int
            Stop training if validation loss doesn't improve
        verbose : int
            Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns
        -------
        history : keras.History
            Training history
        """
        
        # Standardize the data
        original_shape = X_train.shape
        X_flat = X_train.reshape(X_train.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        X_train = X_scaled.reshape(original_shape)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        print("\nTraining autoencoder on good data...")
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        # Calculate reconstruction error threshold on training data
        train_reconstructions = self.model.predict(X_train, verbose=0)
        train_errors = np.mean((X_train - train_reconstructions) ** 2, axis=(1, 2))
        
        # Set threshold at 95th percentile of training errors
        self.threshold = np.percentile(train_errors, 95)
        
        print(f"\nTraining complete!")
        print(f"  Reconstruction error threshold: {self.threshold:.6f}")
        print(f"  (95th percentile of training data)")
        
        return self.history
    
    def predict_anomaly(self, X, return_details=False):
        """
        Detect anomalies in new data.
        
        Parameters
        ----------
        X : ndarray
            Data to check (n_samples, n_timesteps, n_channels)
        return_details : bool
            If True, return reconstruction errors and reconstructions
        
        Returns
        -------
        is_anomaly : ndarray
            Boolean array indicating anomalies (True = fault detected)
        [Optional] errors : ndarray
            Reconstruction errors for each sample
        [Optional] reconstructions : ndarray
            Reconstructed signals
        """
        
        # Standardize
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Reconstruct
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction error (MSE per sample)
        errors = np.mean((X_scaled - reconstructions) ** 2, axis=(1, 2))
        
        # Classify as anomaly if error > threshold
        is_anomaly = errors > self.threshold
        
        if return_details:
            # Inverse transform reconstructions for visualization
            recon_flat = reconstructions.reshape(reconstructions.shape[0], -1)
            recon_original = self.scaler.inverse_transform(recon_flat)
            recon_original = recon_original.reshape(original_shape)
            return is_anomaly, errors, recon_original
        
        return is_anomaly
    
    def predict_anomaly_scores(self, X):
        """
        Get anomaly scores (normalized reconstruction errors).
        
        Parameters
        ----------
        X : ndarray
            Data to score
        
        Returns
        -------
        scores : ndarray
            Anomaly scores (0 = normal, >1 = anomaly)
        """
        is_anomaly, errors, _ = self.predict_anomaly(X, return_details=True)
        
        # Normalize by threshold
        scores = errors / self.threshold
        
        return scores
    
    def get_latent_representation(self, X):
        """
        Get compressed (latent) representation of data.
        
        Parameters
        ----------
        X : ndarray
            Input data
        
        Returns
        -------
        latent : ndarray
            Latent representations (n_samples, latent_dim)
        """
        # Standardize
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Encode
        latent = self.encoder.predict(X_scaled, verbose=0)
        
        return latent
    
    def save(self, filepath='sxr_autoencoder.keras'):
        """Save the trained model."""
        # Save Keras model
        self.model.save(filepath)
        
        # Save additional attributes
        metadata = {
            'n_channels': self.n_channels,
            'n_timesteps': self.n_timesteps,
            'latent_dim': self.latent_dim,
            'threshold': self.threshold,
            'scaler': self.scaler
        }
        
        with open(filepath.replace('.keras', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='sxr_autoencoder.keras'):
        """Load a trained model."""
        # Load Keras model
        self.model = keras.models.load_model(filepath)
        
        # Rebuild encoder
        self.encoder = models.Model(
            self.model.input,
            self.model.get_layer('latent').output
        )
        
        # Load metadata
        with open(filepath.replace('.keras', '_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.n_channels = metadata['n_channels']
        self.n_timesteps = metadata['n_timesteps']
        self.latent_dim = metadata['latent_dim']
        self.threshold = metadata['threshold']
        self.scaler = metadata['scaler']
        
        print(f"Model loaded from {filepath}")


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation loss."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Autoencoder Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    
    return fig


def plot_anomaly_detection(X_test, is_anomaly, errors, threshold,
                           pulse_ids=None, save_path='anomaly_detection.png'):
    """
    Visualize anomaly detection results.
    
    Parameters
    ----------
    X_test : ndarray
        Test data
    is_anomaly : ndarray
        Anomaly flags
    errors : ndarray
        Reconstruction errors
    threshold : float
        Detection threshold
    pulse_ids : list, optional
        Pulse IDs for labeling
    save_path : str
        Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Reconstruction errors
    ax = axes[0, 0]
    x = np.arange(len(errors))
    colors = ['red' if a else 'green' for a in is_anomaly]
    ax.scatter(x, errors, c=colors, alpha=0.6)
    ax.axhline(y=threshold, color='black', linestyle='--', 
               label=f'Threshold = {threshold:.6f}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Anomaly Detection Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error histogram
    ax = axes[0, 1]
    normal_errors = errors[~is_anomaly]
    anomaly_errors = errors[is_anomaly]
    
    if len(normal_errors) > 0:
        ax.hist(normal_errors, bins=30, alpha=0.6, color='green', label='Normal')
    if len(anomaly_errors) > 0:
        ax.hist(anomaly_errors, bins=30, alpha=0.6, color='red', label='Anomaly')
    ax.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Example normal signal
    ax = axes[1, 0]
    if len(normal_errors) > 0:
        normal_idx = np.where(~is_anomaly)[0][0]
        # Plot a few channels
        for i in range(0, min(10, X_test.shape[2]), 2):
            ax.plot(X_test[normal_idx, :, i], alpha=0.7, label=f'Ch {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Signal')
        ax.set_title(f'Example Normal Signal (Error: {errors[normal_idx]:.6f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Example anomaly signal
    ax = axes[1, 1]
    if len(anomaly_errors) > 0:
        anomaly_idx = np.where(is_anomaly)[0][0]
        for i in range(0, min(10, X_test.shape[2]), 2):
            ax.plot(X_test[anomaly_idx, :, i], alpha=0.7, label=f'Ch {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Signal')
        ax.set_title(f'Example Anomaly Signal (Error: {errors[anomaly_idx]:.6f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Anomaly detection plot saved to {save_path}")
    
    return fig


def create_faulty_data(X_clean, fault_type='gain', fault_fraction=0.3):
    """
    Create synthetic faulty data for testing.
    
    Parameters
    ----------
    X_clean : ndarray
        Clean data
    fault_type : str
        Type of fault: 'gain', 'dead', 'noisy', 'offset'
    fault_fraction : float
        Fraction of channels to corrupt
    
    Returns
    -------
    X_faulty : ndarray
        Data with injected faults
    """
    X_faulty = X_clean.copy()
    n_channels = X_clean.shape[2]
    n_faulty = int(n_channels * fault_fraction)
    
    # Randomly select channels to corrupt
    faulty_channels = np.random.choice(n_channels, n_faulty, replace=False)
    
    for ch in faulty_channels:
        if fault_type == 'gain':
            # Random gain error (0.3x to 3x)
            gain_error = np.random.uniform(0.3, 3.0)
            if np.random.rand() > 0.5:
                gain_error = 1.0 / gain_error
            X_faulty[:, :, ch] *= gain_error
            
        elif fault_type == 'dead':
            # Dead channel
            X_faulty[:, :, ch] = 0
            
        elif fault_type == 'noisy':
            # Add excessive noise
            noise = np.random.normal(0, 0.3 * X_faulty[:, :, ch].std(), 
                                    X_faulty[:, :, ch].shape)
            X_faulty[:, :, ch] += noise
            
        elif fault_type == 'offset':
            # Constant offset
            offset = np.random.uniform(-0.5, 0.5)
            X_faulty[:, :, ch] += offset
    
    return X_faulty


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SXR ANOMALY DETECTION - NEURAL NETWORK CLASSIFIER")
    print("="*70)
    
    if not HAS_TENSORFLOW:
        print("\nERROR: TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        exit(1)
    
    print("\nThis module provides an autoencoder-based anomaly detector.")
    print("\nKey features:")
    print("  - Trains only on good (fault-free) data")
    print("  - Learns normal signal patterns")
    print("  - Detects faults via reconstruction error")
    print("  - Handles multiple fault types (gain, dead channels, noise, etc.)")
    
    print("\n" + "="*70)
    print("EXAMPLE WORKFLOW:")
    print("="*70)
    print("""
# 1. Load your clean data
from download_sxr_data import load_sxr_data
data = load_sxr_data('sxr_clean_data.npz')

# 2. Create and train detector
detector = SXRAutoencoder(n_channels=360, n_timesteps=50, latent_dim=32)
X_train = detector.prepare_data(data)
history = detector.train(X_train, epochs=100, batch_size=32)

# 3. Save the trained model
detector.save('sxr_detector.keras')

# 4. Test on new data
data_test = load_sxr_data('sxr_test_data.npz')
X_test = detector.prepare_data(data_test)
is_anomaly = detector.predict_anomaly(X_test)

print(f"Detected {np.sum(is_anomaly)} anomalies out of {len(is_anomaly)} samples")

# 5. Visualize results
is_anomaly, errors, reconstructions = detector.predict_anomaly(X_test, return_details=True)
plot_anomaly_detection(X_test, is_anomaly, errors, detector.threshold)
    """)
    
    print("="*70)