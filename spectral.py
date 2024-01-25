import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

def spectral_subtraction(noisy_audio, alpha=2.0):
    # Calculate magnitude spectrum of the noisy audio
    noisy_spectrum = np.abs(librosa.stft(noisy_audio))

    # Calculate noise profile (average magnitude spectrum)
    noise_spectrum = np.mean(noisy_spectrum, axis=1)

    # Perform spectral subtraction
    clean_spectrum = np.maximum(noisy_spectrum - alpha * np.expand_dims(noise_spectrum, axis=1), 0)

    # Synthesize clean audio
    clean_audio = librosa.istft(clean_spectrum)

    return clean_audio

# Load the noisy audio
noisy_audio, sr = librosa.load('/content/drive/MyDrive/audios/iru2_05012024.wav', sr=None)

# Apply spectral subtraction
clean_audio = spectral_subtraction(noisy_audio)

# Plot the original and denoised waveforms
plt.figure(figsize=(12, 8))

# Original audio waveform
plt.subplot(2, 1, 1)
plt.title('Original Audio Waveform')
librosa.display.waveshow(noisy_audio, sr=sr)

# Denoised audio waveform
plt.subplot(2, 1, 2)
plt.title('Denoised Audio Waveform')
librosa.display.waveshow(clean_audio, sr=sr)

plt.tight_layout()
plt.show()

# Play the original audio
Audio(noisy_audio, rate=sr)

# Play the denoised audio
Audio(clean_audio, rate=sr)
