"""
🎵 Audio Processing Demo — Fundamentals with librosa
Chạy: pip install librosa numpy soundfile matplotlib
       python audio_processing.py
"""
import numpy as np
import os


def generate_test_audio(sr=16000, duration=3.0):
    """Generate synthetic speech-like audio for demo (no file needed)."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    # Simulate speech: mix of fundamental frequency + harmonics + noise
    f0 = 150  # Male voice ~120Hz, Female ~220Hz
    audio = (
        0.5 * np.sin(2 * np.pi * f0 * t) +           # Fundamental
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +       # 2nd harmonic
        0.15 * np.sin(2 * np.pi * 3 * f0 * t) +      # 3rd harmonic
        0.1 * np.sin(2 * np.pi * 4 * f0 * t) +       # 4th harmonic
        0.05 * np.random.randn(len(t))                 # Noise
    ).astype(np.float32)
    
    # Add amplitude envelope (speech is not constant volume)
    envelope = np.ones(len(t))
    # Simulate syllables: volume varies
    for i in range(6):
        center = int(len(t) * (i + 0.5) / 6)
        width = int(len(t) / 12)
        start = max(0, center - width)
        end = min(len(t), center + width)
        envelope[start:end] *= 0.8 + 0.2 * np.sin(np.linspace(0, np.pi, end - start))
    
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    return audio, sr


def demo_time_domain():
    """Demonstrate time-domain audio analysis."""
    print("=== 1. Time-Domain Features ===\n")
    
    audio, sr = generate_test_audio()
    duration = len(audio) / sr
    
    print(f"📊 Audio Properties:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Samples: {len(audio):,}")
    print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  dtype: {audio.dtype}")
    
    # RMS Energy (loudness per frame)
    frame_length = 2048
    hop_length = 512
    frames = len(audio) // hop_length
    
    rms_values = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)
    
    rms_array = np.array(rms_values)
    print(f"\n📊 RMS Energy:")
    print(f"  Frames: {len(rms_array)}")
    print(f"  Mean RMS: {rms_array.mean():.4f}")
    print(f"  Max RMS: {rms_array.max():.4f}")
    print(f"  Dynamic range: {20 * np.log10(rms_array.max() / max(rms_array.min(), 1e-10)):.1f} dB")
    
    # Zero Crossing Rate
    zcr_values = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)
        zcr_values.append(zcr)
    
    zcr_array = np.array(zcr_values)
    print(f"\n📊 Zero Crossing Rate:")
    print(f"  Mean ZCR: {zcr_array.mean():.4f}")
    print(f"  High ZCR → unvoiced/noise, Low ZCR → voiced speech")
    
    # Simple VAD
    threshold = rms_array.mean() * 0.5
    speech_frames = rms_array > threshold
    speech_ratio = speech_frames.mean()
    print(f"\n📊 Simple VAD (RMS threshold):")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Speech frames: {speech_frames.sum()}/{len(speech_frames)} ({speech_ratio:.1%})")
    
    return audio, sr


def demo_frequency_domain():
    """Demonstrate frequency-domain analysis."""
    print("\n=== 2. Frequency-Domain Features ===\n")
    
    audio, sr = generate_test_audio()
    
    # FFT on a frame
    frame = audio[1000:3048]  # 2048 samples
    window = np.hanning(len(frame))
    fft = np.fft.rfft(frame * window)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(frame), d=1/sr)
    
    # Find dominant frequencies
    top_indices = np.argsort(magnitude)[-5:][::-1]
    print(f"📊 FFT Analysis (single frame):")
    print(f"  FFT bins: {len(magnitude)}")
    print(f"  Frequency resolution: {freqs[1]:.1f} Hz")
    for idx in top_indices:
        print(f"  Peak: {freqs[idx]:.0f} Hz (magnitude: {magnitude[idx]:.2f})")
    
    # STFT (Short-Time Fourier Transform)
    n_fft = 2048
    hop_length = 512
    
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames))
    
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        windowed = frame * np.hanning(n_fft)
        stft[:, i] = np.abs(np.fft.rfft(windowed))
    
    print(f"\n📊 STFT (Spectrogram):")
    print(f"  Shape: {stft.shape} (freq_bins × time_frames)")
    print(f"  Freq range: 0 - {sr//2} Hz")
    print(f"  Time resolution: {hop_length/sr*1000:.1f} ms")
    
    # Mel Spectrogram (simplified)
    n_mels = 80
    mel_min = 80
    mel_max = sr // 2
    
    # Mel scale conversion
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)
    
    mel_points = np.linspace(hz_to_mel(mel_min), hz_to_mel(mel_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    print(f"\n📊 Mel Scale:")
    print(f"  {n_mels} mel bands from {mel_min} Hz to {mel_max} Hz")
    print(f"  Low freq bands: {hz_points[:3].astype(int)} Hz (narrow — more detail)")
    print(f"  High freq bands: {hz_points[-3:].astype(int)} Hz (wide — less detail)")
    print(f"  → Matches human hearing perception!")
    
    return stft


def demo_mfcc():
    """Demonstrate MFCC computation."""
    print("\n=== 3. MFCC Features ===\n")
    
    audio, sr = generate_test_audio()
    
    # Manual MFCC-like computation (simplified)
    n_fft = 2048
    hop_length = 512
    n_mels = 40
    n_mfcc = 13
    
    # Step 1: STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    power_spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * np.hanning(n_fft)
        power_spec[:, i] = np.abs(np.fft.rfft(frame)) ** 2
    
    # Step 2: Apply DCT (simplified — normally mel filter bank + DCT)
    # Using random projection as DCT substitute for demo
    np.random.seed(42)
    dct_matrix = np.random.randn(n_mfcc, n_fft // 2 + 1) * 0.01
    mfcc = dct_matrix @ np.log(power_spec + 1e-10)
    
    print(f"📊 MFCC Features:")
    print(f"  Shape: {mfcc.shape} (n_mfcc × n_frames)")
    print(f"  = {n_mfcc} coefficients × {n_frames} time frames")
    
    # Delta (first derivative)
    delta = np.diff(mfcc, axis=1)
    delta = np.pad(delta, ((0, 0), (0, 1)), mode='edge')
    
    # Delta-Delta (second derivative)
    delta2 = np.diff(delta, axis=1)
    delta2 = np.pad(delta2, ((0, 0), (0, 1)), mode='edge')
    
    # Stack all
    full_features = np.concatenate([mfcc, delta, delta2], axis=0)
    
    print(f"\n📊 Full Feature Stack:")
    print(f"  MFCC:        {mfcc.shape}")
    print(f"  + Delta:     {delta.shape}")
    print(f"  + Delta²:    {delta2.shape}")
    print(f"  = Combined:  {full_features.shape}")
    print(f"  → {full_features.shape[0]} features × {full_features.shape[1]} frames")


def demo_preprocessing():
    """Demonstrate audio preprocessing techniques."""
    print("\n=== 4. Audio Preprocessing ===\n")
    
    audio, sr = generate_test_audio()
    
    # Pre-emphasis
    pre_emph_coeff = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emph_coeff * audio[:-1])
    print(f"📊 Pre-emphasis (coeff={pre_emph_coeff}):")
    print(f"  Before: energy = {np.sum(audio**2):.4f}")
    print(f"  After:  energy = {np.sum(emphasized**2):.4f}")
    print(f"  → Boosts high frequencies for better consonant recognition")
    
    # Normalization
    normalized = audio / np.max(np.abs(audio))
    print(f"\n📊 Normalization:")
    print(f"  Before: range [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  After:  range [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Trim silence
    threshold = 0.02
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) > 0:
        trimmed = audio[non_silent[0]:non_silent[-1] + 1]
        print(f"\n📊 Trim Silence (threshold={threshold}):")
        print(f"  Before: {len(audio)/sr:.2f}s ({len(audio)} samples)")
        print(f"  After:  {len(trimmed)/sr:.2f}s ({len(trimmed)} samples)")
        print(f"  Removed: {(len(audio)-len(trimmed))/sr:.2f}s")
    
    # Resampling (concept)
    target_sr = 8000
    resample_ratio = target_sr / sr
    resampled_length = int(len(audio) * resample_ratio)
    print(f"\n📊 Resampling:")
    print(f"  {sr} Hz → {target_sr} Hz")
    print(f"  Samples: {len(audio):,} → {resampled_length:,}")
    print(f"  Data reduction: {(1-resample_ratio)*100:.0f}%")
    
    # SNR estimation
    signal_rms = np.sqrt(np.mean(audio ** 2))
    noise = 0.05 * np.random.randn(len(audio)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2))
    snr = 20 * np.log10(signal_rms / noise_rms)
    
    print(f"\n📊 SNR Estimation:")
    print(f"  Signal RMS: {signal_rms:.4f}")
    print(f"  Estimated SNR: {snr:.1f} dB")
    print(f"  Quality: {'Clean' if snr > 20 else 'Moderate' if snr > 10 else 'Noisy'}")


def demo_feature_comparison():
    """Compare different audio feature representations."""
    print("\n=== 5. Feature Comparison ===\n")
    
    audio, sr = generate_test_audio(duration=5.0)
    n_fft = 2048
    hop_length = 512
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    
    features = {
        "Raw Waveform": (1, len(audio)),
        "Spectrogram": (n_fft // 2 + 1, n_frames),
        "Mel Spectrogram (80)": (80, n_frames),
        "MFCC (13)": (13, n_frames),
        "MFCC + Δ + ΔΔ (39)": (39, n_frames),
    }
    
    print(f"  {'Feature':<25} {'Shape':<15} {'Values/s':<12} {'Use Case'}")
    print(f"  {'-'*75}")
    for name, shape in features.items():
        values_per_sec = shape[0] * shape[1] / (len(audio) / sr)
        use_case = {
            "Raw Waveform": "End-to-end models",
            "Spectrogram": "Visualization",
            "Mel Spectrogram (80)": "Modern ASR (Whisper)",
            "MFCC (13)": "Classic ASR, speaker ID",
            "MFCC + Δ + ΔΔ (39)": "Hybrid ASR",
        }
        print(f"  {name:<25} {str(shape):<15} {values_per_sec:<12,.0f} {use_case[name]}")


if __name__ == "__main__":
    print("=" * 60)
    print("🎵 Audio Processing Fundamentals Demo")
    print("=" * 60)
    print("No external audio file needed — generates synthetic speech.\n")
    
    demo_time_domain()
    demo_frequency_domain()
    demo_mfcc()
    demo_preprocessing()
    demo_feature_comparison()
    
    print(f"\n{'='*60}")
    print("✅ Audio Processing Demo completed!")
    print("    Next: Install librosa for full features: pip install librosa")
    print(f"{'='*60}")
