"""
🎤 Whisper ASR Demo — Speech-to-Text transcription
Chạy: pip install openai-whisper torch
       python whisper_demo.py

NOTE: Generates synthetic audio for demo. Replace with real audio for actual use.
"""
import numpy as np
import os
import sys
import time


def generate_speech_like_audio(sr=16000, duration=5.0):
    """Generate speech-like audio with varying frequencies (simulates speech)."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    # Multiple formants (speech-like frequencies)
    formants = [
        (200, 0.4),   # F0 (fundamental)
        (800, 0.3),   # F1
        (1200, 0.2),  # F2
        (2500, 0.1),  # F3
    ]
    
    audio = np.zeros(len(t), dtype=np.float32)
    for freq, amp in formants:
        # Add vibrato for more natural sound
        vibrato = 5 * np.sin(2 * np.pi * 5 * t)
        audio += amp * np.sin(2 * np.pi * (freq + vibrato) * t)
    
    # Amplitude envelope (syllable-like)
    segments = 8
    envelope = np.zeros(len(t))
    for i in range(segments):
        center = int(len(t) * (i + 0.5) / segments)
        width = int(len(t) / (segments * 2.5))
        start = max(0, center - width)
        end = min(len(t), center + width)
        envelope[start:end] = np.sin(np.linspace(0, np.pi, end - start))
    
    audio *= envelope
    audio += 0.01 * np.random.randn(len(t)).astype(np.float32)  # Light noise
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return audio, sr


def demo_whisper_concepts():
    """Demonstrate Whisper ASR concepts without requiring model download."""
    print("=== 1. Whisper Architecture Concepts ===\n")
    
    print("📊 Whisper Pipeline:")
    print("  Audio → Log Mel Spectrogram (80 × 3000) → Encoder → Decoder → Text\n")
    
    models = [
        ("tiny",     "39M",   "~1 GB",  "~32x", "~10%"),
        ("base",     "74M",   "~1 GB",  "~16x", "~7%"),
        ("small",    "244M",  "~2 GB",  "~6x",  "~5%"),
        ("medium",   "769M",  "~5 GB",  "~2x",  "~4%"),
        ("large-v3", "1.55B", "~10 GB", "~1x",  "~3%"),
        ("turbo",    "809M",  "~6 GB",  "~8x",  "~3.5%"),
    ]
    
    print(f"  {'Model':<12} {'Params':<8} {'VRAM':<8} {'Speed':<8} {'WER (en)'}")
    print(f"  {'-'*50}")
    for name, params, vram, speed, wer in models:
        marker = " ⭐" if name in ("large-v3", "turbo") else ""
        print(f"  {name:<12} {params:<8} {vram:<8} {speed:<8} {wer}{marker}")
    
    print(f"\n  💡 Recommendations:")
    print(f"     Best quality:     large-v3")
    print(f"     Best speed/quality: turbo")
    print(f"     Edge/mobile:      tiny or base")


def demo_mel_spectrogram():
    """Generate and analyze a Mel spectrogram."""
    print("\n=== 2. Mel Spectrogram (Whisper Input) ===\n")
    
    audio, sr = generate_speech_like_audio()
    
    # STFT parameters (Whisper uses these)
    n_fft = 400       # 25ms at 16kHz
    hop_length = 160   # 10ms at 16kHz
    n_mels = 80        # Whisper uses 80 mel bands
    
    # Compute STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames))
    
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * np.hanning(n_fft)
        stft[:, i] = np.abs(np.fft.rfft(windowed)) ** 2
    
    print(f"📊 Spectrogram Shape: {stft.shape}")
    print(f"   Frequency bins: {stft.shape[0]}")
    print(f"   Time frames: {stft.shape[1]}")
    print(f"   Frame duration: {hop_length/sr*1000:.1f}ms")
    
    # Whisper normalizes to 30-second chunks
    whisper_frames = 3000  # 30s * 100 frames/s
    print(f"\n📊 Whisper Input Format:")
    print(f"   Shape: ({n_mels}, {whisper_frames})")
    print(f"   = 80 mel bands × 3000 time frames (30 seconds)")
    print(f"   Padded: our audio has {n_frames} frames → pad to {whisper_frames}")


def demo_transcription_simulation():
    """Simulate the transcription process."""
    print("\n=== 3. Transcription Process ===\n")
    
    audio, sr = generate_speech_like_audio()
    
    # Simulate Whisper's pipeline
    print("📊 Processing Pipeline:")
    
    steps = [
        ("Load audio", 0.1, f"Duration: {len(audio)/sr:.1f}s, SR: {sr}Hz"),
        ("Resample to 16kHz", 0.05, "Already at 16kHz ✓"),
        ("Pad/trim to 30s", 0.05, f"Pad {len(audio)/sr:.1f}s → 30.0s"),
        ("Compute log Mel spectrogram", 0.2, "Shape: (80, 3000)"),
        ("Encode (Transformer encoder)", 0.8, "12-32 layers, multi-head attention"),
        ("Detect language", 0.1, "Detected: Vietnamese (0.95)"),
        ("Decode (autoregressive)", 1.5, "Beam search, temperature=0"),
        ("Apply timestamps", 0.1, "Word-level alignment"),
    ]
    
    total_time = 0
    for step_name, step_time, detail in steps:
        print(f"  {'✓':>3} {step_name:<35} [{step_time*1000:.0f}ms] {detail}")
        total_time += step_time
        time.sleep(0.05)
    
    print(f"\n  Total: {total_time*1000:.0f}ms ({total_time/len(audio)*sr:.1f}x real-time)")
    
    # Simulated output
    result = {
        "text": "Xin chào, đây là bản demo của Whisper ASR",
        "language": "vi",
        "language_probability": 0.95,
        "segments": [
            {"start": 0.0, "end": 1.2, "text": "Xin chào,"},
            {"start": 1.3, "end": 3.0, "text": "đây là bản demo"},
            {"start": 3.1, "end": 5.0, "text": "của Whisper ASR"},
        ],
    }
    
    print(f"\n📊 Transcription Result:")
    print(f"  Language: {result['language']} ({result['language_probability']:.0%})")
    print(f"  Text: \"{result['text']}\"")
    print(f"\n  Segments:")
    for seg in result["segments"]:
        print(f"    [{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")


def demo_wer_calculation():
    """Demonstrate WER (Word Error Rate) calculation."""
    print("\n=== 4. WER Evaluation ===\n")
    
    test_cases = [
        {
            "reference": "xin chào tôi là AI engineer",
            "hypothesis": "xin chào tôi là AI engine",
        },
        {
            "reference": "machine learning rất thú vị",
            "hypothesis": "machine learning rất thú vị",
        },
        {
            "reference": "hôm nay trời đẹp quá",
            "hypothesis": "hôm nay trời đẹp",
        },
    ]
    
    print(f"📊 WER = (S + I + D) / N")
    print(f"   S = Substitutions, I = Insertions, D = Deletions, N = Reference words\n")
    
    for i, case in enumerate(test_cases, 1):
        ref_words = case["reference"].split()
        hyp_words = case["hypothesis"].split()
        
        # Simple WER calculation (edit distance)
        n = len(ref_words)
        m = len(hyp_words)
        
        # Dynamic programming
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i2 in range(n + 1):
            dp[i2][0] = i2
        for j in range(m + 1):
            dp[0][j] = j
        
        for i2 in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i2-1] == hyp_words[j-1]:
                    dp[i2][j] = dp[i2-1][j-1]
                else:
                    dp[i2][j] = 1 + min(
                        dp[i2-1][j],     # Deletion
                        dp[i2][j-1],     # Insertion
                        dp[i2-1][j-1],   # Substitution
                    )
        
        errors = dp[n][m]
        wer = errors / n if n > 0 else 0
        
        print(f"  Case {i}:")
        print(f"    REF: \"{case['reference']}\"")
        print(f"    HYP: \"{case['hypothesis']}\"")
        print(f"    WER: {errors}/{n} = {wer:.1%} {'✅' if wer == 0 else '⚠️'}")
        print()


def demo_faster_whisper_config():
    """Show faster-whisper configuration options."""
    print("=== 5. Faster-Whisper Configuration ===\n")
    
    configs = {
        "Best Quality": {
            "model": "large-v3",
            "compute_type": "float16",
            "beam_size": 5,
            "vad_filter": True,
            "language": "vi",
            "word_timestamps": True,
        },
        "Balanced": {
            "model": "turbo",
            "compute_type": "float16",
            "beam_size": 3,
            "vad_filter": True,
            "language": "vi",
            "word_timestamps": False,
        },
        "Fastest": {
            "model": "base",
            "compute_type": "int8",
            "beam_size": 1,
            "vad_filter": True,
            "language": "vi",
            "word_timestamps": False,
        },
        "Edge/CPU": {
            "model": "tiny",
            "compute_type": "int8",
            "beam_size": 1,
            "vad_filter": False,
            "language": "vi",
            "word_timestamps": False,
        },
    }
    
    for name, config in configs.items():
        print(f"  📋 {name}:")
        for key, value in config.items():
            print(f"     {key}: {value}")
        print()
    
    print("  💡 Code Template:")
    print("""
    from faster_whisper import WhisperModel
    
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, info = model.transcribe(
        "audio.wav",
        language="vi",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    for segment in segments:
        print(f"[{segment.start:.1f}s] {segment.text}")
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("🎤 Whisper ASR Demo")
    print("=" * 60)
    print("Demonstrates ASR concepts without downloading models.\n")
    
    demo_whisper_concepts()
    demo_mel_spectrogram()
    demo_transcription_simulation()
    demo_wer_calculation()
    demo_faster_whisper_config()
    
    print(f"{'='*60}")
    print("✅ Whisper ASR Demo completed!")
    print("   To run real transcription: pip install faster-whisper")
    print(f"{'='*60}")
