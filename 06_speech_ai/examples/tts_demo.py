"""
🔊 TTS Demo — Text-to-Speech concepts and synthesis
Chạy: python tts_demo.py

Demonstrates TTS concepts with synthetic audio generation.
For real TTS: pip install f5-tts or pip install TTS
"""
import numpy as np
import os
import struct
import time


def save_wav(filename: str, audio: np.ndarray, sr: int = 16000):
    """Save numpy array as WAV file (no dependencies needed)."""
    audio_int16 = (audio * 32767).astype(np.int16)
    n_samples = len(audio_int16)
    
    with open(filename, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + n_samples * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))         # Chunk size
        f.write(struct.pack('<H', 1))          # PCM
        f.write(struct.pack('<H', 1))          # Mono
        f.write(struct.pack('<I', sr))         # Sample rate
        f.write(struct.pack('<I', sr * 2))     # Byte rate
        f.write(struct.pack('<H', 2))          # Block align
        f.write(struct.pack('<H', 16))         # Bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', n_samples * 2))
        f.write(audio_int16.tobytes())


def simple_synthesizer(text: str, sr: int = 16000) -> np.ndarray:
    """Simple TTS synthesizer — maps characters to tones (educational demo)."""
    char_to_freq = {
        'a': 440, 'b': 494, 'c': 523, 'd': 587, 'e': 659,
        'f': 698, 'g': 784, 'h': 880, 'i': 988, 'j': 1047,
        'k': 1175, 'l': 1319, 'm': 1397, 'n': 1480, 'o': 1568,
        'p': 1661, 'q': 1760, 'r': 1865, 's': 1976, 't': 2093,
        'u': 2217, 'v': 2349, 'w': 2489, 'x': 2637, 'y': 2794, 'z': 2960,
        ' ': 0,  # Silence for spaces
    }
    
    char_duration = 0.08  # 80ms per character
    audio = np.array([], dtype=np.float32)
    
    for char in text.lower():
        freq = char_to_freq.get(char, 440)
        n_samples = int(sr * char_duration)
        t = np.linspace(0, char_duration, n_samples, dtype=np.float32)
        
        if freq == 0:
            tone = np.zeros(n_samples, dtype=np.float32)
        else:
            # Generate with harmonics for richer sound
            tone = (
                0.5 * np.sin(2 * np.pi * freq * t) +
                0.2 * np.sin(2 * np.pi * 2 * freq * t) +
                0.1 * np.sin(2 * np.pi * 3 * freq * t)
            ).astype(np.float32)
            
            # Apply envelope (fade in/out)
            envelope = np.ones(n_samples, dtype=np.float32)
            fade = int(n_samples * 0.1)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            tone *= envelope
        
        audio = np.concatenate([audio, tone])
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio


def demo_tts_pipeline():
    """Demonstrate TTS pipeline concepts."""
    print("=== 1. TTS Pipeline ===\n")
    
    pipeline_steps = [
        ("Text Input", "Hello, how are you today?"),
        ("Text Normalization", "hello comma how are you today question-mark"),
        ("G2P (Grapheme-to-Phoneme)", "h ɛ l oʊ | h aʊ ɑːr j uː t ə d eɪ"),
        ("Prosody Prediction", "H1 L H2  |  H3 L L H4 L H5"),
        ("Duration Model", "[120ms, 80ms, 60ms, 100ms, ...]  (per phoneme)"),
        ("Acoustic Model", "Mel Spectrogram (80 × N frames)"),
        ("Vocoder", "Waveform (1 × N samples at 24kHz)"),
    ]
    
    print("  Text → Sound Pipeline:")
    for i, (step, detail) in enumerate(pipeline_steps):
        connector = "  →  " if i > 0 else "     "
        print(f"  {i+1}. {step}")
        print(f"     Output: {detail}")
    
    print(f"\n  ⚡ Modern end-to-end models (F5-TTS, VITS) combine steps 3-7")


def demo_tts_models():
    """Compare TTS models."""
    print("\n=== 2. TTS Model Comparison ===\n")
    
    models = [
        ("F5-TTS",     "2025", "Diffusion",    "Zero-shot, streaming",     "4.5+", "Open"),
        ("XTTS v2",    "2024", "GPT-like",     "17 languages, clone",      "4.3",  "Open"),
        ("Bark",       "2023", "GPT-like",     "Emotions, music, effects", "4.0",  "Open"),
        ("VITS2",      "2023", "End-to-end",   "Fast, lightweight",        "4.2",  "Open"),
        ("OpenAI TTS", "2024", "Proprietary",  "6 voices, HD quality",     "4.6",  "API"),
        ("ElevenLabs", "2024", "Proprietary",  "Clone, emotions, SFX",     "4.7",  "API"),
    ]
    
    print(f"  {'Model':<14} {'Year':<6} {'Architecture':<14} {'Features':<28} {'MOS':<5} {'Type'}")
    print(f"  {'-'*80}")
    for name, year, arch, features, mos, type_ in models:
        print(f"  {name:<14} {year:<6} {arch:<14} {features:<28} {mos:<5} {type_}")
    
    print(f"\n  💡 Recommendations:")
    print(f"     Best open-source: F5-TTS (quality) or VITS2 (speed)")
    print(f"     Best API quality: ElevenLabs")
    print(f"     Best cost/quality: OpenAI TTS")


def demo_voice_cloning():
    """Demonstrate voice cloning concepts."""
    print("\n=== 3. Voice Cloning ===\n")
    
    print("  📋 Voice Cloning Pipeline:")
    print("""
    1. Reference Audio (5-15 seconds)
       └─→ Speaker Encoder → Speaker Embedding (256-dim vector)
                                    ↓
    2. Input Text ───────→ TTS Model ←── Speaker Embedding
                                    ↓
    3. Output: New speech in the reference voice
    """)
    
    print("  📋 Requirements for Good Reference Audio:")
    requirements = [
        ("Duration", "5-15 seconds", "Too short → poor quality, too long → diminishing returns"),
        ("Quality", "Clean, no noise", "SNR > 20dB, no background music"),
        ("Content", "Natural speech", "Not whispering, not shouting"),
        ("Emotion", "Neutral or target", "Reference emotion affects output"),
        ("Language", "Any (cross-lingual)", "Vietnamese ref → English output works!"),
    ]
    
    for req, value, note in requirements:
        print(f"    {req:<12} {value:<20} → {note}")
    
    print(f"\n  ⚠️  Ethical Considerations:")
    print(f"     • Always get consent from voice owners")
    print(f"     • Add audio watermarks for synthetic speech detection")
    print(f"     • Don't impersonate real people")
    print(f"     • Some jurisdictions require disclosure of AI-generated speech")


def demo_synthesis():
    """Synthesize audio and save as WAV."""
    print("\n=== 4. Audio Synthesis Demo ===\n")
    
    texts = [
        "hello world",
        "ai engineering",
        "speech synthesis",
    ]
    
    for text in texts:
        start = time.time()
        audio = simple_synthesizer(text)
        elapsed = time.time() - start
        duration = len(audio) / 16000
        rtf = elapsed / duration  # Real-time factor
        
        filename = f"tts_output_{text.replace(' ', '_')}.wav"
        save_wav(filename, audio, sr=16000)
        
        print(f"  🔊 \"{text}\"")
        print(f"     Duration: {duration:.2f}s, Samples: {len(audio):,}")
        print(f"     Synthesis: {elapsed*1000:.1f}ms (RTF: {rtf:.3f})")
        print(f"     Saved: {filename}")
        print()
    
    return texts


def demo_prosody():
    """Demonstrate prosody control concepts."""
    print("=== 5. Prosody Control ===\n")
    
    print("  📋 Prosody Elements:")
    elements = [
        ("Pitch (F0)", "Voice highness/lowness", "Question ↑ vs Statement →"),
        ("Duration", "Speaking speed", "Emphasis = slower, casual = faster"),
        ("Energy", "Volume/loudness", "Important = louder"),
        ("Pause", "Silence between words", "Comma = short, period = long"),
    ]
    
    for elem, desc, example in elements:
        print(f"    {elem:<18} {desc:<28} e.g., {example}")
    
    print(f"\n  📋 SSML (Speech Synthesis Markup Language):")
    print("""
    <speak>
        <prosody rate="slow" pitch="+2st">
            Welcome to the meeting.
        </prosody>
        <break time="500ms"/>
        <emphasis level="strong">
            This is very important.
        </emphasis>
        <prosody rate="fast">
            Let me quickly summarize.
        </prosody>
    </speak>
    """)
    
    print("  📋 Emotion Tags (Bark-style):")
    emotion_tags = [
        "[laughs]", "[sighs]", "[music]", "[clears throat]",
        "[gasps]", "[crying]", "[whispers]", "[cheers]"
    ]
    print(f"    Supported: {', '.join(emotion_tags)}")


def demo_evaluation():
    """Demonstrate TTS evaluation metrics."""
    print("\n=== 6. TTS Evaluation ===\n")
    
    metrics = [
        ("MOS", "Mean Opinion Score", "1-5 scale", "Human judges", "4.5+ = near-human"),
        ("PESQ", "Perceptual quality", "1-4.5 scale", "Automated", "Needs reference audio"),
        ("STOI", "Intelligibility", "0-1 scale", "Automated", "How clear/understandable"),
        ("Speaker Sim", "Voice similarity", "0-1 cosine", "Embedding", ">0.7 = good clone"),
        ("RTF", "Real-time factor", "ratio", "Timing", "<1 = faster than real-time"),
    ]
    
    print(f"  {'Metric':<14} {'Measures':<22} {'Scale':<12} {'Method':<12} {'Notes'}")
    print(f"  {'-'*75}")
    for name, measures, scale, method, notes in metrics:
        print(f"  {name:<14} {measures:<22} {scale:<12} {method:<12} {notes}")
    
    # RTF calculation example
    print(f"\n  📊 RTF Example:")
    print(f"     5 seconds of audio generated in 0.5 seconds")
    print(f"     RTF = 0.5 / 5.0 = 0.1 (10x faster than real-time)")
    print(f"     Lower RTF = better (faster synthesis)")


if __name__ == "__main__":
    print("=" * 60)
    print("🔊 Text-to-Speech Demo")
    print("=" * 60)
    print("Demonstrates TTS concepts + simple synthesis.\n")
    
    demo_tts_pipeline()
    demo_tts_models()
    demo_voice_cloning()
    texts = demo_synthesis()
    demo_prosody()
    demo_evaluation()
    
    # Cleanup generated files
    for text in texts:
        filename = f"tts_output_{text.replace(' ', '_')}.wav"
        if os.path.exists(filename):
            os.remove(filename)
    
    print(f"\n{'='*60}")
    print("✅ TTS Demo completed!")
    print("   For real TTS: pip install f5-tts")
    print(f"{'='*60}")
