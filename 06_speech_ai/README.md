# 🎙️ 06 — Speech AI

> ASR (Speech-to-Text), TTS (Text-to-Speech), Speaker Diarization, Voice Agents, Audio Processing.

## 📚 Docs

| File | Chủ đề | Thời gian đọc |
|------|--------|---------------|
| [01_audio_fundamentals.md](docs/01_audio_fundamentals.md) | Sampling Rate, Spectrogram, MFCC, Audio Preprocessing | ~12 min |
| [02_asr_speech_to_text.md](docs/02_asr_speech_to_text.md) | Whisper, faster-whisper, Streaming ASR, Evaluation (WER) | ~15 min |
| [03_tts_text_to_speech.md](docs/03_tts_text_to_speech.md) | F5-TTS, Bark, XTTS, Voice Cloning, Prosody Control | ~12 min |
| [04_speaker_diarization.md](docs/04_speaker_diarization.md) | Pyannote, Speaker Embedding, Overlap Handling | ~10 min |
| [05_voice_agents.md](docs/05_voice_agents.md) | Real-time Voice Pipeline, Latency Budget, WebSocket | ~15 min |
| [06_production_speech.md](docs/06_production_speech.md) | Deployment, Noise Handling, Multilingual, Edge Cases | ~10 min |

## 💻 Examples

```bash
cd 06_speech_ai/examples
python audio_processing.py    # Audio fundamentals with librosa
python whisper_demo.py         # Whisper ASR transcription
python tts_demo.py             # Text-to-Speech synthesis
```

## ✅ Checklist
- [ ] Đọc hết 6 docs
- [ ] Chạy 3 examples
- [ ] Trả lời 25+ câu trong `interview/speech_qa.md`
