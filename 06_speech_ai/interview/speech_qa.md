# 🎯 Speech AI — Câu Hỏi Phỏng Vấn (25+)

---

## Audio Fundamentals (6 câu)

### Q1: Sample rate 16kHz tại sao đủ cho speech?
**A**: Nyquist: Fs ≥ 2 × Fmax. Speech content nằm dưới 8kHz. 16kHz captures tới 8kHz → đủ cho speech. 44.1kHz chỉ cần cho music.

### Q2: Mel Spectrogram vs MFCC?
**A**: Mel Spectrogram: frequency info trên mel scale, dùng cho modern DL (Whisper). MFCC: compressed (DCT on mel), dùng cho classic ML, speaker ID. Modern ASR chủ yếu dùng Mel Spec.

### Q3: Pre-emphasis filter dùng để làm gì?
**A**: Boost high frequencies (y[n] = x[n] - 0.97*x[n-1]). High freq mang consonant info nhưng energy thấp → amplify chúng → ASR tốt hơn.

### Q4: VAD (Voice Activity Detection) hoạt động thế nào?
**A**: Detect speech vs non-speech frames. Simple: RMS energy threshold. Advanced: Silero VAD (neural net). Dùng để: skip silence, reduce processing, save cost, trigger ASR.

### Q5: Spectrogram đọc thế nào?
**A**: X-axis: time, Y-axis: frequency, Color: intensity/energy. Vowels: horizontal bands (formants). Consonants: vertical bursts. Silence: dark/no color.

### Q6: Tại sao dùng log scale cho Mel Spectrogram?
**A**: Human perception is logarithmic (Weber-Fechner law). 10dB difference sounds same ở mọi volume. Log scale → matches human hearing → model learns better.

---

## ASR / Speech-to-Text (7 câu)

### Q7: Whisper architecture?
**A**: Encoder-decoder Transformer. Input: 30s log Mel spectrogram (80×3000). Encoder: process audio features. Decoder: autoregressive text generation. Multitask: transcribe, translate, timestamp, language detect.

### Q8: faster-whisper nhanh hơn Whisper gốc thế nào?
**A**: CTranslate2 backend: INT8 quantization, efficient attention, optimized C++ inference. ~4x faster, 50% less VRAM. Same accuracy (same weights).

### Q9: WER vs CER?
**A**: WER: Word Error Rate = (S+I+D)/N at word level. CER: Character Error Rate at char level. CER tốt hơn cho agglutinative/tonal languages (Vietnamese, Chinese) vì word boundaries ambiguous.

### Q10: Streaming ASR challenges?
**A**: (1) Partial words ở chunk boundaries, (2) No future context (causal), (3) Latency vs accuracy tradeoff, (4) Need VAD for smart chunking, (5) Endpointing (khi nào user nói xong?).

### Q11: Whisper hallucination?
**A**: On silence/noise, Whisper may generate random text ("Thank you for watching", repeated phrases). Fix: VAD filter trước, check repetition, confidence threshold, energy-based rejection.

### Q12: Multilingual ASR strategies?
**A**: (1) Auto language detect (Whisper built-in), (2) Force language if known, (3) Code-switching: use large-v3 + vi primary, (4) Post-processing: normalize bilingual terms.

### Q13: ASR post-processing?
**A**: (1) Punctuation restoration, (2) Capitalization, (3) Number formatting (5 triệu → 5,000,000), (4) Disfluency removal (uh, um), (5) Speaker tag insertion.

---

## TTS / Text-to-Speech (5 câu)

### Q14: TTS pipeline hiện đại?
**A**: Text → Normalize → Acoustic Model (generate Mel spectrogram) → Vocoder (Mel → waveform). Modern: end-to-end models (VITS, F5-TTS) combine all steps.

### Q15: Voice cloning cần gì?
**A**: 5-15s reference audio (clean, natural speech). Speaker encoder extracts voice embedding → condition TTS model → generate new speech in that voice. F5-TTS, XTTS support zero-shot cloning.

### Q16: MOS score là gì?
**A**: Mean Opinion Score: human judges rate naturalness 1-5. Gold standard for TTS evaluation. 4.5+ = near-human quality. Current SOTA: ~4.5 MOS.

### Q17: Streaming TTS tại sao quan trọng?
**A**: User hears first audio chunk nhanh (300-500ms) instead of waiting for entire response. Buffer LLM tokens → sentence boundary → synthesize → stream. Critical for voice agents.

### Q18: Voice cloning ethical concerns?
**A**: Deepfakes, fraud, impersonation. Mitigations: consent, watermarking, detection models, disclosure laws. Some jurisdictions require labeling AI-generated speech.

---

## Speaker Diarization (3 câu)

### Q19: Speaker diarization pipeline?
**A**: VAD → Segmentation (phát hiện speaker change) → Speaker Embedding (extract voice vectors) → Clustering (group same speakers) → Alignment. Pyannote 3.1 = SOTA.

### Q20: DER (Diarization Error Rate)?
**A**: DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Duration. False Alarm: detected speech where none. Missed: missed speech. Confusion: wrong speaker. Good: DER < 10%.

### Q21: Speaker overlap handling?
**A**: Hard problem. 2+ people talking simultaneously. Pyannote 3.1 detects overlap regions. Solutions: source separation, output both speakers for overlapping timeframes.

---

## Voice Agents & Production (5 câu)

### Q22: Voice agent latency budget?
**A**: Target total <1.2s. Breakdown: VAD 20ms + ASR 300ms + LLM 500ms + TTS 300ms + Network 50ms. Streaming reduces perceived latency (first chunk <500ms).

### Q23: Barge-in handling?
**A**: User speaks during AI playback → VAD detect → stop TTS playback → cancel current generation → restart ASR pipeline. Essential for natural conversation feel.

### Q24: WebSocket vs REST cho voice?
**A**: WebSocket: persistent bidirectional connection, low latency, stream audio both ways. REST: stateless, higher latency, can't stream. Voice agents = always WebSocket.

### Q25: Production noise handling?
**A**: (1) Noise reduction (noisereduce/spectral subtraction), (2) SNR estimation → reject if <10dB, (3) High-pass filter (remove rumble), (4) Normalization. Process before ASR.

### Q26: Speech AI cost optimization?
**A**: (1) Local models cho high volume (vs API), (2) Cache frequent TTS phrases, (3) INT8 quantization, (4) VAD to skip silence, (5) Batch processing, (6) distil-whisper for speed.
