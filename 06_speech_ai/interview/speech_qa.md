# 🎯 Speech AI — Câu Hỏi Phỏng Vấn (35+)

> Mỗi câu quan trọng có: giải thích kỹ thuật → numerical details → code/example → follow-up.

---

## Audio Fundamentals (6 câu)

### Q1: Sample rate 16kHz tại sao đủ cho speech?
**A**: **Nyquist theorem**: Fs ≥ 2 × Fmax. Speech energy tập trung dưới 8kHz.
- 16kHz captures tới 8kHz → đủ cho speech intelligibility
- 44.1kHz cho music (human hearing lên tới 20kHz)
- 8kHz cho telephony (đủ hiểu, quality thấp)
- **Whisper** input: resample to 16kHz bất kể source sample rate
- **Follow-up**: "Bit depth ảnh hưởng gì?" → 16-bit = 96dB dynamic range (standard). 24-bit cho recording studios. ML models hầu hết dùng float32 internally.

### Q2: Mel Spectrogram vs MFCC?
**A**: 

| | Mel Spectrogram | MFCC |
|-|----------------|------|
| **What** | Power spectrum on mel scale | DCT compression of mel spectrum |
| **Dimensions** | (n_mels × time_steps), typically 80×3000 | (n_mfcc × time_steps), typically 13×T |
| **Information** | Rich, preserves fine details | Compressed, removes correlated info |
| **Used by** | Modern DL: Whisper, Wav2Vec2 | Classic ML: GMM-HMM, some speaker ID |
| **Why mel?** | Matches human perception (log freq) | Same + decorrelation |

**2026 standard**: Mel Spectrogram cho almost everything. MFCC chỉ còn dùng cho legacy systems hoặc resource-constrained edge devices.

### Q3: Pre-emphasis filter dùng để làm gì?
**A**: Boost high frequencies: `y[n] = x[n] - α*x[n-1]` (α ≈ 0.97).
- High frequencies carry consonant info nhưng energy thấp → amplified → better ASR
- **Modern models (Whisper)**: không cần pre-emphasis (learned in neural net)
- Mainly used in: traditional MFCC pipelines, some preprocessing

### Q4: VAD (Voice Activity Detection)?
**A**: Detect speech vs non-speech frames.
```python
# Silero VAD — state-of-the-art, lightweight
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, _, _, _) = utils

speech_timestamps = get_speech_timestamps(
    audio_tensor,  # torch.Tensor, 16kHz
    model,
    threshold=0.5,           # Confidence threshold
    min_speech_duration_ms=250,  # Ignore very short (<250ms)
    min_silence_duration_ms=100,
)
# Returns: [{'start': 1000, 'end': 5000}, {'start': 7000, 'end': 12000}]
```
**Use cases**: skip silence (reduce ASR cost), trigger recording, endpoint detection (voice agents), noise filtering.

### Q5: Spectrogram đọc thế nào?
**A**: X-axis = time, Y-axis = frequency, Color = intensity/energy.
- **Vowels**: horizontal bands (formants — resonant frequencies)
- **Consonants**: vertical bursts (plosives like /p/, /t/, /k/) or noise (fricatives /s/, /f/)
- **Silence**: dark/no color regions
- **Follow-up**: "Formants là gì?" → Resonant frequencies of vocal tract. F1 (300-700Hz) + F2 (700-2500Hz) distinguish vowels. F1/F2 plot = vowel space.

### Q6: Tại sao dùng log scale cho Mel Spectrogram?
**A**: Human perception is **logarithmic** (Weber-Fechner law).
- Doubling power = perceive "slightly louder" (not 2× louder)
- 10dB difference sounds same ở mọi volume level
- Log scale → matches how humans hear → model learns relevant features
- Without log: model wastes capacity on perceptually irrelevant energy differences

---

## ASR / Speech-to-Text (8 câu)

### Q7: Whisper architecture chi tiết?
**A**: Encoder-decoder Transformer trained on **680,000 hours** of labeled audio.

```
Input: 30s audio → 16kHz → log Mel spectrogram (80 mel × 3000 frames)
       ↓
Encoder: 2 Conv1D layers → N Transformer blocks → audio features
       ↓
Decoder: autoregressive, generates tokens with special tokens:
         <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> Hello world
```

| Model | Params | English WER | Multilingual | VRAM |
|-------|:------:|:-----------:|:------------:|:----:|
| tiny | 39M | ~8% | 99 langs | 1GB |
| base | 74M | ~6% | 99 langs | 1GB |
| small | 244M | ~4.5% | 99 langs | 2GB |
| medium | 769M | ~3.5% | 99 langs | 5GB |
| large-v3 | 1.55B | ~3% | 99 langs | 10GB |
| large-v3-turbo | 809M | ~3.2% | 99 langs | 6GB |

- **Multitask**: transcribe, translate (→English), language detect, timestamp
- **Follow-up**: "Tại sao Whisper hallucinate?" → On silence/noise, decoder generates random common phrases ("Thank you for watching"). Fix: VAD pre-filter + repetition detection + confidence thresholding.

### Q8: faster-whisper vs Whisper gốc?
**A**: CTranslate2 backend — same weights, optimized inference.
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", 
    beam_size=5,
    word_timestamps=True,      # Word-level timing
    vad_filter=True,           # Built-in Silero VAD
    language="vi",             # Force Vietnamese
)
for segment in segments:
    print(f"[{segment.start:.1f}s → {segment.end:.1f}s] {segment.text}")
```
- **~4× faster**, **50% less VRAM**, same accuracy
- Supports: INT8 quantization (`compute_type="int8"`), batched inference
- **When to use**: production serving, real-time applications, GPU-constrained environment

### Q9: WER vs CER?
**A**: 
- **WER** = (Substitutions + Insertions + Deletions) / Total Reference Words
- **CER** = Same but at character level

``` 
Reference:  "the cat sat on the mat"        (6 words)
Hypothesis: "the cat sat in a mat"           
             S=0  S=0  S=0  S=1 I=1 S=0     WER = (1+1+0)/6 = 33%

CER better for:
  - Vietnamese, Chinese (word boundary ambiguous)
  - Agglutinative languages (Turkish, Finnish)
  - Short utterances (1 word error = 50% WER but only 10% CER)
```

### Q10: Streaming ASR challenges?
**A**: 5 core challenges:
1. **Chunk boundaries**: words split across audio chunks → incomplete words
2. **No future context**: causal-only (can't look ahead) → less accurate
3. **Latency vs accuracy**: larger chunks = better accuracy but higher latency
4. **Endpointing**: when has user finished speaking? (silence threshold)
5. **VAD integration**: smart chunking based on speech/silence

```python
# Streaming ASR pattern
class StreamingASR:
    def __init__(self, chunk_size_ms=1000):
        self.buffer = []
        self.chunk_size = chunk_size_ms
        self.vad = load_silero_vad()
    
    def process_chunk(self, audio_chunk):
        self.buffer.append(audio_chunk)
        
        if self.vad.is_speech_end(audio_chunk):  # Endpoint detected
            full_audio = concat(self.buffer)
            transcript = whisper.transcribe(full_audio)
            self.buffer = []
            return {"final": True, "text": transcript}
        elif len(self.buffer) > self.max_buffer:
            # Force transcribe (prevent unlimited buffering)
            partial = whisper.transcribe(concat(self.buffer))
            return {"final": False, "text": partial}
```

### Q11: Whisper hallucination?
**A**: On silence/noise, Whisper generates random text ("Thank you for watching", repeated phrases).
- **Causes**: model trained on YouTube with such phrases, no speech = uncharted territory
- **Fixes**: 
  1. VAD filter trước Whisper (skip non-speech)
  2. Repetition detection (same phrase 3× → skip)
  3. Log-probability threshold (low confidence → discard)
  4. Energy-based rejection (very low RMS → no speech)
- `faster-whisper` built-in: `vad_filter=True` handles most cases.

### Q12: Multilingual ASR strategies?
**A**: 
1. **Auto language detect**: Whisper built-in (first 30s → detect)
2. **Force language**: `language="vi"` (faster, no misdetection)
3. **Code-switching** (mixing languages): use large-v3, don't force language
4. **Post-processing**: normalize bilingual terms, number formatting per locale

### Q13: ASR post-processing pipeline?
**A**: Raw transcript → polished text:
1. **Punctuation restoration** (Whisper includes này, hoặc separate model)
2. **Capitalization** (sentence start, proper nouns)
3. **Number formatting**: "năm triệu" → "5,000,000"
4. **Disfluency removal**: "uh", "um", repeated words
5. **Speaker tag insertion** (if diarization available)
6. **PII detection**: mask phone numbers, emails automatically

### Q14: ASR evaluation beyond WER?
**A**: 
- **SER** (Sentence Error Rate): % sentences with ≥1 error (stricter)
- **RTF** (Real-Time Factor): processing_time / audio_duration. RTF < 1 = real-time capable
- **Word-level timing accuracy**: alignment precision (for subtitles)
- **Domain-specific accuracy**: medical terms, product names (custom test sets)
- **Follow-up**: "WER đủ chưa?" → WER alone misleading. "a" vs "the" error = same weight as "cancer" vs "dancer". Weighted WER or semantic error metrics better for production.

---

## TTS / Text-to-Speech (6 câu)

### Q15: TTS pipeline hiện đại?
**A**: 
```
Traditional:  Text → Normalize → Acoustic Model → Vocoder → Audio
              (G2P)   (FastSpeech2)    (HiFi-GAN)

End-to-End:   Text → Single Model → Audio
              (VITS, F5-TTS, CosyVoice)
```
- End-to-end simpler, better quality, harder to control
- Traditional: more control (prosody, speed, pitch adjustable at each stage)

### Q16: Voice cloning cần gì?
**A**: 
- **Zero-shot**: 5-15s reference audio → clone voice immediately (F5-TTS, XTTS)
- **Fine-tuned**: 1-5 hours data → train custom voice (highest quality, expensive)
- Requirements: clean audio, natural speech pace, minimal background noise
- Quality depends on: reference audio quality > model capability > text similarity
- **Follow-up**: "Speaker similarity metrics?" → Cosine similarity of speaker embeddings. > 0.85 = very similar voice. DNSMOS for naturalness.

### Q17: MOS và evaluation metrics cho TTS?
**A**: 
| Metric | Type | What it measures | Range |
|--------|------|------------------|-------|
| **MOS** | Human | Overall naturalness | 1-5 (4.5+ ≈ human) |
| **PESQ** | Automated | Perceptual quality | -0.5 to 4.5 |
| **DNSMOS** | Neural (automated) | MOS prediction | 1-5 |
| **Speaker similarity** | Embedding | Voice match | 0-1 cosine |
| **WER of TTS** | ASR-based | Intelligibility | Lower = clearer speech |

- **Production**: DNSMOS for automated monitoring, periodic human MOS evaluation

### Q18: Streaming TTS tại sao quan trọng?
**A**: User hears first audio chunk trong 300-500ms instead of waiting 2-5s cho toàn bộ response.
- **Pattern**: LLM generates text token-by-token → buffer until sentence boundary → TTS synthesize sentence → stream audio chunk
- **Chunking strategy**: split at sentence boundaries (". ", "? ", "! ") cho natural prosody. Never split mid-word.

### Q19: Voice cloning ethical concerns?
**A**: 
- **Risks**: deepfakes, fraud (bank voice auth), impersonation, non-consensual porn
- **Mitigations**: consent requirement, audio watermarking, detection models, disclosure laws
- **Regulation**: EU AI Act requires labeling synthetic voice. Some US states: criminal for fraud
- **Interview tip**: Always mention ethics proactively — shows maturity.

### Q20: SSML (Speech Synthesis Markup Language)?
**A**: XML markup to control TTS output:
```xml
<speak>
  Welcome to <emphasis level="strong">AI Engineering</emphasis>.
  <break time="500ms"/>
  The model achieved <say-as interpret-as="number">95.7</say-as> percent accuracy.
  <prosody rate="slow" pitch="+2st">This is important.</prosody>
</speak>
```
- Control: pauses, emphasis, pronunciation, speed, pitch
- Supported by: Google Cloud TTS, Amazon Polly, Azure TTS. Limited support in open-source models.

---

## Speaker Diarization (4 câu)

### Q21: Speaker diarization pipeline?
**A**: "Who spoke when?" — 5-stage pipeline:
```
Audio → VAD → Segmentation → Speaker Embedding → Clustering → Alignment
         ↓         ↓              ↓                 ↓            ↓
    Detect    Find speaker    Extract d-vector/    Group same    Assign labels
    speech    change points   x-vector per seg     speakers      to timestamps
```
- **SOTA**: Pyannote 3.1 (end-to-end, handles overlap)
- **Alternative**: NeMo, simple-diarizer
- **Follow-up**: "Online diarization?" → Incremental processing as audio streams in. Harder: can't recluster past segments. Use windowed clustering with speaker memory.

### Q22: DER (Diarization Error Rate)?
**A**: DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Reference Duration

| Component | Meaning | Example |
|-----------|---------|---------|
| **False Alarm** | Detected speech where none exists | Model labels silence as speech |
| **Missed Speech** | Missed real speech | Model misses soft-spoken segment |
| **Speaker Confusion** | Correct speech, wrong speaker label | Model assigns SPEAKER_1 when it's SPEAKER_2 |

- **Good**: DER < 10%. **Acceptable**: < 20%. **Pyannote 3.1**: ~12% average on AMI benchmark.

### Q23: Speaker overlap handling?
**A**: Hardest diarization problem: 2+ people talking simultaneously.

| Scenario | Overlap Rate | Strategy |
|----------|:-----------:|----------|
| Meetings | 15-30% | Pyannote OSD, output both speakers |
| Call center | 5-10% | Energy-based: keep louder speaker |
| Podcasts | 10-20% | Host detection + guest separation |

- **Impact on ASR**: clean = 3-5% WER, overlapped = 15-30% WER
- **Advanced**: Source separation models (SepFormer) → separate audio streams → ASR each

### Q24: Speaker verification vs identification?
**A**: 
- **Verification (1:1)**: "Is this Speaker X?" → compare embedding vs enrolled reference. Yes/no.
- **Identification (1:N)**: "WHO is this?" → compare embedding vs N enrolled speakers. Return match.
- **Clustering (unsupervised)**: "How many speakers? Group them." → diarization use case.
- All based on **speaker embeddings** (ECAPA-TDNN, 192-256 dims). Cosine similarity > 0.7 = same speaker.

---

## Voice Agents & Production (6 câu)

### Q25: Voice agent latency budget?
**A**: Target total < 1.2s (human conversation turn-taking ~1.5s).

```
Component breakdown:
  VAD + endpoint detection:  50-100ms
  ASR (streaming):           300-500ms
  LLM generation:            300-800ms  ← bottleneck
  TTS (first chunk):         200-400ms
  Network overhead:          50-100ms
  ─────────────────────────────────────
  Total perceived:           400-900ms (with streaming)
```

**Optimization strategies**:
1. Streaming ASR → LLM → TTS (pipeline, don't wait for each to finish)
2. Speculative TTS: start generating speech from first LLM tokens
3. Pre-cache common responses ("Hello, how can I help?")
4. Use fastest models: distil-whisper, GPT-4o-mini, edge TTS

### Q26: Barge-in handling?
**A**: User speaks during AI playback → interrupt and restart pipeline.
```
Normal flow:    AI speaks ──────────────────────────→ end
Barge-in:       AI speaks ──── [user starts] ──→ STOP AI
                                    ↓
                              Cancel TTS playback
                              Cancel LLM generation
                              Start ASR on new audio
                              Process new user intent
```
Critical for natural voice experience. Without it: AI talks over user → frustrating.

### Q27: WebSocket vs REST cho voice?
**A**: 
- **WebSocket**: persistent bidirectional, stream audio both ways, low latency. **Always for voice agents**.
- **REST**: stateless, higher overhead per request, can't stream audio
- **Hybrid pattern**: WebSocket for audio stream, REST for non-realtime operations (history, config)

### Q28: Production noise handling?
**A**: Process pipeline BEFORE ASR:
1. **Noise reduction** (noisereduce library / spectral subtraction)
2. **SNR estimation** → reject if < 10dB (too noisy)
3. **High-pass filter** (remove low-freq rumble, < 80Hz)
4. **Normalization** (consistent volume level)
5. **Echo cancellation** (if speaker + mic in same room)

### Q29: Edge deployment cho speech?
**A**: Run ASR on-device instead of cloud:
- **whisper.cpp**: C++ port, runs on CPU. Apple Silicon optimized.
- **ONNX Runtime**: Whisper ONNX export, cross-platform
- **distil-whisper**: 50% faster, 6× smaller, ~1% WER degradation
- **Use case**: privacy-sensitive (healthcare), offline (field work), latency-critical

### Q30: Multimodal audio (GPT-4o, Gemini)?
**A**: Direct audio input to LLM → no ASR step needed.
- **GPT-4o audio mode**: native audio understanding, real-time conversation
- **Gemini 2.0**: audio + video + text multimodal
- **Impact**: potentially replaces ASR→LLM→TTS pipeline with single model call
- **Trade-off**: less control (can't tune ASR/TTS separately), vendor lock-in, cost
- **Current state (2026)**: promising but latency still higher than dedicated pipeline for production voice agents

---

## Evaluation Tổng Hợp (5 câu) 🆕

### Q31: Speech AI evaluation framework?
**A**: 

| Task | Primary Metric | Secondary | Tools |
|------|---------------|-----------|-------|
| ASR | WER / CER | RTF, SER | jiwer, whisper_normalizer |
| TTS | MOS (human) | DNSMOS, PESQ | pesq library, DNSMOS |
| Diarization | DER | JER, coverage | pyannote.metrics |
| Speaker Verification | EER | minDCF | speechbrain |
| Voice Agent | End-to-end latency | User satisfaction | Custom |

### Q32: WER benchmarks — các mốc quan trọng?
**A**: 
- **Human transcription**: ~4% WER (English, clean speech)
- **Whisper large-v3**: ~3% WER (English, clean) — **superhuman!**
- **Noisy speech**: 10-25% WER (depends on SNR)
- **Accented speech**: 5-15% WER (model dependent)
- **Vietnamese**: ~8-12% WER (Whisper large-v3, less training data)
- **Key insight**: WER < 5% là production-ready cho hầu hết applications.

### Q33: EER (Equal Error Rate) cho speaker verification?
**A**: Point where False Accept Rate = False Reject Rate.
- Lower EER = better system. SOTA: ~1% EER (VoxCeleb benchmark)
- **FAR** (False Accept): impostor accepted as genuine
- **FRR** (False Reject): genuine speaker rejected
- **Threshold tuning**: security-critical → low FAR (stricter). UX-critical → low FRR (more lenient).

### Q34: Vietnamese speech challenges?
**A**: 
- **Tonal language**: 6 tones, small pitch difference = different meaning
- **Word segmentation**: compound words ambiguous ("bàn chải" = toothbrush, not "table" + "brush")
- **Code-switching**: Vietnamese + English mixing common in tech conversations
- **Data scarcity**: less labeled data than English → higher WER
- **Solutions**: Whisper large-v3 (multilingual), fine-tune on Vietnamese data, Vietnamese-specific post-processing

### Q35: Speech AI cost comparison?
**A**: 

| Solution | Cost (per hour audio) | Latency | Quality |
|----------|:--------------------:|:-------:|:-------:|
| Whisper API (OpenAI) | $0.36/hr | ~30s/hr | Best |
| Faster-whisper (self-host) | GPU cost only (~$0.10) | ~8s/hr | Same |
| Google Cloud STT | $1.44/hr | Real-time | Good |
| Deepgram | $0.75/hr | Real-time | Good |
| Whisper local (CPU) | Free | ~60s/hr | Same |

- **Decision**: high volume → self-host. Low volume → API. Real-time → Deepgram/Google. Budget → local.
