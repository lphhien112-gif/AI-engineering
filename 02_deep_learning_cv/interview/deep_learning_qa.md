# 🎯 Deep Learning & CV — Câu Hỏi Phỏng Vấn (40+)

> Mỗi câu có trả lời chi tiết, follow-up, comparison tables, code snippets.
> Focus: Neural Networks, CNN, Transformer (Vision), Segmentation, Detection, Deployment.

---

## Neural Networks (12 câu)

### Q1: Backpropagation hoạt động thế nào?
**A**: Forward pass → compute loss. Backward pass → chain rule tính ∂Loss/∂w cho mỗi weight.
```
Forward:  x → z = wx+b → a = ReLU(z) → ... → loss = L(ŷ, y)
Backward: ∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂a × ∂a/∂z × ∂z/∂w  (chain rule)
```
PyTorch: `loss.backward()` computes all gradients. `optimizer.step()` updates weights.
- **Follow-up**: "Why autograd?" → Dynamic computational graph, tracks all operations on tensors with `requires_grad=True`.

### Q2: Vanishing gradient xử lý?
**A**: Gradients → 0 as they backprop through many layers → early layers don't learn.
- **Causes**: Sigmoid/Tanh derivatives < 1, multiply through many layers → exponential shrink
- **Solutions** (priority order):
  1. ReLU activation (gradient = 1 for positive z)
  2. Residual connections (gradient shortcut path)
  3. Batch/Layer Normalization (normalize activations)
  4. Proper initialization (He for ReLU, Xavier for tanh)
  5. Gradient clipping (prevent exploding gradients)
- **Follow-up**: "Exploding gradients?" → Opposite problem. Fix: gradient clipping (`torch.nn.utils.clip_grad_norm_`), proper init.

### Q3: ReLU variants — khi nào dùng?
**A**:
| Activation | Formula | Advantage | Use |
|-----------|---------|-----------|-----|
| ReLU | max(0, x) | Simple, fast | Default for CNN |
| LeakyReLU | max(αx, x), α=0.01 | No dying neurons | When ReLU dies |
| GELU | x·Φ(x) | Smooth, stochastic | **Transformers** (GPT, BERT) |
| SiLU/Swish | x·σ(x) | Smooth, self-gated | EfficientNet |
| Mish | x·tanh(softplus(x)) | Best for some tasks | YOLOv4+ |

**Follow-up**: "Dying ReLU?" → If input always negative → output=0 → gradient=0 → neuron dead forever. Fix: LeakyReLU or lower learning rate.

### Q4: BatchNorm vs LayerNorm vs GroupNorm?
**A**:
```
Input shape: (Batch, Channels, Height, Width)

BatchNorm:  normalize across Batch dim — per channel
LayerNorm:  normalize across C,H,W dims — per sample
GroupNorm:  normalize across groups of channels — per sample
InstanceNorm: normalize across H,W — per channel per sample

LayerNorm(C,H,W)     BatchNorm(Batch)    GroupNorm(groups)
     ↓                    ↓                   ↓
[B, C, H, W]        [B, C, H, W]        [B, G, C/G, H, W]
```
| | BatchNorm | LayerNorm | GroupNorm |
|-|-----------|-----------|-----------|
| Normalize over | Batch | Features | Feature groups |
| Batch dependent | ✅ Yes (need stats) | ❌ No | ❌ No |
| Small batch OK | ❌ Noisy stats | ✅ | ✅ |
| Best for | CNN (batch ≥ 16) | **Transformers** | CNN (small batch) |

### Q5: Dropout hoạt động thế nào?
**A**: Training: randomly zero p% neurons → force redundancy → regularization.
- `model.train()`: dropout active
- `model.eval()`: dropout disabled, scale outputs by (1-p)
- Typical: 0.1-0.3 for Transformer, 0.3-0.5 for FC layers
- **Follow-up**: "DropPath (Stochastic Depth)?" → Drop entire residual branches. Used in ViT, EfficientNet. More effective than dropout for residual networks.

### Q6: Adam vs AdamW vs SGD?
**A**:
| | SGD+Momentum | Adam | AdamW |
|-|-------------|------|-------|
| Adaptive LR | ❌ | ✅ (per-param) | ✅ |
| Weight decay | Correct | ❌ Wrong (coupled) | ✅ **Correct** (decoupled) |
| Convergence | Slower, better minima | Fast, OK minima | Fast, **good minima** |
| Best for | CNN (ResNet) | General | **Transformers** |

**Follow-up**: "Why AdamW for Transformers?" → Adam's L2 regularization is wrong — it scales with adaptive LR. AdamW decouples weight decay → proper regularization.

### Q7: Learning Rate Scheduling?
**A**:
1. **Warmup** (5-10% steps): avoid instability at start (large random gradients)
2. **Cosine Decay**: smooth decrease, no hyperparameter (just total steps)
3. **Step Decay**: drop at milestones (e.g., ×0.1 at epoch 30, 60)
4. **OneCycleLR**: ramp up → ramp down (great for CNN fine-tuning)
5. **ReduceOnPlateau**: drop when validation metric stalls

```python
# Modern default: warmup + cosine
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Or linear warmup + cosine (manually)
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=10000
)
```

### Q8: Label Smoothing?
**A**: Hard targets `[0, 0, 1, 0]` → Soft targets `[0.025, 0.025, 0.925, 0.025]` (ε=0.1)
- Prevents overconfidence, improves KL divergence
- Improves calibration + generalization
- `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Follow-up**: "Side effect?" → Slightly hurts knowledge distillation (teacher menos confident).

### Q9: Gradient Accumulation?
**A**: GPU memory limited → batch=4. But need effective batch=32.
```python
accumulation_steps = 8  # Effective batch = 4 × 8 = 32

for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps  # Scale loss!
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q10: Mixed Precision Training (AMP)?
**A**: Forward/backward in FP16 (fast, less memory). Weight update in FP32 (maintain precision).
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for x, y in dataloader:
    with autocast():  # FP16 forward
        loss = model(x, y)
    scaler.scale(loss).backward()      # Scaled FP16 backward
    scaler.step(optimizer)              # Unscale + FP32 update
    scaler.update()                     # Adjust scale factor
```
- **Benefits**: 2-3x faster, ~40-50% less GPU memory
- **Follow-up**: "BF16 vs FP16?" → BF16 has same range as FP32 (no overflow), less precision. Better for training. A100/H100 support.

### Q11: Weight Initialization?
**A**:
| Init | Formula | Best For |
|------|---------|----------|
| **He (Kaiming)** | W ~ N(0, √(2/fan_in)) | ReLU activations |
| **Xavier (Glorot)** | W ~ N(0, √(2/(fan_in + fan_out))) | Tanh/Sigmoid |
| **Zero** | Never for weights! | Only for biases |
| **Pretrained** | Load from checkpoint | **Almost always best** |

### Q12: Loss function selection?
**A**:
| Task | Loss | When |
|------|------|------|
| Binary classification | BCEWithLogitsLoss | 2 classes |
| Multi-class | CrossEntropyLoss | N classes, single label |
| Multi-label | BCEWithLogitsLoss | N classes, multiple labels |
| Regression | MSELoss / L1Loss | Continuous target |
| Segmentation | CE + DiceLoss | Per-pixel classification |
| Detection | Focal + GIoU | Boxes + classes |
| Imbalanced | FocalLoss(γ=2) | Class imbalance |

---

## CNN & CV (12 câu)

### Q13: Conv2d parameters tính thế nào?
**A**: `Params = C_in × C_out × K × K + C_out (bias)`
- Conv2d(3→16, 3×3) = 3×16×3×3 + 16 = **448 params**
- Output size: `(W - K + 2P) / S + 1`
  - W=224, K=3, P=1, S=1 → (224-3+2)/1 + 1 = **224** (same size)

### Q14: Receptive Field?
**A**: Input region 1 output neuron "sees". Deeper layers → larger RF.
- 1 layer Conv(3×3): RF = 3×3
- 2 layers Conv(3×3): RF = 5×5
- 3 layers Conv(3×3): RF = 7×7
- **Follow-up**: "3 layers of 3×3 vs 1 layer of 7×7?" → Same RF but 3×(3²C²) = 27C² vs 49C² params. 3 layers = fewer params + more non-linearity + deeper features.

### Q15: ResNet skip connections?
**A**: 
1. **Gradient flow**: direct path → train 100+ layers (without: max ~20)
2. **Residual learning**: learn F(x) = H(x) - x. If optimal = identity, just learn F(x)=0 (easier than H(x)=x)
3. **Lower bound**: identity shortcut → cant be WORSE than shallower net
- **Follow-up**: "Pre-activation ResNet (v2)?" → BN→ReLU→Conv (better gradient flow than Conv→BN→ReLU)

### Q16: 1×1 Convolution purpose?
**A**: 
- **Channel mixing**: combine information across channels (like FC per-pixel)
- **Dimensionality**: reduce C_in → C_out (bottleneck)
- **Cheap**: only C_in × C_out params (no spatial params)
- Used in: Inception (bottleneck), ResNet (bottleneck block), MobileNet (pointwise)

### Q17: Depthwise Separable Convolution?
**A**: Standard Conv: C_in × C_out × K × K. Depthwise Sep = 2 steps:
1. **Depthwise**: K×K conv per-channel (C_in × K × K params)
2. **Pointwise**: 1×1 conv across channels (C_in × C_out params)
- **Compression**: K²×C_in×C_out → K²×C_in + C_in×C_out ≈ **8-9x fewer params** (K=3)
- Used: MobileNet, EfficientNet, Xception

### Q18: Transfer Learning strategies?
**A**:
1. **Feature extraction**: freeze backbone, train new head (minimal data)
2. **Fine-tune last N layers**: unfreeze gradually (moderate data)
3. **Discriminative LR**: low LR for early layers, high for late layers
4. **Full fine-tune**: unfreeze all (need sufficient data)
- **Rule**: < 1K images → feature extraction. 1K-10K → partial fine-tune. > 10K → full fine-tune.
- **Follow-up**: "Which layers learn what?" → Early = edges/textures (universal). Late = task-specific features. That's why we freeze early layers.

### Q19: Vision Transformer (ViT) vs CNN?
**A**:
| | CNN | ViT |
|-|-----|-----|
| Inductive bias | Translation invariance, locality | None (learn from data) |
| Data hunger | Works with less data | Needs LARGE data (or good pretrained) |
| Global context | Limited by receptive field | From layer 1 (self-attention) |
| Compute | O(n) | O(n²) self-attention |
| Best | Small data, edge deploy | Large data, SOTA |

- **Hybrid** (best): CNN early layers + Transformer late layers (CoAtNet, EfficientFormer)
- **Follow-up**: "Why ViT needs more data?" → No inductive bias → must learn translation invariance from data → needs more examples.

### Q20: mIoU tại sao tốt cho segmentation?
**A**: Pixel accuracy misleading — background = 80% pixels → predict all BG → 80% accuracy, 0% for objects. mIoU: each class equal weight → penalizes ignoring minority classes. **Always report per-class IoU + mIoU.**

### Q21: Focal Loss vs CE?
**A**: 
```
CE:    -log(p_t)                          → equal weight all pixels
Focal: -α(1-p_t)^γ × log(p_t)          → down-weight easy pixels (p_t→1)

γ=0 → standard CE
γ=2 → default, significant down-weighting of easy examples
α   → class balancing factor
```
**When**: severe class imbalance (detection: 1 object vs 1000s background).

### Q22: EfficientNet compound scaling?
**A**: Scale 3 dimensions simultaneously: depth (d), width (w), resolution (r).
- Constraint: d × w² × r² ≈ 2 (FLOPS budget)
- B0 → B7: progressively larger, better accuracy
- Innovation: NAS found optimal scaling coefficients
- **Follow-up**: "EfficientNetV2?" → Training-aware NAS + progressive learning (start small, grow resolution).

### Q23: Image Resolution impact?
**A**: Higher resolution → see more detail + smaller objects. But: quadratic memory (2× res → 4× memory). 
- Detection: 640→1280 = significant improvement on small objects
- Segmentation: 512 standard, 768-1024 for fine details
- Classification: 224-384 typical
- **Progressive resizing**: train at 224 → fine-tune at 384 (faster training, better accuracy)

### Q24: Attention in CNN (SE, CBAM)?
**A**: 
- **SE (Squeeze-Excitation)**: channel attention → learn "which channels are important"
- **CBAM**: channel + spatial attention → which channels + where to focus
- Added to each block, minimal overhead (~1% more params), consistent ~1% accuracy gain
- **Self-attention (Non-Local)**: full spatial attention → expensive but powerful

---

## Deployment & Production (10 câu)

### Q25: ONNX why production?
**A**: Framework-agnostic → any hardware. Optimized graph (constant folding, dead node elimination). ONNX Runtime: 2-5x faster. Hardware providers (NVIDIA, Intel, Apple) optimize for ONNX.

### Q26: Quantization deep dive?
**A**:
- **Dynamic PTQ**: quantize weights statically, activations dynamically. Easiest. Good for LSTM/Linear.
- **Static PTQ**: calibrate with representative data. Better accuracy. Needs ~100-1000 samples.
- **QAT**: insert fake quant nodes during training. Best accuracy but requires retraining.
- **INT8**: 3-4x faster, ~25% model size, <1% accuracy drop (usually)
- **INT4**: for LLMs (GPTQ, AWQ). 6-8x compression, some quality drop.

### Q27: TensorRT optimization?
**A**: NVIDIA-specific. Fuses layers, optimizes kernels for specific GPU. 
- FP16: 2-3x over PyTorch (automatic)
- INT8: 4-6x (needs calibration)
- Best via: `torch.compile(backend="tensorrt")` or `trtexec --onnx=model.onnx`

### Q28: Model serving patterns?
**A**:
1. **Synchronous**: FastAPI + ONNX Runtime. Simple. <100 QPS.
2. **Async batching**: Triton. Collect requests → batch → GPU inference → split. 5-10x throughput.
3. **Serverless**: Cloud Run / Lambda. Scale to zero. Cold start issue for ML.
4. **Streaming**: Kafka → model → Kafka. Real-time pipelines.

### Q29: A/B testing models?
**A**: (1) Define primary metric (accuracy? latency?). (2) Split traffic (90/10, not 50/50). (3) Run sufficient time (statistical power). (4) Paired test (p<0.05). (5) Check guardrails (latency regression?). (6) Roll out or rollback.

### Q30: Knowledge Distillation?
**A**: Large teacher → small student. `Loss = α × CE(student, labels) + (1-α) × KL(student_soft, teacher_soft)`. Temperature T softens distributions → more information transfer. Deploy student (10-100x smaller).
- **Self-distillation**: same architecture, teach yourself → also improves!

### Q31: Model monitoring in production?
**A**: 5 metrics:
1. **Prediction distribution**: drift from training distribution?
2. **Latency**: P50, P95, P99 response time
3. **Error rate**: failed predictions / total
4. **Input quality**: null values, out-of-range features
5. **Business metrics**: CTR, conversion, revenue

### Q32: GPU memory optimization?
**A**: When model doesn't fit in GPU:
1. Mixed Precision (AMP) → ~40% reduction
2. Gradient Accumulation → smaller batch, same effective batch
3. Gradient Checkpointing → recompute vs store (2x slower, 3-4x less memory)
4. Model parallelism → split across GPUs
5. DeepSpeed ZeRO → shard optimizer states
6. Reduce input resolution

### Q33: Docker for ML?
**A**: Multi-stage Dockerfile:
```dockerfile
# Stage 1: install deps
FROM python:3.11-slim AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: runtime (smaller image)
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY model.onnx app.py ./
CMD ["python", "app.py"]
```
Always: pin versions, use slim/distroless images, copy only needed files.

### Q34: CI/CD for ML?
**A**:
```
Push code → Run tests (unit + integration)
         → Train on sample data (smoke test)
         → Evaluate on benchmark (quality gate)
         → Build Docker image
         → Deploy to staging → integration tests
         → Canary deployment (10% traffic)
         → Full rollout
```
