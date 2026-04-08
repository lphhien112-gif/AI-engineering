# 🧠 02 — Deep Learning & Computer Vision

> Neural Networks, CNN, Transfer Learning, Semantic Segmentation, Object Detection, Model Deployment.

## 📚 Docs

| File | Chủ đề | Thời gian đọc |
|------|--------|---------------|
| [01_neural_networks.md](docs/01_neural_networks.md) | Perceptron, Backprop, Activations, Optimizers, Regularization | ~15 min |
| [02_cnn_architectures.md](docs/02_cnn_architectures.md) | Conv, Pooling, ResNet, EfficientNet, SegFormer | ~15 min |
| [03_transfer_learning.md](docs/03_transfer_learning.md) | Pretrained Models, Fine-tuning Strategies, Feature Extraction | ~10 min |
| [04_segmentation_detection.md](docs/04_segmentation_detection.md) | U-Net, YOLO, mIoU, Data Augmentation | ~12 min |
| [05_training_recipes.md](docs/05_training_recipes.md) | Training Loop, AMP, LR Schedule, Gradient Accumulation | ~12 min |
| [06_model_deployment.md](docs/06_model_deployment.md) | ONNX, TorchScript, Quantization, TensorRT | ~10 min |

## 💻 Examples

```bash
cd 02_deep_learning_cv/examples
python pytorch_basics.py        # Tensors, autograd, simple model
python cnn_classifier.py        # CIFAR-10 CNN training
python transfer_learning.py     # Fine-tune pretrained ResNet
python onnx_export.py           # Export & inference with ONNX
```

## ✅ Checklist
- [ ] Đọc hết 6 docs
- [ ] Chạy 4 examples
- [ ] Trả lời 30+ câu trong `interview/deep_learning_qa.md`
