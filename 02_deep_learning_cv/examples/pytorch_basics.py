"""
🔥 PyTorch Basics Demo — Tensors, Autograd, Simple Model
Chạy: pip install torch torchvision
       python pytorch_basics.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


def tensor_demo():
    """Demonstrate tensor operations."""
    print("=== 1. Tensor Operations ===\n")
    
    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Tensor: {x}, dtype: {x.dtype}, device: {x.device}")
    
    # Random tensors
    rand_t = torch.randn(3, 4)  # Normal distribution
    print(f"Random shape: {rand_t.shape}")
    
    # Device management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    x_gpu = x.to(device)
    
    # Reshaping
    batch = torch.randn(8, 3, 32, 32)  # [B, C, H, W]
    flat = batch.view(8, -1)            # [B, 3*32*32]
    print(f"Reshape: {batch.shape} → {flat.shape}")
    
    # Matrix operations
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = A @ B  # Matrix multiplication
    print(f"MatMul: ({A.shape}) @ ({B.shape}) = {C.shape}")


def autograd_demo():
    """Demonstrate autograd (automatic differentiation)."""
    print("\n=== 2. Autograd Demo ===\n")
    
    # requires_grad=True → track gradients
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3*x + 1  # y = x² + 3x + 1
    
    y.backward()  # Compute dy/dx
    print(f"x = {x.item()}")
    print(f"y = x² + 3x + 1 = {y.item()}")
    print(f"dy/dx = 2x + 3 = {x.grad.item()}")  # 2(2) + 3 = 7
    
    # Neural net gradient flow
    w = torch.randn(3, 2, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    inp = torch.randn(5, 3)
    
    out = inp @ w + b
    loss = out.sum()
    loss.backward()
    
    print(f"\nWeight gradient shape: {w.grad.shape}")
    print(f"Bias gradient shape: {b.grad.shape}")


def simple_model_demo():
    """Train a simple neural network on synthetic data."""
    print("\n=== 3. Simple Model Training ===\n")
    
    # Generate synthetic data (binary classification)
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 4).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.float32)
    
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1)
    
    # Split
    train_size = 800
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define model
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_dataset)} samples...\n")
    
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_preds = (torch.sigmoid(test_output) > 0.5).float()
                accuracy = (test_preds == y_test).float().mean()
                avg_loss = epoch_loss / len(train_loader)
                print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, "
                      f"Accuracy={accuracy:.4f}, LR={scheduler.get_last_lr()[0]:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_preds = (torch.sigmoid(test_output) > 0.5).float()
        final_acc = (test_preds == y_test).float().mean()
    
    print(f"\n🎯 Final Test Accuracy: {final_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "simple_model.pth")
    print("💾 Model saved to simple_model.pth")


def model_inspection():
    """Inspect model architecture and parameters."""
    print("\n=== 4. Model Inspection ===\n")
    
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable:,}")
    
    # Layer-by-layer
    print("\nArchitecture:")
    for name, layer in model.named_children():
        params = sum(p.numel() for p in layer.parameters())
        print(f"  [{name}] {layer.__class__.__name__:20s} → {params:,} params")
    
    # Test forward pass
    dummy = torch.randn(2, 3, 32, 32)
    output = model(dummy)
    print(f"\nInput: {dummy.shape} → Output: {output.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("🔥 PyTorch Basics Demo")
    print("=" * 60)
    
    tensor_demo()
    autograd_demo()
    simple_model_demo()
    model_inspection()
    
    # Cleanup
    import os
    if os.path.exists("simple_model.pth"):
        os.remove("simple_model.pth")
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)
