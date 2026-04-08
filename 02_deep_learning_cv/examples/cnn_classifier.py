"""
🖼️ CNN Classifier — CIFAR-10 với Training Loop đầy đủ
Chạy: pip install torch torchvision tqdm
       python cnn_classifier.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 (32×32×3 → 10 classes)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3→32, 32×32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # → 16×16
            nn.Dropout2d(0.25),
            
            # Block 2: 32→64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # → 8×8
            nn.Dropout2d(0.25),
            
            # Block 3: 64→128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → 1×1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    print("=" * 60)
    print("🖼️ CNN CIFAR-10 Classifier")
    print("=" * 60)
    
    # Config
    batch_size = 128
    epochs = 10
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # Datasets
    print("\n📥 Downloading CIFAR-10...")
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    classes = train_dataset.classes
    print(f"   Classes: {classes}")
    print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    model = SimpleCNN(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\n🏋️ Training for {epochs} epochs...")
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            marker = " 🏆"
        else:
            marker = ""
        
        print(f"  Epoch {epoch:2d} │ Train: loss={avg_train_loss:.3f} acc={train_acc:.1f}% │ "
              f"Test: loss={avg_test_loss:.3f} acc={test_acc:.1f}%{marker}")
    
    print(f"\n🎯 Best Test Accuracy: {best_acc:.1f}%")
    
    # Cleanup
    import shutil
    if __name__ == "__main__":
        shutil.rmtree("./data", ignore_errors=True)
    
    print("=" * 60)
    print("✅ Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
