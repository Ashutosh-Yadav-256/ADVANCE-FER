print("Script starting...", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.fer_model import FERModel
from data.fer_dataset import FER2013Dataset
from tqdm import tqdm
import os
import torchvision.transforms as transforms

def train_one_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, landmarks, labels, va_targets in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        va_targets = va_targets.to(device)
        
        optimizer.zero_grad()
        
        logits, va_preds = model(images, landmarks)
        
        loss_cls = criterion_cls(logits, labels)
        # loss_reg = criterion_reg(va_preds, va_targets) # FER2013 has no VA labels, so skip or use dummy
        # For now, just optimize classification
        loss = loss_cls
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, criterion_cls, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, landmarks, labels, va_targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            logits, _ = model(images, landmarks)
            loss = criterion_cls(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(dataloader), 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 16 # Small batch size because of on-the-fly MediaPipe
    LR = 0.0001
    EPOCHS = 10
    DATA_DIR = os.path.join("data", "fer2013")
    
    if not os.path.exists(DATA_DIR):
        print(f"Dataset not found at {DATA_DIR}. Please run 'python download_data.py' first.")
        return

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("Loading datasets...")
    train_dataset = FER2013Dataset(DATA_DIR, split='train', transform=transform)
    test_dataset = FER2013Dataset(DATA_DIR, split='test', transform=transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("Error: Train dataset is empty!")
        return
    
    # DataLoaders
    # num_workers=0 is crucial because MediaPipe is not fork-safe
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Initializing model...")
    # Model
    model = FERModel(num_classes=7).to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion_cls, criterion_reg, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion_cls, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/fer_model_best.pth")
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == '__main__':
    main()
