import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class CrackDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'Non-cracked': 0, 'Cracked': 1}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label = self.label_map[self.labels_df.iloc[idx, 1]]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
# Load datasets
train_dataset = CrackDataset('/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/artifact_folder/train/labels.csv', '/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/artifact_folder/train/images', transform)
val_dataset = CrackDataset('/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/artifact_folder/val/labels.csv', '/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/artifact_folder/val/images', transform)
test_dataset = CrackDataset('/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/artifact_folder/test/labels.csv', '/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/artifact_folder/test/images', transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: Cracked / Non-cracked
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Load the state_dict
model.load_state_dict(torch.load('/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/model/resnet_cos_crack_classifier.pth'))

# Set model to evaluation mode
model.eval()
class SoftF1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()

        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        soft_f1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        return 1 - soft_f1  # we want to minimize loss, so use 1 - F1
from sklearn.metrics import f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds)

        if scheduler:
            scheduler.step(val_loss)  # Optionally use val_loss or convert to custom scheduler

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = SoftF1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10)
torch.save(model.state_dict(), '/Users/nanphattongsirisukool/Documents/GitHub/Structural-Defects-Network-MLOps/model/resnet_SoftF1Loss_crack_classifier.pth')