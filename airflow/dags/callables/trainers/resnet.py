import torch
import os
import logging
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, save_path="../model/resnet_model.pth"):
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds)
        scheduler.step(val_loss)

        logger.info(
            f"[ResNet] Epoch {epoch+1}: "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f}, F1 {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Best model saved at epoch {epoch+1} with F1 {best_f1:.4f}")
