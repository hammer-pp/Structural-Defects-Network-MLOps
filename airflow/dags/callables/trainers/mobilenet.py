import torch
import os
import logging
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
          num_epochs=10, save_path="/opt/airflow/model/mobilenet_model.pth"):
    
    best_f1 = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

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

        logger.info(f"[MobileNet] Epoch {epoch+1}: "
                    f"Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
                    f"Val Loss {val_loss:.4f}, F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Saved best model at epoch {epoch+1} with F1 {val_f1:.4f}")
