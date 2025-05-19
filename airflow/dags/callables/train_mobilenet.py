import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from callables.datasets.crack_dataset import get_dataloaders
from callables.utils.model_utils import get_mobilenet_model
from callables.trainers.mobilenet import train
from airflow.models import Variable
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def train_mobilenet(**kwargs):
    """
    Trains a MobileNet model using training and validation data loaders, evaluates its F1 score on the validation set, 
    and pushes relevant metadata to Airflow XComs and Variables.

    Args:
        **kwargs: Arbitrary keyword arguments, expected to include 'ti' (Airflow TaskInstance).

    Workflow:
        - Selects device (GPU if available, else CPU).
        - Loads training and validation data loaders.
        - Initializes MobileNet model, loss function, optimizer, and learning rate scheduler.
        - Trains the model using the provided data loaders.
        - Evaluates the trained model on the validation set and computes the F1 score.
        - Pushes the model path and validation F1 score to Airflow XCom.
        - Updates Airflow Variable with the last retrain timestamp.
        - Logs completion and best F1 score.

    Side Effects:
        - Saves model metadata and metrics to Airflow XCom and Variables.
        - Logs training completion and F1 score.

    Returns:
        None
    """
    ti = kwargs["ti"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = get_mobilenet_model().to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Train model (returns best val F1)
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    # Evaluate F1 on val set for XCom
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_f1 = f1_score(all_labels, all_preds)

    # Save Airflow metadata
    model_path = "../model/mobilenet_model.pth"
    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="val_f1", value=val_f1)
    Variable.set("last_retrain_time", datetime.utcnow().isoformat())
    logger.info(f"✅ MobileNet training completed. Best F1: {val_f1:.4f}")
