import os
import json
import logging
from airflow.models import Variable
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from callables.datasets.crack_dataset import CrackDataset, get_dataloaders
from callables.utils.model_utils import get_resnet_model
from callables.trainers.resnet import train
from callables.evaluation.evaluate import evaluate_model_on_split
logger = logging.getLogger(__name__)

def train_resnet(**kwargs):
    """
    Trains a ResNet model for crack classification, evaluates its performance, and logs metadata.
    This function performs the following steps:
    1. Retrieves data loaders for training, validation, and testing.
    2. Initializes a ResNet model, loss function, optimizer, and learning rate scheduler.
    3. Trains the model using the provided data loaders.
    4. Evaluates the trained model on the validation set.
    5. Pushes the model path and validation accuracy to Airflow XComs.
    6. Updates the Airflow Variable 'last_retrain_time' with the current timestamp.
    Args:
        **kwargs: Arbitrary keyword arguments, expects 'ti' (Airflow TaskInstance) for XCom and Variable operations.
    Returns:
        None
    """
    
    ti = kwargs['ti']
    
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_resnet_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    num_epochs = 10
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs, save_path="/opt/airflow/model/resnet_model.pth")
    
    
    # Save model metadata to XCom for logging
    model_path = "/opt/airflow/model/resnet_model.pth"
    metrics = evaluate_model_on_split(model, 'val', device)

    ti.xcom_push(key="model_type", value="resnet")
    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="accuracy", value=metrics["Accuracy"])  # or test
    ti.xcom_push(key="optimizer", value="Adam")
    ti.xcom_push(key="learning_rate", value=optimizer.param_groups[0]['lr'])
    ti.xcom_push(key="epochs", value=num_epochs)
    ti.xcom_push(key="metrics", value=metrics)

    Variable.set("last_retrain_time", datetime.now().isoformat())
    logger.info(f"ResNet model and metrics saved. Path: {model_path}")