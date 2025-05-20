import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torchvision import transforms

def evaluate_model_on_split(model, split_name, device):
    """
    Evaluates a PyTorch classification model on a specified data split.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        split_name (str): The name of the data split (e.g., 'train', 'val', 'test') to evaluate on.
        device (torch.device or str): The device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - "Accuracy" (float): The accuracy score.
            - "Precision" (float): The precision score.
            - "Recall" (float): The recall score.
            - "F1-Score" (float): The F1 score.
            - "AUC-ROC" (float): The Area Under the ROC Curve.
            - "Confusion Matrix" (list): The confusion matrix as a nested list.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    base_path = f"opt/airflow/artifact_folder/images/{split_name}"
    labels_df = pd.read_csv(os.path.join(base_path, "labels.csv"))
    images_dir = os.path.join(base_path, "images")

    y_true, y_pred, y_score = [], [], []

    for _, row in labels_df.iterrows():
        img_path = os.path.join(images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)[0][1].item()

        label = 1 if row["label"].lower() == "cracked" else 0
        y_true.append(label)
        y_pred.append(pred)
        y_score.append(prob)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_score),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }
