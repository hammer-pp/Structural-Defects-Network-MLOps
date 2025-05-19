import mlflow
import mlflow.pytorch
import logging
import torch
from callables.utils.model_utils import get_resnet_model, get_mobilenet_model

logger = logging.getLogger(__name__)

def log_model(**kwargs):
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids=kwargs['task_id'], key='model_path')
    accuracy = ti.xcom_pull(task_ids=kwargs['task_id'], key='accuracy')
    optimizer_name = ti.xcom_pull(task_ids=kwargs['task_id'], key='optimizer')
    learning_rate = ti.xcom_pull(task_ids=kwargs['task_id'], key='learning_rate')
    epochs = ti.xcom_pull(task_ids=kwargs['task_id'], key='epochs')
    model_type = ti.xcom_pull(task_ids=kwargs['task_id'], key='model_type')

    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    logger.info(f"📦 Logging {model_type} model to MLflow...")

    # Load appropriate model
    if model_type == "resnet":
        model = get_resnet_model()
    elif model_type == "mobilenet":
        model = get_mobilenet_model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with mlflow.start_run(run_name=f"{model_type.capitalize()}_Crack_Classifier"):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=f"{model_type.capitalize()}ConcreteCrackClassifierModel"
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

    logger.info("✅ Model and metadata logged to MLflow.")
