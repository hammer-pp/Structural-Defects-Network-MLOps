import sys
import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable

sys.path.append(os.path.join(os.path.dirname(__file__)))  # Add the current directory to sys.path
from callables.preprocess import preprocess_data_callable
from callables.check_img import check_new_images
from callables.log_model import log_model
from callables.s3_to_csv import generate_dataset_csv
from callables.train_resnet import train_resnet
from callables.train_mobilenet import train_mobilenet
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'Blabla',
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='crack_cnn_training_pipeline',
    default_args=default_args,
    description='Retrain model when enough new data is available',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['concrete', 'dag'],
) as dag:

    start = EmptyOperator(task_id='start')

    # return s3_to_csv if there are more than 10 imgs, stop_no_data otherwise
    check_data = BranchPythonOperator(
        task_id='check_new_data_from_cloudinary',
        python_callable=check_new_images,
    )
    
    load_new_img_url = PythonOperator(
        task_id='load_new_img',
        python_callable = generate_dataset_csv,
    )
    
    # Define dummy task if not enough data
    skip_training = EmptyOperator(task_id='skip_training')

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_callable,
        op_kwargs={
            'dataset_root': '/opt/airflow/dataset',
            'artifact_folder': '/opt/airflow/artifact_folder',
            'categories': ['Decks', 'Walls', 'Pavements'],
        },
    )

    ResNet_train = EmptyOperator(
        task_id="ResNet_training",
        python_callable = train_resnet,
    )
    MobileNet_train = EmptyOperator(
        task_id="MobileNet_training",
        python_callable = train_mobilenet,
    )

    log = PythonOperator(
        task_id='log_model',
        python_callable=log_model,
        trigger_rule='none_failed_min_one_success',
    )

    end = EmptyOperator(task_id='end')

    # DAG flow with branching
    start >> check_data >> [load_new_img_url, skip_training]
    load_new_img_url >> preprocess >> [ResNet_train, MobileNet_train] >> log >> end
    skip_training >> log >> end