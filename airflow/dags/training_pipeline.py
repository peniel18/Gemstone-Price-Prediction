from __future__ import annotations
import json 
from textwrap import dedent
import pendulum 
from airflow import DAG 
from airflow.operators.python import PythonOperator 
from src.pipeline.training_pipeline import TrainingPipeline


trainingPipeline = TrainingPipeline()

with DAG(
    dag_id = "gemstone_training_pipeline", 
    deflaut_args = {"retries": 2}, 
    description = "Training Pipeline", 
    schedule = "@weekly", 
    start_data = pendulum.datetime(2024, 10, 25), 
    catchup = False, 
    tags= ["machine_learning", "regression", "gemstone"]
):
    pass 