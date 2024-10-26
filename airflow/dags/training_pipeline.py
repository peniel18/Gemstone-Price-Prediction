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
) as dag: 
    
    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]  # task instance 
        trainDataPath, testDataPath = trainingPipeline.start_data_ingestion()
        ti.xcom_push(
            key = "data_ingestion_artifacts",
            value = {"train_data_path": trainDataPath, "test_data_path": testDataPath}
              )

    
    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifacts = ti.xcom_pull(
            task_ids="data_ingestion",
            key = "data_ingestion_artifacts"
            )
        trainData, testData = trainingPipeline.start_data_transformation(
            trainPath = data_ingestion_artifacts["train_data_path"], 
            testPath = data_ingestion_artifacts["test_data_path"]
            )

        trainData = trainData.tolist()
        testData = testData.tolist()
        transformationArtifact = {
            "trainData": trainData, 
            "testData" : testData
            }
        ti.xcom_push(
            key = "data_transformation_artifacts",
            value =  transformationArtifact
            )

    def model_trainer(**kwargs):
        import numpy as np 
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(
            task_ids = "data_transformation", 
            key = "data_transformation_artifacts"
        )
        trainData = np.array(data_transformation_artifact["trainData"])
        testData = np.array(data_transformation_artifact["testData"]) 
        trainingPipeline.start_model_training(trainData=trainData, testData=testData)