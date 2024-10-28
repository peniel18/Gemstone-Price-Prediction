from __future__ import annotations
import json 
from textwrap import dedent
import pendulum 
from airflow import DAG 
from airflow.hooks.S3_hook import S3Hook
from airflow.operators.python import PythonOperator 
from src.pipeline.training_pipeline import TrainingPipeline


trainingPipeline = TrainingPipeline()

with DAG(
    dag_id = "gemstone_training_pipeline", 
    default_args = {"retries": 2}, 
    description = "Training Pipeline", 
    schedule = "@weekly", 
    start_date = pendulum.datetime(2024, 10, 25), 
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


    def model_evaluation(**kwargs):
        import numpy as np 
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(
            task_ids = "data_transformation", 
            key = "data_transformation_artifacts"
        )
        trainData = np.array(data_transformation_artifact["trainData"])
        testData = np.array(data_transformation_artifact["testData"])
        trainingPipeline.eval_model_metrics(trainData=trainData, testData=testData)
    ## push to cloud 

    def push_artifacts_to_s3(**kwargs):
        import os 
        bucket_name = "gemsartifacts"
        artifacts_folder = "/app/artifacts"


    data_ingestion_task = PythonOperator(
        task_id = "data_ingestion", 
        python_callable = data_ingestion
    )

    data_ingestion_task.doc_md  = dedent(
        """
        \ ingestion task 
        this task creates a train and test file 
        """
    )

    data_transformation_task = PythonOperator(
        task_id = "data_transformation", 
        python_callable = data_transformation
    )
    data_transformation_task.doc_md = dedent(
        """
        \ Transformation Task 
        performs the transformation 
        """
    )

    model_trainer_task = PythonOperator(
        task_id = "model_trainer", 
        python_callable = model_trainer
    )

    model_trainer_task.doc_md = dedent(
        """
        \
        model trainer 
        performs model training 
        """
    )

    model_evaluation_task = PythonOperator(
        task_id = "model_evaluation", 
        python_callable = model_evaluation
        )

    model_evaluation_task.doc_md = dedent(
        """
        \ 
        model evaluation

        performs model evaluation

        """
    )



    # push to cloud task here 



data_ingestion_task >> data_transformation_task >> model_trainer_task >> model_evaluation_task