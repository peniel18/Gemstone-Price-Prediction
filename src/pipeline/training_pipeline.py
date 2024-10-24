import os 
import sys
import pandas as pd 
from src.logger.Logging import logging
from src.exception.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

IngestionObj = DataIngestion()
#get train and test data paths 
# data ingestion
train_data_path, test_data_path = IngestionObj.InitialDataIngestion()
# data transformation 
dataTransformation = DataTransformation()
trainData, testData = dataTransformation.InitialDataTransformation(train_path=train_data_path, test_path=test_data_path)
# data training 
modelTrainer = ModelTrainer()
modelTrainer.InitiateModelTraining(trainData=trainData, testData=testData)
# model evaluation 
modelEvaluation = ModelEvaluation()
modelEvaluation.InitiateModelEvaluation(trainData=trainData, testData=testData)

class TrainingPipeline: 
    def start_data_ingestion(self):
        try: 
            dataIngestion = DataIngestion()
            trainPath, testPath = dataIngestion.InitialDataIngestion()
            return trainPath, testPath
        except Exception as e:
            logging.error("Error occured during data ingestion") 
            CustomException(e, sys)

    def start_data_transformation(self):
        pass 