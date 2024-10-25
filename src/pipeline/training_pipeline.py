import os 
import sys
import pandas as pd 
from src.logger.Logging import logging
from src.exception.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline: 
    def __init__(self):
        pass 

    def start_data_ingestion(self):
        try: 
            dataIngestion = DataIngestion()
            trainPath, testPath = dataIngestion.InitialDataIngestion()
            return trainPath, testPath
        except Exception as e:
            logging.error("Error occured during data ingestion") 
            CustomException(e, sys)

    def start_data_transformation(self, trainPath, testPath):
        try: 
            logging.info("Running start_data_transformation")
            dataTransformation = DataTransformation()
            trainData, testData = dataTransformation.InitialDataTransformation(train_path=trainPath, test_path=testPath)
            return trainData, testData
        except Exception as e: 
            logging.error("Error occured in start_data_transformation")
            CustomException(e, sys)


    def start_model_training(self, trainData, testData):
        try: 
            logging.info("start model training")
            modelTrainer = ModelTrainer()
            modelTrainer.InitiateModelTraining(trainData=trainData, testData=testData)
        except Exception as e: 
            logging.error("Error occured from function start_model_training")
            CustomException(e, sys)

    def start_training(self):
        try: 
            logging.info("Started Training model")
            trainPath, testPath = self.start_data_ingestion()
            trainData, testData = self.start_data_transformation(trainPath, testPath)
            self.start_model_training(trainData, testData)
        except Exception as e: 
            logging.error("Error occured during training process")
            CustomException(e, sys)
        

        