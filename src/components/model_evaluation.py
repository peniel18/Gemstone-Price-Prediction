import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import numpy as np 
import pandas as pd 
from src.logger.Logging import logging
import pickle
from src.utils.utils import load_object
from src.exception.exception import CustomException
import mlflow 
import mlflow.sklearn
from urllib.parse import urlparse 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 


@dataclass
class ModelEvaluationConfig:
    pass 


class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation has started")

        
    def eval_metrics(self,y_true, preds):
        rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true=y_true, y_pred=preds)
        logging.info("Evaluating metrics")
        return rmse, mae, r2 
    
    def InitiateModelEvaluation(self, trainData, testData):
        try: 
            X_test, y_test = testData[:,:-1], testData[:, -1]
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run(): 
                predictions = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, preds=predictions)
                # log metrics 
                mlflow.log_metrics("rmse", rmse) 
                mlflow.log_metrics("mae", mae)
                mlflow.log_metrics("r2", r2)   

        except Exception as e: 
            logging.info()
            raise CustomException(e, sys)
    
