import os 
import sys 
import pandas as pd 
from src.exception.exception import CustomException
from src.logger.Logging import logging
from src.utils.utils import load_object


class PredictionPipeline: 
    def __init__(self) -> None:
        pass 

    def predict(self, features):
        try: 
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(preprocessor_path)
            # transform the features 
            TransformedFeatures = preprocessor.transform(features)
            y_hat = model.predict(TransformedFeatures)

            
        except Exception as e: 
            logging.info()
            CustomException(e, sys)


class CustomData: 
    def __init__(self) -> None:
        pass

    def getDataAsDataFrame():
        pass 


if __name__ == "__main__":
    PredictionPipeline().predict()