import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import numpy as np 
from src.logger.Logging import logging
from src.exception.exception import CustomException
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 



@dataclass
class DataTransformationConfig:
    pass 


class DataTransformation:
    def __init__(self):
        pass

    def InitialDataIngestion(self):
        try: 
            pass 
        except Exception as e: 
            logging.info()
            raise CustomException(e, sys)
