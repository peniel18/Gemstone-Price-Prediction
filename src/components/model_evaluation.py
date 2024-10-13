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




@dataclass
class ModelEvaluationConfig:
    pass 


class ModelEvaluation:
    def __init__(self):
        pass

    def InitiateModelEvaluation(self):
        try: 
            pass 
        except Exception as e: 
            logging.info()
            raise CustomException(e, sys)
