import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import numpy as np 
import pandas as pd 
from src.logger.Logging import logging
from src.utils.utils import save_object, evaluate_model
from src.exception.exception import CustomException


@dataclass
class ModelTrainerConfig:
    pass 


class ModelTrainer:
    def __init__(self):
        pass

    def InitiateModelTraining(self):
        try: 
            pass 
        except Exception as e: 
            logging.info()
            raise CustomException(e, sys)
