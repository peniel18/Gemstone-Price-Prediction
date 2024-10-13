import os 
import sys 
from dataclasses import dataclass
from pathlib import Path
import pandas as pd 
#import numpy as np 
#from src.logger.Logging import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split



@dataclass
class DataIngestionConfig:
    train_data_source: str = Path("./experiment/train.csv")
    test_data_source: str = Path("./experiments/test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        


    def InitialDataIngestion(self):
        #logging.info("Data Ingestion Started")
        try: 
            # read data from source eg. from API or github
            data = pd.read_csv("https://raw.githubusercontent.com/sunnysavita10/fsdsmendtoend/main/notebooks/data/gemstone.csv")
           # logging.info("Reading data")

            # make a dir from the directory of raw.csv
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            # save data 
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
           # logging.info("Data saved in artifacts folder")

            # Split data 
            train_data, test_data  = train_test_split(data, test_size=0.25)
           # logging.info("Train test split Completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

        except Exception as e: 

            raise CustomException(e, sys)
        
        return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )


if __name__ == "__main__":
    obj = DataIngestion()
    # start data ingestion
    obj.InitialDataIngestion()