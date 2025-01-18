import sys
import os
import pandas as pd 
import numpy as np
from src.fifty_k.exception import CustomException
from src.fifty_k.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.fifty_k.components.data_transformation import DataTransformation
from src.fifty_k.components.model_tranier import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path =os.path.join("artifacts",'train.csv')
    test_data_path =os.path.join("artifacts",'test.csv')
    raw_data_path =os.path.join("artifacts",'raw.csv')
    
class DataIngestion:
    
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        
        logging.info("data ingestion started")
        try:
            df =pd.read_csv("notebook/adult.csv")
            logging.info("read the data")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path ,index=False)
            
            logging.info("train test split initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion complete")
            return(
                self.ingestion_config.train_data_path,
                
                self.ingestion_config.test_data_path
            
            )
        
        except Exception as e:
            
            logging.info("error occured in data ingestion ")
            raise CustomException(e,sys)
         
         
         #++++++++++++++++++++++++++++++++++++++++++
         
if __name__ =='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    
    
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    
    modeltrainer=ModelTrainer()
    
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

