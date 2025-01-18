import os, sys
from src.fifty_k.logger import logging
from src.fifty_k.exception import CustomException
from src.fifty_k.components.data_ingestion import DataIngestion
from src.fifty_k.components.data_transformation import DataTransformation
from src.fifty_k.components.model_tranier import ModelTrainer
from dataclasses import dataclass

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_training = ModelTrainer()
    model_training.initiate_model_trainer(train_arr, test_arr)
