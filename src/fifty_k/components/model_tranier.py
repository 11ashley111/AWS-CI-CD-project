import os 
import sys
import pandas as pd
import numpy as np
from src.fifty_k.logger import logging
from src.fifty_k.exception import CustomException
from src.fifty_k.utils import save_object
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.fifty_k.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting our data into dependent and independent features")
            X_train, y_train, X_test, y_test  = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            models={
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic": LogisticRegression()
            }
            
            params= {
                "Random Forest":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                "Decision Tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },
                "Logistic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                
                }
            }
            
            model_report:dict =evaluate_model(X_train,y_train,X_test,y_test,models,params)    
            
            
           # Get the best model from the report
            best_model_score = max(model_report.values())
            best_model_name = [key for key, val in model_report.items() if val == best_model_score][0]
            best_model = models[best_model_name]

            
            
            print(f"Best Model Found: {best_model_name} with Accuracy: {best_model_score}")
            logging.info(f"Best Model Found: {best_model_name} with Accuracy: {best_model_score}")

            logging.info("Fitting the best model on the full training data")
            best_model.fit(X_train, y_train)

            save_object(file_path=self.model_trainer_config.train_model_file_path,obj=best_model)
            
            
        
        except Exception as e:
            raise CustomException(e,sys)
