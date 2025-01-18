import os 
import sys
import pandas as pd
import numpy as np
from src.fifty_k.logger import logging
from src.fifty_k.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.base import BaseEstimator ,TransformerMixin
from src.fifty_k.utils import save_object



@dataclass

class DataTransformerConfig :
    preprocess_obj_file_path =os.path.join("artifacts","preprocessor.pkl")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers
            X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
            X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
        return X.values

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()
        
        
    def get_data_transformation_obj(self):
        try:
                
            numerical_features=["age","education_num","capital_gain",
            "capital_loss","hours_per_week"]
            
            categorical_features=["workclass","marital_status","occupation","relationship","race","sex","native_country"]
            
            
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                
                ("outlier_handler",OutlierHandler(columns=numerical_features)),
                
                ("scaler",StandardScaler())
                
            ])
            
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore')),
            ("scaler",StandardScaler(with_mean=False))
            ])
            
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_features),
                ("cat_pipeline",cat_pipeline,categorical_features)
            ]
                
            )
            return preprocessor
            
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            
            
            # Replace '?' with NaN
            train_data.replace('?', np.nan, inplace=True)
            test_data.replace('?', np.nan, inplace=True)
            
            logging.info("? -> nan complete")
            
            # Rename columns to replace '.' with '_'
            train_data.columns = train_data.columns.str.replace('.', '_', regex=False)
            test_data.columns = test_data.columns.str.replace('.', '_', regex=False)

            logging.info(". -> _ complete")

            
            # Map the 'income' column values to 0 and 1
            train_data['income'] = train_data['income'].map({'<=50K': 0, '>50K': 1})
            test_data['income'] = test_data['income'].map({'<=50K': 0, '>50K': 1})
            
            logging.info("income mapping complete ")
            
            preprocessor_obj=self.get_data_transformation_obj()
            
            target_column_name= "income"
            
            ## divide the train dataset to independent and dependent feature
            
            input_features_train_data=train_data.drop(columns=[target_column_name],axis=1)
            target_features_train_data=train_data[target_column_name]
            
            ## divide the train dataset to independent and dependent feature
            
            input_features_test_data=test_data.drop(columns=[target_column_name],axis=1)
            target_features_test_data=test_data[target_column_name]
            
            logging.info("Applying Preprocessing on training and test dataframe")
            
            input_features_train_data_arr =preprocessor_obj.fit_transform(input_features_train_data)
            input_features_train_test_arr=preprocessor_obj.transform(input_features_test_data)
            
           # Ensure target features are reshaped to match the input features
            target_features_train_data = target_features_train_data.values.reshape(-1, 1)
            target_features_test_data = target_features_test_data.values.reshape(-1, 1)

            # Convert to NumPy arrays if they are not already
            input_features_train_data_arr = np.array(input_features_train_data_arr)
            target_features_train_data = np.array(target_features_train_data)

            # Concatenate input and target features for training and testing
            train_arr = np.c_[input_features_train_data_arr, target_features_train_data]
            test_arr = np.c_[input_features_train_test_arr, target_features_test_data]

            
            save_object(
                file_path= self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                test_arr,
                train_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )
            
        
        except Exception as e:
            raise CustomException(e,sys)
        
        