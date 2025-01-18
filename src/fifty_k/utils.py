from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from src.fifty_k.logger import logging
from src.fifty_k.exception import CustomException
import os ,sys
import pickle
from sklearn.model_selection import GridSearchCV
from src.fifty_k.exception import CustomException
import sys

def save_object(file_path, obj):
    try:
        # Open the file in binary write mode
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb' ) as file_objt:
            return pickle.load(file_objt)
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            para = params[model_name]

            # Perform GridSearchCV
            GS = GridSearchCV(model, para, cv=5, scoring="accuracy", n_jobs=-1)
            GS.fit(X_train, y_train)

            # Get the best model
            best_model = GS.best_estimator_

            # Make predictions
            y_pred = best_model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_pred)

            # Store the accuracy in the report
            report[model_name] = test_model_accuracy

        return report

    except Exception as e:
        raise CustomException(e,sys)