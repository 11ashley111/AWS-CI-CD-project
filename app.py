from src.fifty_k.exception import CustomException
from src.fifty_k.logger import logging
import sys

if __name__=='__main__':
    logging.info("the execution is started")
    
    try:
        a=10+20
        
    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)
    