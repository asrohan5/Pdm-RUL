import sys
from src.logger import logging

def error_message_details(error, error_detail:sys):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = 'Unknown'
        line_number = 'Unknown'
    
    error_message = f"Error Occured in Python Script Name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(str(error_message))
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

#if __name__ == "__main__":
#    try:
#        a = 1/0
#    except Exception as e:
#        logging.info("divide by zero error")
#        raise CustomException(e,sys)