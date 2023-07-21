# logger.py

import logging

class Logger:
    def __init__(self, log_file_name, log_level=logging.INFO, terminal=False, logger_name=''):
        # Create a custom logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Create handlers
        if terminal:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(log_level)

        f_handler = logging.FileHandler(log_file_name)
        f_handler.setLevel(log_level)

        # Create formatters and add it to handlers
        if terminal: 
            c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(c_format)
    
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        if terminal:
            self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def get_logger(self):
        return self.logger
