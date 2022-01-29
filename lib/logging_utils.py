import logging
from contextlib import contextmanager
import time

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    FORMATS = {
        logging.ERROR: "%(levelname)s - %(message)s",
        logging.WARNING: "%(levelname)s - %(message)s",
        logging.INFO: "%(levelname)s - %(message)s",
        logging.DEBUG: "%(name)s - %(levelname)s - %(message)s",
        "DEFAULT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logging_level(level):

    if level == 'debug':
        lv = logging.DEBUG
    elif level == 'info':
        lv = logging.INFO
    elif level == 'warning':
        lv = logging.WARNING
    elif level == 'error':
        lv = logging.ERROR
    elif level == 'critical':
        lv = logging.CRITICAL
    else:
        raise ValueError(f'Unknown logging level - {level}')

    return lv

def setup_console_logger(logger=None, level='warning'):

    level = get_logging_level(level)    

    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)


def setup_file_logger(fn, logger=None, level='warning'):

    # formatter=logging.Formatter(
    #     "{'time':'%(asctime)s', 'name': '%(name)s', \
    #     'level': '%(levelname)s', 'message': '%(message)s'}"
    # )

    formatter=logging.Formatter(
        "{'message': '%(message)s'}"
    )

    level = get_logging_level(level)

    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(fn)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
