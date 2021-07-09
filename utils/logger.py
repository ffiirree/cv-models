import time
import os
import logging
from os.path import dirname, abspath, exists, join

__all__ = ['make_logger']


def make_logger(run_name, log_dir='logs', rank: int = 0):
    logger = logging.getLogger(run_name)
    logger.propagate = False

    log_filepath = join(
        log_dir, f'{run_name}_{time.strftime("%Y%m%d_%H%M%S", time.localtime()) }.log')

    log_dir = dirname(abspath(log_filepath))
    if not exists(log_dir):
        os.makedirs(log_dir)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(
            log_filepath, mode='a', encoding='utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s:%(lineno)d[%(levelname)s]: %(message)s',
            datefmt='%H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    return logger
