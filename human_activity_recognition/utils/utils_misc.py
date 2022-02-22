import logging
import tensorflow as tf

def remove_handler():
    logger = logging.getLogger()
    logger.removeHandler(logger.handlers[-1])

def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False, del_prev_handler=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        if del_prev_handler:
            logger.removeHandler(logger.handlers[-1])
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)