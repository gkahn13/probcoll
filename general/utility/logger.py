import logging
from colorlog import ColoredFormatter

_color_formatter = ColoredFormatter(
    "%(log_color)s%(name)-16s %(levelname)-8s%(reset)s %(white)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

_normal_formatter = logging.Formatter(
    '%(name)-16s %(levelname)-8s %(message)s'
)

DEBUG_LOG_PATH = '/tmp/log.txt'
def set_log_path(path):
    global DEBUG_LOG_PATH
    DEBUG_LOG_PATH = path

_LOGGERS = {}
def get_logger(name, lvl=logging.INFO, log_path=DEBUG_LOG_PATH):
    global _LOGGERS

    if isinstance(lvl, str):
        lvl = lvl.lower().strip()
        if lvl == 'debug': lvl = logging.DEBUG
        elif lvl == 'info': lvl = logging.INFO
        elif lvl == 'warn' or lvl == 'warning': lvl = logging.WARN
        elif lvl == 'error': lvl = logging.ERROR
        elif lvl == 'fatal' or lvl == 'critical': lvl = logging.CRITICAL
        else: raise ValueError('unknown logging level')

    logger = _LOGGERS.get(name, None)
    if logger is not None: return logger

    # file_handler = logging.FileHandler(GET_FILE_MAN().debug_log_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_normal_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    console_handler.setFormatter(_color_formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    _LOGGERS[name] = logger
    return logger
