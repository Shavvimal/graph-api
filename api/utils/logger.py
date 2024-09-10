import logging
from api.utils.col import Col

class CustomFormatter(logging.Formatter):
    """Custom Formatter that adds colors to log levels."""

    # Initialize the color class
    col = Col()

    # Define color formats for different log levels
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    FORMATS = {
        logging.DEBUG: f'{col.HEADER}%(asctime)s - %(name)s - %(levelname)s - %(message)s{col.ENDC}',
        logging.INFO: f'{col.CYAN}%(asctime)s - %(name)s - %(levelname)s - %(message)s{col.ENDC}',
        logging.WARNING: f'{col.WARNING}%(asctime)s - %(name)s - %(levelname)s - %(message)s{col.ENDC}',
        logging.ERROR: f'{col.FAIL}%(asctime)s - %(name)s - %(levelname)s - %(message)s{col.ENDC}',
        logging.CRITICAL: f'{col.FAIL}{col.BOLD}%(asctime)s - %(name)s - %(levelname)s - %(message)s{col.ENDC}',
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(debug_console: bool = False) -> logging.Logger:
    """Sets up the logger with optional debug output to the console."""
    logger = logging.getLogger("API")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_console else logging.INFO)
    console_handler.setFormatter(CustomFormatter())

    logger.addHandler(console_handler)
    return logger

