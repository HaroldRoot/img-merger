import logging

from colorama import init, Fore

init(autoreset=True)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        prefix_color = {
            logging.INFO: Fore.GREEN,
            logging.ERROR: Fore.RED
        }.get(record.levelno, Fore.RESET)
        formatted_message = super().format(record)
        return f"{prefix_color}{formatted_message}{Fore.RESET}"


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = CustomFormatter('[%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
