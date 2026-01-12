import logging


class Logger:
    @staticmethod
    def get_logger(name: str):
        """
        Returns a logger instance with value INFO

        :param name: Logger name (ex: 'MAIN', 'ML', 'DATA')
        :return: logger
        """
        logger = logging.getLogger(name)

        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            ch.setFormatter(formatter)

            logger.addHandler(ch)

        return logger
