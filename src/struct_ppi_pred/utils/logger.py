import logging

def setup_logger(log_level=logging.INFO):
    """
    Sets up a logger to log messages to the console only.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # Create a logger
    logger = logging.getLogger("PPI_Project")
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Define log format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(console_handler)

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Console-only logger has been successfully set up!")
    logger.debug("This is a debug message.")
    logger.error("An error occurred.")
