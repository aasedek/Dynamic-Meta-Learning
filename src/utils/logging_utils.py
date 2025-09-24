"""
Logging utilities for DML project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """Centralized logging setup for the DML project."""

    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
    ) -> logging.Logger:
        """
        Set up a logger with specified configuration.

        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            format_string: Optional custom format string

        Returns:
            Configured logger instance
        """
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers to avoid duplication
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(format_string)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


def get_project_logger(name: str) -> logging.Logger:
    """Get a project logger with standard configuration."""
    return LoggerSetup.setup_logger(
        name=name, log_file="dml_training.log", level=logging.INFO
    )
