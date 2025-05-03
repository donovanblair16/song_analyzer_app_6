# debug_utils.py
"""
Debug utilities for the Audio Analyzer Tool.

This module provides functions for consistent debugging across the application,
respecting the debug settings in main_app_config.py.
"""

import logging
import time
import functools
import inspect
from main_app_config import (
    DEBUG_ENABLED,
    DEBUG_AREAS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_TO_CONSOLE,
    LOG_TO_FILE,
    LOG_FILE,
)

# Set up logging
logger = logging.getLogger("audio_analyzer")
logger.setLevel(LOG_LEVEL)

# Create handlers
if not logger.handlers:  # Avoid adding handlers multiple times
    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)

    if LOG_TO_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)


def debug_print(area, message, *args):
    """
    Print a debug message if debugging is enabled for the specified area.

    Args:
        area (str): The debug area ('UI', 'ANALYSIS', etc.)
        message (str): The message to print
        *args: Additional values to include in the message
    """
    if DEBUG_ENABLED and DEBUG_AREAS.get(area, False):
        # Get caller information
        frame = inspect.currentframe().f_back
        caller_module = inspect.getmodule(frame)
        caller_function = frame.f_code.co_name
        caller_line = frame.f_lineno

        # Format the message with additional context
        formatted_message = f"[{area}] {caller_module.__name__}.{caller_function}:{caller_line} - {message}"
        if args:
            formatted_message += " " + " ".join(str(arg) for arg in args)

        logger.debug(formatted_message)


def debug_timing(area):
    """
    Decorator to measure and log the execution time of a function.

    Args:
        area (str): The debug area this timing belongs to

    Returns:
        function: The decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not (
                DEBUG_ENABLED
                and DEBUG_AREAS.get(area, False)
                and DEBUG_AREAS.get("TIMING", False)
            ):
                return func(*args, **kwargs)

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            execution_time = end_time - start_time
            debug_print("TIMING", f"{func.__name__} took {execution_time:.4f} seconds")

            return result

        return wrapper

    return decorator


def function_trace(area):
    """
    Decorator to trace function entry and exit.

    Args:
        area (str): The debug area this trace belongs to

    Returns:
        function: The decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not (DEBUG_ENABLED and DEBUG_AREAS.get(area, False)):
                return func(*args, **kwargs)

            arg_str = ", ".join(
                [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
            )
            debug_print(area, f"ENTER {func.__name__}({arg_str})")

            result = func(*args, **kwargs)

            debug_print(area, f"EXIT {func.__name__} -> {repr(result)}")
            return result

        return wrapper

    return decorator
