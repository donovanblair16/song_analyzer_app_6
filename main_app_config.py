# main_app_config.py
"""
Configuration module for the Audio Analyzer Tool.

This module centralizes all configuration parameters, paths, and constants
used throughout the application that were previously defined in main_app.py.
"""

import os
import logging

# Base folder paths
ANALYSIS_BASE_FOLDER = (
    "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"
)
PROJECT_BASE_FOLDER = os.path.dirname(ANALYSIS_BASE_FOLDER)
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")

# Subfolder names (used by FileManager/SectionEditor)
PERFECT_SUBFOLDER = "Perfect"
WIP_SUBFOLDER = "WIP"

# HMM model configuration
N_FEATURES_EXPECTED = 6
N_MIXTURES_EXPECTED = 1
CLEANING_FLAGS_EXPECTED = "out_sh_con"

# Construct the filename parts
feature_str = f"{N_FEATURES_EXPECTED}f"
mixture_str = f"{N_MIXTURES_EXPECTED}m"
base_filename = f"gmmhmm_{feature_str}_{mixture_str}_{CLEANING_FLAGS_EXPECTED}"

# Construct the full paths using the base filename
GMMHMM_MODEL_PATH = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_model.joblib")
GMMHMM_AUX_PATH = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_aux.joblib")

# Minimum section duration allowed after a split (in seconds)
MIN_SPLIT_SECTION_DURATION_SEC = 1.0

# ===== Debug Configuration =====
# Main debug setting - set to True to enable all debugging
DEBUG_ENABLED = True

# Specific debug areas - these are only checked if DEBUG_ENABLED is True
DEBUG_AREAS = {
    "INITIALIZATION": False,  # Initialization and setup
    "UI": True,  # UI state changes, button updates, etc.
    "ANALYSIS": True,  # Analysis process details
    "PLAYBACK": True,  # Audio playback operations
    "PLOT": True,  # Plot generation and updates
    "FILE": True,  # File operations
    "SECTION": True,  # Section manipulation
    "HMM": True,  # HMM prediction
    "TIMING": False,  # Performance timing measurements
}

# Logging configuration
LOG_LEVEL = logging.DEBUG if DEBUG_ENABLED else logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_BASE_FOLDER, "app.log")
LOG_TO_CONSOLE = True
LOG_TO_FILE = False
