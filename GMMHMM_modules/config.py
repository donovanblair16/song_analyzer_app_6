# =============================================================================
# FILE: config.py
# Parent: GMMHMM_modules
# Purpose: Centralize configuration settings for GMMHMM training.
# ADDED: spectral_centroid_avg feature definition.
# =============================================================================

import os

# --- Path Configuration ---
# Assumes config.py is inside GMMHMM_modules, which is inside the main project folder
PROJECT_BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_BASE_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "completed_analyses")
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")
TRANSITION_MATRIX_PATH = os.path.join(HMM_OUTPUT_FOLDER, "transition_matrix.npy")
LABELS_TO_IGNORE = [
    "Fade Out",
    "Start",
    "Error",
    "Fill",
]  # Labels excluded during feature prep/training

# --- Feature Configuration ---
# List of ALL features the application calculates and could potentially be used for training.
ALL_FEATURE_KEYS = [
    "avg_rms",  # Average loudness
    "relative_rms",  # Loudness relative to track max
    "relative_position",  # Position in track (0=start, 1=end)
    "position_context",  # Emphasizes start/end (1=ends, 0=middle)
    "label_proportion",  # Proportion of this label in the track
    "low_energy_norm",  # Normalized low frequency energy (<150Hz)
    "rms_std_dev_section",  # Loudness variation within the section
    "centroid_std_dev_section",  # Timbre variation within the section
    "delta_rms",  # Change in avg_rms from previous section
    "delta_centroid",  # Change in spectral_centroid_avg from previous section
    "rms_trend",  # Slope of RMS within the section (rising/falling)
    "crest_factor",  # Peak-to-average ratio within the section
    "spectral_centroid_slope",  # Slope of spectral centroid within the section
    "spectral_centroid_avg",  # <-- ADDED: Average spectral centroid (brightness)
]

# Descriptions for the features (used in GUI tooltips, etc.)
FEATURE_DESCRIPTIONS = {
    "avg_rms": "Average Loudness (RMS)",
    "relative_rms": "Relative Loudness (to track max)",
    "relative_position": "Relative Position (0=start, 1=end)",
    "position_context": "Position Emphasis (1=Ends, 0=Middle)",
    "label_proportion": "Label Proportion (Count[Label]/Total Sections)",
    "low_energy_norm": "Low Frequency Energy (<150Hz)",
    "rms_std_dev_section": "Loudness Variation (within section)",
    "centroid_std_dev_section": "Timbre Variation (within section)",
    "delta_rms": "Loudness Change (from previous)",
    "delta_centroid": "Timbre Change (from previous)",
    "rms_trend": "Loudness Trend (within section)",
    "crest_factor": "Peak/Average Ratio (dynamics)",
    "spectral_centroid_slope": "Brightness Trend (within section)",
    "spectral_centroid_avg": "Average Spectral Centroid (Brightness)",  # <-- ADDED
}

# Default feature weights (used if trainer GUI allows weighting)
DEFAULT_FEATURE_WEIGHTS = {
    "avg_rms": 1.0,
    "relative_rms": 5.0,
    "relative_position": 0.01,
    "position_context": 2.0,
    "label_proportion": 1.5,
    "low_energy_norm": 2.0,
    "rms_std_dev_section": 1.0,
    "centroid_std_dev_section": 0.5,
    "delta_rms": 2.0,
    "delta_centroid": 1.0,
    "rms_trend": 1.5,
    "crest_factor": 1.0,
    "spectral_centroid_slope": 1.0,
    "spectral_centroid_avg": 1.0,  # <-- ADDED: Default weight
}

# Default features selected ON in the trainer GUI
# NOTE: spectral_centroid_avg is NOT added here by default.
#       You will need to manually select it in the trainer GUI.
DEFAULT_FEATURES_ON = [
    "relative_rms",
    "position_context",
    "low_energy_norm",
    "centroid_std_dev_section",
    "delta_rms",
    "crest_factor",
    "spectral_centroid_slope",
    "label_proportion",
]


# --- Default Data Cleaning Configuration (used in GUI) ---
DEFAULT_ENABLE_OUTLIER_REMOVAL = True
DEFAULT_OUTLIER_Z_THRESHOLD = 3.0
DEFAULT_ENABLE_SHORT_SECTION_REMOVAL = True
DEFAULT_SHORT_SECTION_MIN_BARS = 2
DEFAULT_ENABLE_CONSISTENCY_FILTERING = True

# --- Default HMM Training Parameters (used in GUI) ---
DEFAULT_HMM_N_MIXTURES = 1
DEFAULT_HMM_MIN_COVAR = 0.1
DEFAULT_HMM_COVARIANCE_TYPE = "diag"  # 'diag', 'full', 'tied', 'spherical'
DEFAULT_HMM_N_ITER = 100
DEFAULT_HMM_TOL = 1e-3
DEFAULT_HMM_RANDOM_STATE = 42
DEFAULT_USE_TRANSITION_PRIOR = True  # Use pre-calculated transition matrix
