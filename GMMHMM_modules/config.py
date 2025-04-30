# =============================================================================
# FILE: config.py
# Purpose: Centralize configuration settings for GMMHMM training.
# ADDED: label_proportion feature
# =============================================================================

import os

# --- Path Configuration ---
PROJECT_BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_BASE_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "completed_analyses")
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")
TRANSITION_MATRIX_PATH = os.path.join(HMM_OUTPUT_FOLDER, "transition_matrix.npy")
LABELS_TO_IGNORE = ["Fade Out", "Start", "Error", "Fill"]

# --- Feature Configuration ---
ALL_FEATURE_KEYS = [
    "avg_rms",
    "relative_rms",
    "relative_position",
    "position_context",
    "label_proportion",  # <-- ADDED
    "low_energy_norm",
    "rms_std_dev_section",
    "centroid_std_dev_section",
    "delta_rms",
    "delta_centroid",
    "rms_trend",
    "crest_factor",
    "spectral_centroid_slope",
]

FEATURE_DESCRIPTIONS = {
    "avg_rms": "Average Loudness (RMS)",
    "relative_rms": "Relative Loudness (to track max)",
    "relative_position": "Relative Position (0=start, 1=end)",
    "position_context": "Position Emphasis (1=Ends, 0=Middle)",
    "label_proportion": "Label Proportion (Count[Label]/Total Sections)",  # <-- ADDED
    "low_energy_norm": "Low Frequency Energy",
    "rms_std_dev_section": "Loudness Variation",
    "centroid_std_dev_section": "Spectral Variation",
    "delta_rms": "Loudness Change (from previous section)",
    "delta_centroid": "Timbre Change (from previous section)",
    "rms_trend": "Loudness Trend (rising/falling)",
    "crest_factor": "Peak/Average Ratio (dynamics)",
    "spectral_centroid_slope": "Brightness Trend (rising/falling)",
}

# Default feature weights (adjust as needed)
DEFAULT_FEATURE_WEIGHTS = {
    "avg_rms": 1.0,  # Reduced, often redundant with relative_rms
    "relative_rms": 5.0,  # High weight for loudness context
    "relative_position": 0.01,  # Low weight, info captured by position_context
    "position_context": 2.0,  # Moderate weight for position emphasis
    "label_proportion": 1.5,  # <-- ADDED: Moderate starting weight
    "low_energy_norm": 2.0,
    "rms_std_dev_section": 1.0,
    "centroid_std_dev_section": 0.5,  # Reduced based on previous test
    "delta_rms": 2.0,
    "delta_centroid": 1.0,  # Reduced slightly
    "rms_trend": 1.5,
    "crest_factor": 1.0,
    "spectral_centroid_slope": 1.0,
}

# Default features ON (based on last successful run + new feature)
DEFAULT_FEATURES_ON = [
    "relative_rms",
    "position_context",
    "low_energy_norm",
    "centroid_std_dev_section",
    "delta_rms",
    "crest_factor",
    "spectral_centroid_slope",
    "label_proportion",  # <-- ADDED
]


# --- Default Data Cleaning Configuration (used in GUI) ---
DEFAULT_ENABLE_OUTLIER_REMOVAL = False
DEFAULT_OUTLIER_Z_THRESHOLD = 3.0
DEFAULT_ENABLE_SHORT_SECTION_REMOVAL = False
DEFAULT_SHORT_SECTION_MIN_BARS = 2
DEFAULT_ENABLE_CONSISTENCY_FILTERING = False

# --- Default HMM Training Parameters (used in GUI) ---
DEFAULT_HMM_N_MIXTURES = 1  # Defaulting to 1 based on user trying this
DEFAULT_HMM_MIN_COVAR = 0.1  # Defaulting higher based on previous discussion
DEFAULT_HMM_COVARIANCE_TYPE = "diag"
DEFAULT_HMM_N_ITER = 100
DEFAULT_HMM_TOL = 1e-3
DEFAULT_HMM_RANDOM_STATE = 42
DEFAULT_USE_TRANSITION_PRIOR = True  # Default to using the prior if available
