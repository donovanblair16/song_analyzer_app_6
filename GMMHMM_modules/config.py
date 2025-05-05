# =============================================================================
# FILE: config.py
# Parent: GMMHMM_modules
# Purpose: Centralize configuration settings for GMMHMM training.
# MODIFIED: Updated feature lists (ALL_FEATURE_KEYS, FEATURE_DESCRIPTIONS,
#           DEFAULT_FEATURE_WEIGHTS) to include all 21 features identified
#           from the user's screenshot (2025-05-04).
# MODIFIED: Reordered ALL_FEATURE_KEYS to place user's preferred features first.
#           Updated DEFAULT_FEATURES_ON to match user preference.
# =============================================================================

import os

# --- Path Configuration ---
# Assumes config.py is inside GMMHMM_modules, which is inside the main project folder
PROJECT_BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_BASE_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "completed_analyses")
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")
TRANSITION_MATRIX_PATH = os.path.join(HMM_OUTPUT_FOLDER, "transition_matrix.npy")

# --- Label Configuration ---
LABELS_TO_IGNORE = [
    "Fade Out",
    "Start",
    "Error",
    "Fill",
]  # Labels excluded during feature prep/training


# --- Feature Configuration ---

# List of ALL 21 features expected to be pre-calculated in the input .joblib files.
# Reordered to place preferred features at the top for GUI display.
ALL_FEATURE_KEYS = [
    # --- Preferred Features ---
    "relative_rms",  # Section loudness relative to track's max section avg_rms
    "low_energy_norm",  # Normalized energy in low frequency band (e.g., <150Hz)
    "delta_rms",  # Change in Avg RMS from previous section
    "label_proportion",  # Proportion of this label within the track's sections
    "relative_position",  # Relative start position in track (0=start, 1=end)
    "position_context",  # Positional emphasis (1 near ends, 0 near middle)
    # --- Other Features ---
    "avg_rms",  # Average Loudness (RMS) across the section
    "centroid_std_dev_section",  # Timbre Variation (Std Dev of Spectral Centroid within section)
    "crest_factor",  # Peak/Average amplitude ratio (Dynamics within section)
    "delta_centroid",  # Change in Avg Spectral Centroid from previous section
    "high_end_ratio",  # Ratio of high-frequency energy to total energy
    "low_end_ratio",  # Ratio of low-frequency energy to total energy
    "peak_rms",  # Peak RMS value observed within the section
    "rms_std_dev",  # Standard Deviation of frame-level RMS values (Overall track variation)
    "rms_std_dev_section",  # Standard Deviation of frame-level RMS values within the section
    "rms_trend",  # Loudness Trend (Linear regression slope of RMS within section)
    "spectral_bandwidth_avg",  # Average Spectral Bandwidth (Spread of spectrum around centroid)
    "spectral_centroid_avg",  # Average Spectral Centroid (Brightness measure)
    "spectral_centroid_slope",  # Brightness Trend (Slope of spectral centroid within section)
    "spectral_centroid_std_dev",  # Standard Deviation of frame-level Spectral Centroid (Overall timbre variation)
    "spectral_contrast_avg",  # Average Spectral Contrast (Difference between spectral peaks and valleys)
]

# Descriptions for the features (used in GUI tooltips, etc.)
# Keeping these aligned with the ALL_FEATURE_KEYS order for easier reading,
# though dictionary order doesn't strictly matter for lookup.
FEATURE_DESCRIPTIONS = {
    "relative_rms": "Section loudness relative to track's max section avg_rms",
    "low_energy_norm": "Normalized energy in low frequency band (e.g., <150Hz)",
    "delta_rms": "Loudness Change (Avg RMS from previous section)",
    "label_proportion": "Proportion of this label within the track's sections",
    "relative_position": "Relative start position in track (0=start, 1=end)",
    "position_context": "Positional emphasis (1 near ends, 0 near middle)",
    "avg_rms": "Average Loudness (RMS) across the section",
    "centroid_std_dev_section": "Timbre Variation (Std Dev of Spectral Centroid within section)",
    "crest_factor": "Peak/Average amplitude ratio (Dynamics within section)",
    "delta_centroid": "Timbre Change (Avg Spectral Centroid from previous section)",
    "high_end_ratio": "Ratio of high-frequency energy to total energy",
    "low_end_ratio": "Ratio of low-frequency energy to total energy",
    "peak_rms": "Peak RMS value observed within the section",
    "rms_std_dev": "Standard Deviation of frame-level RMS values (Overall track variation)",
    "rms_std_dev_section": "Standard Deviation of frame-level RMS values within the section",
    "rms_trend": "Loudness Trend (Linear regression slope of RMS within section)",
    "spectral_bandwidth_avg": "Average Spectral Bandwidth (Spread of spectrum around centroid)",
    "spectral_centroid_avg": "Average Spectral Centroid (Brightness measure)",
    "spectral_centroid_slope": "Brightness Trend (Slope of spectral centroid within section)",
    "spectral_centroid_std_dev": "Standard Deviation of frame-level Spectral Centroid (Overall timbre variation)",
    "spectral_contrast_avg": "Average Spectral Contrast (Difference between spectral peaks and valleys)",
}

# Default feature weights (used if trainer GUI allows weighting)
# Keeping these aligned with the ALL_FEATURE_KEYS order for easier reading.
DEFAULT_FEATURE_WEIGHTS = {
    "relative_rms": 5.0,  # Previous default
    "low_energy_norm": 2.0,  # Previous default
    "delta_rms": 2.0,  # Previous default
    "label_proportion": 1.5,  # Previous default
    "relative_position": 0.01,  # Previous default
    "position_context": 2.0,  # Previous default
    "avg_rms": 1.0,  # Previous default
    "centroid_std_dev_section": 0.5,  # Previous default
    "crest_factor": 1.0,  # Previous default
    "delta_centroid": 1.0,  # Previous default
    "high_end_ratio": 1.0,  # New feature default
    "low_end_ratio": 1.0,  # New feature default
    "peak_rms": 1.0,  # New feature default
    "rms_std_dev": 1.0,  # New feature default
    "rms_std_dev_section": 1.0,  # Previous default
    "rms_trend": 1.5,  # Previous default
    "spectral_bandwidth_avg": 1.0,  # New feature default
    "spectral_centroid_avg": 1.0,  # Previous default
    "spectral_centroid_slope": 1.0,  # Previous default
    "spectral_centroid_std_dev": 0.5,  # New feature default
    "spectral_contrast_avg": 1.0,  # New feature default
}

# Default features selected ON in the trainer GUI
# Updated to exactly match the user's preferred list.
DEFAULT_FEATURES_ON = [
    "relative_rms",
    "low_energy_norm",
    "delta_rms",
    "label_proportion",
    "relative_position",
    "position_context",
]
# Default setting for undersampling
DEFAULT_ENABLE_UNDERSAMPLING = False

# --- Default Data Cleaning Configuration (used in GUI) ---
DEFAULT_ENABLE_OUTLIER_REMOVAL = True
DEFAULT_OUTLIER_Z_THRESHOLD = 3.0
DEFAULT_ENABLE_SHORT_SECTION_REMOVAL = True
# NOTE: Short section removal requires 'duration_bars' feature to be present in joblibs.
# Add 'duration_bars' to ALL_FEATURE_KEYS if it exists and you want this cleaning enabled.
DEFAULT_SHORT_SECTION_MIN_BARS = 2
# NOTE: Consistency filtering requires 'avg_rms' feature to be present.
DEFAULT_ENABLE_CONSISTENCY_FILTERING = True


# --- Default HMM Training Parameters (used in GUI) ---
DEFAULT_HMM_N_MIXTURES = 1
DEFAULT_HMM_MIN_COVAR = 0.1
DEFAULT_HMM_COVARIANCE_TYPE = "diag"  # 'diag', 'full', 'tied', 'spherical'
DEFAULT_HMM_N_ITER = 100
DEFAULT_HMM_TOL = 1e-3
DEFAULT_HMM_RANDOM_STATE = 42
DEFAULT_USE_TRANSITION_PRIOR = True  # Use pre-calculated transition matrix, if found
