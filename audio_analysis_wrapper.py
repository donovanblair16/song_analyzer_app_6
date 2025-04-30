# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_wrapper.py

"""
Audio Analysis Module Wrapper

This is a wrapper module that imports and exposes all the necessary functions
from the audio_analysis_modules package. This maintains backward compatibility
with existing code that imports functions from audio_analysis.py.
"""

# Import all the key functions from the audio_analysis_modules package
from audio_analysis_modules import (
    load_and_preprocess,
    detect_sections,
    analyze_chroma_and_clusters,
    results_on_failure,
    analyze_stereo_and_hpss,
    extract_section_features,
    calculate_bar_features,
)

# Import constants for backward compatibility
from audio_analysis_modules.constants import (
    MIN_OUTRO_BARS,
    MIN_FADEOUT_BARS,
    FADEOUT_RMS_THRESHOLD,
    FADE_OUT_COLOR_HEX,
    FALLBACK_COLOR_HEX,
    DARK_RED_HEX,
    RED_HEX,
)

# Re-export all imported functions and constants
__all__ = [
    # Functions
    "load_and_preprocess",
    "detect_sections",
    "analyze_chroma_and_clusters",
    "results_on_failure",
    "analyze_stereo_and_hpss",
    "extract_section_features",
    "calculate_bar_features",
    # Constants
    "MIN_OUTRO_BARS",
    "MIN_FADEOUT_BARS",
    "FADEOUT_RMS_THRESHOLD",
    "FADE_OUT_COLOR_HEX",
    "FALLBACK_COLOR_HEX",
    "DARK_RED_HEX",
    "RED_HEX",
]
