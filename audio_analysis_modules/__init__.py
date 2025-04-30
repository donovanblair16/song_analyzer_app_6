# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/__init__.py

"""
Audio Analysis Modules Package

This package contains modules for audio analysis, including loading, preprocessing,
section detection, chroma analysis, spectral analysis, and feature extraction.
"""

# Import key functions to expose at the package level
from .loading import load_and_preprocess
from .section_detection import detect_sections
from .chroma_analysis import analyze_chroma_and_clusters, results_on_failure
from .spectral_analysis import analyze_stereo_and_hpss
from .feature_extraction import extract_section_features, calculate_bar_features

# Expose all key functions at the package level
__all__ = [
    "load_and_preprocess",
    "detect_sections",
    "analyze_chroma_and_clusters",
    "results_on_failure",
    "analyze_stereo_and_hpss",
    "extract_section_features",
    "calculate_bar_features",
]
