# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/constants.py

"""
Constants used throughout the audio analysis modules.
"""

# Thresholds used in labeling logic
MIN_OUTRO_BARS = 4
MIN_FADEOUT_BARS = 4  # Sections shorter than this at the end might be fade outs
FADEOUT_RMS_THRESHOLD = 0.05  # Avg RMS below this might indicate a fade out

# Color hex codes for consistency
FADE_OUT_COLOR_HEX = "#8A2BE2"  # BlueViolet/Purple
FALLBACK_COLOR_HEX = "#808080"  # Grey for unknown labels
DARK_RED_HEX = "#8B0000"
RED_HEX = "#FF0000"

# Note names for chroma analysis
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
