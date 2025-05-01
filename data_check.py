"""Module for performing data integrity checks on analysis files."""

import joblib
import os
import numpy as np  # Needed to check array type

# --- Configuration: Set this to your 'Perfect' folder path ---
PERFECT_FOLDER_PATH = (
    "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses/Perfect"
)
# --- Choose a file to inspect ---
# *** REPLACE THIS with the actual filename you want to check ***
FILENAME_TO_CHECK = "ACRAZE Don Toliver - Bandit Extended Club Mix.analysis.joblib"

# Construct the full path
file_path = os.path.join(PERFECT_FOLDER_PATH, FILENAME_TO_CHECK)

print(f"--- Checking file: {FILENAME_TO_CHECK} ---")

if not os.path.exists(file_path):
    print(f"ERROR: File not found at {file_path}")
else:
    try:
        # Load the data from the joblib file
        track_data = joblib.load(file_path)

        # Check if the loaded data is a dictionary
        if isinstance(track_data, dict):
            print("File loaded successfully. Checking for required keys...")

            # Check for 'rms' key
            has_rms = "rms" in track_data
            print(f"  - Contains 'rms' key: {has_rms}")
            if has_rms:
                # Optionally check if it's a numpy array and its shape
                rms_data = track_data["rms"]
                is_array = isinstance(rms_data, np.ndarray)
                print(f"    - Is 'rms' a numpy array: {is_array}")
                if is_array:
                    print(f"    - Shape of 'rms' array: {rms_data.shape}")

            # Check for 'rms_times' key
            has_rms_times = "rms_times" in track_data
            print(f"  - Contains 'rms_times' key: {has_rms_times}")
            if has_rms_times:
                # Optionally check if it's a numpy array and its shape
                rms_times_data = track_data["rms_times"]
                is_array = isinstance(rms_times_data, np.ndarray)
                print(f"    - Is 'rms_times' a numpy array: {is_array}")
                if is_array:
                    print(f"    - Shape of 'rms_times' array: {rms_times_data.shape}")

            # Conclusion based on checks
            if has_rms and has_rms_times:
                print(
                    "\nRESULT: Looks like the necessary frame-level RMS data IS PRESENT."
                )
            else:
                print(
                    "\nRESULT: The necessary frame-level 'rms' and/or 'rms_times' data IS MISSING."
                )

            # Optional: Print all available keys
            # print("\nAvailable keys in the file:", list(track_data.keys()))

        else:
            print(
                f"ERROR: Loaded data is not a dictionary (type: {type(track_data)}). Cannot check keys."
            )

    except Exception as e:
        print(f"ERROR: Failed to load or inspect file: {e}")
        import traceback

        traceback.print_exc()

print("--- Check complete ---")
