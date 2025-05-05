# =============================================================================
# FILE: create_lean_cache.py
# Purpose: Creates a lean cache file containing only essential data
#          (semantic_labels, section_features) from the full analysis
#          .joblib files in the 'Perfect' directory. This speeds up
#          loading for the GMM-HMM trainer.
# =============================================================================

import os
import sys
import joblib
import traceback
from collections import defaultdict

# --- Configuration ---
# Try to import paths from the GMMHMM_modules config
try:
    # Add project root to sys.path if necessary to find GMMHMM_modules
    # Assumes this script is run from the project root directory
    # Adjust path logic if running from a different location
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    GMMHMM_MODULES_PATH = os.path.join(PROJECT_ROOT, "GMMHMM_modules")
    if GMMHMM_MODULES_PATH not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)  # Add project root to find the package

    from GMMHMM_modules.config import PERFECT_FOLDER_PATH, HMM_OUTPUT_FOLDER

    print(f"Successfully imported paths from GMMHMM_modules.config:")
    print(f" -> PERFECT_FOLDER_PATH: {PERFECT_FOLDER_PATH}")
    print(f" -> HMM_OUTPUT_FOLDER:   {HMM_OUTPUT_FOLDER}")

except ImportError:
    print("ERROR: Could not import from GMMHMM_modules.config.")
    print("Please define paths manually below or ensure the script")
    print("can find the GMMHMM_modules package.")
    # --- Define Paths Manually If Import Fails ---
    # IMPORTANT: Update these paths if the import fails
    PROJECT_BASE_FOLDER = (
        "/Users/donovanblair/Desktop/song_analyzer_app_6"  # Example path
    )
    ANALYSIS_BASE_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "completed_analyses")
    PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")
    HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")
    # --- End Manual Path Definition ---
    print("\nUsing manually defined paths:")
    print(f" -> PERFECT_FOLDER_PATH: {PERFECT_FOLDER_PATH}")
    print(f" -> HMM_OUTPUT_FOLDER:   {HMM_OUTPUT_FOLDER}")


# Define the name for the lean cache file
LEAN_CACHE_FILENAME = "hmm_lean_training_data_cache.joblib"
LEAN_CACHE_FILE_PATH = os.path.join(HMM_OUTPUT_FOLDER, LEAN_CACHE_FILENAME)

# Define the essential keys to extract from each analysis file
ESSENTIAL_KEYS = ["semantic_labels", "section_features"]


# --- Main Script Logic ---
def create_lean_cache():
    """
    Loads full analysis files, extracts essential data, and saves to a lean cache.
    """
    print("\n--- Starting Lean Cache Creation ---")

    # --- Validate Input Path ---
    if not os.path.isdir(PERFECT_FOLDER_PATH):
        print(f"ERROR: 'Perfect' analysis folder not found at: {PERFECT_FOLDER_PATH}")
        print("Cache creation aborted.")
        return

    # --- Ensure Output Directory Exists ---
    try:
        os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True)
        print(f"Ensured output directory exists: {HMM_OUTPUT_FOLDER}")
    except OSError as e:
        print(f"ERROR: Could not create output directory: {HMM_OUTPUT_FOLDER}")
        print(f" -> Error: {e}")
        print("Cache creation aborted.")
        return

    # --- Load and Process Files ---
    lean_cache_data = {}  # Dictionary to store {filename: {essential_data}}
    skipped_files = []
    processed_files = 0

    try:
        print(f"Scanning for .joblib files in: {PERFECT_FOLDER_PATH}")
        analysis_files = [
            f for f in os.listdir(PERFECT_FOLDER_PATH) if f.endswith(".joblib")
        ]
    except FileNotFoundError:
        print(f"ERROR: Cannot access directory: {PERFECT_FOLDER_PATH}")
        return
    except Exception as e:
        print(f"ERROR scanning directory: {e}")
        return

    if not analysis_files:
        print("No .joblib files found in the 'Perfect' directory.")
        print("Cache creation aborted.")
        return

    print(f"Found {len(analysis_files)} .joblib files. Processing...")

    for filename in analysis_files:
        file_path = os.path.join(PERFECT_FOLDER_PATH, filename)
        print(f" -> Processing: {filename}...", end="")
        try:
            # Load the full data
            full_data = joblib.load(file_path)

            # Validate it's a dictionary and contains essential keys
            if not isinstance(full_data, dict):
                print(" [SKIPPED - Not a dictionary]")
                skipped_files.append(f"{filename} (Not a dictionary)")
                continue

            # Convert defaultdict to dict just in case (important for saving later)
            full_data = dict(full_data)

            missing_keys = [key for key in ESSENTIAL_KEYS if key not in full_data]
            if missing_keys:
                print(f" [SKIPPED - Missing keys: {', '.join(missing_keys)}]")
                skipped_files.append(
                    f"{filename} (Missing keys: {', '.join(missing_keys)})"
                )
                continue

            # Extract only the essential data
            lean_data = {key: full_data[key] for key in ESSENTIAL_KEYS}

            # Basic validation of extracted data types (optional but recommended)
            if not isinstance(lean_data.get("semantic_labels"), list) or not isinstance(
                lean_data.get("section_features"), list
            ):
                print(" [SKIPPED - Invalid data types for labels/features]")
                skipped_files.append(f"{filename} (Invalid data types)")
                continue
            if len(lean_data["semantic_labels"]) != len(lean_data["section_features"]):
                print(" [SKIPPED - Label/feature length mismatch]")
                skipped_files.append(f"{filename} (Length mismatch)")
                continue

            # Store the lean data in the cache dictionary
            lean_cache_data[filename] = lean_data
            processed_files += 1
            print(" [OK]")

        except Exception as e:
            print(f" [ERROR: {e}]")
            traceback.print_exc(limit=1)  # Print limited traceback for load errors
            skipped_files.append(f"{filename} (Load/Processing Error)")
            continue

    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed: {processed_files} files")
    if skipped_files:
        print(f"Skipped files ({len(skipped_files)}):")
        for skipped in skipped_files:
            print(f"  - {skipped}")

    # --- Save the Lean Cache ---
    if not lean_cache_data:
        print("\nNo valid data was processed. Lean cache file will not be created.")
        return

    try:
        print(f"\nSaving lean cache data to: {LEAN_CACHE_FILE_PATH}")
        joblib.dump(
            lean_cache_data, LEAN_CACHE_FILE_PATH, compress=3
        )  # Use compression
        print("Lean cache created successfully!")
    except Exception as e:
        print(f"ERROR: Failed to save lean cache file: {e}")
        traceback.print_exc()

    print("\n--- Lean Cache Creation Finished ---")


# --- Run the script ---
if __name__ == "__main__":
    create_lean_cache()
