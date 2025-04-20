# =============================================================================
# FILE: hmm_trainer.py
# Purpose: Load 'Perfect' analysis files, prepare data, train, and save an HMM
#          to a dedicated 'hmm_model' subfolder.
# =============================================================================

import os
import glob
import joblib
import numpy as np
import traceback
from collections import Counter, defaultdict
from hmmlearn import hmm # Using hmmlearn library
from sklearn.preprocessing import StandardScaler # To scale features

# --- Configuration ---

# Directory where the 'Perfect' analysis files are saved
# Note: Assumes the structure created by main_app.py
# WARNING: Hardcoded path from main_app.py - consider making this more robust
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/Song Analyzer App 6/completed_analyses"
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")

# *** Define the main project folder and the HMM output subfolder ***
# Assumes ANALYSIS_BASE_FOLDER is inside the main project folder
PROJECT_BASE_FOLDER = os.path.dirname(ANALYSIS_BASE_FOLDER)
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model") # New folder for outputs

# Labels to ignore during HMM training (not considered structural states)
LABELS_TO_IGNORE = ["Fade Out", "Start", "Error", "Fill"] # Added Fill as it's transitional

# HMM Parameters (can be tuned later)
N_MIXTURES = 3 # Number of Gaussian mixtures per state in the GMMHMM
COVARIANCE_TYPE = "diag" # Covariance type for GMMs ('diag', 'full', 'tied', 'spherical')
N_ITER = 50 # Number of iterations for Baum-Welch training
TOL = 1e-3 # Convergence tolerance for training
HMM_RANDOM_STATE = 42 # For reproducibility

# *** Update output file paths to use the new folder ***
MODEL_SAVE_PATH = os.path.join(HMM_OUTPUT_FOLDER, "trained_hmm_model.joblib")
LABEL_MAP_SAVE_PATH = os.path.join(HMM_OUTPUT_FOLDER, "hmm_label_map.joblib") # To map HMM state indices back to labels

# --- Helper Functions ---

def load_perfect_analyses(folder_path):
    """Loads all .joblib analysis files from the 'Perfect' subfolder."""
    all_track_data = []
    search_pattern = os.path.join(folder_path, "*.joblib")
    analysis_files = glob.glob(search_pattern)
    print(f"Found {len(analysis_files)} analysis files in '{folder_path}'.")

    for f_path in analysis_files:
        try:
            print(f"Loading: {os.path.basename(f_path)}...")
            data = joblib.load(f_path)
            if isinstance(data, dict) and 'semantic_labels' in data and 'section_starts' in data:
                # Basic validation - check if essential keys exist
                # Added check for spectral_contrast_frames - If adding more features, add checks here
                if ('bar_starts_absolute' in data and data['bar_starts_absolute'] is not None and
                    'bar_rms_data' in data and data['bar_rms_data'] is not None and
                    'spectral_centroid_frames' in data and data['spectral_centroid_frames'] is not None and
                    # 'spectral_contrast_frames' in data and data['spectral_contrast_frames'] is not None and # Add check if using contrast
                    'times_absolute' in data and data['times_absolute'] is not None):
                     all_track_data.append(data)
                     print(f" -> Successfully loaded and validated.")
                else:
                     # Find missing keys for better debugging
                     missing_keys = [k for k in ['bar_starts_absolute', 'bar_rms_data', 'spectral_centroid_frames', 'times_absolute'] if k not in data or data[k] is None]
                     print(f" -> Skipping: Missing required feature keys: {missing_keys}.")
            else:
                print(f" -> Skipping: Invalid format or missing essential keys.")
        except Exception as e:
            print(f" -> Error loading file {f_path}: {e}")
            # traceback.print_exc() # Uncomment for detailed errors
    print(f"Successfully loaded {len(all_track_data)} valid analysis files.")
    return all_track_data

def calculate_bar_features(track_data):
    """
    Calculates features averaged over each bar.
    Current features: Avg RMS, Avg Spectral Centroid.
    Returns features and corresponding labels per bar.
    """
    bar_starts = track_data.get("bar_starts_absolute")
    bar_rms = track_data.get("bar_rms_data")
    section_starts = track_data.get("section_starts")
    semantic_labels = track_data.get("semantic_labels")
    spec_centroid = track_data.get("spectral_centroid_frames")
    # spec_contrast = track_data.get("spectral_contrast_frames") # Uncomment if adding contrast
    frame_times_abs = track_data.get("times_absolute") # Absolute times for spectral features

    # Add checks for all required data
    required_data = {
        "bar_starts": bar_starts, "bar_rms": bar_rms, "section_starts": section_starts,
        "semantic_labels": semantic_labels, "spec_centroid": spec_centroid,
        "frame_times_abs": frame_times_abs
        # "spec_contrast": spec_contrast # Add if using contrast
    }
    if any(v is None for v in required_data.values()):
        missing = [k for k, v in required_data.items() if v is None]
        print(f"Warning: Missing data for bar feature calculation: {missing}. Skipping track.")
        return [], []

    num_bars = len(bar_starts)
    if len(bar_rms) != num_bars:
        print(f"Warning: Mismatch between bar_starts ({num_bars}) and bar_rms ({len(bar_rms)}). Skipping track.")
        return [], []
    if len(spec_centroid) != len(frame_times_abs):
         print(f"Warning: Mismatch between spec_centroid ({len(spec_centroid)}) and frame_times ({len(frame_times_abs)}). Skipping track.")
         return [], []
    # Add contrast length check if using it

    bar_features_list = []
    bar_labels_list = []
    # section_ends = list(section_starts[1:]) + [float('inf')] # Not strictly needed

    current_section_idx = 0
    for i in range(num_bars):
        bar_start_time = bar_starts[i]
        bar_end_time = bar_starts[i+1] if i + 1 < num_bars else track_data['duration_processed'] + track_data['trim_offset_sec']

        # Find the label for the current bar
        while current_section_idx + 1 < len(section_starts) and bar_start_time >= section_starts[current_section_idx + 1]:
            current_section_idx += 1
        # Ensure index is valid
        if current_section_idx >= len(semantic_labels):
             print(f"Warning: Section index {current_section_idx} out of bounds for labels. Assigning fallback.")
             current_label = "Unknown" # Assign a fallback or skip
        else:
             current_label = semantic_labels[current_section_idx]


        # --- Calculate Features for the Bar ---
        # 1. Average RMS (already calculated)
        avg_rms_bar = bar_rms[i] if np.isfinite(bar_rms[i]) else 0.0

        # 2. Average Spectral Centroid
        frame_mask = (frame_times_abs >= bar_start_time) & (frame_times_abs < bar_end_time)
        frames_in_bar_centroid = spec_centroid[frame_mask]
        avg_centroid_bar = np.mean(frames_in_bar_centroid) if frames_in_bar_centroid.size > 0 and np.all(np.isfinite(frames_in_bar_centroid)) else 0.0
        if not np.isfinite(avg_centroid_bar): avg_centroid_bar = 0.0

        # 3. Average Spectral Contrast (Example - uncomment and adapt if needed)
        # frames_in_bar_contrast = spec_contrast[:, frame_mask] # Get contrast bands for frames in bar
        # avg_contrast_bar_bands = np.mean(frames_in_bar_contrast, axis=1) if frames_in_bar_contrast.size > 0 else np.zeros(spec_contrast.shape[0])
        # avg_contrast_bar = np.mean(avg_contrast_bar_bands) # Overall average contrast
        # if not np.isfinite(avg_contrast_bar): avg_contrast_bar = 0.0

        # Store features and label for this bar
        # Adjust feature list if adding more features
        bar_features_list.append([avg_rms_bar, avg_centroid_bar])
        bar_labels_list.append(current_label)

    return bar_features_list, bar_labels_list


def prepare_hmm_data(all_track_data):
    """
    Processes loaded track data to extract bar-level features and state sequences,
    filtering ignored labels and formatting for hmmlearn.
    """
    all_observations = []
    all_states_str = []
    sequence_lengths = []

    # --- Determine the set of valid states ---
    all_possible_labels = set()
    for data in all_track_data:
        if data and 'semantic_labels' in data:
            all_possible_labels.update(data['semantic_labels'])

    valid_states = sorted([label for label in all_possible_labels if label not in LABELS_TO_IGNORE])
    if not valid_states:
         print("Error: No valid states found after filtering ignored labels. Check LABELS_TO_IGNORE and your data.")
         return None, None, None, None, None

    label_to_int = {label: i for i, label in enumerate(valid_states)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    n_states = len(valid_states)
    print(f"HMM States ({n_states}): {valid_states}")
    print(f"Label to Int Map: {label_to_int}")

    # --- Extract features and states for each track ---
    for i, track_data in enumerate(all_track_data):
        print(f"Processing track {i+1}/{len(all_track_data)} for HMM data...")
        bar_features, bar_labels = calculate_bar_features(track_data)

        if not bar_features or not bar_labels:
            print(f" -> No bar features/labels extracted. Skipping track.")
            continue

        track_observations = []
        track_states_int = []
        track_states_str_filtered = [] # Store filtered string labels temporarily

        for features, label in zip(bar_features, bar_labels):
            if label not in LABELS_TO_IGNORE:
                track_observations.append(features)
                # Map label to integer state, handle potential KeyError if label somehow invalid
                state_int = label_to_int.get(label)
                if state_int is None:
                    print(f"Warning: Label '{label}' not found in valid states map. Skipping bar.")
                    # Need to remove corresponding observation if skipping state
                    track_observations.pop() # Remove last added observation
                    continue
                track_states_int.append(state_int)
                track_states_str_filtered.append(label) # Store the valid label string

        if track_observations: # Only add if not empty after filtering
            all_observations.extend(track_observations)
            all_states_str.extend(track_states_str_filtered) # Extend with filtered strings
            sequence_lengths.append(len(track_observations))
            print(f" -> Added sequence of length {len(track_observations)}")
        else:
            print(f" -> Track resulted in empty sequence after filtering. Skipping.")

    if not all_observations:
        print("Error: No valid observation sequences generated.")
        return None, None, None, None, None

    # Concatenate observations into a single NumPy array
    X = np.array(all_observations)
    print(f"Total observations (bars): {X.shape[0]}, Features per observation: {X.shape[1]}")

    # Scale features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")

    # Reconstruct integer state sequence from the filtered string sequence
    all_states_int_filtered = [label_to_int[lbl] for lbl in all_states_str]

    if len(all_states_int_filtered) != X_scaled.shape[0]:
         print("!!! ERROR: Mismatch between length of observations and states after processing!")
         # This indicates a bug in the filtering/appending logic above
         return None, None, None, None, None


    return X_scaled, sequence_lengths, int_to_label, np.array(all_states_int_filtered), scaler


def train_hmm(X, lengths, n_states):
    """Initializes and trains a GMMHMM."""
    print(f"\n--- Training HMM ---")
    print(f"Number of states: {n_states}")
    print(f"Number of sequences: {len(lengths)}")
    print(f"Total observations: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"GMM Mixtures per state: {N_MIXTURES}")

    # Initialize GMMHMM model
    model = hmm.GMMHMM(n_components=n_states, n_mix=N_MIXTURES,
                       covariance_type=COVARIANCE_TYPE,
                       random_state=HMM_RANDOM_STATE, # For reproducibility
                       n_iter=N_ITER, tol=TOL, verbose=True,
                       init_params="mcw", # Initialize means, covars, weights (startprob/transmat often need custom init)
                       params="stmcw") # Train startprob, transmat, means, covars, weights

    # TODO: Consider supervised initialization using known state sequences if available
    # This often leads to better results than random initialization.
    # Requires calculating initial probabilities and transition counts from `state_sequence`
    # Requires calculating means/covars per state from `X` segmented by `state_sequence`
    # Example (Conceptual - requires state_sequence from prepare_hmm_data):
    # startprob = calculate_start_probabilities(state_sequence, lengths, n_states)
    # transmat = calculate_transition_matrix(state_sequence, lengths, n_states)
    # means, covars = calculate_emission_params(X, state_sequence, n_states, N_MIXTURES)
    # model.startprob_ = startprob
    # model.transmat_ = transmat
    # model.means_ = means
    # model.covars_ = covars
    # model.init_params = "" # Don't re-initialize these if set manually
    # model.params = "w" # Only train weights if fully initialized? Or 'stmcw' still?

    print("Fitting HMM...")
    try:
        model.fit(X, lengths)
        print("HMM Training Complete.")
        print(f"Log Likelihood: {model.score(X, lengths)}")
        # print("\nLearned Transition Matrix:")
        # print(np.round(model.transmat_, 3))
        return model
    except Exception as e:
        print(f"HMM Training Failed: {e}")
        traceback.print_exc()
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting HMM Training Process ---")

    # 1. Load Data
    all_data = load_perfect_analyses(PERFECT_FOLDER_PATH)
    if not all_data:
        print("No valid analysis files found or loaded. Exiting.")
        exit()

    # 2. Prepare Data
    # Returns: X_scaled, lengths, int_to_label, state_sequence, scaler
    prepared_data = prepare_hmm_data(all_data)
    if prepared_data is None or prepared_data[0] is None:
        print("Data preparation failed. Exiting.")
        exit()
    X_scaled, lengths, int_to_label, state_sequence, scaler = prepared_data

    n_unique_states = len(int_to_label)

    # 3. Train HMM
    trained_model = train_hmm(X_scaled, lengths, n_unique_states)

    # 4. Save Model (if training successful)
    if trained_model:
        print(f"\n--- Saving Trained Model and Maps ---")
        try:
            # *** Create the output directory if it doesn't exist ***
            os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True)
            print(f"Ensured output directory exists: {HMM_OUTPUT_FOLDER}")

            # Save the HMM model itself
            joblib.dump(trained_model, MODEL_SAVE_PATH)
            print(f"Trained HMM model saved to: {MODEL_SAVE_PATH}")

            # Save the label map (int -> label name) and the scaler
            save_dict = {
                'int_to_label': int_to_label,
                'scaler': scaler
            }
            joblib.dump(save_dict, LABEL_MAP_SAVE_PATH)
            print(f"Label map and scaler saved to: {LABEL_MAP_SAVE_PATH}")

        except Exception as e:
            print(f"Error creating directory or saving model/maps: {e}")
            traceback.print_exc()

    print("\n--- HMM Training Process Finished ---")

