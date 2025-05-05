"""
This script processes 'Perfect' analysis data to extract and visualize
transition matrices for state transitions in music tracks.
MODIFIED: Loads data from the lean cache file created by inspect_data.py
          ('feature_inspector_lean_cache.joblib') for faster processing.
It counts state transitions, calculates probabilities, applies smoothing,
and saves the transition matrix.
"""

# =============================================================================
# FILE: extract_transitions.py
# Purpose: Load lean cached 'Perfect' analysis data, count state transitions,
#          calculate probabilities, apply smoothing, visualize,
#          and save the *initial/smoothed* (non-enhanced) transition matrix.
# MODIFIED: Loads from lean cache instead of full joblib files.
# MODIFIED: Removed load_perfect_analyses function.
# FIXED: NameError by using PROJECT_BASE_FOLDER instead of PROJECT_ROOT.
# FIXED: Hardcoded the correct LEAN_CACHE_FILE_PATH.
# =============================================================================

import os
import sys
import numpy as np
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm

# --- Configuration ---

# Determine the script's own directory to make paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory to find the main project root
PROJECT_BASE_FOLDER = os.path.dirname(SCRIPT_DIR)

# Try to import HMM_OUTPUT_FOLDER from GMMHMM_modules config, define manually otherwise
try:
    GMMHMM_MODULES_PATH = os.path.join(PROJECT_BASE_FOLDER, "GMMHMM_modules")
    if GMMHMM_MODULES_PATH not in sys.path:
        sys.path.insert(0, PROJECT_BASE_FOLDER)
    from GMMHMM_modules.config import HMM_OUTPUT_FOLDER

    print(f"Successfully imported HMM_OUTPUT_FOLDER: {HMM_OUTPUT_FOLDER}")
except ImportError:
    print("Warning: Could not import HMM_OUTPUT_FOLDER from GMMHMM_modules.config.")
    HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")
    print(f"Using manually defined HMM_OUTPUT_FOLDER: {HMM_OUTPUT_FOLDER}")
except NameError as ne:
    print(f"Error during import setup: {ne}")
    print("Defining HMM_OUTPUT_FOLDER manually as fallback.")
    try:
        HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")
        print(f"Using manually defined HMM_OUTPUT_FOLDER: {HMM_OUTPUT_FOLDER}")
    except NameError:
        print(
            "CRITICAL ERROR: PROJECT_BASE_FOLDER not defined. Cannot set HMM_OUTPUT_FOLDER."
        )
        HMM_OUTPUT_FOLDER = "/tmp/hmm_model_fallback"  # Absolute fallback
        print(f"Using absolute fallback HMM_OUTPUT_FOLDER: {HMM_OUTPUT_FOLDER}")


# --- Lean Cache File Path ---
# *** FIX: Hardcode the exact path provided by the user ***
LEAN_CACHE_FILE_PATH = "/Users/donovanblair/Desktop/song_analyzer_app_6/inspect_data_program/feature_inspector_lean_cache.joblib"
# *** End FIX ***


# State mapping - ensure this EXACTLY matches your HMM trainer/predictor
LABEL_TO_IDX = {
    "Body": 0,
    "Breakdown": 1,
    "Build": 2,
    "Drop": 3,
    "Intro": 4,
    "Outro": 5,
    # Add any other valid labels present in your 'Perfect' analyses if needed
    # Example: "Fill": 6
}
# Generate the reverse mapping
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
N_STATES = len(LABEL_TO_IDX)  # Number of states based on the map

# Smoothing parameter (alpha) for Laplace/additive smoothing
SMOOTHING_ALPHA = 0  # Set to 0 to disable smoothing

# Output filename for the final matrix
OUTPUT_MATRIX_FILENAME = "transition_matrix.npy"

# --- Functions ---


def extract_transition_counts(lean_cache_data):
    """
    Extracts raw transition counts and state occurrence counts from the
    lean cache data ({filename: {'semantic_labels': [...], ...}}).
    Only considers labels defined in LABEL_TO_IDX.

    Args:
        lean_cache_data (dict): The dictionary loaded from the lean cache file.

    Returns:
        tuple: (transition_counts_matrix, state_counts_array) or (None, None) on error.
    """
    if N_STATES == 0:
        print("ERROR: LABEL_TO_IDX map is empty. Cannot count transitions.")
        return None, None
    transition_counts = np.zeros((N_STATES, N_STATES), dtype=int)
    state_counts = np.zeros(N_STATES, dtype=int)
    print("\nExtracting transitions from lean cache...")
    tracks_processed = 0
    total_valid_labels = 0
    total_transitions_counted = 0

    for filename, track_data in lean_cache_data.items():
        if not isinstance(track_data, dict):
            print(f"Warning: Skipping item for {filename}, not a dictionary.")
            continue
        labels = track_data.get("semantic_labels", [])
        if not isinstance(labels, list):
            print(
                f"Warning: Skipping item for {filename}, 'semantic_labels' is not a list."
            )
            continue

        filtered_labels = [label for label in labels if label in LABEL_TO_IDX]
        total_valid_labels += len(filtered_labels)
        if not filtered_labels:
            continue

        tracks_processed += 1
        for label in filtered_labels:
            try:
                state_counts[LABEL_TO_IDX[label]] += 1
            except KeyError:
                pass

        for i in range(len(filtered_labels) - 1):
            try:
                from_idx = LABEL_TO_IDX[filtered_labels[i]]
                to_idx = LABEL_TO_IDX[filtered_labels[i + 1]]
                transition_counts[from_idx, to_idx] += 1
                total_transitions_counted += 1
            except KeyError as ke:
                print(
                    f"WARNING: KeyError during transition counting (label '{ke}' not in map) in {filename}."
                )

    print(f"Processed {tracks_processed} tracks from cache.")
    print(
        f"Total valid labels encountered (matching LABEL_TO_IDX): {total_valid_labels}"
    )
    print(
        f"Total transitions counted (between valid labels): {total_transitions_counted}"
    )
    return transition_counts, state_counts


def create_transition_matrix(transition_counts):
    """
    Convert raw transition counts into a probability matrix.
    """
    if transition_counts is None or N_STATES == 0:
        return None
    transition_probs = np.zeros_like(transition_counts, dtype=float)
    print("\nCalculating initial transition probabilities...")
    for i in range(transition_counts.shape[0]):
        row_sum = np.sum(transition_counts[i])
        state_label = IDX_TO_LABEL.get(i, f"Unknown Index {i}")
        if row_sum > 0:
            transition_probs[i] = transition_counts[i] / row_sum
        else:
            print(
                f" -> State '{state_label}': No outgoing transitions observed. Assigning uniform probability."
            )
            if N_STATES > 0:
                transition_probs[i] = 1.0 / N_STATES
            else:
                print("ERROR: N_STATES is zero.")
                transition_probs[i] = 0
    return transition_probs


def apply_smoothing(transition_probs, alpha=SMOOTHING_ALPHA):
    """
    Apply additive (Laplace-like) smoothing.
    """
    if transition_probs is None or alpha <= 0:
        print("\nSkipping smoothing (alpha <= 0 or no input matrix).")
        return transition_probs
    if N_STATES == 0:
        print("ERROR: N_STATES is zero.")
        return transition_probs
    print(f"\nApplying smoothing with alpha = {alpha}...")
    smoothed_probs = transition_probs.copy()
    smoothed_probs = (1 - alpha) * smoothed_probs + alpha / N_STATES
    row_sums = np.sum(smoothed_probs, axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0
    smoothed_probs = smoothed_probs / row_sums
    return smoothed_probs


def incorporate_prior_knowledge(transition_probs):
    """
    Adjusts transition probabilities based on domain knowledge.
    """
    if transition_probs is None or N_STATES == 0:
        return None
    print("\nIncorporating prior knowledge (enhancements/reductions)...")
    enhanced_probs = transition_probs.copy()
    enhancements = {
        ("Intro", "Drop"): 1.0,
        ("Drop", "Body"): 1.0,
        ("Body", "Build"): 1.0,
        ("Build", "Drop"): 1.0,
        ("Drop", "Breakdown"): 1.0,
        ("Breakdown", "Build"): 1.0,
    }
    reductions = {
        ("Outro", "Intro"): 1.0,
        ("Drop", "Intro"): 1.0,
    }

    # NEW: Transitions to make impossible (set to zero)
    impossible_transitions = {
        ("Build", "Body"): 0.0,
        ("Outro", "Drop"): 0.0,
        ("Outro", "Body"): 0.0,
        ("Outro", "Build"): 0.0,
        ("Outro", "Breakdown"): 0.0,
        ("Outro", "Intro"): 0.0
    }
        # Add any other transitions you want to make impossible
        # For example: ("Outro", "Drop"), ("Intro", "Outro"), etc.
        
    
    #apply enhancements
    for (from_label, to_label), factor in enhancements.items():
        if from_label in LABEL_TO_IDX and to_label in LABEL_TO_IDX:
            from_idx, to_idx = LABEL_TO_IDX[from_label], LABEL_TO_IDX[to_label]
            original_prob = enhanced_probs[from_idx, to_idx]
            if original_prob > 1e-9:
                enhanced_probs[from_idx, to_idx] *= factor
                print(
                    f" -> Enhanced P({to_label}|{from_label}): {original_prob:.3f} * {factor:.1f} -> {enhanced_probs[from_idx, to_idx]:.3f}"
                )
        else:
            print(
                f" -> Warning: Cannot apply enhancement '{from_label}'->'{to_label}', label not in map."
            )
    for (from_label, to_label), factor in reductions.items():
        if from_label in LABEL_TO_IDX and to_label in LABEL_TO_IDX:
            from_idx, to_idx = LABEL_TO_IDX[from_label], LABEL_TO_IDX[to_label]
            original_prob = enhanced_probs[from_idx, to_idx]
            enhanced_probs[from_idx, to_idx] *= factor
            print(
                f" -> Reduced P({to_label}|{from_label}): {original_prob:.3f} * {factor:.1f} -> {enhanced_probs[from_idx, to_idx]:.3f}"
            )
        else:
            print(
                f" -> Warning: Cannot apply reduction '{from_label}'->'{to_label}', label not in map."
            )
    # NEW: Apply impossible transitions (set to zero)
    for (from_label, to_label), value in impossible_transitions.items():
        if from_label in LABEL_TO_IDX and to_label in LABEL_TO_IDX:
            from_idx, to_idx = LABEL_TO_IDX[from_label], LABEL_TO_IDX[to_label]
            original_prob = enhanced_probs[from_idx, to_idx]
            enhanced_probs[from_idx, to_idx] = value  # Set to the value (0.0)
            print(
                f" -> Made transition P({to_label}|{from_label}) impossible: {original_prob:.3f} -> {value:.3f}"
            )
        else:
            print(
                f" -> Warning: Cannot make transition '{from_label}'->'{to_label}' impossible, label not in map."    
            )
    
    print(" -> Renormalizing rows after enhancements/reductions...")
    row_sums = np.sum(enhanced_probs, axis=1, keepdims=True)
    zero_rows = row_sums < 1e-9
    if np.any(zero_rows):
        zero_indices = np.where(zero_rows)[0]
        print(
            f"WARNING: Found rows summing near zero after enhancements: {zero_indices}. Setting uniform probability."
        )
        if N_STATES > 0:
            enhanced_probs[zero_rows.flatten(), :] = 1.0 / N_STATES
        else:
            enhanced_probs[zero_rows.flatten(), :] = 0.0
        row_sums[zero_rows] = 1.0
    enhanced_probs = enhanced_probs / row_sums
    return enhanced_probs


def visualize_transition_matrix(transition_probs, title="Transition Matrix"):
    """Creates and saves a visual heatmap representation."""
    if transition_probs is None or N_STATES == 0:
        print(f"Cannot visualize matrix '{title}': No data or states.")
        return
    try:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(
            transition_probs,
            cmap="viridis",
            aspect="auto",
            origin="upper",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, label="Transition Probability")
        for i in range(transition_probs.shape[0]):
            for j in range(transition_probs.shape[1]):
                prob = transition_probs[i, j]
                text_color = "white" if prob < 0.5 else "black"
                plt.text(
                    j,
                    i,
                    f"{prob:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        tick_labels = [IDX_TO_LABEL.get(idx, f"Idx {idx}") for idx in range(N_STATES)]
        plt.xticks(np.arange(N_STATES), tick_labels, rotation=45, ha="right")
        plt.yticks(np.arange(N_STATES), tick_labels)
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.title(title)
        plt.tight_layout()
        os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
        save_path = os.path.join(HMM_OUTPUT_FOLDER, f"{safe_title}.png")
        plt.savefig(save_path)
        print(f" -> Saved visualization to {save_path}")
        plt.close()
    except Exception as e:
        print(f"ERROR creating visualization '{title}': {e}")
        traceback.print_exc()
        plt.close()


def save_transition_matrix(transition_matrix, filename):
    """Saves the NumPy transition matrix."""
    if transition_matrix is None:
        print("Cannot save matrix: No data.")
        return
    try:
        os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True)
        save_path = os.path.join(HMM_OUTPUT_FOLDER, filename)
        np.save(save_path, transition_matrix)
        print(f"Saved final transition matrix to {save_path}")
    except Exception as e:
        print(f"ERROR saving transition matrix to {save_path}: {e}")
        traceback.print_exc()


def print_matrix(matrix, title):
    """Prints a formatted matrix."""
    if matrix is None or N_STATES == 0:
        print(f"Cannot print matrix '{title}': No data or states.")
        return
    print(f"\n--- {title} ---")
    labels = [IDX_TO_LABEL.get(i, f"Idx {i}") for i in range(N_STATES)]
    header = f"{'From \\ To':<10}" + "".join([f" | {label:<10}" for label in labels])
    print(header)
    print("-" * len(header))
    for i in range(matrix.shape[0]):
        from_label = IDX_TO_LABEL.get(i, f"Idx {i}")
        row_str = f"{from_label:<10}" + "".join(
            [f" | {matrix[i, j]:<10.3f}" for j in range(matrix.shape[1])]
        )
        print(row_str)
    print("-" * len(header))


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Transition Matrix Extraction (using Lean Cache) ---")

    # --- Load Lean Cache Data ---
    print(
        f"Attempting to load lean cache from: {LEAN_CACHE_FILE_PATH}"
    )  # Path uses correct logic now
    if not os.path.exists(LEAN_CACHE_FILE_PATH):
        print(f"ERROR: Lean cache file not found at {LEAN_CACHE_FILE_PATH}")
        print(
            "Please run 'inspect_data.py' (or the cache creation script) first to generate the cache."
        )
        sys.exit(1)

    try:
        cached_content = joblib.load(LEAN_CACHE_FILE_PATH)
        if not isinstance(cached_content, dict) or "data" not in cached_content:
            raise ValueError("Lean cache file format is invalid (missing 'data' key).")
        lean_data_cache = cached_content["data"]
        if not lean_data_cache:
            raise ValueError("Lean cache 'data' dictionary is empty.")
        print(f"Successfully loaded lean cache data for {len(lean_data_cache)} files.")
    except Exception as e:
        print(f"ERROR: Failed to load or validate lean cache file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Proceed with processing ---
    transition_counts, state_counts = extract_transition_counts(lean_data_cache)

    if transition_counts is None or N_STATES == 0:
        print(
            "\nERROR: Failed to count transitions or no states defined. Cannot proceed."
        )
        sys.exit(1)
    if np.sum(state_counts) == 0:
        print(
            "\nWARNING: No valid labels found in the cached data matching the state map."
        )

    print("\n--- State Occurrence Counts ---")
    for state_idx, count in enumerate(state_counts):
        print(f"  {IDX_TO_LABEL.get(state_idx, 'Unknown')}: {int(count)}")
    print_matrix(transition_counts, "Raw Transition Counts")

    transition_probs = create_transition_matrix(transition_counts)
    if transition_probs is None:
        print("ERROR: Failed to create transition probability matrix.")
        sys.exit(1)
    print_matrix(
        transition_probs, "Initial Transition Probabilities (Normalized Counts)"
    )
    visualize_transition_matrix(transition_probs, "Initial Transition Probabilities")

    smoothed_probs = apply_smoothing(transition_probs, alpha=SMOOTHING_ALPHA)
    if smoothed_probs is None:
        smoothed_probs = transition_probs  # Fallback
    if SMOOTHING_ALPHA > 0:
        print_matrix(
            smoothed_probs,
            f"Smoothed Transition Probabilities (Alpha={SMOOTHING_ALPHA})",
        )
        visualize_transition_matrix(
            smoothed_probs, f"Smoothed Transition Probabilities Alpha={SMOOTHING_ALPHA}"
        )
    else:
        print("\nSmoothing was skipped (alpha=0).")

    enhanced_probs = incorporate_prior_knowledge(smoothed_probs)
    if enhanced_probs is None:
        enhanced_probs = smoothed_probs  # Fallback
    print_matrix(
        enhanced_probs, "Enhanced Transition Probabilities (Initial/Smoothed + Priors)"
    )
    visualize_transition_matrix(enhanced_probs, "Enhanced Transition Probabilities")

    print("\n*** Saving the Initial/Smoothed (Non-Enhanced) Matrix ***")
    save_transition_matrix(
        smoothed_probs, OUTPUT_MATRIX_FILENAME
    )  # Save the smoothed (or initial if alpha=0) matrix

    print("\n--- Transition matrix analysis and saving complete! ---")
