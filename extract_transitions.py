""" 
This script processes 'Perfect' analysis files to extract and visualize 
transition matrices for state transitions in music tracks.
It loads the analysis files, counts state transitions, calculates   
probabilities, applies smoothing, and saves the transition matrix.
The script is designed to be run in a specific environment where the
analysis files are stored in a predefined directory structure.
It also includes error handling and logging for better debugging.
The script is modular, with functions for loading data,
extracting transitions, creating matrices, applying smoothing,
visualizing results, and saving the final matrix.

"""

# =============================================================================
# FILE: extract_transitions.py
# Purpose: Load 'Perfect' analysis files, count state transitions,
#          calculate probabilities, apply smoothing, visualize,
#          and save the *initial/smoothed* (non-enhanced) transition matrix.
# MODIFIED: Saves 'smoothed_probs' instead of 'enhanced_probs'.
# =============================================================================

import os
import numpy as np
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
import traceback  # Added for better error reporting

# --- Configuration ---
# *** Ensure these paths are correct for your system ***
# Use the correct path for your setup
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"  # Corrected path
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")
# Ensure HMM_OUTPUT_FOLDER exists and is where the main app expects the matrix
PROJECT_BASE_FOLDER = os.path.dirname(
    os.path.dirname(ANALYSIS_BASE_FOLDER)
)  # Go up two levels
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")

# State mapping - ensure this EXACTLY matches your HMM trainer/predictor
LABEL_TO_IDX = {
    "Body": 0,
    "Breakdown": 1,
    "Build": 2,
    "Drop": 3,
    "Intro": 4,
    "Outro": 5,
}
# Generate the reverse mapping
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
N_STATES = len(LABEL_TO_IDX)  # Number of states based on the map

# Smoothing parameter (alpha) for Laplace/additive smoothing
# *** SET TO 0 TO DISABLE SMOOTHING ***
SMOOTHING_ALPHA = 0  # Set to 0 to disable smoothing

# Output filename for the final matrix
OUTPUT_MATRIX_FILENAME = "transition_matrix.npy"

# --- Functions ---


def load_perfect_analyses(folder_path):
    """
    Load all .joblib analysis files from the 'Perfect' subfolder.
    Performs basic validation for 'semantic_labels'.
    """
    all_data = []
    if not os.path.isdir(folder_path):
        print(f"ERROR: Perfect analysis folder not found at: {folder_path}")
        return []
    try:
        analysis_files = [f for f in os.listdir(folder_path) if f.endswith(".joblib")]
    except FileNotFoundError:
        print(f"ERROR: Cannot access perfect analysis folder: {folder_path}")
        return []
    print(
        f"Found {len(analysis_files)} .joblib files in '{os.path.basename(folder_path)}'."
    )
    for filename in analysis_files:
        file_path = os.path.join(folder_path, filename)
        try:
            data = joblib.load(file_path)
            if (
                "semantic_labels" in data
                and isinstance(
                    data["semantic_labels"], (list, np.ndarray)
                )  # Allow numpy array
                and len(data["semantic_labels"]) > 0
            ):
                all_data.append(data)
                # print(f" -> Successfully loaded: {filename}") # Optional verbose
            else:
                print(f" -> Skipping {filename}: Missing or empty 'semantic_labels'.")
        except Exception as e:
            print(f" -> Error loading {filename}: {e}")
    print(f"Successfully loaded data from {len(all_data)} files.")
    return all_data


def extract_transition_counts(all_data):
    """
    Extract raw transition counts and state occurrence counts from labeled tracks.
    Only considers labels defined in LABEL_TO_IDX.
    """
    transition_counts = np.zeros((N_STATES, N_STATES), dtype=int)
    state_counts = np.zeros(N_STATES, dtype=int)
    print("\nExtracting transitions...")
    tracks_processed = 0
    total_valid_labels = 0
    total_transitions_counted = 0

    for track_data in all_data:
        labels = track_data.get("semantic_labels", [])
        filtered_labels = [label for label in labels if label in LABEL_TO_IDX]
        total_valid_labels += len(filtered_labels)
        if not filtered_labels:
            continue
        tracks_processed += 1
        for label in filtered_labels:
            state_counts[LABEL_TO_IDX[label]] += 1
        for i in range(len(filtered_labels) - 1):
            try:
                from_idx = LABEL_TO_IDX[filtered_labels[i]]
                to_idx = LABEL_TO_IDX[filtered_labels[i + 1]]
                transition_counts[from_idx, to_idx] += 1
                total_transitions_counted += 1
            except KeyError as ke:
                print(f"WARNING: KeyError during transition counting: {ke}.")

    print(f"Processed {tracks_processed} tracks.")
    print(f"Total valid labels encountered: {total_valid_labels}")
    print(f"Total transitions counted: {total_transitions_counted}")
    return transition_counts, state_counts


def create_transition_matrix(transition_counts):
    """
    Convert raw transition counts into a probability matrix.
    Handles states with zero outgoing transitions by assigning uniform probability.
    """
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
            transition_probs[i] = 1.0 / N_STATES
    return transition_probs


def apply_smoothing(transition_probs, alpha=SMOOTHING_ALPHA):
    """
    Apply additive (Laplace-like) smoothing to avoid zero probabilities.
    Ensures rows sum to 1 after smoothing.
    If alpha is 0 or less, smoothing is skipped.
    """
    if alpha <= 0:
        print("\nSkipping smoothing (alpha <= 0).")
        return transition_probs
    print(f"\nApplying smoothing with alpha = {alpha}...")
    smoothed_probs = transition_probs.copy()
    smoothed_probs = (1 - alpha) * smoothed_probs + alpha / N_STATES
    row_sums = np.sum(smoothed_probs, axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1.0
    smoothed_probs = smoothed_probs / row_sums
    return smoothed_probs


def incorporate_prior_knowledge(transition_probs):
    """
    Adjusts transition probabilities based on domain knowledge (e.g., music theory).
    Multiplies specific transition probabilities by enhancement/reduction factors.
    Ensures rows are renormalized afterwards.
    """
    print("\nIncorporating prior knowledge (enhancements/reductions)...")
    enhanced_probs = transition_probs.copy()
    enhancements = {
        ("Intro", "Drop"): 1.5,
        ("Drop", "Body"): 1.2,
        ("Body", "Build"): 1.3,
        ("Build", "Drop"): 1.5,
        ("Drop", "Breakdown"): 1.2,
        ("Breakdown", "Build"): 1.4,
    }
    reductions = {
        ("Outro", "Intro"): 0.1,
        ("Drop", "Intro"): 0.5,
    }

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

    print(" -> Renormalizing rows after enhancements/reductions...")
    row_sums = np.sum(enhanced_probs, axis=1)[:, np.newaxis]
    zero_rows = row_sums < 1e-9
    if np.any(zero_rows):
        zero_indices = np.where(zero_rows)[0]
        print(
            f"WARNING: Found rows summing near zero after enhancements: {zero_indices}. Setting uniform probability for these rows."
        )
        enhanced_probs[zero_rows.flatten(), :] = 1.0 / N_STATES
        row_sums[zero_rows] = 1.0
    enhanced_probs = enhanced_probs / row_sums
    return enhanced_probs


def visualize_transition_matrix(transition_probs, title="Transition Matrix"):
    """Creates and saves a visual heatmap representation of the transition matrix."""
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
    """Saves the NumPy transition matrix to the specified file."""
    try:
        os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True)
        save_path = os.path.join(HMM_OUTPUT_FOLDER, filename)
        np.save(save_path, transition_matrix)
        print(f"Saved final transition matrix to {save_path}")
    except Exception as e:
        print(f"ERROR saving transition matrix to {save_path}: {e}")
        traceback.print_exc()


def print_matrix(matrix, title):
    """Prints a formatted matrix with state labels for console output."""
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
    print("--- Starting Transition Matrix Extraction ---")
    perfect_data = load_perfect_analyses(PERFECT_FOLDER_PATH)
    if not perfect_data:
        print(
            "\nNo valid 'Perfect' track data found. Cannot generate transition matrix. Exiting."
        )
        exit()

    transition_counts, state_counts = extract_transition_counts(perfect_data)
    print("\n--- State Occurrence Counts ---")
    for state_idx, count in enumerate(state_counts):
        print(f"  {IDX_TO_LABEL.get(state_idx, 'Unknown')}: {int(count)}")
    print_matrix(transition_counts, "Raw Transition Counts")

    transition_probs = create_transition_matrix(transition_counts)
    print_matrix(
        transition_probs, "Initial Transition Probabilities (Normalized Counts)"
    )
    visualize_transition_matrix(transition_probs, "Initial Transition Probabilities")

    smoothed_probs = apply_smoothing(transition_probs, alpha=SMOOTHING_ALPHA)
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
        smoothed_probs = transition_probs  # Ensure variable holds correct matrix

    # --- Enhancement Step (Calculated but NOT saved) ---
    enhanced_probs = incorporate_prior_knowledge(smoothed_probs)
    print_matrix(
        enhanced_probs, "Enhanced Transition Probabilities (Initial/Smoothed + Priors)"
    )
    visualize_transition_matrix(enhanced_probs, "Enhanced Transition Probabilities")

    # --- Save the NON-ENHANCED matrix ---
    # Use 'smoothed_probs' which holds either the smoothed or the initial probabilities
    print(
        "\n*** Saving the Initial/Smoothed (Non-Enhanced) Matrix ***"
    )  # Added print statement
    save_transition_matrix(smoothed_probs, OUTPUT_MATRIX_FILENAME)

    print("\n--- Transition matrix analysis and saving complete! ---")
