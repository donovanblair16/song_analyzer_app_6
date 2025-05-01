# =============================================================================
# FILE: data_utils.py
# Purpose: Handle data loading and cleaning operations for GMMHMM training.
# =============================================================================

import os
import joblib
import numpy as np
from collections import defaultdict

# Import constants from config within the same package
from .config import LABELS_TO_IGNORE

# Attempt to import the feature extraction function from the root level
# This assumes 'audio_analysis_wrapper.py' is in the project root
try:
    # Need to adjust path if running this module directly vs importing it
    import sys
    # Add project root to path to find audio_analysis_wrapper
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from audio_analysis_wrapper import extract_section_features
    print("Successfully imported 'extract_section_features' from 'audio_analysis_wrapper'.")
except ImportError:
    print("\n *** WARNING: Could not import 'extract_section_features' from 'audio_analysis_wrapper.py'. ***")
    print(" *** Using a DUMMY function instead. Ensure the real function provides required features. ***\n")

    # Define a dummy function if the real one isn't available
    def extract_section_features(track_data):
        print("WARNING: Using DUMMY extract_section_features function!")
        labels = track_data.get("semantic_labels", ["Intro", "Drop", "Outro", "Fill", "Drop"])
        features = []
        for i, label in enumerate(labels):
            features.append({
                "avg_rms": np.random.rand() * 0.5 + (0.1 if label == "Intro" else 0.4),
                "relative_position": i / len(labels),
                "low_energy_norm": np.random.rand(),
                "rms_std_dev_section": np.random.rand() * 0.1,
                "centroid_std_dev_section": np.random.rand() * 100,
                "delta_rms": (np.random.rand() - 0.5) * 0.1,
                "delta_centroid": (np.random.rand() - 0.5) * 100,
                "rms_trend": (np.random.rand() - 0.5) * 0.05,
                "crest_factor": 1.0 + np.random.rand() * 2.0,
                "spectral_centroid_slope": (np.random.rand() - 0.5) * 200,
                "duration_bars": np.random.randint(1, 8),
                "original_label": label,
            })
        return features, labels

# --- Data Loading ---
def load_perfect_analyses(folder_path):
    """
    Loads all .joblib analysis files from the specified folder.

    Args:
        folder_path (str): Path to the folder containing .joblib analysis files.

    Returns:
        list: A list of tuples: [(filename, data_dict), ...]. Returns empty list on error.
    """
    all_data_with_filenames = []  # Store tuples of (filename, data)
    if not os.path.isdir(folder_path):
        print(f"ERROR: Analysis folder not found at: {folder_path}")
        return []
    try:
        analysis_files = [f for f in os.listdir(folder_path) if f.endswith(".joblib")]
    except FileNotFoundError:
        print(f"ERROR: Cannot access analysis folder: {folder_path}")
        return []

    print(f"Found {len(analysis_files)} .joblib files in '{os.path.basename(folder_path)}'.")
    for filename in analysis_files:
        file_path = os.path.join(folder_path, filename)
        try:
            data = joblib.load(file_path)
            # Validate essential data
            if (
                "semantic_labels" in data
                and isinstance(data["semantic_labels"], list)
                and len(data["semantic_labels"]) > 0
            ):
                # Append filename along with data
                all_data_with_filenames.append((filename, data))
            else:
                print(f" -> Skipping {filename}: Missing or empty 'semantic_labels'.")
        except Exception as e:
            print(f" -> Error loading {filename}: {e}")

    print(f"Successfully loaded data from {len(all_data_with_filenames)} files.")
    return all_data_with_filenames

# --- Data Cleaning Functions ---
def remove_feature_outliers(section_features_list, section_labels, song_name, z_threshold=3.0):
    """
    Remove sections with extreme feature values based on Z-score.

    Args:
        section_features_list (list): List of feature dictionaries for sections.
        section_labels (list): List of corresponding labels.
        song_name (str): Name of the song for reporting.
        z_threshold (float): Z-score threshold for outlier detection.

    Returns:
        tuple: (cleaned_features, cleaned_labels, removal_details)
                removal_details is a list of dictionaries describing removed sections.
    """
    print(f"Applying outlier removal with z-threshold = {z_threshold}...")
    original_count = len(section_features_list)

    # Extract arrays for each feature across all sections in the list
    feature_arrays = defaultdict(list)
    for section_dict in section_features_list:
        for key, value in section_dict.items():
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                feature_arrays[key].append(float(value)) # Ensure float conversion

    # Calculate means and std deviations for each feature
    feature_stats = {}
    for key, values in feature_arrays.items():
        if len(values) > 1: # Need at least 2 values for std
            np_values = np.array(values)
            mean = np.mean(np_values)
            std = np.std(np_values)
            if std > 1e-9: # Avoid division by zero or near-zero std
                feature_stats[key] = {'mean': mean, 'std': std}

    # Check each section for outliers
    cleaned_features = []
    cleaned_labels = []
    removal_details = []
    removed_count = 0

    for idx, (section_dict, label) in enumerate(zip(section_features_list, section_labels)):
        is_outlier = False
        outlier_feature = None
        outlier_value = None

        for key, stats in feature_stats.items():
            if key in section_dict:
                value = section_dict[key]
                # Check if value is numeric and finite before calculating z-score
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    value_float = float(value)
                    z_score = abs((value_float - stats['mean']) / stats['std'])
                    if z_score > z_threshold:
                        is_outlier = True
                        outlier_feature = key
                        outlier_value = value_float
                        break # Found an outlier feature, no need to check others for this section
                else:
                    # Handle non-numeric or non-finite values if necessary, or just skip
                    pass


        if is_outlier:
            removed_count += 1
            removal_details.append({
                "song": song_name,
                "section_index": idx,
                "label": label,
                "reason": f"Outlier (z-score > {z_threshold})",
                "feature": outlier_feature,
                "value": f"{outlier_value:.4f}" if outlier_value is not None else "N/A"
            })
        else:
            cleaned_features.append(section_dict)
            cleaned_labels.append(label)

    print(f"Removed {removed_count} of {original_count} sections as outliers.")
    return cleaned_features, cleaned_labels, removal_details

def remove_short_sections(section_features_list, section_labels, song_name, min_bars=2):
    """
    Remove sections shorter than minimum bar count.

    Args:
        section_features_list (list): List of feature dictionaries.
        section_labels (list): List of corresponding labels.
        song_name (str): Name of the song for reporting.
        min_bars (float): Minimum duration in bars.

    Returns:
        tuple: (cleaned_features, cleaned_labels, removal_details)
    """
    print(f"Removing sections shorter than {min_bars} bars...")
    original_count = len(section_features_list)

    cleaned_features = []
    cleaned_labels = []
    removal_details = []

    for idx, (section, label) in enumerate(zip(section_features_list, section_labels)):
        section_bars = section.get('duration_bars', 0) # Default to 0 if key missing
        # Ensure section_bars is numeric before comparison
        is_short = False
        if isinstance(section_bars, (int, float, np.number)) and np.isfinite(section_bars):
            if float(section_bars) < min_bars:
                is_short = True
        else:
            # Handle non-numeric duration_bars if needed, e.g., treat as short or log warning
            print(f"Warning: Non-numeric duration_bars '{section_bars}' for section {idx} in {song_name}. Treating as short.")
            is_short = True # Or handle differently

        if not is_short:
            cleaned_features.append(section)
            cleaned_labels.append(label)
        else:
            removal_details.append({
                "song": song_name,
                "section_index": idx,
                "label": label,
                "reason": f"Too short (<{min_bars} bars)",
                "feature": "duration_bars",
                "value": f"{float(section_bars):.1f}" if isinstance(section_bars, (int, float, np.number)) and np.isfinite(section_bars) else str(section_bars)
            })

    removed_count = original_count - len(cleaned_features)
    print(f"Removed {removed_count} of {original_count} sections for being too short.")
    return cleaned_features, cleaned_labels, removal_details


def filter_consistency(section_features_list, section_labels, song_name):
    """
    Remove sections where features (e.g., RMS) don't match typical values for their label.
    This is a basic example checking RMS; could be expanded.

    Args:
        section_features_list (list): List of feature dictionaries.
        section_labels (list): List of corresponding labels.
        song_name (str): Name of the song for reporting.

    Returns:
        tuple: (cleaned_features, cleaned_labels, removal_details)
    """
    print("Applying consistency filtering (basic RMS check)...")
    original_count = len(section_features_list)

    label_feature_means = defaultdict(lambda: defaultdict(list))

    # Calculate average feature values per label across the input list
    for section, label in zip(section_features_list, section_labels):
        # Only consider labels that are NOT in LABELS_TO_IGNORE for calculating typicals
        if label not in LABELS_TO_IGNORE:
            for feature, value in section.items():
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                        label_feature_means[label][feature].append(float(value))

    # Convert lists to means
    for label in label_feature_means:
        for feature in label_feature_means[label]:
            values = label_feature_means[label][feature]
            if values:
                label_feature_means[label][feature] = np.mean(values)
            else:
                # Handle case where a feature might have no valid values for a label
                label_feature_means[label][feature] = None # Or some other indicator

    # Filter sections
    filtered_features = []
    filtered_labels = []
    removal_details = []

    for idx, (section, label) in enumerate(zip(section_features_list, section_labels)):
        consistent = True
        inconsistent_feature = None
        inconsistent_value = None

        # Check consistency only for labels we care about (not ignored ones)
        if label not in LABELS_TO_IGNORE and label in label_feature_means:
            # Example: Check if RMS is within 0.5x to 2.0x of typical for this label
            if 'avg_rms' in section and 'avg_rms' in label_feature_means[label]:
                typical_rms = label_feature_means[label]['avg_rms']
                section_rms = section['avg_rms']

                # Ensure both values are valid numbers before comparing
                if typical_rms is not None and isinstance(section_rms, (int, float, np.number)) and np.isfinite(section_rms) and typical_rms > 1e-6: # Avoid division by zero/small numbers
                    section_rms_float = float(section_rms)
                    if not (0.5 * typical_rms <= section_rms_float <= 2.0 * typical_rms):
                        consistent = False
                        inconsistent_feature = 'avg_rms'
                        inconsistent_value = section_rms_float
                elif typical_rms is None:
                    # Cannot perform check if typical value wasn't calculated
                    pass
                elif not (isinstance(section_rms, (int, float, np.number)) and np.isfinite(section_rms)):
                        # Handle case where section RMS is invalid
                        consistent = False # Treat invalid RMS as inconsistent
                        inconsistent_feature = 'avg_rms'
                        inconsistent_value = section_rms # Keep original value for reporting

            # Add more consistency checks here for other features if desired

        # Keep section if consistent
        if consistent:
            filtered_features.append(section)
            filtered_labels.append(label)
        else:
            removal_details.append({
                "song": song_name,
                "section_index": idx,
                "label": label,
                "reason": "Inconsistent feature values (e.g., RMS)",
                "feature": inconsistent_feature,
                "value": f"{inconsistent_value:.4f}" if isinstance(inconsistent_value, (float, np.number)) else str(inconsistent_value)
            })

    removed_count = original_count - len(filtered_features)
    print(f"Removed {removed_count} of {original_count} sections for inconsistency.")
    return filtered_features, filtered_labels, removal_details
