# =============================================================================
# FILE: data_utils.py
# Purpose: Handle data loading and cleaning operations for GMMHMM training.
# MODIFIED: Removed internal/external feature extraction logic. Assumes
#           features are pre-calculated in loaded joblib files.
# =============================================================================

import os
import joblib
import numpy as np
from collections import defaultdict

# Import constants from config within the same package
from .config import LABELS_TO_IGNORE


# --- Data Loading ---
def load_perfect_analyses(folder_path):
    """
    Loads all .joblib analysis files from the specified folder.
    Expects files to contain 'semantic_labels' and 'section_features' (list of dicts).

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

    print(
        f"Found {len(analysis_files)} .joblib files in '{os.path.basename(folder_path)}'."
    )
    for filename in analysis_files:
        file_path = os.path.join(folder_path, filename)
        try:
            data = joblib.load(file_path)
            # Validate essential data
            # Check for semantic_labels and section_features
            if (
                "semantic_labels" in data
                and isinstance(data["semantic_labels"], list)
                and len(data["semantic_labels"]) > 0
                and "section_features" in data  # Check for pre-calculated features
                and isinstance(data["section_features"], list)
                and len(data["section_features"])
                == len(data["semantic_labels"])  # Ensure lengths match
            ):
                # Append filename along with data
                all_data_with_filenames.append((filename, data))
                print(
                    f" -> Successfully loaded {filename} with {len(data['semantic_labels'])} sections."
                )
            else:
                missing_keys = []
                if (
                    "semantic_labels" not in data
                    or not isinstance(data.get("semantic_labels"), list)
                    or not data.get("semantic_labels")
                ):
                    missing_keys.append("'semantic_labels' (missing or empty list)")
                if "section_features" not in data or not isinstance(
                    data.get("section_features"), list
                ):
                    missing_keys.append("'section_features' (missing or not a list)")
                elif (
                    "semantic_labels" in data
                    and "section_features" in data
                    and len(data.get("section_features", []))
                    != len(data.get("semantic_labels", []))
                ):
                    missing_keys.append(
                        "'section_features' and 'semantic_labels' length mismatch"
                    )

                print(
                    f" -> Skipping {filename}: Missing or invalid required data: {', '.join(missing_keys)}."
                )
        except Exception as e:
            print(f" -> Error loading {filename}: {e}")

    print(f"Successfully loaded valid data from {len(all_data_with_filenames)} files.")
    return all_data_with_filenames


# --- Data Cleaning Functions ---
def remove_feature_outliers(
    section_features_list, section_labels, song_name, z_threshold=3.0
):
    """
    Remove sections with extreme feature values based on Z-score.
    Operates on the provided list of feature dictionaries.

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
    if original_count == 0:
        return [], [], []

    # Extract arrays for each feature across all sections in the list
    feature_arrays = defaultdict(list)
    # Use the first valid section to determine available numeric features
    available_numeric_features = set()
    for section_dict in section_features_list:
        if isinstance(section_dict, dict):
            for key, value in section_dict.items():
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    available_numeric_features.add(key)
        # Optimization: break after finding the first valid dict's features
        # if available_numeric_features:
        #     break

    if not available_numeric_features:
        print(
            "Warning: No numeric features found in the first section to calculate outlier stats."
        )
        return section_features_list, section_labels, []

    # Populate feature arrays only for available numeric features
    for section_dict in section_features_list:
        if isinstance(section_dict, dict):
            for key in available_numeric_features:
                value = section_dict.get(key)  # Use .get() for safety
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    feature_arrays[key].append(float(value))  # Ensure float conversion
                # Optionally handle missing values for a feature within a section if needed
                # else: feature_arrays[key].append(np.nan) # or skip

    # Calculate means and std deviations for each feature
    feature_stats = {}
    for key, values in feature_arrays.items():
        if len(values) > 1:  # Need at least 2 values for std
            np_values = np.array(values)
            # Filter out potential NaNs added above if handling missing values that way
            np_values = np_values[np.isfinite(np_values)]
            if len(np_values) > 1:
                mean = np.mean(np_values)
                std = np.std(np_values)
                if std > 1e-9:  # Avoid division by zero or near-zero std
                    feature_stats[key] = {"mean": mean, "std": std}

    # Check each section for outliers
    cleaned_features = []
    cleaned_labels = []
    removal_details = []
    removed_count = 0

    for idx, (section_dict, label) in enumerate(
        zip(section_features_list, section_labels)
    ):
        is_outlier = False
        outlier_feature = None
        outlier_value = None

        # Check only against features for which we could calculate stats
        if isinstance(section_dict, dict):  # Ensure it's a dictionary
            for key, stats in feature_stats.items():
                if key in section_dict:
                    value = section_dict[key]
                    # Check if value is numeric and finite before calculating z-score
                    if isinstance(value, (int, float, np.number)) and np.isfinite(
                        value
                    ):
                        value_float = float(value)
                        # Use the calculated std dev which should be > 1e-9
                        z_score = abs((value_float - stats["mean"]) / stats["std"])
                        if z_score > z_threshold:
                            is_outlier = True
                            outlier_feature = key
                            outlier_value = value_float
                            break  # Found an outlier feature, no need to check others
                    else:
                        # Handle non-numeric or non-finite values if necessary, or just skip
                        pass
        else:
            # Handle cases where an element isn't a dictionary (shouldn't happen with checks in load)
            print(
                f"Warning: Item at index {idx} is not a dictionary. Skipping outlier check for this item."
            )

        if is_outlier:
            removed_count += 1
            removal_details.append(
                {
                    "song": song_name,
                    "section_index": idx,
                    "label": label,
                    "reason": f"Outlier (z-score > {z_threshold})",
                    "feature": outlier_feature,
                    "value": (
                        f"{outlier_value:.4f}" if outlier_value is not None else "N/A"
                    ),
                }
            )
        else:
            # Only append if it wasn't identified as an outlier
            cleaned_features.append(section_dict)
            cleaned_labels.append(label)

    print(f"Removed {removed_count} of {original_count} sections as outliers.")
    return cleaned_features, cleaned_labels, removal_details


def remove_short_sections(section_features_list, section_labels, song_name, min_bars=2):
    """
    Remove sections shorter than minimum bar count, assuming 'duration_bars'
    feature exists in the pre-calculated features.

    Args:
        section_features_list (list): List of feature dictionaries.
        section_labels (list): List of corresponding labels.
        song_name (str): Name of the song for reporting.
        min_bars (float): Minimum duration in bars.

    Returns:
        tuple: (cleaned_features, cleaned_labels, removal_details)
    """
    print(
        f"Removing sections shorter than {min_bars} bars (requires 'duration_bars' feature)..."
    )
    original_count = len(section_features_list)

    cleaned_features = []
    cleaned_labels = []
    removal_details = []

    has_duration_bars_feature = False  # Flag to check if the feature exists

    for idx, (section, label) in enumerate(zip(section_features_list, section_labels)):
        # Check if the section is a dict and contains the key on the first iteration
        if idx == 0 and isinstance(section, dict) and "duration_bars" in section:
            has_duration_bars_feature = True
        elif idx == 0:
            print(
                "Warning: 'duration_bars' feature not found in the first section. Cannot remove short sections."
            )

        if not has_duration_bars_feature:
            # If feature missing, keep all sections and exit the loop for this cleaning type
            cleaned_features = section_features_list
            cleaned_labels = section_labels
            break  # Stop checking further sections

        # Proceed if feature exists
        section_bars = section.get(
            "duration_bars", 0
        )  # Default to 0 if key missing (though checked above)
        # Ensure section_bars is numeric before comparison
        is_short = False
        if isinstance(section_bars, (int, float, np.number)) and np.isfinite(
            section_bars
        ):
            if float(section_bars) < min_bars:
                is_short = True
        else:
            # Handle non-numeric duration_bars if needed, e.g., treat as short or log warning
            print(
                f"Warning: Non-numeric duration_bars '{section_bars}' for section {idx} in {song_name}. Treating as short."
            )
            is_short = True  # Or handle differently

        if not is_short:
            cleaned_features.append(section)
            cleaned_labels.append(label)
        else:
            removal_details.append(
                {
                    "song": song_name,
                    "section_index": idx,
                    "label": label,
                    "reason": f"Too short (<{min_bars} bars)",
                    "feature": "duration_bars",
                    "value": (
                        f"{float(section_bars):.1f}"
                        if isinstance(section_bars, (int, float, np.number))
                        and np.isfinite(section_bars)
                        else str(section_bars)
                    ),
                }
            )

    removed_count = original_count - len(cleaned_features)
    if (
        has_duration_bars_feature
    ):  # Only print count if the check was actually performed
        print(
            f"Removed {removed_count} of {original_count} sections for being too short."
        )
    return cleaned_features, cleaned_labels, removal_details


def filter_consistency(section_features_list, section_labels, song_name):
    """
    Remove sections where features (e.g., RMS) don't match typical values for their label.
    This is a basic example checking RMS; requires 'avg_rms' feature.

    Args:
        section_features_list (list): List of feature dictionaries.
        section_labels (list): List of corresponding labels.
        song_name (str): Name of the song for reporting.

    Returns:
        tuple: (cleaned_features, cleaned_labels, removal_details)
    """
    print("Applying consistency filtering (basic RMS check, requires 'avg_rms')...")
    original_count = len(section_features_list)
    if original_count == 0:
        return [], [], []

    # Check if avg_rms exists in the first section's features
    has_avg_rms = False
    if (
        isinstance(section_features_list[0], dict)
        and "avg_rms" in section_features_list[0]
    ):
        has_avg_rms = True
    else:
        print("Warning: 'avg_rms' feature not found. Skipping consistency filtering.")
        return section_features_list, section_labels, []

    label_feature_means = defaultdict(lambda: defaultdict(list))

    # Calculate average feature values per label across the input list
    for section, label in zip(section_features_list, section_labels):
        # Only consider labels that are NOT in LABELS_TO_IGNORE for calculating typicals
        if label not in LABELS_TO_IGNORE and isinstance(
            section, dict
        ):  # Check section is dict
            # Only calculate for avg_rms if doing the basic check
            feature = "avg_rms"
            value = section.get(feature)
            if (
                value is not None
                and isinstance(value, (int, float, np.number))
                and np.isfinite(value)
            ):
                label_feature_means[label][feature].append(float(value))

    # Convert lists to means
    for label in label_feature_means:
        for feature in label_feature_means[label]:
            values = label_feature_means[label][feature]
            if values:
                label_feature_means[label][feature] = np.mean(values)
            else:
                # Handle case where a feature might have no valid values for a label
                label_feature_means[label][feature] = None  # Or some other indicator

    # Filter sections
    filtered_features = []
    filtered_labels = []
    removal_details = []

    for idx, (section, label) in enumerate(zip(section_features_list, section_labels)):
        consistent = True
        inconsistent_feature = None
        inconsistent_value = None

        # Check consistency only for labels we care about and if section is a dict
        if (
            label not in LABELS_TO_IGNORE
            and label in label_feature_means
            and isinstance(section, dict)
        ):
            # Example: Check if RMS is within 0.5x to 2.0x of typical for this label
            feature_to_check = "avg_rms"
            if (
                feature_to_check in section
                and feature_to_check in label_feature_means[label]
            ):
                typical_val = label_feature_means[label][feature_to_check]
                section_val = section[feature_to_check]

                # Ensure both values are valid numbers before comparing
                if (
                    typical_val is not None
                    and isinstance(section_val, (int, float, np.number))
                    and np.isfinite(section_val)
                    and typical_val > 1e-6
                ):  # Avoid division by zero/small numbers
                    section_val_float = float(section_val)
                    if not (
                        0.5 * typical_val <= section_val_float <= 2.0 * typical_val
                    ):
                        consistent = False
                        inconsistent_feature = feature_to_check
                        inconsistent_value = section_val_float
                elif typical_val is None:
                    # Cannot perform check if typical value wasn't calculated
                    pass
                elif not (
                    isinstance(section_val, (int, float, np.number))
                    and np.isfinite(section_val)
                ):
                    # Handle case where section RMS is invalid
                    consistent = False  # Treat invalid value as inconsistent
                    inconsistent_feature = feature_to_check
                    inconsistent_value = (
                        section_val  # Keep original value for reporting
                    )

            # Add more consistency checks here for other features if desired

        # Keep section if consistent
        if consistent:
            filtered_features.append(section)
            filtered_labels.append(label)
        else:
            removal_details.append(
                {
                    "song": song_name,
                    "section_index": idx,
                    "label": label,
                    "reason": "Inconsistent feature values (e.g., RMS)",
                    "feature": inconsistent_feature,
                    "value": (
                        f"{inconsistent_value:.4f}"
                        if isinstance(inconsistent_value, (float, np.number))
                        else str(inconsistent_value)
                    ),
                }
            )

    removed_count = original_count - len(filtered_features)
    print(f"Removed {removed_count} of {original_count} sections for inconsistency.")
    return filtered_features, filtered_labels, removal_details
