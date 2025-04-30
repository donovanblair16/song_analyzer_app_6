# =============================================================================
# FILE: hmm_model.py
# Purpose: Handle HMM-specific data preparation and training logic.
# =============================================================================

import numpy as np
import traceback
from collections import defaultdict
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Import functions/constants from other modules in the package
from .config import LABELS_TO_IGNORE, DEFAULT_HMM_RANDOM_STATE, DEFAULT_HMM_COVARIANCE_TYPE, DEFAULT_HMM_N_ITER, DEFAULT_HMM_TOL
from .data_utils import (
    extract_section_features, remove_feature_outliers,
    remove_short_sections, filter_consistency
)
import os # For path splitting in cleaning

def prepare_multi_feature_data(all_track_data_with_filenames, feature_keys, cleaning_settings):
    """
    Processes loaded track data to extract features, clean, scale, and prepare for HMM.

    Args:
        all_track_data_with_filenames (list): List of tuples (filename, data_dict).
        feature_keys (list): List of feature names (strings) to use.
        cleaning_settings (dict): Dictionary with cleaning parameters:
            {
                'enable_outlier_removal': bool, 'outlier_z_threshold': float or None,
                'enable_short_section_removal': bool, 'short_section_min_bars': float or None,
                'enable_consistency_filtering': bool
            }

    Returns:
        tuple or None: On success, returns a tuple containing:
            (X_scaled (np.ndarray): Scaled feature array (not weighted yet),
                sequence_lengths (list): List of sequence lengths,
                int_to_label (dict): Map from HMM state index to label name,
                label_to_int (dict): Map from label name to HMM state index,
                scaler_multi (StandardScaler): Fitted scaler object,
                data_cleaning_info (dict): Information about data cleaning steps,
                raw_averages (dict): Raw feature averages by label before scaling)
        Returns None on critical failure.
    """
    print("\n--- Preparing Multi-Feature Data for HMM ---")
    if not feature_keys:
        print("ERROR: No feature keys provided for data preparation.")
        return None

    n_features = len(feature_keys)
    print(f"Using {n_features} features: {feature_keys}")

    all_feature_vectors = []
    all_states_str_sequence = [] # Store the sequence of valid state labels
    sequence_lengths = []      # List to store lengths of sequences

    # Track data cleaning details
    data_cleaning_info = {
        "summary": {
            "total_processed": 0, "total_removed": 0,
            "outliers_removed": 0, "short_sections_removed": 0,
            "inconsistent_removed": 0
        },
        "removal_details": []
    }

    # --- Determine the set of valid states ---
    all_possible_labels = set()
    for _, data in all_track_data_with_filenames:
        if data and "semantic_labels" in data:
            all_possible_labels.update(data["semantic_labels"])

    valid_states = sorted([label for label in all_possible_labels if label not in LABELS_TO_IGNORE])
    if not valid_states:
        print("ERROR: No valid states found after filtering ignored labels. Check LABELS_TO_IGNORE and data.")
        return None

    label_to_int = {label: i for i, label in enumerate(valid_states)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    n_states = len(valid_states)
    print(f"HMM States ({n_states}): {valid_states}")
    print(f"Label to Int Map: {label_to_int}")

    # --- Extract Feature vectors and state sequence ---
    skipped_sections_count = 0
    total_cleaned_sections = 0
    label_feature_values = defaultdict(lambda: defaultdict(list)) # For raw averages

    for i, (filename, track_data) in enumerate(all_track_data_with_filenames):
        print(f"\n===== Processing track {i+1}/{len(all_track_data_with_filenames)} ({filename}) =====")
        try:
            # Call feature extraction (might be dummy or real)
            section_feature_dicts, section_labels = extract_section_features(track_data)
            if not section_feature_dicts or not section_labels or len(section_feature_dicts) != len(section_labels):
                print(f" -> Skipping track {i+1}: Invalid data from extract_section_features.")
                continue
        except Exception as e:
            print(f" -> ERROR calling extract_section_features for track {i+1} ({filename}): {e}")
            traceback.print_exc()
            continue

        data_cleaning_info["summary"]["total_processed"] += len(section_feature_dicts)
        original_section_count = len(section_feature_dicts)
        song_name = os.path.splitext(filename)[0] # For reporting

        # --- Apply Data Cleaning (if enabled) ---
        current_features = section_feature_dicts
        current_labels = section_labels

        if cleaning_settings.get('enable_outlier_removal'):
            current_features, current_labels, details = remove_feature_outliers(
                current_features, current_labels, song_name, cleaning_settings.get('outlier_z_threshold', 3.0))
            data_cleaning_info["summary"]["outliers_removed"] += len(details)
            data_cleaning_info["removal_details"].extend(details)

        if cleaning_settings.get('enable_short_section_removal'):
            current_features, current_labels, details = remove_short_sections(
                current_features, current_labels, song_name, cleaning_settings.get('short_section_min_bars', 2))
            data_cleaning_info["summary"]["short_sections_removed"] += len(details)
            data_cleaning_info["removal_details"].extend(details)

        if cleaning_settings.get('enable_consistency_filtering'):
            current_features, current_labels, details = filter_consistency(
                current_features, current_labels, song_name)
            data_cleaning_info["summary"]["inconsistent_removed"] += len(details)
            data_cleaning_info["removal_details"].extend(details)

        # Update final lists after all cleaning steps
        section_feature_dicts = current_features
        section_labels = current_labels

        sections_removed = original_section_count - len(section_feature_dicts)
        total_cleaned_sections += sections_removed
        data_cleaning_info["summary"]["total_removed"] += sections_removed
        if sections_removed > 0:
            print(f" -> Data cleaning removed {sections_removed} of {original_section_count} sections.")

        if not section_feature_dicts:
            print(f" -> Skipping track {i+1}: No sections remain after data cleaning.")
            continue

        # --- Process remaining sections ---
        track_feature_vectors = []
        track_state_labels = []

        for idx, feature_dict in enumerate(section_feature_dicts):
            label = section_labels[idx]

            if label in LABELS_TO_IGNORE or label not in label_to_int:
                if label not in LABELS_TO_IGNORE: # Only count skips if it wasn't explicitly ignored
                        skipped_sections_count += 1
                continue # Skip ignored or invalid labels

            # Collect raw feature values before potential skip
            for key, value in feature_dict.items():
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    # Only collect if key is one of the selected features for this run
                    if key in feature_keys:
                        label_feature_values[label][key].append(float(value))


            # Extract the required features for this section
            current_feature_vector = []
            valid_section = True
            for key in feature_keys:
                value = feature_dict.get(key)
                if value is None or not np.isfinite(float(value)):
                    print(f" -> WARNING: Skipping section {idx} (Label: {label}): Missing or non-finite value for feature '{key}'. Value: {value}")
                    valid_section = False
                    skipped_sections_count += 1
                    # Clear any partially collected raw values for this invalid section if needed
                    # (Currently, raw values are collected before this check)
                    break
                current_feature_vector.append(float(value))

            if valid_section:
                if len(current_feature_vector) != n_features:
                        print(f" -> ERROR: Section {idx} (Label: {label}) yielded wrong number of features ({len(current_feature_vector)} vs {n_features}). Skipping.")
                        skipped_sections_count += 1
                        continue # Skip section with wrong feature count
                track_feature_vectors.append(current_feature_vector)
                track_state_labels.append(label)


        # Add sequence to global lists if valid sections were found
        if track_feature_vectors:
            all_feature_vectors.extend(track_feature_vectors)
            all_states_str_sequence.extend(track_state_labels)
            sequence_lengths.append(len(track_feature_vectors))
            print(f" -> Added sequence of length {len(track_feature_vectors)}")
        else:
            print(f" -> Track {i+1} resulted in empty feature sequence after filtering/cleaning.")

    print("\n===== Finished Processing All Tracks =====")
    if skipped_sections_count > 0:
        print(f"NOTE: Skipped {skipped_sections_count} sections due to ignored labels or missing/invalid feature values.")
    if total_cleaned_sections > 0:
        print(f"NOTE: Data cleaning removed {total_cleaned_sections} sections in total.")

    if not all_feature_vectors:
        print("\n *** ERROR: No valid multi-feature observations generated. Cannot train. ***")
        print(" *** Check warnings, feature extraction, and cleaning settings. ***")
        return None

    # --- Convert to NumPy array and Scale ---
    X = np.array(all_feature_vectors)
    if X.ndim != 2 or X.shape[1] != n_features:
        print(f"ERROR: Final feature array X has unexpected shape {X.shape}. Expected (*, {n_features}).")
        return None
    if X.shape[0] == 0:
        print("Error: Final feature array X is unexpectedly empty.")
        return None

    print(f"\nTotal valid observations: {X.shape[0]}, Features per observation: {X.shape[1]}")

    # --- Calculate Raw Averages ---
    raw_averages = {}
    print("\nCalculating Raw Feature Averages (before scaling):")
    for label in sorted(label_feature_values.keys()): # Sort labels for consistent output
        raw_averages[label] = {}
        print(f"  Label: '{label}'")
        # Sort features for consistent output
        for feature in sorted(label_feature_values[label].keys()):
            values = label_feature_values[label][feature]
            if values:
                mean_val = np.mean(values)
                raw_averages[label][feature] = mean_val
                print(f"    - {feature}: {mean_val:.4f} (from {len(values)} sections)")
            else:
                raw_averages[label][feature] = None # Indicate no data
                print(f"    - {feature}: N/A (no valid sections)")


    # --- Scale Features ---
    print(f"\nScaling {n_features} features using StandardScaler...")
    scaler_multi = StandardScaler()
    try:
        X_scaled = scaler_multi.fit_transform(X)
        print(" -> Features scaled.")
        # Round means and scales for cleaner printing
        means_rounded = np.round(scaler_multi.mean_, 4)
        scales_rounded = np.round(scaler_multi.scale_, 4)
        print(f" -> Scaler Means: {means_rounded}")
        print(f" -> Scaler Scales (StdDev): {scales_rounded}")
    except ValueError as scale_err:
            print(f"ERROR during scaling: {scale_err}")
            print(" -> This might happen if a feature has zero variance after cleaning.")
            # Add check for zero variance columns
            zero_var_cols = np.where(np.std(X, axis=0) < 1e-9)[0]
            if len(zero_var_cols) > 0:
                print(" -> Features with near-zero variance detected at indices:", zero_var_cols)
                print(" -> Corresponding feature keys:", [feature_keys[i] for i in zero_var_cols])
            return None


    # --- Final Checks (before returning) ---
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("ERROR: NaNs or Infs found in scaled feature data!")
        # Find columns with NaN/Inf
        nan_cols = np.where(np.isnan(X_scaled).any(axis=0))[0]
        inf_cols = np.where(np.isinf(X_scaled).any(axis=0))[0]
        print(" -> NaN columns:", [feature_keys[i] for i in nan_cols])
        print(" -> Inf columns:", [feature_keys[i] for i in inf_cols])
        return None
    if len(all_states_str_sequence) != X_scaled.shape[0]:
        print("!!! ERROR: Mismatch between length of final features and full state sequence!")
        return None

    print("\nMulti-feature data preparation complete.")
    return (
        X_scaled, # Scaled features (weighting happens just before training)
        sequence_lengths,
        int_to_label,
        label_to_int,
        scaler_multi,
        data_cleaning_info,
        raw_averages
    )


def train_gmmhmm(X_scaled, feature_weights_map, feature_keys, lengths, n_states, n_mix, min_covar, transmat_prior=None):
    """
    Applies weights and trains a GMMHMM model.

    Args:
        X_scaled (np.ndarray): The scaled multi-feature data (n_observations, n_features).
        feature_weights_map (dict): Dictionary mapping feature keys to their weights.
        feature_keys (list): Ordered list of feature keys corresponding to columns in X_scaled.
        lengths (list): List of sequence lengths.
        n_states (int): The number of hidden states (labels).
        n_mix (int): The number of Gaussian mixtures per state.
        min_covar (float): Minimum covariance value for regularization.
        transmat_prior (np.ndarray, optional): Pre-calculated transition matrix to initialize with.

    Returns:
        tuple: (trained_model (hmm.GMMHMM or None), feature_weight_vector (np.ndarray))
                Returns None for model if training fails.
    """
    print(f"\n--- Training GMMHMM ({n_states} States, {n_mix} Mixtures) ---")
    n_features = X_scaled.shape[1]
    if n_features != len(feature_keys):
            print(f"ERROR: Mismatch between X_scaled features ({n_features}) and feature_keys ({len(feature_keys)})")
            return None, None

    # --- Apply Weighting ---
    print(f"Applying feature weights: {feature_weights_map}")
    # Create weight vector in the correct order
    try:
        feature_weight_vector = np.array([feature_weights_map.get(k, 1.0) for k in feature_keys])
    except Exception as e:
        print(f"ERROR creating weight vector: {e}")
        return None, None

    if len(feature_weight_vector) != n_features:
        print(f"ERROR: Weight vector length ({len(feature_weight_vector)}) != number of features ({n_features})")
        return None, None

    # Apply weights element-wise
    X_weighted = X_scaled * feature_weight_vector
    print(" -> Features weighted.")

    # --- Final Check on Weighted Data ---
    if np.any(np.isnan(X_weighted)) or np.any(np.isinf(X_weighted)):
        print("ERROR: NaNs or Infs found in final weighted feature data!")
        nan_cols = np.where(np.isnan(X_weighted).any(axis=0))[0]
        inf_cols = np.where(np.isinf(X_weighted).any(axis=0))[0]
        print(" -> NaN columns:", [feature_keys[i] for i in nan_cols])
        print(" -> Inf columns:", [feature_keys[i] for i in inf_cols])
        return None, feature_weight_vector # Return weights even if training fails

    print(f"Training data shape: {X_weighted.shape}")
    print(f"Number of sequences: {len(lengths)}")
    print(f"Using Covariance Type: {DEFAULT_HMM_COVARIANCE_TYPE}")
    print(f"Max Iterations: {DEFAULT_HMM_N_ITER}, Tolerance: {DEFAULT_HMM_TOL}, Min Covar: {min_covar}")

    # --- Initialize GMMHMM ---
    model = hmm.GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type=DEFAULT_HMM_COVARIANCE_TYPE,
        min_covar=min_covar,
        random_state=DEFAULT_HMM_RANDOM_STATE,
        n_iter=DEFAULT_HMM_N_ITER,
        tol=DEFAULT_HMM_TOL,
        verbose=True, # Print convergence info
        # Init params: s=startprob, t=transmat, m=means, c=covars, w=weights
        init_params="scmw" if transmat_prior is not None else "stmcw",
        params="stmcw", # Train all parameters
    )

    # --- Set Initial Transition Matrix (if provided) ---
    if transmat_prior is not None:
        print("Initializing transition matrix with provided prior.")
        if transmat_prior.shape == (n_states, n_states):
            # Ensure rows sum to 1 (handle potential floating point issues)
            row_sums = np.sum(transmat_prior, axis=1)[:, np.newaxis]
            # Avoid division by zero for rows that sum to zero (shouldn't happen with valid prior)
            row_sums[row_sums < 1e-9] = 1.0
            valid_transmat = transmat_prior / row_sums
            # Add small epsilon to zero probabilities and renormalize to avoid issues if needed
            # valid_transmat[valid_transmat < 1e-9] = 1e-9
            # valid_transmat /= np.sum(valid_transmat, axis=1)[:, np.newaxis]
            model.transmat_ = valid_transmat
            model.transmat_prior_ = valid_transmat # Also set the prior attribute if needed
        else:
            print(f"WARNING: Provided transition matrix shape {transmat_prior.shape} doesn't match n_states {n_states}. Using default init.")
            model.init_params = "stmcw" # Revert to full default init

    # --- Train the Model ---
    print("Fitting GMMHMM model...")
    try:
        model.fit(X_weighted, lengths)
        if model.monitor_.converged:
                print(f"GMMHMM Training Converged in {model.monitor_.n_iter} iterations.")
        else:
                print(f"WARNING: GMMHMM Training did NOT converge after {model.monitor_.n_iter} iterations.")
        print(f"Final Log Likelihood: {model.score(X_weighted, lengths)}")
        return model, feature_weight_vector
    except ValueError as ve:
        print(f"\n *** GMMHMM Training Failed: ValueError encountered. ***")
        print(f" -> Error Message: {ve}")
        print(" -> This often indicates issues like:")
        print("    - Insufficient data for some states/mixtures (check sequence lengths and cleaning).")
        print("    - Features might be constant or highly correlated after weighting.")
        print(f"    - Degenerate covariance matrices (try increasing min_covar: {min_covar}).")
        print(f"    - Consider reducing n_mix ({n_mix}) or getting more/different data.")
        traceback.print_exc()
        return None, feature_weight_vector
    except Exception as e:
        print(f"\n *** GMMHMM Training Failed: An unexpected error occurred. ***")
        print(f" -> Error Message: {e}")
        traceback.print_exc()
        return None, feature_weight_vector
