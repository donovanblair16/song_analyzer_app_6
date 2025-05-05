# =============================================================================
# FILE: hmm_model.py
# Purpose: Handle HMM-specific data preparation and training logic.
# MODIFIED: Removed internal feature extraction. Assumes features are
#           pre-calculated in loaded joblib files ('section_features').
# ADDED: Optional undersampling logic in prepare_multi_feature_data.
# =============================================================================

import numpy as np
import traceback
from collections import defaultdict, Counter  # Added Counter
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import os  # For path splitting in cleaning
import random  # Added for sampling

# Import functions/constants from other modules in the package
from .config import (
    LABELS_TO_IGNORE,
    DEFAULT_HMM_RANDOM_STATE,
    DEFAULT_HMM_COVARIANCE_TYPE,
    DEFAULT_HMM_N_ITER,
    DEFAULT_HMM_TOL,
)

# Import only cleaning functions from data_utils
from .data_utils import (
    remove_feature_outliers,
    remove_short_sections,
    filter_consistency,
)


def prepare_multi_feature_data(
    all_track_data_with_filenames, feature_keys, data_processing_settings  # Renamed arg
):
    """Processes loaded track data containing pre-calculated features ('section_features'),
       cleans (optional), undersamples (optional), scales, and prepares data for HMM training.

    Args:
        all_track_data_with_filenames (list): List of tuples, where each tuple
            contains (filename, data_dict). data_dict MUST contain
            'semantic_labels' (list) and 'section_features' (list of dicts).
        feature_keys (list): List of feature names (strings) to extract from
            the 'section_features' dictionaries and use for training.
        data_processing_settings (dict): Dictionary with cleaning/sampling parameters. Keys include:
            - 'enable_outlier_removal' (bool): Whether to remove outliers.
            - 'outlier_z_threshold' (float or None): Z-score threshold for outliers.
            - 'enable_short_section_removal' (bool): Whether to remove short sections.
            - 'short_section_min_bars' (float or None): Minimum bars for short sections.
            - 'enable_consistency_filtering' (bool): Whether to filter based on consistency.
            - 'enable_undersampling' (bool): Whether to undersample the majority class.

    Returns:
        tuple or None: On success, returns a tuple containing the following elements
        in order:

        1.  **X_scaled** (np.ndarray): Scaled feature array (potentially undersampled).
        2.  **sequence_lengths** (list or None): List of sequence lengths for HMM input.
            Will be None if undersampling was performed (sequence structure lost).
        3.  **int_to_label** (dict): Map from HMM state integer index to label name (str).
        4.  **label_to_int** (dict): Map from label name (str) to HMM state integer index.
        5.  **scaler_multi** (StandardScaler): Fitted scikit-learn StandardScaler object.
        6.  **data_cleaning_info** (dict): Information about data cleaning steps performed.
        7.  **raw_averages** (dict): Raw feature averages by label before scaling/sampling.

        Returns None on critical failure (e.g., no valid data after cleaning,
        missing essential keys in loaded data).

    """
    print(
        "\n--- Preparing Multi-Feature Data for HMM (Using Pre-calculated Features) ---"
    )
    if not feature_keys:
        print("ERROR: No feature keys provided for data preparation.")
        return None

    n_features = len(feature_keys)
    print(f"Expecting {n_features} pre-calculated features: {feature_keys}")

    all_feature_vectors_raw = []  # Store all valid feature vectors before sampling
    all_labels_str_raw = []  # Store corresponding string labels before sampling
    sequence_lengths_original = []  # Store original lengths before potential sampling

    # Track data cleaning details
    data_cleaning_info = {
        "summary": {
            "total_processed": 0,
            "total_removed": 0,
            "outliers_removed": 0,
            "short_sections_removed": 0,
            "inconsistent_removed": 0,
            "undersampled_removed": 0,  # Added for undersampling
        },
        "removal_details": [],
    }

    # --- Determine the set of valid states ---
    all_possible_labels = set()
    valid_tracks_count = 0
    for _, data in all_track_data_with_filenames:
        # Basic check: ensure data is a dictionary and has labels
        if (
            isinstance(data, dict)
            and "semantic_labels" in data
            and isinstance(data["semantic_labels"], list)
        ):
            all_possible_labels.update(data["semantic_labels"])
            valid_tracks_count += 1
        # No need to check section_features here, load_perfect_analyses already did

    if valid_tracks_count == 0:
        print("ERROR: No tracks with valid 'semantic_labels' found in the loaded data.")
        return None

    valid_states = sorted(
        [label for label in all_possible_labels if label not in LABELS_TO_IGNORE]
    )
    if not valid_states:
        print(
            "ERROR: No valid states found after filtering ignored labels. Check LABELS_TO_IGNORE and data."
        )
        return None

    label_to_int = {label: i for i, label in enumerate(valid_states)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    n_states = len(valid_states)
    print(f"HMM States ({n_states}): {valid_states}")
    print(f"Label to Int Map: {label_to_int}")

    # --- Extract Feature vectors and state sequence from loaded data ---
    skipped_sections_count = 0
    total_cleaned_sections = 0
    label_feature_values = defaultdict(lambda: defaultdict(list))  # For raw averages

    for i, (filename, track_data) in enumerate(all_track_data_with_filenames):
        print(
            f"\n===== Processing track {i+1}/{len(all_track_data_with_filenames)} ({filename}) ====="
        )

        # Directly get pre-calculated features and labels
        # Validation happened during load_perfect_analyses
        section_feature_dicts = track_data.get(
            "section_features"
        )  # Should be list of dicts
        section_labels = track_data.get("semantic_labels")  # Should be list

        if section_feature_dicts is None or section_labels is None:
            print(
                f" -> Skipping track {i+1}: Critical data 'section_features' or 'semantic_labels' missing (should have been caught earlier)."
            )
            continue

        data_cleaning_info["summary"]["total_processed"] += len(section_feature_dicts)
        original_section_count = len(section_feature_dicts)
        song_name = os.path.splitext(filename)[0]  # For reporting

        # --- Apply Data Cleaning (if enabled) ---
        current_features = section_feature_dicts
        current_labels = section_labels

        if data_processing_settings.get("enable_outlier_removal"):
            current_features, current_labels, details = remove_feature_outliers(
                current_features,
                current_labels,
                song_name,
                data_processing_settings.get("outlier_z_threshold", 3.0),
            )
            data_cleaning_info["summary"]["outliers_removed"] += len(details)
            data_cleaning_info["removal_details"].extend(details)

        if data_processing_settings.get("enable_short_section_removal"):
            # This cleaning step requires 'duration_bars' to be in the feature dicts
            current_features, current_labels, details = remove_short_sections(
                current_features,
                current_labels,
                song_name,
                data_processing_settings.get("short_section_min_bars", 2),
            )
            data_cleaning_info["summary"]["short_sections_removed"] += len(details)
            data_cleaning_info["removal_details"].extend(details)

        if data_processing_settings.get("enable_consistency_filtering"):
            # This cleaning step requires 'avg_rms' to be in the feature dicts
            current_features, current_labels, details = filter_consistency(
                current_features, current_labels, song_name
            )
            data_cleaning_info["summary"]["inconsistent_removed"] += len(details)
            data_cleaning_info["removal_details"].extend(details)

        # Update final lists after all cleaning steps
        section_feature_dicts_cleaned = current_features
        section_labels_cleaned = current_labels

        sections_removed = original_section_count - len(section_feature_dicts_cleaned)
        total_cleaned_sections += sections_removed
        data_cleaning_info["summary"]["total_removed"] += sections_removed
        if sections_removed > 0:
            print(
                f" -> Data cleaning removed {sections_removed} of {original_section_count} sections."
            )

        if not section_feature_dicts_cleaned:
            print(f" -> Skipping track {i+1}: No sections remain after data cleaning.")
            continue

        # --- Process remaining sections ---
        track_feature_vectors = []
        track_state_labels = []

        for idx, feature_dict in enumerate(section_feature_dicts_cleaned):
            label = section_labels_cleaned[idx]

            if label in LABELS_TO_IGNORE or label not in label_to_int:
                if (
                    label not in LABELS_TO_IGNORE
                ):  # Only count skips if it wasn't explicitly ignored
                    skipped_sections_count += 1
                continue  # Skip ignored or invalid labels

            # Ensure feature_dict is actually a dictionary before proceeding
            if not isinstance(feature_dict, dict):
                print(
                    f" -> WARNING: Skipping section {idx} (Label: {label}): Item is not a dictionary. Value: {feature_dict}"
                )
                skipped_sections_count += 1
                continue

            # Collect raw feature values before potential skip
            # Calculate raw averages based on *cleaned* data.
            for key in feature_keys:  # Only consider selected features
                value = feature_dict.get(key)
                if (
                    value is not None
                    and isinstance(value, (int, float, np.number))
                    and np.isfinite(value)
                ):
                    label_feature_values[label][key].append(float(value))

            # Extract the required features for this section directly from the dict
            current_feature_vector = []
            valid_section = True
            missing_keys_in_section = []
            non_finite_keys_in_section = []

            for key in feature_keys:
                value = feature_dict.get(key)  # Use .get() for safety
                if value is None:
                    missing_keys_in_section.append(key)
                    valid_section = False
                elif not np.isfinite(float(value)):  # Check for NaN/Inf
                    non_finite_keys_in_section.append(key)
                    valid_section = False
                else:
                    current_feature_vector.append(float(value))  # Append if valid

            if not valid_section:
                reason = []
                if missing_keys_in_section:
                    reason.append(f"missing keys: {missing_keys_in_section}")
                if non_finite_keys_in_section:
                    reason.append(f"non-finite keys: {non_finite_keys_in_section}")
                print(
                    f" -> WARNING: Skipping section {idx} (Label: {label}): Invalid/Missing features ({', '.join(reason)})."
                )
                skipped_sections_count += 1
                continue  # Skip this section

            # This check should theoretically pass if the loop above succeeded
            if len(current_feature_vector) != n_features:
                print(
                    f" -> ERROR: Section {idx} (Label: {label}) yielded wrong number of features ({len(current_feature_vector)} vs {n_features}) after checks. Skipping."
                )
                skipped_sections_count += 1
                continue  # Skip section with wrong feature count

            track_feature_vectors.append(current_feature_vector)
            track_state_labels.append(label)

        # Add sequence to global lists if valid sections were found
        if track_feature_vectors:
            all_feature_vectors_raw.extend(track_feature_vectors)
            all_labels_str_raw.extend(track_state_labels)
            sequence_lengths_original.append(len(track_feature_vectors))
            print(f" -> Added sequence of length {len(track_feature_vectors)}")
        else:
            print(
                f" -> Track {i+1} resulted in empty feature sequence after filtering/cleaning."
            )

    print("\n===== Finished Processing All Tracks =====")
    if skipped_sections_count > 0:
        print(
            f"NOTE: Skipped {skipped_sections_count} sections due to ignored labels or missing/invalid feature values in loaded data."
        )
    if total_cleaned_sections > 0:
        print(
            f"NOTE: Data cleaning removed {total_cleaned_sections} sections in total."
        )

    if not all_feature_vectors_raw:
        print(
            "\n *** ERROR: No valid multi-feature observations generated from loaded data. Cannot train. ***"
        )
        print(
            " *** Check loaded .joblib files, selected features, and cleaning settings. ***"
        )
        return None

    # --- Calculate Raw Averages (Based on data *after* cleaning, *before* sampling) ---
    raw_averages = {}
    print(
        "\nCalculating Raw Feature Averages (based on data POST cleaning, PRE sampling):"
    )
    for label in sorted(
        label_feature_values.keys()
    ):  # Sort labels for consistent output
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
                raw_averages[label][feature] = None  # Indicate no data
                print(f"    - {feature}: N/A (no valid sections)")

    # --- Optional Undersampling ---
    X_final = np.array(all_feature_vectors_raw)
    labels_final = all_labels_str_raw
    sequence_lengths_final = sequence_lengths_original  # Default to original lengths

    if data_processing_settings.get("enable_undersampling", False):
        print("\n--- Applying Undersampling (Majority Class) ---")
        print(
            "WARNING: Undersampling is applied globally and breaks the original sequence structure."
        )
        print(
            "         HMM will be trained treating all sections as one long sequence."
        )

        label_counts = Counter(labels_final)
        print(f"Label counts BEFORE undersampling: {dict(label_counts)}")

        if len(label_counts) < 2:
            print(" -> Skipping undersampling: Less than 2 classes present.")
        else:
            # Find majority class and target count (second most frequent)
            most_common = label_counts.most_common()
            majority_label, majority_count = most_common[0]
            target_count = most_common[1][1]  # Count of the second most frequent

            print(f" -> Majority class: '{majority_label}' ({majority_count} sections)")
            print(f" -> Target count (based on 2nd most frequent): {target_count}")

            if majority_count > target_count:
                # Get indices for each class
                indices_by_label = defaultdict(list)
                for idx, label in enumerate(labels_final):
                    indices_by_label[label].append(idx)

                # Sample majority class indices
                majority_indices_sampled = random.sample(
                    indices_by_label[majority_label], target_count
                )
                print(
                    f" -> Undersampling '{majority_label}' from {majority_count} down to {target_count}."
                )
                data_cleaning_info["summary"]["undersampled_removed"] = (
                    majority_count - target_count
                )

                # Combine sampled majority with all minority indices
                final_indices_to_keep = []
                for label, indices in indices_by_label.items():
                    if label == majority_label:
                        final_indices_to_keep.extend(majority_indices_sampled)
                    else:
                        final_indices_to_keep.extend(indices)

                # Shuffle the final indices to mix classes
                random.shuffle(final_indices_to_keep)

                # Select the corresponding features and labels
                X_final = X_final[final_indices_to_keep]
                labels_final = [labels_final[i] for i in final_indices_to_keep]
                sequence_lengths_final = None  # Indicate sequence structure is lost

                print(
                    f" -> Data after undersampling: {X_final.shape[0]} sections total."
                )
                print(
                    f" -> Label counts AFTER undersampling: {dict(Counter(labels_final))}"
                )
                print(f" -> Sequence lengths set to None due to undersampling.")

            else:
                print(
                    f" -> Skipping undersampling: Majority class count ({majority_count}) not greater than target ({target_count})."
                )
    else:
        print("\nUndersampling disabled.")

    # --- Convert final labels to integers ---
    # This needs to happen AFTER potential sampling
    all_states_int_sequence = []
    valid_obs_count = 0
    for label in labels_final:
        if label in label_to_int:
            all_states_int_sequence.append(label_to_int[label])
            valid_obs_count += 1
        else:
            # This shouldn't happen if filtering worked, but good to check
            print(
                f"WARNING: Label '{label}' not in label_to_int map during final conversion."
            )

    if valid_obs_count != X_final.shape[0]:
        print(
            f"ERROR: Mismatch between final feature count ({X_final.shape[0]}) and valid integer state count ({valid_obs_count})."
        )
        return None

    # --- Convert to NumPy array and Scale ---
    X = X_final  # Use the potentially undersampled data
    if X.ndim != 2 or X.shape[1] != n_features:
        print(
            f"ERROR: Final feature array X has unexpected shape {X.shape}. Expected (*, {n_features})."
        )
        return None
    if X.shape[0] == 0:
        print("Error: Final feature array X is unexpectedly empty.")
        return None

    print(
        f"\nTotal observations for scaling/training: {X.shape[0]}, Features per observation: {X.shape[1]}"
    )

    # --- Scale Features ---
    print(f"\nScaling {n_features} features using StandardScaler...")
    scaler_multi = StandardScaler()
    try:
        # Fit ONLY on the final (potentially undersampled) data
        X_scaled = scaler_multi.fit_transform(X)
        print(" -> Features scaled.")
        # Round means and scales for cleaner printing
        means_rounded = np.round(scaler_multi.mean_, 4)
        scales_rounded = np.round(scaler_multi.scale_, 4)
        print(f" -> Scaler Means (based on final data): {means_rounded}")
        print(f" -> Scaler Scales (StdDev) (based on final data): {scales_rounded}")
    except ValueError as scale_err:
        print(f"ERROR during scaling: {scale_err}")
        print(
            " -> This might happen if a feature has zero variance after cleaning/sampling."
        )
        # Add check for zero variance columns
        zero_var_cols = np.where(np.std(X, axis=0) < 1e-9)[0]
        if len(zero_var_cols) > 0:
            print(
                " -> Features with near-zero variance detected at indices:",
                zero_var_cols,
            )
            print(
                " -> Corresponding feature keys:",
                [feature_keys[i] for i in zero_var_cols],
            )
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
    # Check length against the integer sequence, not the original string sequence
    if len(all_states_int_sequence) != X_scaled.shape[0]:
        print(
            f"!!! ERROR: Mismatch between length of final features ({X_scaled.shape[0]}) and final integer state sequence ({len(all_states_int_sequence)})!"
        )
        return None

    print("\nMulti-feature data preparation complete.")
    return (
        X_scaled,
        sequence_lengths_final,  # Return potentially modified lengths
        int_to_label,
        label_to_int,
        scaler_multi,
        data_cleaning_info,
        raw_averages,  # Return raw averages calculated BEFORE sampling
    )


def train_gmmhmm(
    X_scaled,
    feature_weights_map,
    feature_keys,
    lengths,  # Can be None if undersampling was used
    n_states,
    n_mix,
    min_covar,
    transmat_prior=None,
):
    """
    Applies weights and trains a GMMHMM model.
    Handles case where lengths is None (due to undersampling).

    Args:
        X_scaled (np.ndarray): The scaled multi-feature data (n_observations, n_features).
        feature_weights_map (dict): Dictionary mapping feature keys to their weights.
        feature_keys (list): Ordered list of feature keys corresponding to columns in X_scaled.
        lengths (list or None): List of sequence lengths, or None if undersampled.
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
        print(
            f"ERROR: Mismatch between X_scaled features ({n_features}) and feature_keys ({len(feature_keys)})"
        )
        return None, None

    # --- Apply Weighting ---
    print(f"Applying feature weights: {feature_weights_map}")
    # Create weight vector in the correct order
    try:
        # Ensure weight map uses keys consistent with selected feature_keys
        feature_weight_vector = np.array(
            [feature_weights_map.get(k, 1.0) for k in feature_keys]
        )
    except Exception as e:
        print(f"ERROR creating weight vector: {e}")
        return None, None

    if len(feature_weight_vector) != n_features:
        print(
            f"ERROR: Weight vector length ({len(feature_weight_vector)}) != number of features ({n_features})"
        )
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
        return None, feature_weight_vector  # Return weights even if training fails

    print(f"Training data shape: {X_weighted.shape}")
    if lengths:
        print(f"Number of sequences: {len(lengths)}")
        if sum(lengths) != X_weighted.shape[0]:
            print(
                f"ERROR: Sum of lengths ({sum(lengths)}) != total observations ({X_weighted.shape[0]})"
            )
            return None, feature_weight_vector
    else:
        print(
            "WARNING: No sequence lengths provided (likely due to undersampling). Training as a single sequence."
        )
        # If lengths is None, hmmlearn treats X as a single sequence
        # No need to explicitly set lengths=[X_weighted.shape[0]]

    print(f"Using Covariance Type: {DEFAULT_HMM_COVARIANCE_TYPE}")
    print(
        f"Max Iterations: {DEFAULT_HMM_N_ITER}, Tolerance: {DEFAULT_HMM_TOL}, Min Covar: {min_covar}"
    )

    # --- Initialize GMMHMM ---
    model = hmm.GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type=DEFAULT_HMM_COVARIANCE_TYPE,
        min_covar=min_covar,
        random_state=DEFAULT_HMM_RANDOM_STATE,
        n_iter=DEFAULT_HMM_N_ITER,
        tol=DEFAULT_HMM_TOL,
        verbose=True,  # Print convergence info
        # Init params: s=startprob, t=transmat, m=means, c=covars, w=weights
        # Don't initialize 't' if prior is provided
        init_params="scmw" if transmat_prior is not None else "stmcw",
        # Train all parameters
        params="stmcw",
    )

    # --- Set Initial Transition Matrix (if provided) ---
    if transmat_prior is not None:
        print("Initializing transition matrix with provided prior.")
        if transmat_prior.shape == (n_states, n_states):
            # Ensure rows sum to 1 (handle potential floating point issues)
            row_sums = np.sum(
                transmat_prior, axis=1, keepdims=True
            )  # Keep dims for broadcasting
            # Avoid division by zero for rows that sum to zero (shouldn't happen with valid prior)
            # Add small epsilon where sum is zero before division
            row_sums[row_sums < 1e-9] = (
                1.0  # Avoid division by zero, but normalization might slightly change values
            )
            valid_transmat = transmat_prior / row_sums

            # Check again after division, just in case
            if not np.allclose(np.sum(valid_transmat, axis=1), 1.0):
                print(
                    "WARNING: Transition matrix prior rows do not sum perfectly to 1 after normalization attempt. Using as is."
                )
                # Optional: Implement more robust normalization if needed
            model.transmat_ = valid_transmat
            # Setting the prior attribute directly might also be needed depending on hmmlearn version behavior
            # model.transmat_prior = valid_transmat
        else:
            print(
                f"WARNING: Provided transition matrix prior shape {transmat_prior.shape} doesn't match n_states {n_states}. Using default init."
            )
            model.init_params = "stmcw"  # Revert to full default init

    # --- Train the Model ---
    print("Fitting GMMHMM model...")
    try:
        # Pass lengths=None if undersampling was done
        model.fit(X_weighted, lengths=lengths)
        if hasattr(model, "monitor_") and model.monitor_.converged:
            print(f"GMMHMM Training Converged in {model.monitor_.n_iter} iterations.")
        elif hasattr(model, "monitor_"):
            print(
                f"WARNING: GMMHMM Training did NOT converge after {model.monitor_.n_iter} iterations."
            )
        else:
            print("WARNING: Model monitor not available to check convergence.")

        # Check score after fitting
        final_score = model.score(X_weighted, lengths=lengths)
        print(f"Final Log Likelihood: {final_score}")
        if not np.isfinite(final_score):
            print(
                "ERROR: Model fitting resulted in non-finite log likelihood. Training likely failed."
            )
            return None, feature_weight_vector

        return model, feature_weight_vector
    except ValueError as ve:
        print(f"\n *** GMMHMM Training Failed: ValueError encountered. ***")
        print(f" -> Error Message: {ve}")
        print(" -> This often indicates issues like:")
        print(
            "    - Insufficient data for some states/mixtures (check sequence lengths and cleaning)."
        )
        print("    - Features might be constant or highly correlated after weighting.")
        print(
            f"    - Degenerate covariance matrices (try increasing min_covar: {min_covar})."
        )
        print(
            f"    - Consider reducing n_mix ({n_mix}) or getting more/different data."
        )
        traceback.print_exc()
        return None, feature_weight_vector
    except Exception as e:
        print(f"\n *** GMMHMM Training Failed: An unexpected error occurred. ***")
        print(f" -> Error Message: {e}")
        traceback.print_exc()
        return None, feature_weight_vector
