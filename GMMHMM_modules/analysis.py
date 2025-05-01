# =============================================================================
# FILE: analysis.py
# Purpose: Analyze trained GMMHMM models and export results, including
#          details for each mixture component.
# =============================================================================

import os
import numpy as np
import json
import datetime
import traceback


# Helper function for JSON serialization of NumPy types
def np_encoder(obj):
    """Custom JSON encoder for NumPy types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle potential NaN/Inf before converting to float
        if np.isnan(obj):
            return "NaN"  # Represent NaN as a string
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"  # Represent Inf as a string
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()  # Serialize dates/datetimes
    try:
        # Attempt default serialization for other types
        return json.JSONEncoder().encode(obj)
    except TypeError:
        # Fallback for types not handled above
        return str(obj)


def analyze_trained_gmmhmm(
    model, int_to_label, scaler, feature_keys, feature_weights_vector, raw_averages=None
):
    """
    Analyzes the trained GMMHMM model to describe state characteristics,
    including details for each mixture component within each state.

    Args:
        model (hmm.GMMHMM): The trained GMMHMM model.
        int_to_label (dict): Dictionary mapping state indices to labels.
        scaler (StandardScaler): The scaler fitted on the multi-feature data.
        feature_keys (list): List of feature names in the correct order.
        feature_weights_vector (np.ndarray): NumPy array of weights applied to features.
        raw_averages (dict, optional): Dictionary containing raw feature averages by label.

    Returns:
        dict: Dictionary containing model analysis information, including state
              summaries, mixture details (weights, means, std devs), raw averages,
              and transitions. Returns an empty dict if analysis fails significantly.
    """
    print("\n--- Analyzing Trained GMMHMM (with Mixture Details) ---")
    analysis_results = {"states": [], "transitions": None}

    if not model or not hasattr(model, "n_components"):
        print("ERROR: Invalid or untrained model provided for analysis.")
        return analysis_results

    n_states = model.n_components
    n_mix = model.n_mix

    # Safely get n_features
    try:
        if hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
        elif hasattr(model, "means_") and model.means_ is not None:
            n_features = model.means_.shape[-1]
        else:
            raise AttributeError("Cannot determine number of features from model.")
    except AttributeError as e:
        print(f"ERROR: {e}")
        return analysis_results  # Cannot proceed

    if n_features != len(feature_keys) or n_features != len(feature_weights_vector):
        print(
            f"ERROR: Mismatch in feature counts: Model({n_features}), Keys({len(feature_keys)}), Weights({len(feature_weights_vector)})."
        )
        return analysis_results

    print(f"Model: {n_states} states, {n_mix} mixtures, {n_features} features.")
    print(f"Features: {feature_keys}")
    print(f"Weights Applied: {np.round(feature_weights_vector, 4)}")

    # --- Analyze Learned Parameters (Means, Covariances, Weights) for each Mixture ---
    print("\nAnalyzing Learned State & Mixture Characteristics...")
    if (
        not all(hasattr(model, attr) for attr in ["means_", "weights_", "covars_"])
        or model.means_ is None
        or model.weights_ is None
        or model.covars_ is None
    ):
        print(
            "ERROR: Model missing 'means_', 'weights_', or 'covars_' attributes. Analysis skipped."
        )
        return analysis_results

    # Model parameters are in the scaled+weighted space
    scaled_weighted_means_all = model.means_  # Shape (n_states, n_mix, n_features)
    mixture_weights = model.weights_  # Shape (n_states, n_mix)
    scaled_weighted_covars_all = model.covars_  # Shape depends on covariance_type

    # Validate shapes
    if scaled_weighted_means_all.shape != (
        n_states,
        n_mix,
        n_features,
    ) or mixture_weights.shape != (n_states, n_mix):
        print("ERROR: Unexpected shapes for model means or weights.")
        return analysis_results

    # Handle different covariance types to get variances/std_devs
    scaled_weighted_vars_all = None
    if model.covariance_type == "diag":
        if scaled_weighted_covars_all.shape == (n_states, n_mix, n_features):
            scaled_weighted_vars_all = (
                scaled_weighted_covars_all  # Variances are stored directly
            )
        else:
            print("ERROR: Unexpected shape for diagonal covariances.")
            return analysis_results
    elif model.covariance_type == "full":
        if scaled_weighted_covars_all.shape == (
            n_states,
            n_mix,
            n_features,
            n_features,
        ):
            # Extract diagonal variances from full covariance matrices
            scaled_weighted_vars_all = np.array(
                [
                    [np.diag(scaled_weighted_covars_all[i, j]) for j in range(n_mix)]
                    for i in range(n_states)
                ]
            )  # Shape: (n_states, n_mix, n_features)
        else:
            print("ERROR: Unexpected shape for full covariances.")
            return analysis_results
    # Add handling for 'tied' or 'spherical' if needed, though 'diag' is common
    else:
        print(
            f"WARNING: Covariance type '{model.covariance_type}' variance extraction not fully implemented for analysis."
        )
        # Attempt to use covars directly if shape matches diag, otherwise skip std dev analysis
        if scaled_weighted_covars_all.shape == (n_states, n_mix, n_features):
            scaled_weighted_vars_all = scaled_weighted_covars_all
        else:
            scaled_weighted_vars_all = None  # Cannot determine variances

    # --- Process Each State ---
    for i in range(n_states):
        label = int_to_label.get(i, f"State {i}")
        print(f"\n===== State: '{label}' =====")

        state_info = {
            "name": label,
            "mixture_details": [],  # List to hold info for each mixture
            "raw_averages": {},
        }

        # Add raw averages if available
        if raw_averages and label in raw_averages:
            state_info["raw_averages"] = {
                k: v for k, v in raw_averages[label].items() if v is not None
            }

        # --- Process Each Mixture within the State ---
        for j in range(n_mix):
            print(f"--- Mixture {j+1} ---")
            mixture_info = {
                "mixture_index": j,
                "mixture_weight": None,
                "mean_features_unscaled": {},
                "std_dev_features_unscaled": {},
            }

            # 1. Mixture Weight
            mixture_weight = mixture_weights[i, j]
            mixture_info["mixture_weight"] = mixture_weight
            print(f"  Weight (Prior Probability): {mixture_weight:.4f}")

            # 2. Mixture Mean (Unscale and Unweight)
            scaled_weighted_mean_mix = scaled_weighted_means_all[
                i, j
            ]  # Shape (n_features,)

            # Reverse Weighting
            weights_safe = feature_weights_vector.copy()
            weights_safe[weights_safe == 0] = 1.0  # Avoid division by zero
            scaled_mean_mix = scaled_weighted_mean_mix / weights_safe
            if np.any(feature_weights_vector == 0):
                print(
                    "    Warning: Feature weight of 0 detected; unweighted mean value might be inaccurate."
                )

            # Reverse Scaling
            try:
                # Reshape for scaler (expects 2D array)
                unscaled_mean_mix = scaler.inverse_transform(
                    scaled_mean_mix.reshape(1, -1)
                )[0]
                print("  Mean Feature Values (Unscaled):")
                for k, key in enumerate(feature_keys):
                    value = unscaled_mean_mix[k]
                    print(f"    - {key}: {value:.4f}")
                    mixture_info["mean_features_unscaled"][key] = value
            except Exception as e:
                print(f"    ERROR unscaling mean for mixture {j}: {e}")
                mixture_info["mean_features_unscaled"] = {"error": str(e)}

            # 3. Mixture Standard Deviation (Unscale and Unweight)
            if scaled_weighted_vars_all is not None:
                scaled_weighted_var_mix = scaled_weighted_vars_all[
                    i, j
                ]  # Shape (n_features,)
                # Variance is scaled by weight^2
                weights_sq_safe = weights_safe**2
                scaled_var_mix = scaled_weighted_var_mix / weights_sq_safe

                # Unscaling variance: var_orig = var_scaled * (scaler_scale**2)
                scaler_scale_sq = scaler.scale_**2
                unscaled_var_mix = scaled_var_mix * scaler_scale_sq

                # Get standard deviation (sqrt of variance), handle potential negative variance from numerical issues
                unscaled_std_dev_mix = np.sqrt(
                    np.maximum(unscaled_var_mix, 0)
                )  # Ensure non-negative before sqrt

                print("  Feature Standard Deviations (Unscaled):")
                for k, key in enumerate(feature_keys):
                    value = unscaled_std_dev_mix[k]
                    print(f"    - {key}: {value:.4f}")
                    mixture_info["std_dev_features_unscaled"][key] = value
            else:
                print("  Feature Standard Deviations: Could not be determined.")
                mixture_info["std_dev_features_unscaled"] = {
                    "error": "Could not determine variances"
                }

            # Add mixture info to the state's list
            state_info["mixture_details"].append(mixture_info)

        # Add the completed state info to the main results
        analysis_results["states"].append(state_info)

    # --- Analyze Transitions (Same as before) ---
    print("\nAnalyzing Learned Transitions...")
    transition_info = None
    if hasattr(model, "transmat_") and model.transmat_ is not None:
        if model.transmat_.shape == (n_states, n_states):
            transition_info = {
                "state_names": [
                    int_to_label.get(i, f"State {i}") for i in range(n_states)
                ],
                "matrix": model.transmat_,  # Keep as numpy array for now, convert to list for JSON later
            }
            analysis_results["transitions"] = (
                transition_info  # Store raw matrix for now
            )
            # Print transitions
            header = f"{'From \\ To':<12}" + "".join(
                [f" | {name:<10}" for name in transition_info["state_names"]]
            )
            print(header)
            print("-" * len(header))
            for i in range(n_states):
                row_str = f"{transition_info['state_names'][i]:<12}"
                for j in range(n_states):
                    row_str += f" | {model.transmat_[i, j]:<10.3f}"
                print(row_str)
            print("-" * len(header))
        else:
            print(
                "WARNING: Transition matrix shape mismatch. Skipping transition analysis."
            )
    else:
        print("Transition matrix not found in model. Skipping transition analysis.")

    print("--- GMMHMM Analysis Complete ---")
    return analysis_results


def export_analysis_results(
    analysis_results,
    data_cleaning_info,
    cleaning_settings,
    feature_keys,
    feature_weights_map,
    model_params,
    output_dir,
    base_filename,
):
    """
    Exports the model analysis (including mixture details), cleaning info,
    and settings to a JSON file.

    Args:
        analysis_results (dict): Results from analyze_trained_gmmhmm.
        data_cleaning_info (dict): Dictionary containing data cleaning summary and details.
        cleaning_settings (dict): Dictionary with the cleaning settings used.
        feature_keys (list): List of feature names used.
        feature_weights_map (dict): Dictionary of feature weights used.
        model_params (dict): Dictionary containing HMM parameters (n_mix, min_covar, etc.).
        output_dir (str): Directory to save the JSON file.
        base_filename (str): Base name for the output file (e.g., "gmmhmm_10f_3m").

    Returns:
        str or None: Path to the exported JSON file, or None on failure.
    """
    print("\n--- Exporting Detailed Analysis Results ---")
    output_path = os.path.join(output_dir, f"{base_filename}_analysis.json")

    # Create the full export data structure
    export_data = {
        "export_timestamp": datetime.datetime.now().isoformat(),
        "model_base_name": base_filename,
        "configuration": {
            "feature_keys": feature_keys,
            "feature_weights": {
                k: float(v) for k, v in feature_weights_map.items()
            },  # Ensure float
            "model_parameters": model_params,
            "data_cleaning_settings": cleaning_settings,
        },
        "analysis": analysis_results,  # Contains states (with mixture_details) and transitions
        "data_cleaning_report": data_cleaning_info,
    }

    # Convert numpy arrays in transitions matrix to lists for JSON compatibility
    # (np_encoder handles arrays within mixture details now)
    if (
        export_data["analysis"]
        and "transitions" in export_data["analysis"]
        and export_data["analysis"]["transitions"] is not None
    ):
        if isinstance(export_data["analysis"]["transitions"]["matrix"], np.ndarray):
            export_data["analysis"]["transitions"]["matrix"] = export_data["analysis"][
                "transitions"
            ]["matrix"].tolist()

    # Save to file using the custom encoder
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        with open(output_path, "w") as f:
            # Use the custom encoder which handles numpy types within the nested structure
            json.dump(export_data, f, indent=2, default=np_encoder)
        print(f"Detailed analysis results successfully exported to: {output_path}")
        return output_path
    except TypeError as te:
        print(f"\nERROR: Failed to serialize detailed analysis results to JSON: {te}")
        print(" -> Check for non-standard data types in the results.")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"\nERROR: Failed to write detailed analysis results file: {e}")
        traceback.print_exc()
        return None
