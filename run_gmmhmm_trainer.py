# =============================================================================
# FILE: run_gmmhmm_trainer.py
# Purpose: Main script to run the GMMHMM training process by orchestrating
#          calls to the modules within GMMHMM_modules.
# ADDED: Option to disable use of transition prior.
# MODIFIED: To load data from a pre-generated lean cache file
#           (hmm_lean_training_data_cache.joblib) if it exists,
#           otherwise falls back to loading full analysis files.
# ADDED: Handling for undersampling option from GUI.
# =============================================================================

import os
import sys
import joblib
import numpy as np
import traceback

# --- Add project root to sys.path if necessary ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    print(f"[DEBUG] Adding project root to sys.path: {PROJECT_ROOT}")
    sys.path.append(PROJECT_ROOT)

# --- Import necessary components from GMMHMM_modules ---
try:
    print("[DEBUG] Attempting to import from GMMHMM_modules...")
    from GMMHMM_modules.config import (
        PERFECT_FOLDER_PATH,
        HMM_OUTPUT_FOLDER,
        TRANSITION_MATRIX_PATH,
        LABELS_TO_IGNORE,
        DEFAULT_HMM_COVARIANCE_TYPE,
    )
    from GMMHMM_modules.data_utils import load_perfect_analyses  # Keep for fallback
    from GMMHMM_modules.gui import (
        show_feature_selection_dialog,
        show_model_analysis_dialog,
    )
    from GMMHMM_modules.hmm_model import prepare_multi_feature_data, train_gmmhmm
    from GMMHMM_modules.analysis import analyze_trained_gmmhmm, export_analysis_results

    print("[DEBUG] Successfully imported modules.")
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules from GMMHMM_modules: {e}")
    print(
        "Ensure 'run_gmmhmm_trainer.py' is in the project root and 'GMMHMM_modules' exists."
    )
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Define Lean Cache File Path ---
LEAN_CACHE_FILENAME = "hmm_lean_training_data_cache.joblib"
LEAN_CACHE_FILE_PATH = os.path.join(HMM_OUTPUT_FOLDER, LEAN_CACHE_FILENAME)


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting GMMHMM Multi-Feature Training Process ---")
    print("[DEBUG] Entered main execution block.")

    # 1. Show Configuration Dialog
    print("[DEBUG] Preparing to call show_feature_selection_dialog()...")
    try:
        user_config = show_feature_selection_dialog()
        print(
            f"[DEBUG] show_feature_selection_dialog() returned. Result: {'Config received' if user_config else 'None (Cancelled?)'}"
        )
    except Exception as gui_error:
        print(f"\n!!! ERROR: An exception occurred during the GUI dialog call !!!")
        print(f" -> Error Type: {type(gui_error).__name__}")
        print(f" -> Error Message: {gui_error}")
        traceback.print_exc()
        print("!!! Exiting due to GUI error. !!!")
        sys.exit(1)

    if user_config is None:
        print(
            "User cancelled configuration or dialog failed to return config. Training aborted."
        )
        sys.exit(0)

    # Extract configuration values from the returned dictionary
    print("[DEBUG] Processing configuration returned from dialog...")
    feature_keys = user_config["feature_keys"]
    feature_weights_map = user_config["feature_weights"]
    hmm_n_mixtures = user_config["hmm_n_mixtures"]
    hmm_min_covar = user_config["hmm_min_covar"]
    # <<< MODIFIED: Get data processing settings dict >>>
    data_processing_settings = user_config["data_processing_settings"]
    use_transition_prior = user_config["use_transition_prior"]
    n_features = len(feature_keys)

    # Construct base filename for output files based on config
    feature_str = f"{n_features}f"
    mixture_str = f"{hmm_n_mixtures}m"
    processing_flags = []
    if data_processing_settings.get("enable_outlier_removal"):
        processing_flags.append("out")
    if data_processing_settings.get("enable_short_section_removal"):
        processing_flags.append("sh")
    if data_processing_settings.get("enable_consistency_filtering"):
        processing_flags.append("con")
    # <<< ADDED: Undersampling flag to filename >>>
    if data_processing_settings.get("enable_undersampling"):
        processing_flags.append("under")
    processing_str = "_".join(processing_flags) if processing_flags else "noclean"
    prior_flag = "" if use_transition_prior else "_noprior"
    base_filename = f"gmmhmm_{feature_str}_{mixture_str}_{processing_str}{prior_flag}"
    model_save_path = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_model.joblib")
    aux_save_path = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_aux.joblib")

    print(f"\n--- Configuration Summary ---")
    print(f"Selected Features ({n_features}): {feature_keys}")
    print(f"Feature Weights: {feature_weights_map}")
    print(f"HMM Mixtures: {hmm_n_mixtures}")
    print(f"HMM Min Covariance: {hmm_min_covar}")
    print(f"Use Transition Prior: {use_transition_prior}")
    print(f"Data Processing Settings:")
    print(
        f"  - Outlier Removal: {'ENABLED (z=' + str(data_processing_settings.get('outlier_z_threshold', 'N/A')) + ')' if data_processing_settings.get('enable_outlier_removal') else 'DISABLED'}"
    )
    print(
        f"  - Short Section Removal: {'ENABLED (min_bars=' + str(data_processing_settings.get('short_section_min_bars', 'N/A')) + ')' if data_processing_settings.get('enable_short_section_removal') else 'DISABLED'}"
    )
    print(
        f"  - Consistency Filtering: {'ENABLED' if data_processing_settings.get('enable_consistency_filtering') else 'DISABLED'}"
    )
    # <<< ADDED: Undersampling status print >>>
    print(
        f"  - Undersampling: {'ENABLED' if data_processing_settings.get('enable_undersampling') else 'DISABLED'}"
    )
    print(f"Model Output Base Name: {base_filename}")
    print(f"Model Save Path: {model_save_path}")
    print(f"Aux Data Save Path: {aux_save_path}")
    print("---------------------------\n")

    # 2. Load Data (Try Lean Cache First)
    all_track_data_for_prep = None  # Initialize variable

    print(f"[DEBUG] Checking for lean cache file: {LEAN_CACHE_FILE_PATH}")
    if os.path.exists(LEAN_CACHE_FILE_PATH):
        print(f"--- Found lean cache file. Loading data from cache... ---")
        try:
            # Load the dictionary: {filename: {'semantic_labels': [...], 'section_features': [...]}}
            lean_cache_data = joblib.load(LEAN_CACHE_FILE_PATH)
            if not isinstance(lean_cache_data, dict) or not lean_cache_data:
                print(
                    "ERROR: Lean cache file is empty or not a valid dictionary. Falling back to full load."
                )
                all_track_data_for_prep = None  # Ensure fallback happens
            else:
                # Convert the loaded cache dict into the list of tuples format expected by prepare_multi_feature_data
                # [(filename, data_dict), ...] where data_dict contains only labels and features
                all_track_data_for_prep = list(lean_cache_data.items())
                print(
                    f"--- Successfully loaded lean data for {len(all_track_data_for_prep)} tracks from cache. ---"
                )

        except Exception as e:
            print(f"ERROR: Failed to load or process lean cache file: {e}")
            traceback.print_exc()
            print("--- Falling back to loading full analysis files. ---")
            all_track_data_for_prep = None  # Ensure fallback happens
    else:
        print(
            f"--- Lean cache file not found. Proceeding to load full analysis files from 'Perfect' folder... ---"
        )
        all_track_data_for_prep = None  # Ensure fallback happens

    # Fallback to loading full files if cache wasn't loaded successfully
    if all_track_data_for_prep is None:
        print("[DEBUG] Loading full 'Perfect' analyses (fallback)...")
        # This is the original loading method
        all_data_with_filenames = load_perfect_analyses(PERFECT_FOLDER_PATH)
        if not all_data_with_filenames:
            print(
                "ERROR: No valid 'Perfect' track data found (fallback method). Cannot train HMM. Exiting."
            )
            sys.exit(1)
        print(f"[DEBUG] Loaded full data for {len(all_data_with_filenames)} tracks.")
        all_track_data_for_prep = all_data_with_filenames  # Use the fully loaded data

    # 3. Prepare Multi-Feature Data (using either cached or fully loaded data)
    print("[DEBUG] Preparing multi-feature data...")
    # prepare_multi_feature_data expects a list of tuples: [(filename, data_dict), ...]
    # The data_dict should contain 'semantic_labels' and 'section_features'
    # <<< MODIFIED: Pass data_processing_settings dict >>>
    prep_result = prepare_multi_feature_data(
        all_track_data_for_prep, feature_keys, data_processing_settings
    )

    if prep_result is None:
        print("ERROR: Multi-feature data preparation failed. Exiting.")
        sys.exit(1)
    print("[DEBUG] Data preparation successful.")

    (
        X_scaled,
        sequence_lengths,  # This might be None now
        int_to_label,
        label_to_int,
        scaler_multi,
        data_cleaning_info,  # This now includes undersampling info if done
        raw_averages,
    ) = prep_result

    n_unique_states = len(int_to_label)
    if n_unique_states <= 0:
        print("ERROR: No valid states found after data preparation. Exiting.")
        sys.exit(1)
    print(f"[DEBUG] Found {n_unique_states} unique states.")

    # 4. Load Optional Transition Matrix Prior (Conditional)
    transition_matrix_prior = None
    if use_transition_prior:
        print("[DEBUG] Checking for transition matrix prior (user enabled)...")
        if os.path.exists(TRANSITION_MATRIX_PATH):
            try:
                loaded_prior = np.load(TRANSITION_MATRIX_PATH)
                print(
                    f" -> Loaded transition matrix prior from {TRANSITION_MATRIX_PATH}"
                )
                if loaded_prior.shape == (n_unique_states, n_unique_states):
                    if np.allclose(np.sum(loaded_prior, axis=1), 1.0, atol=1e-6):
                        transition_matrix_prior = loaded_prior
                        print(" -> Prior matrix validated (shape and row sums).")
                    else:
                        print(
                            " -> WARNING: Loaded transition matrix prior rows do not sum to 1. Ignoring prior."
                        )
                else:
                    print(
                        f" -> WARNING: Loaded prior shape {loaded_prior.shape} != expected ({n_unique_states}, {n_unique_states}). Ignoring prior."
                    )
            except Exception as e:
                print(
                    f" -> WARNING: Error loading or validating transition matrix prior: {e}. Proceeding without prior."
                )
        else:
            print(
                f" -> Transition matrix prior file not found at {TRANSITION_MATRIX_PATH}. Proceeding without prior."
            )
    else:
        print(
            "[DEBUG] User disabled use of transition matrix prior. HMM will initialize transitions."
        )

    # 5. Train the GMMHMM
    print("[DEBUG] Starting GMMHMM training...")
    # <<< MODIFIED: Pass potentially None sequence_lengths >>>
    train_result = train_gmmhmm(
        X_scaled,
        feature_weights_map,
        feature_keys,
        sequence_lengths,  # Pass the potentially modified lengths
        n_unique_states,
        hmm_n_mixtures,
        hmm_min_covar,
        transmat_prior=transition_matrix_prior,
    )

    if train_result is None:
        print("\nERROR: GMMHMM training function returned None. Exiting.")
        sys.exit(1)

    trained_model, feature_weight_vector = train_result
    print("[DEBUG] GMMHMM training function returned a model.")

    # 6. Analyze, Save Model, and Export Results (if training succeeded)
    if trained_model:
        print("\n--- Post-Training Analysis and Saving ---")
        model_analysis_info = None
        try:
            print("[DEBUG] Analyzing trained model...")
            model_analysis_info = analyze_trained_gmmhmm(
                trained_model,
                int_to_label,
                scaler_multi,
                feature_keys,
                feature_weight_vector,
                raw_averages,
            )
            if model_analysis_info:
                print("[DEBUG] Displaying analysis dialog...")
                show_model_analysis_dialog(
                    model_analysis_info, data_cleaning_info, raw_averages
                )
                print("[DEBUG] Analysis dialog closed.")
            else:
                print("WARNING: Model analysis returned no info, skipping dialog.")
        except Exception as analyze_e:
            print(f"\nERROR during model analysis or display: {analyze_e}")
            traceback.print_exc()

        # --- Saving Model and Aux Data ---
        try:
            print("[DEBUG] Ensuring output directory exists...")
            os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True)
            print(f"Ensured output directory exists: {HMM_OUTPUT_FOLDER}")

            print(f"[DEBUG] Saving model to {model_save_path}...")
            joblib.dump(trained_model, model_save_path)
            print(f"Trained GMMHMM model saved to: {model_save_path}")

            aux_data = {
                "int_to_label": int_to_label,
                "label_to_int": label_to_int,
                "scaler_multi": scaler_multi,
                "feature_keys": feature_keys,
                "feature_weights_vector": feature_weight_vector,
                "feature_weights_map": feature_weights_map,
                "labels_ignored": LABELS_TO_IGNORE,
                "hmm_params": {
                    "type": "GMMHMM",
                    "n_components": trained_model.n_components,
                    "n_mixtures": trained_model.n_mix,
                    "covariance_type": trained_model.covariance_type,
                    "min_covar": hmm_min_covar,
                    "used_transition_prior": (transition_matrix_prior is not None),
                },
                # <<< MODIFIED: Save data_processing_settings >>>
                "data_processing_settings_used": data_processing_settings,
                "raw_averages": raw_averages,
            }
            print(f"[DEBUG] Saving auxiliary data to {aux_save_path}...")
            joblib.dump(aux_data, aux_save_path)
            print(f"Auxiliary data saved to: {aux_save_path}")

        except Exception as save_e:
            print(f"\n *** ERROR: Failed saving model or auxiliary data: {save_e} ***")
            traceback.print_exc()

        # --- Export Analysis to JSON ---
        if model_analysis_info:
            print("[DEBUG] Exporting analysis results to JSON...")
            model_params_export = {
                "n_mixtures": hmm_n_mixtures,
                "min_covar": hmm_min_covar,
                "covariance_type": DEFAULT_HMM_COVARIANCE_TYPE,
                "n_iter": (
                    trained_model.monitor_.n_iter
                    if hasattr(trained_model, "monitor_")
                    else "N/A"
                ),
                "converged": (
                    trained_model.monitor_.converged
                    if hasattr(trained_model, "monitor_")
                    else "N/A"
                ),
                "used_transition_prior": (transition_matrix_prior is not None),
            }
            export_path = export_analysis_results(
                model_analysis_info,
                data_cleaning_info,
                data_processing_settings,  # Pass updated dict
                feature_keys,
                feature_weights_map,
                model_params_export,
                HMM_OUTPUT_FOLDER,
                base_filename,
            )
            if export_path:
                print(f"\nAnalysis data exported to JSON: {export_path}")
            else:
                print("\nFailed to export analysis data to JSON.")
        else:
            print(
                "\nSkipping JSON export because model analysis failed or produced no results."
            )

    else:
        print("\n *** GMMHMM Training Failed. Model not saved or analyzed. ***")

    print("\n--- GMMHMM Multi-Feature Training Process Finished ---")
    sys.exit(0)
