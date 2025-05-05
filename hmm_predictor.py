# =============================================================================
# FILE: hmm_predictor.py
# Contains the HMMPredictor class for loading a trained HMM (Gaussian or GMMHMM)
# and predicting section labels for new tracks using multiple features.
# Uses a fixed color map for HMM prediction results.
# FIXED: Changed key for loading feature weights to 'feature_weights_vector'.
# MODIFIED: To load model/aux files based on paths set externally via select_hmm_model.
# ADDED: Debugging print statements to trace alignment issues.
# =============================================================================

import os
import joblib
import numpy as np
import traceback
from collections import Counter, defaultdict
from tkinter import messagebox  # For showing errors if model loading fails

# Attempt to import necessary components from other modules
try:
    # This function should return (list_of_dictionaries, list_of_labels)
    # The dictionaries MUST contain keys matching FEATURE_KEYS loaded from aux data.
    from audio_analysis_wrapper import extract_section_features

    print(
        "Successfully imported 'extract_section_features' from 'audio_analysis_wrapper'."
    )
except ImportError as e:
    print(f"ERROR importing required modules in hmm_predictor.py: {e}")
    messagebox.showerror(
        "Import Error", "HMMPredictor failed to import 'extract_section_features'."
    )

    # Define dummy function if necessary for structure testing
    def extract_section_features(track_data):
        print("WARNING: Using DUMMY extract_section_features in predictor!")
        labels = track_data.get(
            "semantic_labels", ["Intro", "Drop", "Outro", "Fill", "Drop"]
        )
        features = [
            {  # Provide dummy values for keys expected by default model
                "relative_rms": np.random.rand(),
                "low_energy_norm": np.random.rand(),
                "delta_rms": (np.random.rand() - 0.5) * 0.1,
                "label_proportion": np.random.rand(),
                "relative_position": i / len(labels),
                "position_context": 1.0 if label in ["Intro", "Outro"] else 0.0,
                "avg_rms": np.random.rand(),
                "centroid_std_dev_section": np.random.rand() * 100,
                "crest_factor": 1.0 + np.random.rand() * 2.0,
                "delta_centroid": (np.random.rand() - 0.5) * 100,
                "high_end_ratio": np.random.rand(),
                "low_end_ratio": np.random.rand(),
                "peak_rms": np.random.rand(),
                "rms_std_dev": np.random.rand() * 0.1,
                "rms_std_dev_section": np.random.rand() * 0.1,
                "rms_trend": (np.random.rand() - 0.5) * 0.05,
                "spectral_bandwidth_avg": np.random.rand() * 1000,
                "spectral_centroid_avg": np.random.rand() * 2000,
                "spectral_centroid_slope": (np.random.rand() - 0.5) * 200,
                "spectral_centroid_std_dev": np.random.rand() * 100,
                "spectral_contrast_avg": np.random.rand() * 10,
            }
            for i, label in enumerate(labels)
        ]
        return features, labels


# Define the fixed color map specifically for HMM results visualization
HMM_PURPLE_COLOR = "#8A2BE2"  # Define Purple locally for clarity
HMM_RESULT_COLOR_MAP = {
    "Intro": "#FF0000",  # Red
    "Outro": "#FF0000",  # Red
    "Body": "#014421",  # Dark Green
    "Build": "#FFA500",  # Orange
    "Drop": "#8B0000",  # Dark Red
    "Breakdown": "#ADD8E6",  # Light Blue
    "Fill": HMM_PURPLE_COLOR,  # Purple (May occur if original section was Fill)
    "Fade Out": HMM_PURPLE_COLOR,  # Purple (Shouldn't be predicted by HMM)
    "Unknown": HMM_PURPLE_COLOR,  # Purple (If prediction fails for a state)
    "Skipped": "#808080",  # Grey for sections skipped during prediction
}
HMM_FALLBACK_COLOR = HMM_PURPLE_COLOR  # Purple for any other unexpected labels


class HMMPredictor:
    """
    Handles loading a GMM-HMM model and predicting section labels for new tracks
    using multiple features, based on user-selected model files.
    """

    def __init__(self, hmm_output_folder):
        """
        Initializes the predictor. Stores the path to the HMM model directory.

        Args:
            hmm_output_folder (str): Path to the directory where HMM model
                                     and auxiliary files are stored.
        """
        self.hmm_output_folder = (
            hmm_output_folder  # Store for browsing in _select_hmm_model
        )
        self.model = None
        self.scaler = None  # Renamed from scaler_multi for consistency
        self.feature_keys = None
        self.int_to_label = None
        self.label_to_int = None  # Added for completeness, loaded from aux
        self.feature_weights_vector = None  # Renamed from feature_weights
        self.labels_ignored = None
        # --- Attributes to store selected paths ---
        self.model_path = None  # Will be set by main_app._select_hmm_model
        self.aux_path = None  # Will be set by main_app._select_hmm_model
        # --- End Attributes ---
        self.is_loaded = False  # Flag to track if the selected model is loaded
        print(f"HMMPredictor initialized. Models expected in: {self.hmm_output_folder}")

    def load_model(self):
        """
        Loads the GMM-HMM model and auxiliary data from the paths
        previously set by the user via main_app._select_hmm_model.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        # Reset previous model state
        self.model = None
        self.scaler = None
        self.feature_keys = None
        self.int_to_label = None
        self.label_to_int = None
        self.feature_weights_vector = None
        self.labels_ignored = None
        self.is_loaded = False  # Reset loaded flag

        # --- Check if model and aux paths have been set by the selection process ---
        if not self.model_path or not self.aux_path:
            print(
                "Error: HMM model or auxiliary file path not selected via 'Select HMM Model'."
            )
            # Don't show messagebox here, let the calling function (_trigger_hmm_prediction) handle UI feedback
            return False
        # --- End Check ---

        print(f"Attempting to load HMM model from: {self.model_path}")
        print(f"Attempting to load HMM aux data from: {self.aux_path}")

        try:
            # Load the model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
            print(f" -> Model loaded. Type: {type(self.model)}")

            # Load the auxiliary data
            if not os.path.exists(self.aux_path):
                raise FileNotFoundError(f"Auxiliary file not found: {self.aux_path}")
            aux_data = joblib.load(self.aux_path)
            print(f" -> Aux data loaded. Keys: {list(aux_data.keys())}")

            # Store required auxiliary data as attributes
            self.scaler = aux_data.get("scaler_multi")  # Load the multi-feature scaler
            self.feature_keys = aux_data.get(
                "feature_keys"
            )  # Load the list of feature keys
            self.int_to_label = aux_data.get("int_to_label")
            self.label_to_int = aux_data.get("label_to_int")  # Load label->int map too
            self.feature_weights_vector = aux_data.get(
                "feature_weights_vector"
            )  # Load the weights vector
            self.labels_ignored = aux_data.get(
                "labels_ignored", []
            )  # Load ignored labels

            # --- Validation ---
            missing_aux = []
            if self.model is None:
                missing_aux.append("Model Object")
            if self.scaler is None:
                missing_aux.append("Scaler (scaler_multi)")
            if self.feature_keys is None:
                missing_aux.append("Feature Keys (feature_keys)")
            if self.int_to_label is None:
                missing_aux.append("Int->Label Map (int_to_label)")
            if self.label_to_int is None:
                missing_aux.append(
                    "Label->Int Map (label_to_int)"
                )  # Check label_to_int
            if self.feature_weights_vector is None:
                missing_aux.append("Feature Weights (feature_weights_vector)")

            if missing_aux:
                error_msg = f"Failed to load essential data from aux file.\nMissing: {', '.join(missing_aux)}\nFile: {self.aux_path}"
                messagebox.showerror("HMM Load Error", error_msg)
                print(f"Error: {error_msg}")
                # Clear paths if load failed critically, so user must re-select
                self.model_path = None
                self.aux_path = None
                return False

            # Further validation
            if not hasattr(self.scaler, "transform"):
                raise TypeError(
                    "Loaded 'scaler_multi' object does not have a 'transform' method."
                )
            if not isinstance(self.feature_weights_vector, np.ndarray):
                raise TypeError("Loaded 'feature_weights_vector' is not a NumPy array.")
            if len(self.feature_keys) != len(self.feature_weights_vector):
                raise ValueError(
                    f"Mismatch between feature_keys ({len(self.feature_keys)}) and feature_weights_vector ({len(self.feature_weights_vector)})."
                )

            # Check model's expected feature count if possible
            model_n_features = -1
            if hasattr(self.model, "n_features_in_"):
                model_n_features = self.model.n_features_in_
            elif hasattr(self.model, "n_features"):
                model_n_features = self.model.n_features
            if model_n_features != -1 and model_n_features != len(self.feature_keys):
                raise ValueError(
                    f"Mismatch between model's expected features ({model_n_features}) and loaded feature_keys ({len(self.feature_keys)})."
                )

            print(
                "DEBUG HMMPredictor: HMM model and multi-feature aux data loaded successfully."
            )
            print(f" -> Loaded Feature Keys: {self.feature_keys}")
            print(f" -> Loaded Feature Weights: {self.feature_weights_vector}")
            print(f" -> Loaded Scaler Means: {self.scaler.mean_}")
            print(f" -> Loaded Scaler Scales: {self.scaler.scale_}")
            print(f" -> Loaded Label Map: {self.int_to_label}")
            print(f" -> Loaded Ignored Labels: {self.labels_ignored}")

            self.is_loaded = True  # Set flag indicating successful load
            return True

        except FileNotFoundError as fnf_error:
            messagebox.showerror(
                "HMM Load Error", f"File not found during loading:\n{fnf_error}"
            )
            print(f"Error loading HMM files: {fnf_error}")
            self.model_path = None  # Clear paths on error
            self.aux_path = None
            self.is_loaded = False
            return False
        except Exception as e:
            messagebox.showerror(
                "HMM Load Error",
                f"An unexpected error occurred loading HMM files:\n{e}",
            )
            print(f"Error loading HMM files: {e}")
            traceback.print_exc()
            self.model_path = None  # Clear paths on error
            self.aux_path = None
            self.is_loaded = False
            return False

    def predict(self, track_data):
        """
        Runs HMM prediction on the provided track data using the loaded model
        and auxiliary data (scaler, feature keys, weights).

        Args:
            track_data (dict): The dictionary containing analysis results for the track.
                               Must contain 'section_features' (list of dicts) and
                               'section_starts'.

        Returns:
            tuple: (hmm_section_starts, hmm_semantic_labels, hmm_label_colors, posteriors)
                   on success, or None on failure.
                   Returns ([], [], [], None) if no valid sections found.
        """
        # Use self.is_loaded flag which is set by load_model()
        if not self.is_loaded:
            print("ERROR HMMPredictor: Model not loaded successfully. Cannot predict.")
            # Messagebox might be redundant if load_model already showed one
            # messagebox.showerror("HMM Error", "HMM Model not loaded. Cannot predict.")
            return None

        # Double-check essential components just in case
        if (
            self.model is None
            or self.scaler is None
            or self.feature_keys is None
            or self.feature_weights_vector is None
            or self.int_to_label is None
        ):
            messagebox.showerror(
                "Prediction Error",
                "Internal Error: HMM components missing despite load success flag.",
            )
            print("ERROR: Predict called but essential model components are None.")
            return None

        print("\n--- Running HMM Prediction ---")
        n_features = len(self.feature_keys)
        print(f"DEBUG HMM: Using {n_features} features: {self.feature_keys}")

        try:
            # 1. Extract Features from Track Data
            section_feature_dicts = track_data.get("section_features", [])
            original_section_labels = track_data.get(
                "semantic_labels", []
            )  # Get original labels too
            original_section_starts = track_data.get(
                "section_starts", []
            )  # Get original starts
            trim_offset = track_data.get(
                "trim_offset_sec", "N/A"
            )  # Get trim offset for context

            # --- DEBUG PRINT 1: Input Verification ---
            print(f"DEBUG HMM: Trim Offset Received = {trim_offset}")
            print(
                f"DEBUG HMM: Original Starts Received (len {len(original_section_starts)}, first 5): {original_section_starts[:5]}"
            )
            print(
                f"DEBUG HMM: Original Labels Received (len {len(original_section_labels)}, first 5): {original_section_labels[:5]}"
            )
            # --- END DEBUG ---

            if not section_feature_dicts:
                print("Prediction Warning: No 'section_features' found in track data.")
                return [], [], [], None  # Return empty lists and None for posteriors

            if len(original_section_starts) != len(section_feature_dicts):
                print(
                    f"Warning HMMPredictor: Mismatch between section_starts ({len(original_section_starts)}) and section_features ({len(section_feature_dicts)})."
                )
                # Let's try to proceed but alignment might fail later.

            # 2. Prepare feature sequence, skipping ignored labels and invalid sections
            feature_vectors_for_hmm = []
            indices_of_kept_sections = []  # Track original indices of sections used

            for i, feature_dict in enumerate(section_feature_dicts):
                if not isinstance(feature_dict, dict):
                    continue  # Skip non-dicts

                # Get original label safely
                label = (
                    original_section_labels[i]
                    if i < len(original_section_labels)
                    else ""
                )

                # Skip sections with ignored labels
                if label in self.labels_ignored:
                    # print(f"DEBUG HMM Loop: Skipping Index {i}, Label '{label}' (Ignored)") # Optional
                    continue

                # Extract the required features in the correct order
                current_feature_vector = []
                valid_section = True
                for key in self.feature_keys:
                    value = feature_dict.get(key)
                    try:
                        if value is None or not np.isfinite(float(value)):
                            raise ValueError("None or non-finite")
                        current_feature_vector.append(float(value))
                    except (TypeError, ValueError):
                        # print(f"DEBUG HMM Loop: Skipping Index {i}, Label '{label}', Invalid feature '{key}': {value}") # Optional
                        valid_section = False
                        break  # Stop processing this section

                if valid_section and len(current_feature_vector) == n_features:
                    feature_vectors_for_hmm.append(current_feature_vector)
                    indices_of_kept_sections.append(i)
                # else: print(f"DEBUG HMM Loop: Section {i} skipped (valid={valid_section}, len={len(current_feature_vector)})") # Optional

            # --- DEBUG PRINT 2: Sections Kept ---
            print(
                f"DEBUG HMM: Indices Kept (first 10): {indices_of_kept_sections[:10]}"
            )
            print(
                f"DEBUG HMM: Total sections originally: {len(original_section_starts)}"
            )
            print(
                f"DEBUG HMM: Total sections kept for HMM: {len(indices_of_kept_sections)}"
            )
            if not feature_vectors_for_hmm:
                print("WARNING HMM: No valid feature vectors to process.")
                messagebox.showwarning(
                    "HMM Warning",
                    "No valid sections found for HMM prediction in this track (check features and ignored labels).",
                )
                return [], [], [], None  # Return empty if nothing to process
            # --- END DEBUG ---

            X_raw = np.array(feature_vectors_for_hmm)
            if X_raw.shape[1] != n_features:
                raise ValueError(
                    f"Prepared feature array shape mismatch: {X_raw.shape[1]} vs {n_features}"
                )

            # 3. Scale Features
            print("DEBUG HMM: Scaling features...")
            X_scaled = self.scaler.transform(X_raw)
            print(f" -> Scaled features shape: {X_scaled.shape}")

            # 4. Apply Weights
            print(f"DEBUG HMM: Applying feature weights...")
            if len(self.feature_weights_vector) != X_scaled.shape[1]:
                raise ValueError(
                    f"Weight vector length ({len(self.feature_weights_vector)}) != Scaled features ({X_scaled.shape[1]})"
                )
            X_predict = X_scaled * self.feature_weights_vector
            print(f" -> Final feature shape for prediction: {X_predict.shape}")

            # 5. Predict States
            print("DEBUG HMM: Running HMM model.predict()...")
            X_predict_contiguous = np.ascontiguousarray(
                X_predict
            )  # Ensure C-contiguous
            predicted_states_int = self.model.predict(X_predict_contiguous)

            # 5b. Get posterior probabilities
            posteriors = None  # Initialize to None
            try:
                # Use predict_proba if available (standard for sklearn HMMs)
                if hasattr(self.model, "predict_proba"):
                    posteriors = self.model.predict_proba(X_predict_contiguous)
                    print(
                        f"DEBUG HMM: Got posteriors via predict_proba, shape: {posteriors.shape}"
                    )
                # Fallback: Try _compute_log_likelihood (might be specific to hmmlearn)
                elif hasattr(self.model, "_compute_log_likelihood"):
                    log_probs = self.model._compute_log_likelihood(X_predict_contiguous)
                    print(
                        f"DEBUG HMM: Got log_probs via _compute_log_likelihood, shape: {log_probs.shape}"
                    )
                    # Convert log probs to normalized probabilities
                    probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
                    posteriors = probs / np.sum(probs, axis=1, keepdims=True)
                    print(
                        f"DEBUG HMM: Converted log_probs to posteriors, shape: {posteriors.shape}"
                    )
                else:
                    print(
                        "Warning HMM: Model has neither 'predict_proba' nor '_compute_log_likelihood'. Cannot get posteriors."
                    )

            except Exception as e:
                print(f"Warning HMM: Could not extract posterior probabilities: {e}")
                # traceback.print_exc() # Keep this commented unless needed
                posteriors = None

            # --- DEBUG PRINT 3: Prediction Output ---
            print(f"DEBUG HMM: Predicted states length: {len(predicted_states_int)}")
            if posteriors is not None:
                print(f"DEBUG HMM: Posteriors shape: {posteriors.shape}")
                # Verify length match immediately after prediction
                if posteriors.shape[0] != len(predicted_states_int):
                    print(
                        f"CRITICAL WARNING HMM: Posteriors rows ({posteriors.shape[0]}) != Predicted states len ({len(predicted_states_int)}) !!"
                    )
            else:
                print(f"DEBUG HMM: Posteriors are None")
            # --- END DEBUG ---

            # 6. Map Predicted State Indices to Labels
            print("DEBUG HMM: Mapping predicted states to labels...")
            predicted_labels = [
                self.int_to_label.get(state_int, "Unknown")
                for state_int in predicted_states_int
            ]
            print(f"DEBUG HMM: Predicted label counts: {Counter(predicted_labels)}")

            # 7. Generate Colors for Predicted Labels
            print("DEBUG HMM: Generating colors...")
            predicted_colors = [
                HMM_RESULT_COLOR_MAP.get(label, HMM_FALLBACK_COLOR)
                for label in predicted_labels
            ]

            # 8. Align predictions back to original section starts
            print("DEBUG HMM: Aligning predictions to original section starts...")
            hmm_section_starts = []  # Initialize
            alignment_fallback_used = False  # Flag to track if fallback occurs

            if not original_section_starts:
                print("ERROR HMM: Original 'section_starts' not found. Cannot align.")
                hmm_section_starts = [0.0] * len(
                    predicted_labels
                )  # Placeholder, likely wrong
                alignment_fallback_used = True
            else:
                try:
                    # --- DEBUG PRINT 4: Alignment Check ---
                    if indices_of_kept_sections:
                        first_kept_index = indices_of_kept_sections[0]
                        print(f"DEBUG HMM: First kept index: {first_kept_index}")
                        if first_kept_index < len(original_section_starts):
                            corresponding_original_start = original_section_starts[
                                first_kept_index
                            ]
                            print(
                                f"DEBUG HMM: Corresponding original start for first HMM state: {corresponding_original_start}"
                            )
                        else:
                            print(
                                f"ERROR HMM: First kept index {first_kept_index} out of bounds for original_section_starts (len {len(original_section_starts)})"
                            )
                            # This case should trigger the IndexError below
                    else:
                        print(
                            f"DEBUG HMM: No sections were kept, alignment skipped (hmm_section_starts will be empty)."
                        )
                    # --- END DEBUG ---

                    # THE CORE ALIGNMENT STEP:
                    hmm_section_starts = [
                        original_section_starts[i] for i in indices_of_kept_sections
                    ]

                except IndexError as e:
                    print(
                        f"ERROR HMM: IndexError during alignment list comprehension! {e}"
                    )
                    print(
                        f"DEBUG HMM: Applying Fallback alignment: Using first {len(predicted_labels)} original starts."
                    )
                    hmm_section_starts = original_section_starts[
                        : len(predicted_labels)
                    ]
                    alignment_fallback_used = True  # Mark fallback as used

            # --- DEBUG PRINT 5: Post-Alignment Starts ---
            print(
                f"DEBUG HMM: Final HMM Starts before final length check (len {len(hmm_section_starts)}, first 5): {hmm_section_starts[:5]}"
            )
            # --- END DEBUG ---

            # Final length check after alignment
            # Determine the reliable length based on the prediction output
            prediction_len = len(predicted_states_int)

            # Check if all output lists match the prediction length
            starts_len = len(hmm_section_starts)
            labels_len = len(predicted_labels)
            colors_len = len(predicted_colors)
            posteriors_rows = (
                posteriors.shape[0] if posteriors is not None else prediction_len
            )  # Assume match if None

            if starts_len != prediction_len:
                print(
                    f"ERROR HMM: Length Mismatch! Starts ({starts_len}) vs Predicted States ({prediction_len}). Fallback used: {alignment_fallback_used}"
                )
                # If starts don't match prediction length, something is wrong with alignment.
                # It's safer to truncate everything to the SHORTEST of the available lists.
                min_len = min(starts_len, prediction_len, labels_len, colors_len)
                if posteriors is not None:
                    min_len = min(min_len, posteriors_rows)

                print(
                    f"DEBUG HMM: Trimming ALL outputs to minimum detected length: {min_len}"
                )
                hmm_section_starts = hmm_section_starts[:min_len]
                predicted_labels = predicted_labels[:min_len]
                predicted_colors = predicted_colors[:min_len]
                if posteriors is not None and posteriors.shape[0] > min_len:
                    posteriors = posteriors[:min_len, :]
                # Update lengths after potential trimming
                prediction_len = min_len

            # Check posteriors length separately ONLY IF starts matched prediction length initially
            elif posteriors is not None and posteriors_rows != prediction_len:
                print(
                    f"ERROR HMM: Posteriors rows ({posteriors_rows}) != Predicted States ({prediction_len}). Trimming posteriors."
                )
                # Trim posteriors to match the prediction length
                posteriors = posteriors[:prediction_len, :]

            # Assign potentially trimmed lists
            hmm_semantic_labels = predicted_labels
            hmm_label_colors = predicted_colors

            # --- DEBUG PRINT 6: Final Check Before Return ---
            final_len_starts = len(hmm_section_starts)
            final_len_labels = len(hmm_semantic_labels)
            final_len_colors = len(hmm_label_colors)
            final_post_rows = posteriors.shape[0] if posteriors is not None else -1
            print(
                f"DEBUG HMM: BEFORE RETURN - Lengths - Starts:{final_len_starts}, Labels:{final_len_labels}, Colors:{final_len_colors}, PosteriorRows:{final_post_rows}"
            )
            if final_len_starts > 0:
                print(
                    f"DEBUG HMM: BEFORE RETURN - First Final Start: {hmm_section_starts[0]}"
                )
            else:
                print(f"DEBUG HMM: BEFORE RETURN - No sections being returned.")
            # --- END DEBUG ---

            # Ensure all returned lists have the same final length
            if not (
                final_len_starts
                == final_len_labels
                == final_len_colors
                == (posteriors.shape[0] if posteriors is not None else final_len_starts)
            ):
                print(
                    "CRITICAL ERROR HMM: Final output lists have inconsistent lengths before return!"
                )
                # Depending on severity, might want to return None or raise an error
                # For now, return potentially inconsistent lists, but log the critical error.

            return hmm_section_starts, hmm_semantic_labels, hmm_label_colors, posteriors

        except Exception as e:
            messagebox.showerror(
                "HMM Prediction Error",
                f"An unexpected error occurred during HMM prediction:\n{e}",
            )
            print(f"ERROR during HMM prediction: {e}")
            traceback.print_exc()
            return None  # Indicate failure


    def calculate_feature_importance(self, track_data, hmm_semantic_labels=None, hmm_section_starts=None):
        """
        Calculates feature importance metrics for HMM states and sections.
        Uses the model's covariance matrices, means, and posteriors to provide:
        1. Feature importance per state (from GMM components)
        2. Feature contributions to state assignments for each section
        3. Mahalanobis distances showing section-state matching

        Args:
            track_data (dict): The track data dictionary containing analysis results
            hmm_semantic_labels (list, optional): List of predicted semantic labels
            hmm_section_starts (list, optional): List of predicted section starts

        Returns:
            dict: Dictionary with feature importance metrics
        """
        if not self.is_loaded or self.model is None:
            print("ERROR: HMM model not loaded. Cannot calculate feature importance.")
            return None

        # Initialize the results dictionary
        importance_data = {
            'feature_names': self.feature_keys.copy(),  # Store feature names
            'feature_weights': self.feature_weights_vector.copy(),  # Store feature weights
            'state_to_label_map': self.int_to_label.copy(),  # Store state-label mapping
            'label_to_state_map': self.label_to_int.copy() if self.label_to_int else {},
            'section_feature_contribution': [],  # Will store feature contributions for each section
            'section_mahalanobis_distances': [],  # Will store mahalanobis distances per section to each state
            'state_feature_importance': [],  # Will store feature importance per state
            'state_means': [],  # Will store mean vectors for each state
            'state_variances': [],  # Will store variance diagonals for each state
        }

        try:
            # Get model parameters (means, covariances) for each state
            # Different attribute names based on model type (GaussianHMM vs GMMHMM)
            n_states = self.model.n_components
            n_features = len(self.feature_keys)
            
            print(f"DEBUG: Model type: {type(self.model).__name__}")
            print(f"DEBUG: Model n_components: {n_states}")
            print(f"DEBUG: Feature keys count: {n_features}")

            # Print model attribute shapes to help with debugging
            if hasattr(self.model, 'means_'):
                means_shape = np.array(self.model.means_).shape
                print(f"DEBUG: Model means_ shape: {means_shape}")
            if hasattr(self.model, 'covars_'):
                covars_shape = np.array(self.model.covars_).shape
                print(f"DEBUG: Model covars_ shape: {covars_shape}")
            
            # Check if it's a GMMHMM (has GMM emission distributions) or GaussianHMM
            is_gmmhmm = hasattr(self.model, 'means_') and isinstance(self.model.means_, list)
            
            # Extract means and covariances based on model type
            if is_gmmhmm:
                # For GMMHMM, we have multiple components per state
                # We'll use the means and covars of the most important component
                # (highest weight) for each state
                for state in range(n_states):
                    if hasattr(self.model, 'weights_') and len(self.model.weights_) > state:
                        weights = self.model.weights_[state]
                        max_weight_idx = np.argmax(weights)
                        state_means = self.model.means_[state][max_weight_idx]
                        state_covars = self.model.covars_[state][max_weight_idx]
                    else:
                        # Fallback if weights not available
                        state_means = self.model.means_[state][0]
                        state_covars = self.model.covars_[state][0]
                    
                    # Ensure means is a 1D array of correct length
                    if isinstance(state_means, np.ndarray):
                        state_means = np.array(state_means).flatten()
                        if len(state_means) != n_features:
                            print(f"WARNING: State {state} means has {len(state_means)} features but expected {n_features}")
                            if len(state_means) > n_features:
                                state_means = state_means[:n_features]
                            else:
                                state_means = np.pad(state_means, (0, n_features - len(state_means)), 'constant')
                    else:
                        print(f"WARNING: State {state} means is not a numpy array")
                        state_means = np.zeros(n_features)
                    
                    # Store the mean vector
                    importance_data['state_means'].append(state_means)
                    
                    # Calculate feature importance using the variance
                    # For full covariance matrices, use the diagonal elements
                    if isinstance(state_covars, np.ndarray):
                        if state_covars.ndim == 2:  # Full covariance matrix
                            variances = np.diag(state_covars)
                        elif state_covars.ndim == 1:  # Diagonal covariance
                            variances = state_covars
                        else:  # 3D array or other unexpected shape
                            print(f"WARNING: State {state} has unexpected covariance shape: {state_covars.shape}")
                            # Try to extract diagonal elements if possible
                            try:
                                # If it's a 3D array with shape like (1, n, n), extract the diagonals
                                if state_covars.ndim == 3 and state_covars.shape[0] == 1:
                                    variances = np.diag(state_covars[0])
                                else:
                                    # Otherwise, flatten and use first n_features elements
                                    flat_covars = state_covars.flatten()
                                    variances = flat_covars[:n_features]
                            except Exception as ex:
                                print(f"Error extracting variances: {ex}")
                                variances = np.ones(n_features)  # Fallback
                    else:
                        print(f"WARNING: State {state} covars is not a numpy array")
                        variances = np.ones(n_features)
                    
                    # Ensure variances has the right length
                    if len(variances) != n_features:
                        print(f"WARNING: State {state} variances has {len(variances)} features but expected {n_features}")
                        if len(variances) > n_features:
                            variances = variances[:n_features]
                        else:
                            variances = np.pad(variances, (0, n_features - len(variances)), 'constant', constant_values=1.0)
                    
                    # Store variance for later use
                    importance_data['state_variances'].append(variances)
                    
                    # Calculate the normalized feature importance for this state
                    # Lower variance = higher importance
                    if np.any(variances > 0):
                        importance = 1.0 / (variances + 1e-10)  # Avoid division by zero
                        importance = importance / np.sum(importance)  # Normalize to sum to 1.0
                    else:
                        importance = np.ones_like(variances) / len(variances)
                    
                    importance_data['state_feature_importance'].append(importance)
            else:
                # For GaussianHMM with single Gaussian per state
                if hasattr(self.model, 'means_') and hasattr(self.model, 'covars_'):
                    # Handle different possible structures of GaussianHMM
                    for state in range(n_states):
                        # Extract and normalize means
                        if self.model.means_.ndim == 2:  # Standard shape (n_states, n_features)
                            state_means = self.model.means_[state]
                        elif self.model.means_.ndim == 3:  # Unusual shape
                            state_means = self.model.means_[state].flatten()
                        else:
                            state_means = np.array(self.model.means_[state])
                        
                        # Ensure means is correct length
                        state_means = np.array(state_means).flatten()
                        if len(state_means) != n_features:
                            if len(state_means) > n_features:
                                state_means = state_means[:n_features]
                            else:
                                state_means = np.pad(state_means, (0, n_features - len(state_means)), 'constant')
                        
                        importance_data['state_means'].append(state_means)
                        
                        # Extract variances from covariance matrix
                        if self.model.covars_.ndim == 2:  # Diagonal covariance (n_states, n_features)
                            variances = self.model.covars_[state]
                        elif self.model.covars_.ndim == 3:  # Full covariance (n_states, n_features, n_features)
                            # Extract diagonal elements for feature importance
                            try:
                                variances = np.diag(self.model.covars_[state])
                            except Exception:
                                print(f"WARNING: Couldn't extract diagonal from covars for state {state}")
                                # If covars is 3D with first dimension of 1, it might be (1, n, n)
                                if self.model.covars_[state].shape[0] == 1:
                                    variances = np.diag(self.model.covars_[state][0])
                                else:
                                    # Just use the first n_features elements
                                    flat_covars = self.model.covars_[state].flatten()
                                    variances = flat_covars[:n_features]
                        else:  # Other structure
                            print(f"WARNING: Unexpected covars shape for state {state}: {self.model.covars_[state].shape}")
                            try:
                                # Attempt to flatten and use first n_features elements
                                flat_covars = np.array(self.model.covars_[state]).flatten()
                                variances = flat_covars[:n_features]
                            except Exception:
                                print(f"WARNING: Using identity covariance for state {state}")
                                variances = np.ones(n_features)
                        
                        # Ensure variances has correct length
                        variances = np.array(variances).flatten()
                        if len(variances) != n_features:
                            if len(variances) > n_features:
                                variances = variances[:n_features]
                            else:
                                variances = np.pad(variances, (0, n_features - len(variances)), 'constant', constant_values=1.0)
                        
                        importance_data['state_variances'].append(variances)
                        
                        # Calculate normalized feature importance
                        if np.any(variances > 0):
                            importance = 1.0 / (variances + 1e-10)
                            importance = importance / np.sum(importance)
                        else:
                            importance = np.ones_like(variances) / len(variances)
                        
                        importance_data['state_feature_importance'].append(importance)
                else:
                    print("Warning: HMM model doesn't have expected means_ or covars_ attributes")
                    return None

            # Get section features for comparison with state means
            if track_data is None or not hasattr(self, "feature_keys"):
                print(
                    "Warning: Cannot calculate section-based metrics (missing track_data or feature_keys)"
                )
                return importance_data  # Return what we have so far

            # Use provided parameters or get from track_data
            if hmm_semantic_labels is None or hmm_section_starts is None:
                hmm_semantic_labels = track_data.get("hmm_semantic_labels", [])
                hmm_section_starts = track_data.get("hmm_section_starts", [])

            if not hmm_semantic_labels or not hmm_section_starts:
                print("Warning: No HMM prediction results available for section metrics")
                return importance_data  # Return what we have so far

            # Get posteriors if they exist
            posteriors = track_data.get("hmm_posteriors")
            if posteriors is not None and posteriors.shape[0] != len(hmm_semantic_labels):
                print(
                    f"Warning: Posteriors shape mismatch: {posteriors.shape[0]} vs {len(hmm_semantic_labels)} sections"
                )
                posteriors = None

            # Extract features for each section using the same method as in predict()
            section_features = track_data.get("section_features", [])
            indices_of_kept_sections = track_data.get("hmm_kept_indices", [])

            # If kept_indices is not available, try to reconstruct it
            if not indices_of_kept_sections and section_features:
                # This is a best-effort reconstruction that might not match original
                indices_of_kept_sections = list(range(len(section_features)))
                # Filter out ignored labels if original labels are available
                if "semantic_labels" in track_data and self.labels_ignored:
                    indices_of_kept_sections = [
                        i
                        for i in indices_of_kept_sections
                        if i < len(track_data["semantic_labels"])
                        and track_data["semantic_labels"][i] not in self.labels_ignored
                    ]

            # Make sure we have at least some sections to analyze
            if not section_features or not indices_of_kept_sections:
                print("Warning: No section features available for importance calculation")
                return importance_data

            # Calculate section feature contribution and mahalanobis distance for each section
            for i, idx in enumerate(indices_of_kept_sections):
                if i >= len(hmm_semantic_labels) or idx >= len(section_features):
                    continue  # Skip if indices are out of bounds

                feature_dict = section_features[idx]

                # Extract feature vector in the correct order
                feature_vector = []
                valid_section = True
                for key in self.feature_keys:
                    value = feature_dict.get(key)
                    try:
                        if value is None or not np.isfinite(float(value)):
                            valid_section = False
                            break
                        feature_vector.append(float(value))
                    except (TypeError, ValueError):
                        valid_section = False
                        break

                if not valid_section or len(feature_vector) != len(self.feature_keys):
                    # Skip invalid sections
                    importance_data["section_feature_contribution"].append(None)
                    importance_data["section_mahalanobis_distances"].append(None)
                    continue

                # Create feature vector array and apply scaling/weighting as in predict()
                feature_vector = np.array(feature_vector)
                feature_vector_scaled = self.scaler.transform(
                    feature_vector.reshape(1, -1)
                ).flatten()
                feature_vector_weighted = (
                    feature_vector_scaled * self.feature_weights_vector
                )

                # Get the corresponding state for this section
                label = hmm_semantic_labels[i]
                state = (
                    self.label_to_int.get(label, -1)
                    if hasattr(self, "label_to_int") and self.label_to_int
                    else -1
                )

                # If we can't map the label to a state, try reverse lookup from int_to_label
                if state == -1 and hasattr(self, "int_to_label"):
                    for state_idx, state_label in self.int_to_label.items():
                        if state_label == label:
                            state = state_idx
                            break

                if state == -1 or state >= n_states:
                    # Cannot determine the state, use fallback
                    print(f"Warning: Cannot map label '{label}' to a valid state")
                    importance_data["section_feature_contribution"].append(None)
                    importance_data["section_mahalanobis_distances"].append(None)
                    continue

                # Calculate feature contributions for this section
                # Feature contribution = (feature_value - state_mean) * feature_weight * feature_importance
                state_means = importance_data["state_means"][state]
                state_importance = importance_data["state_feature_importance"][state]
                feature_contribs = (
                    feature_vector_weighted - state_means
                ) * state_importance
                importance_data["section_feature_contribution"].append(feature_contribs)

                # Calculate mahalanobis distances to each state
                mahalanobis_distances = []
                for s in range(n_states):
                    s_means = importance_data["state_means"][s]
                    s_variances = importance_data["state_variances"][s]

                    # Calculate mahalanobis distance using diagonal covariance (simplified)
                    delta = feature_vector_weighted - s_means
                    normalized_delta = delta / np.sqrt(s_variances + 1e-10)
                    mahalanobis_dist = np.sqrt(np.sum(normalized_delta**2))
                    mahalanobis_distances.append(mahalanobis_dist)

                importance_data["section_mahalanobis_distances"].append(
                    mahalanobis_distances
                )

            # Store indices of kept sections for reference
            importance_data["hmm_kept_indices"] = indices_of_kept_sections

            return importance_data

        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            import traceback

            traceback.print_exc()
            return None
