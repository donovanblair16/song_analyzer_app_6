# =============================================================================
# FILE: hmm_predictor.py
# Contains the HMMPredictor class for loading a trained HMM (Gaussian or GMMHMM)
# and predicting section labels for new tracks using multiple features.
# Uses a fixed color map for HMM prediction results.
# FIXED: Changed key for loading feature weights to 'feature_weights_vector'.
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
    from audio_analysis_wrapper import (
        # load_and_preprocess, # Not directly used by predictor
        # detect_sections, # Not directly used by predictor
        # analyze_chroma_and_clusters, # Not directly used by predictor
        # analyze_stereo_and_hpss, # Not directly used by predictor
        # results_on_failure, # Not directly used by predictor
        # calculate_bar_features, # Not directly used by predictor
        extract_section_features,  # Crucial for getting features
    )

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
        # Dummy must provide expected keys (adapt if needed based on loaded feature_keys)
        features = [
            {
                "avg_rms": np.random.rand(),
                "relative_rms": np.random.rand(),
                "relative_position": i / len(labels),
                "low_energy_norm": np.random.rand(),
                "rms_std_dev_section": np.random.rand() * 0.1,
                "centroid_std_dev_section": np.random.rand() * 100,
                "delta_rms": (np.random.rand() - 0.5) * 0.1,
                "delta_centroid": (np.random.rand() - 0.5) * 100,
                "rms_trend": (np.random.rand() - 0.5) * 0.05,
                "crest_factor": 1.0 + np.random.rand() * 2.0,
                "spectral_centroid_slope": (np.random.rand() - 0.5) * 200,
            }
            for i, _ in enumerate(labels)
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
}
HMM_FALLBACK_COLOR = HMM_PURPLE_COLOR  # Purple for any other unexpected labels


class HMMPredictor:
    """
    Handles loading the trained HMM model (Gaussian or GMM) and auxiliary data,
    and predicting section labels for new tracks using multiple features.
    """

    def __init__(self, model_path, aux_data_path):
        """
        Initializes the predictor.

        Args:
            model_path (str): Path to the saved HMM model (.joblib).
            aux_data_path (str): Path to the saved auxiliary data (map, scaler, keys, weights etc.) (.joblib).
        """
        self.model_path = model_path
        self.aux_data_path = aux_data_path
        self.model = None
        # Attributes to load from aux_data
        self.scaler_multi = None  # Scaler fitted on multiple features
        self.int_to_label = None  # Map from state index to label name
        self.feature_keys = None  # List of feature keys used during training (in order)
        self.feature_weights = (
            None  # Numpy array of weights applied <<< Will be loaded now
        )
        self.labels_ignored = None  # List of labels ignored during training
        self.is_loaded = False
        print(
            f"HMMPredictor initialized. Model path: {model_path}, Aux path: {aux_data_path}"
        )

    def load_model(self):
        """Loads the HMM model and auxiliary data from files."""
        if self.is_loaded:
            print("DEBUG HMMPredictor: Model and aux data already loaded.")
            return True

        print("DEBUG HMMPredictor: Attempting to load HMM model and aux data...")
        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"HMM model file not found: {self.model_path}")
            if not os.path.exists(self.aux_data_path):
                raise FileNotFoundError(
                    f"HMM aux data file not found: {self.aux_data_path}"
                )

            # Load the HMM model object (works for GaussianHMM or GMMHMM)
            self.model = joblib.load(self.model_path)
            print(f" -> Model loaded. Type: {type(self.model)}")

            # Load the auxiliary data dictionary
            aux_data = joblib.load(self.aux_data_path)
            print(f" -> Aux data loaded. Keys: {list(aux_data.keys())}")

            # Extract required components from aux_data, looking for multi-feature keys
            self.int_to_label = aux_data.get("int_to_label")
            self.scaler_multi = aux_data.get(
                "scaler_multi"
            )  # Load the multi-feature scaler
            self.feature_keys = aux_data.get(
                "feature_keys"
            )  # Load the list of feature keys
            # <<< FIX: Load using the correct key from the trainer script >>>
            self.feature_weights = aux_data.get("feature_weights_vector")
            self.labels_ignored = aux_data.get(
                "labels_ignored", []
            )  # Load ignored labels

            # Validate that all necessary components were loaded
            missing = []
            if self.model is None:
                missing.append("model")
            if self.scaler_multi is None:
                missing.append("scaler_multi")
            if self.int_to_label is None:
                missing.append("int_to_label map")
            if self.feature_keys is None:
                missing.append("feature_keys list")
            # <<< FIX: Check the correct variable name >>>
            if self.feature_weights is None:
                missing.append(
                    "feature_weights_vector array"
                )  # Updated error message slightly
            if missing:  # Check if the list is not empty
                raise ValueError(
                    f"Loaded HMM auxiliary data is incomplete ({', '.join(missing)} missing)."
                )

            # Validate scaler and weights
            if not hasattr(self.scaler_multi, "transform"):
                raise TypeError(
                    "Loaded 'scaler_multi' object does not have a 'transform' method."
                )
            # <<< FIX: Check the correct variable name >>>
            if not isinstance(self.feature_weights, np.ndarray):
                raise TypeError("Loaded 'feature_weights_vector' is not a NumPy array.")
            # <<< FIX: Check the correct variable name >>>
            if len(self.feature_keys) != len(self.feature_weights):
                raise ValueError(
                    f"Mismatch between feature_keys ({len(self.feature_keys)}) and feature_weights_vector ({len(self.feature_weights)})."
                )
            # Check model's expected feature count if possible (might not exist pre-predict)
            # Use n_features_in_ if available (hmmlearn >= 0.3.0)
            model_n_features = -1
            if hasattr(self.model, "n_features_in_"):
                model_n_features = self.model.n_features_in_
            elif hasattr(self.model, "n_features"):  # Older attribute name
                model_n_features = self.model.n_features

            if model_n_features != -1 and model_n_features != len(self.feature_keys):
                raise ValueError(
                    f"Mismatch between model's expected features ({model_n_features}) and loaded feature_keys ({len(self.feature_keys)})."
                )

            print(
                "DEBUG HMMPredictor: HMM model and multi-feature aux data loaded successfully."
            )
            # Print loaded parameters for verification
            print(f" -> Loaded Feature Keys: {self.feature_keys}")
            print(
                f" -> Loaded Feature Weights: {self.feature_weights}"
            )  # Uses the loaded attribute
            print(f" -> Loaded Scaler Means: {self.scaler_multi.mean_}")
            print(f" -> Loaded Scaler Scales: {self.scaler_multi.scale_}")
            print(f" -> Loaded Label Map: {self.int_to_label}")
            print(f" -> Loaded Ignored Labels: {self.labels_ignored}")

            self.is_loaded = True
            return True

        except FileNotFoundError as fnf_err:
            messagebox.showerror(
                "HMM Load Error",
                f"Could not find required HMM file(s):\n{fnf_err}\n\nPlease ensure the model was trained and files exist.",
            )
            print(f"ERROR: {fnf_err}")
            self.is_loaded = False
            return False
        except Exception as e:
            messagebox.showerror(
                "HMM Load Error", f"An error occurred while loading HMM files:\n{e}"
            )
            print(f"ERROR loading HMM files: {e}")
            traceback.print_exc()
            self.is_loaded = False
            return False

    def predict(self, track_data):
        """
        Runs HMM prediction on the provided track data using multiple features
        defined during training (e.g., RMS, position, low-end).

        Args:
            track_data (dict): The dictionary containing analysis results for the track.
                               Must contain data needed by extract_section_features.

        Returns:
            tuple: (hmm_section_starts, hmm_semantic_labels, hmm_label_colors) on success,
                   or None on failure. Returns ([], [], []) if no valid sections found.
        """
        if not self.is_loaded:
            print("ERROR HMMPredictor: Model not loaded. Cannot predict.")
            messagebox.showerror("HMM Error", "HMM Model not loaded. Cannot predict.")
            return None  # Indicate failure clearly

        print("\nDEBUG HMMPredictor: Running prediction...")
        if not self.feature_keys:  # Check if feature keys were loaded
            print("ERROR HMMPredictor: Feature keys not loaded. Cannot prepare data.")
            messagebox.showerror(
                "HMM Error", "Feature keys missing from loaded HMM data."
            )
            return None
        # <<< FIX: Check the correct variable name >>>
        if not isinstance(self.feature_weights, np.ndarray):
            print(
                "ERROR HMMPredictor: Feature weights not loaded correctly (not an array)."
            )
            messagebox.showerror(
                "HMM Error", "Feature weights missing or invalid in loaded HMM data."
            )
            return None

        n_features = len(self.feature_keys)
        print(
            f"DEBUG HMMPredictor: Expecting {n_features} features: {self.feature_keys}"
        )

        try:
            # 1. Extract Section Feature Dictionaries and Original Labels
            print("DEBUG HMMPredictor: Calling extract_section_features...")
            # extract_section_features should return list[dict], list[str]
            # The dicts must contain the keys listed in self.feature_keys
            section_feature_dicts, original_section_labels = extract_section_features(
                track_data
            )

            if not section_feature_dicts or not original_section_labels:
                print(
                    "Warning HMMPredictor: extract_section_features returned no data."
                )
                return [], [], []  # Return empty lists if no sections extracted

            # Also need original section start times for alignment later
            original_section_starts = track_data.get("section_starts", [])
            if len(original_section_starts) != len(section_feature_dicts):
                print(
                    f"Warning HMMPredictor: Mismatch between section_starts ({len(original_section_starts)}) and extracted features ({len(section_feature_dicts)}). Alignment might be imperfect."
                )
                # Proceed cautiously, alignment logic below handles this

            # 2. Prepare the multi-feature sequence for the HMM
            print(f"DEBUG HMMPredictor: Preparing {n_features}-feature sequence...")
            feature_vectors_for_hmm = []
            indices_of_kept_sections = (
                []
            )  # Keep track of which original sections we use

            for i, feature_dict in enumerate(section_feature_dicts):
                # Basic check if it's a dictionary
                if not isinstance(feature_dict, dict):
                    print(
                        f" -> Warning: Skipping section {i}, item from extract_section_features is not a dict."
                    )
                    continue

                # Get label safely, default to empty string if index out of bounds
                label = (
                    original_section_labels[i]
                    if i < len(original_section_labels)
                    else ""
                )

                # Skip sections with ignored labels (use loaded ignored labels)
                if label in self.labels_ignored:
                    continue

                # Extract the required features in the correct order
                current_feature_vector = []
                valid_section = True
                for key in self.feature_keys:  # Use loaded feature keys
                    value = feature_dict.get(key)
                    # Check if feature exists and is finite
                    # Use explicit float conversion to handle potential numpy types
                    try:
                        float_val = float(value)
                        if value is None or not np.isfinite(float_val):
                            raise ValueError("Value is None or non-finite")
                        current_feature_vector.append(float_val)
                    except (TypeError, ValueError):
                        print(
                            f" -> WARNING: Skipping section {i} (Label: {label}): Missing or non-finite value for feature '{key}'. Value: {value} (Type: {type(value)})"
                        )
                        valid_section = False
                        break  # Stop processing features for this section

                # If all features were valid, keep the vector and original index
                if valid_section:
                    if len(current_feature_vector) == n_features:
                        feature_vectors_for_hmm.append(current_feature_vector)
                        indices_of_kept_sections.append(i)
                    else:  # Should not happen if loop logic is correct
                        print(
                            f" -> WARNING: Skipping section {i} (Label: {label}): Incorrect feature vector length ({len(current_feature_vector)} vs {n_features})."
                        )

            # Check if any valid feature vectors were found
            if not feature_vectors_for_hmm:
                print(
                    "Warning HMMPredictor: No valid feature vectors found in track sections after filtering."
                )
                messagebox.showwarning(
                    "HMM Warning",
                    "No valid sections found for HMM prediction in this track.",
                )
                return [], [], []  # Return empty lists

            print(
                f"DEBUG HMMPredictor: Kept {len(feature_vectors_for_hmm)} sections for HMM input."
            )

            # Convert list of vectors to NumPy array
            X_raw = np.array(feature_vectors_for_hmm)
            # Ensure shape is (n_sections, n_features)
            if X_raw.ndim != 2 or X_raw.shape[1] != n_features:
                raise ValueError(
                    f"Prepared feature array has unexpected shape: {X_raw.shape}. Expected ({len(feature_vectors_for_hmm)}, {n_features})"
                )

            # 3. Scale Features using the loaded multi-feature scaler
            print("DEBUG HMMPredictor: Scaling features...")
            X_scaled = self.scaler_multi.transform(X_raw)
            print(f" -> Scaled features shape: {X_scaled.shape}")

            # 4. Apply Weights using loaded weights
            print(
                f"DEBUG HMMPredictor: Applying feature weights ({self.feature_weights})..."
            )
            # <<< FIX: Check the correct variable name >>>
            X_predict = X_scaled * self.feature_weights  # Element-wise multiplication
            print(f" -> Final feature shape for prediction: {X_predict.shape}")

            # 5. Predict States using the loaded HMM model
            print("DEBUG HMMPredictor: Running HMM model.predict()...")
            # model.predict expects (n_samples, n_features)
            predicted_states_int = self.model.predict(X_predict)
            print(f"DEBUG HMMPredictor: Predicted {len(predicted_states_int)} states.")

            # 6. Map Predicted State Indices to Labels
            print("DEBUG HMMPredictor: Mapping predicted states to labels...")
            predicted_labels = [
                self.int_to_label.get(state_int, "Unknown")  # Use loaded map
                for state_int in predicted_states_int
            ]
            print(
                f"DEBUG HMMPredictor: Predicted label counts: {Counter(predicted_labels)}"
            )

            # 7. Generate Colors for Predicted Labels
            print("DEBUG HMMPredictor: Generating colors...")
            predicted_colors = [
                HMM_RESULT_COLOR_MAP.get(label, HMM_FALLBACK_COLOR)
                for label in predicted_labels
            ]

            # 8. Align predictions back to original section starts
            # Use the indices of the sections we actually processed
            print(
                "DEBUG HMMPredictor: Aligning predictions to original section starts..."
            )
            if not original_section_starts:
                print(
                    "Warning HMMPredictor: Original 'section_starts' not found in track_data. Cannot return aligned starts."
                )
                hmm_section_starts = [0.0] * len(predicted_labels)  # Placeholder starts
            else:
                # Select the start times corresponding to the sections used for prediction
                try:
                    hmm_section_starts = [
                        original_section_starts[i] for i in indices_of_kept_sections
                    ]
                except IndexError:
                    print(
                        "ERROR HMMPredictor: Index error during alignment. Mismatch between original_section_starts and kept indices."
                    )
                    # Fallback or error handling needed
                    hmm_section_starts = original_section_starts[
                        : len(predicted_labels)
                    ]  # Basic fallback

            # Ensure lengths match after alignment (should unless error above)
            if len(hmm_section_starts) != len(predicted_labels):
                print(
                    f"ERROR HMMPredictor: Final length mismatch after alignment! Starts: {len(hmm_section_starts)}, Labels: {len(predicted_labels)}"
                )
                # Handle error: e.g., truncate to shortest length
                min_final_len = min(len(hmm_section_starts), len(predicted_labels))
                hmm_section_starts = hmm_section_starts[:min_final_len]
                predicted_labels = predicted_labels[:min_final_len]
                predicted_colors = predicted_colors[:min_final_len]

            hmm_semantic_labels = predicted_labels
            hmm_label_colors = predicted_colors

            print(
                f"DEBUG HMMPredictor: Prediction successful. Returning {len(hmm_semantic_labels)} sections."
            )
            return hmm_section_starts, hmm_semantic_labels, hmm_label_colors

        except Exception as e:
            # Catch-all for any unexpected errors during prediction
            messagebox.showerror(
                "HMM Prediction Error",
                f"An unexpected error occurred during HMM prediction:\n{e}",
            )
            print(f"ERROR during HMM prediction: {e}")
            traceback.print_exc()
            return None  # Indicate failure
