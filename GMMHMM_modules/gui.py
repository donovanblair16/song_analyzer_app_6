# =============================================================================
# FILE: gui.py
# Purpose: Define Tkinter GUI dialogs for feature selection and model analysis.
# MODIFIED: Set default checkbox state based on a predefined list.
# =============================================================================

import tkinter as tk
from tkinter import ttk, messagebox, BooleanVar, StringVar, DoubleVar
import numpy as np  # Needed for ModelAnalysisDialog raw averages display
import traceback  # For debugging
import os  # Needed for checking transition prior path

# Import defaults and feature lists from config
from .config import (
    ALL_FEATURE_KEYS,
    FEATURE_DESCRIPTIONS,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_HMM_N_MIXTURES,
    DEFAULT_HMM_MIN_COVAR,
    DEFAULT_ENABLE_OUTLIER_REMOVAL,
    DEFAULT_OUTLIER_Z_THRESHOLD,
    DEFAULT_ENABLE_SHORT_SECTION_REMOVAL,
    DEFAULT_SHORT_SECTION_MIN_BARS,
    DEFAULT_ENABLE_CONSISTENCY_FILTERING,
    DEFAULT_USE_TRANSITION_PRIOR,
    TRANSITION_MATRIX_PATH,
)

# <<< Define the features that should be ON by default >>>
DEFAULT_FEATURES_ON = [
    "relative_rms",
    "position_context",
    "low_energy_norm",
    "centroid_std_dev_section",
    "delta_rms",
    "crest_factor",
    "spectral_centroid_slope",
]


# --- Feature Selection Dialog ---
class FeatureSelectionDialog(tk.Toplevel):
    """
    A Tkinter dialog for selecting features, weights, HMM parameters,
    data cleaning options, and transition prior usage.
    """

    def __init__(self, parent):
        print("[DEBUG GUI] FeatureSelectionDialog __init__ started.")  # DEBUG PRINT
        super().__init__(parent)
        self.title("GMMHMM Configuration")
        self.geometry("700x800")  # Adjusted size slightly for new option
        self.resizable(True, True)
        self.parent = parent

        # Result dictionary to store all selections
        self.result_config = None  # Will store dict on confirm

        # --- Tkinter Variables ---
        print("[DEBUG GUI] Initializing Tkinter variables...")  # DEBUG PRINT
        # Feature selection
        self.feature_vars = {}  # Checkbuttons {feature_key: BooleanVar}
        self.weight_vars = {}  # Entry variables {feature_key: StringVar}

        # Data cleaning
        self.enable_outlier_removal = BooleanVar(value=DEFAULT_ENABLE_OUTLIER_REMOVAL)
        self.outlier_z_threshold = StringVar(value=str(DEFAULT_OUTLIER_Z_THRESHOLD))
        self.enable_short_section_removal = BooleanVar(
            value=DEFAULT_ENABLE_SHORT_SECTION_REMOVAL
        )
        self.short_section_min_bars = StringVar(
            value=str(DEFAULT_SHORT_SECTION_MIN_BARS)
        )
        self.enable_consistency_filtering = BooleanVar(
            value=DEFAULT_ENABLE_CONSISTENCY_FILTERING
        )

        # Model parameters
        self.mixture_var = StringVar(value=str(DEFAULT_HMM_N_MIXTURES))
        self.min_covar_var = StringVar(value=str(DEFAULT_HMM_MIN_COVAR))
        self.use_transition_prior_var = BooleanVar(value=DEFAULT_USE_TRANSITION_PRIOR)

        print("[DEBUG GUI] Tkinter variables initialized.")  # DEBUG PRINT

        # --- Create Widgets ---
        print("[DEBUG GUI] Calling create_widgets()...")  # DEBUG PRINT
        self.create_widgets()
        print("[DEBUG GUI] create_widgets() finished.")  # DEBUG PRINT

        # --- Make Modal ---
        print(
            "[DEBUG GUI] Setting up modal behavior (transient, grab_set, protocol)..."
        )  # DEBUG PRINT
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        print("[DEBUG GUI] Modal behavior set.")  # DEBUG PRINT

        # Center the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x = (screen_width // 2) - (width // 2)
            y = (screen_height // 2) - (height // 2)
            self.geometry(f"{width}x{height}+{x}+{y}")
            print(f"[DEBUG GUI] Centered window at {x},{y}")  # DEBUG PRINT
        except tk.TclError as e:
            print(f"[DEBUG GUI] Warning: Could not center window automatically: {e}")

        self.lift()
        self.focus_force()

        print(
            "[DEBUG GUI] FeatureSelectionDialog __init__ finished. Waiting for wait_window()..."
        )  # DEBUG PRINT

    def create_widgets(self):
        """Creates and lays out the widgets in the dialog."""
        print("[DEBUG GUI] create_widgets: Setting up main frame...")  # DEBUG PRINT
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title and description
        ttk.Label(
            main_frame, text="Configure GMMHMM Training", font=("Arial", 14, "bold")
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 15))

        description = (
            "Select features, set weights, configure data cleaning, "
            "and adjust HMM parameters."
        )
        ttk.Label(main_frame, text=description, wraplength=650).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(0, 20)
        )

        # --- Feature selection frame with scrollbar ---
        print("[DEBUG GUI] create_widgets: Setting up feature frame...")  # DEBUG PRINT
        feature_outer_frame = ttk.LabelFrame(
            main_frame, text="Feature Selection & Weights", padding=10
        )
        feature_outer_frame.grid(
            row=2, column=0, columnspan=4, sticky="nsew", pady=(0, 15)
        )
        main_frame.rowconfigure(2, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Canvas for scrollable content
        canvas = tk.Canvas(feature_outer_frame)
        scrollbar = ttk.Scrollbar(
            feature_outer_frame, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header for feature selection columns
        ttk.Label(scrollable_frame, text="Use", font=("Arial", 10, "bold")).grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Label(scrollable_frame, text="Feature", font=("Arial", 10, "bold")).grid(
            row=0, column=1, padx=5, pady=5, sticky="w"
        )
        ttk.Label(scrollable_frame, text="Weight", font=("Arial", 10, "bold")).grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Label(
            scrollable_frame, text="Description", font=("Arial", 10, "bold")
        ).grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Create rows for each feature
        for i, feature in enumerate(ALL_FEATURE_KEYS):
            row = i + 1

            # <<< MODIFIED: Set default checkbox state >>>
            is_default_on = feature in DEFAULT_FEATURES_ON
            self.feature_vars[feature] = BooleanVar(value=is_default_on)
            # <<< END MODIFICATION >>>

            ttk.Checkbutton(scrollable_frame, variable=self.feature_vars[feature]).grid(
                row=row, column=0, padx=5, pady=2
            )

            # Feature name
            ttk.Label(scrollable_frame, text=feature).grid(
                row=row, column=1, padx=5, pady=2, sticky="w"
            )

            # Weight entry (uses default weight from config)
            self.weight_vars[feature] = StringVar(
                value=str(DEFAULT_FEATURE_WEIGHTS.get(feature, 1.0))
            )
            weight_entry = ttk.Entry(
                scrollable_frame, textvariable=self.weight_vars[feature], width=8
            )
            weight_entry.grid(row=row, column=2, padx=5, pady=2)

            # Description
            description = FEATURE_DESCRIPTIONS.get(feature, feature)
            ttk.Label(scrollable_frame, text=description).grid(
                row=row, column=3, padx=5, pady=2, sticky="w"
            )
        print("[DEBUG GUI] create_widgets: Feature rows created.")  # DEBUG PRINT

        # --- Quick selection buttons ---
        quick_frame = ttk.Frame(feature_outer_frame)
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(quick_frame, text="Select All", command=self.select_all).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(quick_frame, text="Unselect All", command=self.unselect_all).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            quick_frame, text="Original 7 Features", command=self.select_original_seven
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            quick_frame, text="Classic Features Only", command=self.select_classic
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            quick_frame,
            text="Recommended Defaults",
            command=self.select_recommended_defaults,
        ).pack(
            side=tk.LEFT, padx=5
        )  # Added Defaults button

        # --- Data Cleaning Options Frame ---
        print(
            "[DEBUG GUI] create_widgets: Setting up data cleaning frame..."
        )  # DEBUG PRINT
        data_cleaning_frame = ttk.LabelFrame(
            main_frame, text="Data Cleaning Options", padding=10
        )
        data_cleaning_frame.grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=(10, 15)
        )
        data_cleaning_frame.columnconfigure(3, weight=1)
        # Outlier Removal
        ttk.Checkbutton(
            data_cleaning_frame,
            text="Enable Outlier Removal",
            variable=self.enable_outlier_removal,
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(data_cleaning_frame, text="Z-score threshold:").grid(
            row=0, column=1, padx=5, pady=5, sticky="e"
        )
        ttk.Entry(
            data_cleaning_frame, textvariable=self.outlier_z_threshold, width=8
        ).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Label(
            data_cleaning_frame, text="(Removes sections with extreme feature values)"
        ).grid(row=0, column=3, sticky="w", padx=5, pady=5)
        # Short Section Removal
        ttk.Checkbutton(
            data_cleaning_frame,
            text="Remove Short Sections",
            variable=self.enable_short_section_removal,
        ).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(data_cleaning_frame, text="Minimum bars:").grid(
            row=1, column=1, padx=5, pady=5, sticky="e"
        )
        ttk.Entry(
            data_cleaning_frame, textvariable=self.short_section_min_bars, width=8
        ).grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ttk.Label(
            data_cleaning_frame,
            text="(Removes sections shorter than specified bar count)",
        ).grid(row=1, column=3, sticky="w", padx=5, pady=5)
        # Consistency Filtering
        ttk.Checkbutton(
            data_cleaning_frame,
            text="Enable Consistency Filtering",
            variable=self.enable_consistency_filtering,
        ).grid(row=2, column=0, sticky="w", padx=5, pady=5, columnspan=2)
        ttk.Label(
            data_cleaning_frame,
            text="(Removes sections with inconsistent features for their label, e.g., RMS)",
        ).grid(row=2, column=3, sticky="w", padx=5, pady=5)

        # --- Model parameters frame ---
        print(
            "[DEBUG GUI] create_widgets: Setting up model params frame..."
        )  # DEBUG PRINT
        params_frame = ttk.LabelFrame(main_frame, text="Model Parameters", padding=10)
        params_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 20))
        params_frame.columnconfigure(2, weight=1)

        # Number of mixtures
        ttk.Label(params_frame, text="Gaussian mixtures per state:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        mixture_entry = ttk.Entry(params_frame, textvariable=self.mixture_var, width=8)
        mixture_entry.grid(row=0, column=1, padx=5, pady=5)

        # Min covar setting
        ttk.Label(params_frame, text="Minimum covariance:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        min_covar_entry = ttk.Entry(
            params_frame, textvariable=self.min_covar_var, width=8
        )
        min_covar_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(params_frame, text="(Regularization)").grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        # Transition Prior Checkbox
        prior_cb = ttk.Checkbutton(
            params_frame,
            text="Use Transition Matrix Prior (if found)",
            variable=self.use_transition_prior_var,
        )
        prior_cb.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        if not os.path.exists(TRANSITION_MATRIX_PATH):
            prior_cb.config(state=tk.DISABLED)
            ttk.Label(params_frame, text="(Prior file not found)").grid(
                row=2, column=2, sticky="w", padx=5, pady=5
            )
        else:
            ttk.Label(
                params_frame, text="(Initialize transitions with pre-calculated matrix)"
            ).grid(row=2, column=2, sticky="w", padx=5, pady=5)

        # --- Final action buttons ---
        print("[DEBUG GUI] create_widgets: Setting up final buttons...")  # DEBUG PRINT
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=4, pady=10)

        ttk.Button(
            button_frame, text="Confirm and Train", command=self.on_confirm
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(
            side=tk.LEFT, padx=10
        )
        print("[DEBUG GUI] create_widgets: Finished setup.")  # DEBUG PRINT

    # --- Quick Selection Methods ---
    def select_all(self):
        """Select all features"""
        for var in self.feature_vars.values():
            var.set(True)

    def unselect_all(self):
        """Unselect all features"""
        for var in self.feature_vars.values():
            var.set(False)

    def select_original_seven(self):
        """Select the original 7 features"""
        original_seven = [
            "avg_rms",
            "relative_position",
            "low_energy_norm",
            "rms_std_dev_section",
            "centroid_std_dev_section",
            "delta_rms",
            "delta_centroid",
        ]
        for feature, var in self.feature_vars.items():
            var.set(feature in original_seven)

    def select_classic(self):
        """Select only classic features (not the new ones)"""
        classic_features = ["avg_rms", "relative_position", "low_energy_norm"]
        if "relative_rms" in ALL_FEATURE_KEYS:
            classic_features.append("relative_rms")
        if "position_context" in ALL_FEATURE_KEYS:
            classic_features.append("position_context")
        for feature, var in self.feature_vars.items():
            var.set(feature in classic_features)

    # <<< ADDED: Method to apply recommended defaults >>>
    def select_recommended_defaults(self):
        """Selects the predefined default features and resets weights."""
        for feature, var in self.feature_vars.items():
            var.set(feature in DEFAULT_FEATURES_ON)
        # Also reset weights to the current defaults from config
        for feature, weight_var in self.weight_vars.items():
            weight_var.set(str(DEFAULT_FEATURE_WEIGHTS.get(feature, 1.0)))

    # --- Validation and Confirmation ---
    # (Validation code remains the same)
    def validate_inputs(self):
        """Validate inputs before closing dialog. Returns config dict or None."""
        print("[DEBUG GUI] Validating inputs...")  # DEBUG PRINT
        config = {}

        # Validate feature selection and weights
        selected_features = [f for f in ALL_FEATURE_KEYS if self.feature_vars[f].get()]
        if not selected_features:
            messagebox.showerror(
                "Selection Error", "You must select at least one feature.", parent=self
            )
            print("[DEBUG GUI] Validation failed: No features selected.")  # DEBUG PRINT
            return None
        config["feature_keys"] = selected_features

        try:
            weights = {}
            for feature in selected_features:
                weight = float(self.weight_vars[feature].get())
                if weight <= 0:
                    messagebox.showerror(
                        "Input Error",
                        f"Weight for {feature} must be positive.",
                        parent=self,
                    )
                    print(
                        f"[DEBUG GUI] Validation failed: Non-positive weight for {feature}."
                    )  # DEBUG PRINT
                    return None
                weights[feature] = weight
            config["feature_weights"] = weights
        except ValueError:
            messagebox.showerror(
                "Input Error", "All weights must be valid numbers.", parent=self
            )
            print(
                "[DEBUG GUI] Validation failed: Invalid weight format."
            )  # DEBUG PRINT
            return None

        # Validate HMM parameters
        try:
            n_mixtures = int(self.mixture_var.get())
            if n_mixtures < 1 or n_mixtures > 10:  # Added upper bound sanity check
                messagebox.showerror(
                    "Input Error",
                    "Number of mixtures must be between 1 and 10.",
                    parent=self,
                )
                print(
                    f"[DEBUG GUI] Validation failed: Invalid mixture count ({n_mixtures})."
                )  # DEBUG PRINT
                return None
            config["hmm_n_mixtures"] = n_mixtures
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Number of mixtures must be a valid integer.",
                parent=self,
            )
            print(
                "[DEBUG GUI] Validation failed: Invalid mixture format."
            )  # DEBUG PRINT
            return None

        try:
            min_covar = float(self.min_covar_var.get())
            if min_covar <= 0:
                messagebox.showerror(
                    "Input Error", "Minimum covariance must be positive.", parent=self
                )
                print(
                    "[DEBUG GUI] Validation failed: Non-positive min_covar."
                )  # DEBUG PRINT
                return None
            config["hmm_min_covar"] = min_covar
        except ValueError:
            messagebox.showerror(
                "Input Error", "Minimum covariance must be a valid number.", parent=self
            )
            print(
                "[DEBUG GUI] Validation failed: Invalid min_covar format."
            )  # DEBUG PRINT
            return None

        # Get transition prior setting
        config["use_transition_prior"] = self.use_transition_prior_var.get()

        # Validate and store data cleaning parameters
        config["cleaning_settings"] = {}
        config["cleaning_settings"][
            "enable_outlier_removal"
        ] = self.enable_outlier_removal.get()
        if config["cleaning_settings"]["enable_outlier_removal"]:
            try:
                z_threshold = float(self.outlier_z_threshold.get())
                if z_threshold <= 0:
                    messagebox.showerror(
                        "Input Error",
                        "Z-score threshold must be positive.",
                        parent=self,
                    )
                    print(
                        "[DEBUG GUI] Validation failed: Non-positive Z-score."
                    )  # DEBUG PRINT
                    return None
                config["cleaning_settings"]["outlier_z_threshold"] = z_threshold
            except ValueError:
                messagebox.showerror(
                    "Input Error",
                    "Z-score threshold must be a valid number.",
                    parent=self,
                )
                print(
                    "[DEBUG GUI] Validation failed: Invalid Z-score format."
                )  # DEBUG PRINT
                return None
        else:
            config["cleaning_settings"][
                "outlier_z_threshold"
            ] = None  # Store None if disabled

        config["cleaning_settings"][
            "enable_short_section_removal"
        ] = self.enable_short_section_removal.get()
        if config["cleaning_settings"]["enable_short_section_removal"]:
            try:
                min_bars = float(self.short_section_min_bars.get())  # Allow float
                if min_bars <= 0:
                    messagebox.showerror(
                        "Input Error", "Minimum bars must be positive.", parent=self
                    )
                    print(
                        "[DEBUG GUI] Validation failed: Non-positive min_bars."
                    )  # DEBUG PRINT
                    return None
                config["cleaning_settings"]["short_section_min_bars"] = min_bars
            except ValueError:
                messagebox.showerror(
                    "Input Error", "Minimum bars must be a valid number.", parent=self
                )
                print(
                    "[DEBUG GUI] Validation failed: Invalid min_bars format."
                )  # DEBUG PRINT
                return None
        else:
            config["cleaning_settings"][
                "short_section_min_bars"
            ] = None  # Store None if disabled

        config["cleaning_settings"][
            "enable_consistency_filtering"
        ] = self.enable_consistency_filtering.get()

        print("[DEBUG GUI] Validation successful.")  # DEBUG PRINT
        return config

    def on_confirm(self):
        """Handle confirm button click"""
        print("[DEBUG GUI] Confirm button clicked.")  # DEBUG PRINT
        validated_config = self.validate_inputs()
        if validated_config:
            self.result_config = validated_config
            print(
                "[DEBUG GUI] Configuration validated, destroying window."
            )  # DEBUG PRINT
            self.destroy()  # Close the dialog

    def on_cancel(self):
        """Handle cancel button click or window close"""
        print("[DEBUG GUI] Cancel button clicked or window closed.")  # DEBUG PRINT
        self.result_config = None  # Indicate cancellation
        print("[DEBUG GUI] Destroying window.")  # DEBUG PRINT
        self.destroy()  # Close the dialog


# --- Model Analysis Dialog ---
# (No changes needed in ModelAnalysisDialog)
class ModelAnalysisDialog(tk.Toplevel):
    """
    A Tkinter dialog to display the results of the GMMHMM analysis,
    including details for each mixture component.
    """

    def __init__(self, parent, model_info, data_cleaning_info, raw_averages=None):
        super().__init__(parent)
        self.title("GMMHMM Model Analysis Results (Mixture Details)")  # Updated Title
        self.geometry("950x750")  # Made wider and taller for more info
        self.resizable(True, True)

        # Store the analysis info
        self.model_info = model_info if model_info else {}  # Handle None case
        self.cleaning_info = (
            data_cleaning_info if data_cleaning_info else {}
        )  # Handle None case
        self.raw_averages = raw_averages if raw_averages else {}  # Handle None case

        # Create widgets
        self.create_widgets()

        # Make modal and visible
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Center and focus
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x = (screen_width // 2) - (width // 2)
            y = (screen_height // 2) - (height // 2)
            self.geometry(f"{width}x{height}+{x}+{y}")
        except tk.TclError:
            pass  # Ignore centering error if screen info not available yet
        self.lift()
        self.focus_force()

        # Wait for this dialog window to close
        self.wait_window()

    def create_widgets(self):
        """Creates and lays out the widgets in the analysis dialog."""
        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook with tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        # States tab
        states_tab = ttk.Frame(notebook, padding=10)
        notebook.add(states_tab, text="State & Mixture Analysis")  # Updated Tab Name

        # Transitions tab
        transitions_tab = ttk.Frame(notebook, padding=10)
        notebook.add(transitions_tab, text="Transition Probabilities")

        # Data Cleaning tab
        cleaning_tab = ttk.Frame(notebook, padding=10)
        notebook.add(cleaning_tab, text="Data Cleaning Report")

        # Fill Tabs
        self.create_states_tab(states_tab)
        self.create_transitions_tab(transitions_tab)
        self.create_cleaning_tab(cleaning_tab)

        # Close button at the bottom
        close_button = ttk.Button(main_frame, text="Close", command=self.on_close)
        close_button.pack(pady=(10, 5))

    def _add_scrollable_frame(self, parent_tab):
        """Helper to create a scrollable frame inside a tab."""
        canvas = tk.Canvas(parent_tab)
        scrollbar_y = ttk.Scrollbar(parent_tab, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(
            parent_tab, orient="horizontal", command=canvas.xview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        parent_tab.grid_rowconfigure(0, weight=1)
        parent_tab.grid_columnconfigure(0, weight=1)

        return scrollable_frame

    def create_states_tab(self, parent_tab):
        """Fills the State Analysis tab, now showing mixture details."""
        scrollable_frame = self._add_scrollable_frame(parent_tab)

        # Title
        ttk.Label(
            scrollable_frame,
            text="State & Mixture Analysis",
            font=("Arial", 14, "bold"),
        ).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10), padx=5
        )  # Span 3 columns now

        # Display state information
        states_info = self.model_info.get("states", [])
        row_num = 1
        if not states_info:
            ttk.Label(
                scrollable_frame, text="No state analysis information available."
            ).grid(row=row_num, column=0, columnspan=3, sticky="w", pady=5, padx=5)
            return

        for state_data in states_info:
            state_name = state_data.get("name", "Unknown State")
            state_frame = ttk.LabelFrame(
                scrollable_frame, text=f"State: {state_name}", padding=10
            )
            state_frame.grid(
                row=row_num, column=0, columnspan=3, sticky="ew", padx=5, pady=8
            )  # Span 3
            state_frame.columnconfigure(
                1, weight=1
            )  # Allow middle column (mixture details) to expand

            # Display Raw Averages for the state (if available)
            raw_features = state_data.get("raw_averages", {})
            raw_frame = ttk.LabelFrame(state_frame, text="Raw Data Averages", padding=5)
            raw_frame.grid(
                row=0, column=0, sticky="nsew", padx=(0, 5), pady=5
            )  # Column 0

            if raw_features:
                raw_feature_text = ""
                # Show all features found in raw data, sorted
                for feature in sorted(raw_features.keys()):
                    value = raw_features[feature]
                    if isinstance(value, (int, float, np.number)) and np.isfinite(
                        value
                    ):
                        raw_feature_text += f"{feature}: {float(value):.4f}\n"
                    else:
                        raw_feature_text += (
                            f"{feature}: N/A\n"  # Handle non-numeric if needed
                        )
                ttk.Label(
                    raw_frame, text=raw_feature_text.strip(), justify="left"
                ).pack(anchor="nw")
            else:
                ttk.Label(raw_frame, text="N/A").pack(anchor="nw")

            # Display Mixture Details
            mixture_details = state_data.get("mixture_details", [])
            if mixture_details:
                # Frame to hold all mixture details for this state
                mixtures_outer_frame = ttk.Frame(state_frame)
                mixtures_outer_frame.grid(
                    row=0, column=1, sticky="nsew", padx=5, pady=5
                )  # Column 1
                mixtures_outer_frame.columnconfigure(
                    0, weight=1
                )  # Allow inner content to expand

                for mix_idx, mix_data in enumerate(mixture_details):
                    mix_frame = ttk.LabelFrame(
                        mixtures_outer_frame,
                        text=f"Mixture {mix_idx+1} (Weight: {mix_data.get('mixture_weight', 0):.3f})",
                        padding=5,
                    )
                    mix_frame.grid(
                        row=mix_idx, column=0, sticky="ew", pady=(0, 5)
                    )  # Stack mixtures vertically
                    mix_frame.columnconfigure(0, weight=1)
                    mix_frame.columnconfigure(1, weight=1)

                    # Mean Features Frame
                    mean_frame = ttk.LabelFrame(
                        mix_frame, text="Mean Features", padding=3
                    )
                    mean_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
                    mean_features = mix_data.get("mean_features_unscaled", {})
                    mean_text = ""
                    if isinstance(mean_features, dict) and "error" not in mean_features:
                        for feature in sorted(mean_features.keys()):
                            value = mean_features[feature]
                            if isinstance(
                                value, (int, float, np.number)
                            ) and np.isfinite(value):
                                mean_text += f"{feature}: {float(value):.4f}\n"
                            else:
                                mean_text += (
                                    f"{feature}: {str(value)}\n"  # Show error/NaN
                                )
                    else:
                        mean_text = mean_features.get("error", "N/A")
                    ttk.Label(mean_frame, text=mean_text.strip(), justify="left").pack(
                        anchor="nw"
                    )

                    # Std Dev Features Frame
                    std_frame = ttk.LabelFrame(
                        mix_frame, text="Std Dev Features", padding=3
                    )
                    std_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
                    std_features = mix_data.get("std_dev_features_unscaled", {})
                    std_text = ""
                    if isinstance(std_features, dict) and "error" not in std_features:
                        for feature in sorted(std_features.keys()):
                            value = std_features[feature]
                            if isinstance(
                                value, (int, float, np.number)
                            ) and np.isfinite(value):
                                std_text += f"{feature}: {float(value):.4f}\n"
                            else:
                                std_text += (
                                    f"{feature}: {str(value)}\n"  # Show error/NaN
                                )
                    else:
                        std_text = std_features.get("error", "N/A")
                    ttk.Label(std_frame, text=std_text.strip(), justify="left").pack(
                        anchor="nw"
                    )

            else:
                # Fallback if mixture details are missing
                ttk.Label(state_frame, text="Mixture details not available.").grid(
                    row=0, column=1, sticky="nsew", padx=5, pady=5
                )

            row_num += 1

    def create_transitions_tab(self, parent_tab):
        """Fills the Transition Probabilities tab."""
        scrollable_frame = self._add_scrollable_frame(parent_tab)

        # Title
        ttk.Label(
            scrollable_frame,
            text="Learned Transition Probabilities",
            font=("Arial", 14, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10), padx=5)

        # Display transition matrix if available
        transitions = self.model_info.get("transitions", None)
        # Check if transitions exist and matrix is a list (as converted in analysis.py)
        if (
            not transitions
            or "matrix" not in transitions
            or not isinstance(transitions.get("matrix"), list)
            or not transitions.get("state_names")
        ):
            ttk.Label(
                scrollable_frame,
                text="Transition matrix not available or in unexpected format.",
            ).grid(row=1, column=0, sticky="w", pady=10, padx=5)
            return

        # Create a frame for the matrix
        matrix_frame = ttk.Frame(scrollable_frame, relief=tk.GROOVE, borderwidth=1)
        matrix_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Get state names and matrix
        state_names = transitions.get("state_names", [])
        matrix = transitions.get("matrix", [])  # Should be a list of lists
        n_states = len(state_names)

        if (
            not matrix
            or len(matrix) != n_states
            or not all(len(row) == n_states for row in matrix)
        ):
            ttk.Label(
                scrollable_frame, text="Transition matrix has incorrect dimensions."
            ).grid(row=1, column=0, sticky="w", pady=10, padx=5)
            return

        # Header row with state names (To states)
        ttk.Label(
            matrix_frame, text="From \\ To", width=12, anchor="center", relief=tk.RAISED
        ).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        for j, name in enumerate(state_names):
            ttk.Label(
                matrix_frame, text=name, width=12, anchor="center", relief=tk.RAISED
            ).grid(row=0, column=j + 1, padx=1, pady=1, sticky="nsew")

        # Matrix rows (From states)
        for i, row_data in enumerate(matrix):
            ttk.Label(
                matrix_frame,
                text=state_names[i],
                width=12,
                anchor="w",
                relief=tk.RAISED,
            ).grid(row=i + 1, column=0, padx=1, pady=1, sticky="nsew")
            for j, value in enumerate(row_data):
                # Basic heat map coloring (optional)
                bg_color = "#FFFFFF"  # White default
                try:
                    # Value should already be float or convertible string ("NaN", "Infinity")
                    if isinstance(value, (int, float)):
                        val_float = float(value)
                        if val_float > 0.75:
                            bg_color = "#B3E5FC"  # Light Blue High
                        elif val_float > 0.5:
                            bg_color = "#E1F5FE"  # Lighter Blue Medium-High
                        elif val_float > 0.25:
                            bg_color = "#FFF9C4"  # Light Yellow Medium-Low
                        elif val_float > 0.05:
                            bg_color = "#FFECB3"  # Lighter Yellow Low
                        text_val = f"{val_float:.3f}"
                    else:
                        text_val = str(value)  # Display NaN/Infinity as string
                except (ValueError, TypeError):
                    text_val = str(value)  # Fallback

                ttk.Label(
                    matrix_frame,
                    text=text_val,
                    width=12,
                    anchor="center",
                    background=bg_color,
                ).grid(row=i + 1, column=j + 1, padx=1, pady=1, sticky="nsew")

        # Configure column weights for the matrix frame for resizing
        for k in range(n_states + 1):
            matrix_frame.columnconfigure(k, weight=1)

    def create_cleaning_tab(self, parent_tab):
        """Fills the Data Cleaning Report tab."""
        scrollable_frame = self._add_scrollable_frame(parent_tab)

        # Title
        ttk.Label(
            scrollable_frame, text="Data Cleaning Report", font=("Arial", 14, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10), padx=5)

        # Summary statistics
        cleaning_summary = self.cleaning_info.get("summary", {})
        summary_frame = ttk.LabelFrame(
            scrollable_frame, text="Cleaning Summary", padding=10
        )
        summary_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        summary_text = f"""
Total sections processed: {cleaning_summary.get('total_processed', 'N/A')}
Total sections removed: {cleaning_summary.get('total_removed', 'N/A')}
  - Outliers removed: {cleaning_summary.get('outliers_removed', 'N/A')}
  - Short sections removed: {cleaning_summary.get('short_sections_removed', 'N/A')}
  - Inconsistent sections removed: {cleaning_summary.get('inconsistent_removed', 'N/A')}
"""
        ttk.Label(summary_frame, text=summary_text.strip(), justify="left").pack(
            anchor="nw", padx=5, pady=5
        )

        # Detailed removal information
        removal_details = self.cleaning_info.get("removal_details", [])
        details_frame = ttk.LabelFrame(
            scrollable_frame, text="Removal Details", padding=10
        )
        details_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        scrollable_frame.rowconfigure(2, weight=1)  # Allow treeview frame to expand

        if removal_details:
            # Create a treeview for detailed information
            columns = ("song", "section_index", "label", "reason", "feature", "value")
            tree = ttk.Treeview(
                details_frame, columns=columns, show="headings", height=15
            )  # Increased height

            # Set column headings
            tree.heading("song", text="Song Name")
            tree.heading("section_index", text="Sec Idx")
            tree.heading("label", text="Label")
            tree.heading("reason", text="Removal Reason")
            tree.heading("feature", text="Feature")
            tree.heading("value", text="Value")

            # Set column widths and anchor
            tree.column("song", width=180, anchor="w")
            tree.column("section_index", width=60, anchor="center")
            tree.column("label", width=100, anchor="w")
            tree.column("reason", width=180, anchor="w")
            tree.column("feature", width=120, anchor="w")
            tree.column("value", width=100, anchor="e")

            # Add scrollbar for the treeview
            tree_scroll_y = ttk.Scrollbar(
                details_frame, orient="vertical", command=tree.yview
            )
            tree_scroll_x = ttk.Scrollbar(
                details_frame, orient="horizontal", command=tree.xview
            )
            tree.configure(
                yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set
            )

            tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            tree_scroll_y.grid(row=0, column=1, sticky="ns")
            tree_scroll_x.grid(row=1, column=0, sticky="ew")

            details_frame.grid_rowconfigure(0, weight=1)
            details_frame.grid_columnconfigure(0, weight=1)

            # Add data to the treeview, sorted by song then index
            sorted_details = sorted(
                removal_details,
                key=lambda x: (x.get("song", ""), x.get("section_index", 0)),
            )
            for detail in sorted_details:
                tree.insert(
                    "",
                    "end",
                    values=(
                        detail.get("song", "Unknown"),
                        detail.get("section_index", "?"),
                        detail.get("label", "Unknown"),
                        detail.get("reason", "Unknown"),
                        detail.get("feature", "N/A"),
                        detail.get("value", "N/A"),
                    ),
                )
        else:
            ttk.Label(
                details_frame, text="No sections were removed by data cleaning."
            ).pack(padx=5, pady=10)

    def on_close(self):
        """Handles closing the analysis dialog."""
        print("[DEBUG GUI] ModelAnalysisDialog closing.")  # DEBUG PRINT
        self.grab_release()
        self.destroy()


# --- Helper Functions to Show Dialogs ---


def show_feature_selection_dialog():
    """
    Shows the feature selection dialog and returns the selected configuration.
    MODIFIED: Does not create/destroy explicit root window.

    Returns:
        dict or None: A dictionary containing the selected configuration
                      (feature_keys, feature_weights, hmm_n_mixtures,
                      hmm_min_covar, cleaning_settings, use_transition_prior)
                      if the user confirms, otherwise None if cancelled.
    """
    dialog = None  # Initialize dialog variable
    result = None  # Initialize result variable
    try:
        print("[DEBUG GUI] Creating feature selection dialog window...")  # DEBUG PRINT
        dialog = FeatureSelectionDialog(None)
        print(
            "[DEBUG GUI] FeatureSelectionDialog object created. Waiting for window..."
        )  # DEBUG PRINT
        dialog.wait_window()
        print("[DEBUG GUI] wait_window() finished.")  # DEBUG PRINT

        if dialog and hasattr(dialog, "result_config"):
            result = dialog.result_config
            print(
                f"[DEBUG GUI] Dialog result retrieved: {'Confirmed' if result else 'Cancelled'}"
            )  # DEBUG PRINT
        else:
            print(
                "[DEBUG GUI] Dialog object or result_config attribute not found after wait_window."
            )
            result = None

    except tk.TclError as e:
        print(f"\n!!! ERROR: A Tkinter TclError occurred: {e} !!!")
        traceback.print_exc()
        result = None
    except Exception as e:
        print(
            f"\n!!! ERROR: An unexpected error occurred in show_feature_selection_dialog: {e} !!!"
        )
        traceback.print_exc()
        result = None
    finally:
        if dialog and dialog.winfo_exists():
            print("[DEBUG GUI] Explicitly destroying lingering dialog window.")
            dialog.destroy()

    return result


def show_model_analysis_dialog(model_info, data_cleaning_info, raw_averages=None):
    """
    Shows a dialog with model analysis results.
    MODIFIED: Does not create/destroy explicit root window.

    Args:
        model_info (dict): Dictionary containing model analysis information.
        data_cleaning_info (dict): Dictionary containing data cleaning information.
        raw_averages (dict, optional): Dictionary containing raw feature averages by label.
    """
    dialog = None
    try:
        print("\n[DEBUG GUI] Displaying model analysis dialog...")
        dialog = ModelAnalysisDialog(None, model_info, data_cleaning_info, raw_averages)
        print("[DEBUG GUI] Model analysis dialog closed.")
    except tk.TclError as e:
        print(
            f"\n!!! ERROR: A Tkinter TclError occurred during analysis dialog: {e} !!!"
        )
        traceback.print_exc()
    except Exception as e:
        print(
            f"\n!!! ERROR: An unexpected error occurred in show_model_analysis_dialog: {e} !!!"
        )
        traceback.print_exc()
    finally:
        if dialog and dialog.winfo_exists():
            print("[DEBUG GUI] Explicitly destroying lingering analysis dialog window.")
            dialog.destroy()
