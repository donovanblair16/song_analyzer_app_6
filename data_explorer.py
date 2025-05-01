# /Users/donovanblair/Desktop/song_analyzer_app_6/data_explorer.py

"""Provides functions or classes for exploring analysis data."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import joblib
import numpy as np
import traceback
from collections import defaultdict

# <<< Import the feature calculation function >>>
try:
    # Assuming data_explorer_gui.py is in the project root,
    # alongside the GMMHMM_modules folder
    from GMMHMM_modules.feature_extraction import extract_section_features

    print("DEBUG Explorer: Successfully imported extract_section_features.")
except ImportError as e:
    print(f"ERROR: Could not import extract_section_features: {e}")
    messagebox.showerror(
        "Import Error",
        "Could not find the feature extraction function.\n"
        "Ensure data_explorer_gui.py is in the correct project folder.",
    )

    # Define a dummy if import fails, so the GUI can still load partially
    def extract_section_features(track_data):
        """Dummy function if the real feature extraction fails to import."""
        print("ERROR: Using dummy extract_section_features in Explorer!")
        return [], []


class DataExplorerApp:
    """
    A GUI application to scan analysis files for sections meeting specific
    feature criteria. Calculates HMM features on-the-fly during scan.
    """

    def __init__(self, master):
        """Initializes the DataExplorerApp GUI.

        Args:
            master (tk.Tk or tk.Toplevel): The root window or top-level widget.

        """
        self.master = master
        master.title("Analysis Data Explorer (HMM Features)")  # Updated title
        master.geometry("800x600")

        # --- State Variables ---
        self.folder_path = tk.StringVar()
        self.all_features = []  # List of features found in data
        self.all_labels = []  # List of labels found in data
        self.selected_feature = tk.StringVar()
        self.selected_label = tk.StringVar()
        self.comparison_type = tk.StringVar(value="<")  # Default to less than
        self.threshold_value = tk.StringVar(value="0.0")

        # --- Build GUI ---
        self.create_widgets()

    def create_widgets(self):
        """Creates and lays out all the GUI widgets for the application."""
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Controls Frame ---
        controls_frame = ttk.LabelFrame(
            main_frame, text="Scan Configuration", padding="10"
        )
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)  # Allow folder path label to expand

        # Folder Selection
        ttk.Button(
            controls_frame, text="Select Folder...", command=self.select_folder
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(
            controls_frame, textvariable=self.folder_path, state="readonly", width=60
        ).grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Feature Selection
        ttk.Label(controls_frame, text="Feature:").grid(
            row=1, column=0, padx=(5, 0), pady=5, sticky="e"
        )
        self.feature_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.selected_feature,
            state="readonly",
            width=25,
        )
        self.feature_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Label Selection
        ttk.Label(controls_frame, text="Label:").grid(
            row=2, column=0, padx=(5, 0), pady=5, sticky="e"
        )
        self.label_combo = ttk.Combobox(
            controls_frame, textvariable=self.selected_label, state="readonly", width=25
        )
        self.label_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Comparison Type & Threshold
        ttk.Label(controls_frame, text="Condition:").grid(
            row=3, column=0, padx=(5, 0), pady=5, sticky="e"
        )
        condition_frame = ttk.Frame(controls_frame)
        condition_frame.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Radiobutton(
            condition_frame, text="Value <", variable=self.comparison_type, value="<"
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(
            condition_frame, text="Value >", variable=self.comparison_type, value=">"
        ).pack(side=tk.LEFT, padx=(0, 10))
        self.threshold_entry = ttk.Entry(
            condition_frame, textvariable=self.threshold_value, width=10
        )
        self.threshold_entry.pack(side=tk.LEFT)

        # Scan Button
        self.scan_button = ttk.Button(
            controls_frame, text="Scan Folder", command=self.run_scan, state=tk.DISABLED
        )
        self.scan_button.grid(row=4, column=0, columnspan=3, pady=10)

        # --- Results Frame ---
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Use Treeview for structured results
        columns = ("filename", "index", "label", "feature", "value")
        self.results_tree = ttk.Treeview(
            results_frame, columns=columns, show="headings", height=15
        )

        # Define headings
        self.results_tree.heading("filename", text="Filename")
        self.results_tree.heading("index", text="Sec Idx")
        self.results_tree.heading("label", text="Label")
        self.results_tree.heading("feature", text="Feature")
        self.results_tree.heading("value", text="Value")

        # Define column widths and alignment
        self.results_tree.column("filename", width=300, anchor=tk.W)
        self.results_tree.column("index", width=60, anchor=tk.CENTER)
        self.results_tree.column("label", width=100, anchor=tk.W)
        self.results_tree.column("feature", width=150, anchor=tk.W)
        self.results_tree.column("value", width=100, anchor=tk.E)

        # Add scrollbars
        vsb = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.results_tree.yview
        )
        hsb = ttk.Scrollbar(
            results_frame, orient="horizontal", command=self.results_tree.xview
        )
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout for treeview and scrollbars
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Configure resizing behavior
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Select a folder to begin.")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_folder(self):
        """Opens a dialog to select the folder containing analysis files.

        Sets the folder path, updates the status bar, triggers feature/label
        extraction, and enables the scan button upon successful selection.
        """
        initial_dir = "."
        try:
            default_base = (
                "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"
            )
            if os.path.isdir(default_base):
                initial_dir = default_base
            elif os.path.isdir(os.path.dirname(default_base)):
                initial_dir = os.path.dirname(default_base)
        except Exception:
            pass

        selected_path = filedialog.askdirectory(
            title="Select Folder Containing Analysis Files (e.g., 'Perfect')",
            initialdir=initial_dir,
        )
        if selected_path:
            self.folder_path.set(selected_path)
            self.status_var.set(
                f"Folder selected. Extracting features/labels from first file..."
            )
            self.master.update_idletasks()
            self.extract_features_and_labels()  # Calls updated logic
            if self.all_features and self.all_labels:
                self.scan_button.config(state=tk.NORMAL)
                self.status_var.set(
                    f"Ready to scan folder: {os.path.basename(selected_path)}"
                )
            else:
                self.scan_button.config(state=tk.DISABLED)
                self.status_var.set(
                    "Error: Could not extract features/labels. Check console."
                )
        else:
            self.status_var.set("Folder selection cancelled.")

    def extract_features_and_labels(self):
        """Loads the first valid joblib file, runs extract_section_features on it,
        and populates the feature and label lists available for selection in the GUI.

        Updates the feature and label comboboxes and the status bar.
        """
        folder = self.folder_path.get()
        if not folder or not os.path.isdir(folder):
            self.status_var.set("Error: Invalid folder path.")
            return

        self.all_features = []  # Reset lists
        self.all_labels = set()

        try:
            found_file_processed = False
            for filename in os.listdir(folder):
                if filename.endswith(".joblib"):
                    file_path = os.path.join(folder, filename)
                    print(
                        f"DEBUG: Attempting to load and process {filename} for feature/label extraction."
                    )
                    try:
                        track_data = joblib.load(file_path)
                        if not isinstance(track_data, dict):
                            print(f"DEBUG: Data in {filename} is not a dict.")
                            continue

                        # <<< Call the HMM feature extraction function >>>
                        calculated_features_list, original_labels = (
                            extract_section_features(track_data)
                        )

                        if not calculated_features_list:
                            print(
                                f"DEBUG: extract_section_features returned no features for {filename}."
                            )
                            continue  # Try next file if feature extraction failed

                        # --- Populate Labels ---
                        if original_labels and isinstance(
                            original_labels, (list, np.ndarray)
                        ):
                            self.all_labels.update(original_labels)

                        # --- Populate Features (from the calculated features) ---
                        found_features_set = set()
                        for section_dict in calculated_features_list:
                            if isinstance(section_dict, dict):
                                for key, value in section_dict.items():
                                    # Check if value is a standard numeric type (excluding complex)
                                    # and not NaN before adding the key
                                    if isinstance(
                                        value, (int, float, np.number)
                                    ) and not isinstance(value, (np.complexfloating)):
                                        is_nan = False
                                        # Check for NaN safely, handling potential type errors
                                        if isinstance(value, (float, np.floating)):
                                            try:
                                                is_nan = np.isnan(value)
                                            except TypeError:
                                                pass  # Ignore if isnan fails for some reason
                                        if not is_nan:
                                            found_features_set.add(key)

                        self.all_features = sorted(list(found_features_set))
                        print(f"DEBUG: Extracted features: {self.all_features}")
                        found_file_processed = True
                        break  # Stop after processing the first valid file

                    except ModuleNotFoundError as mnfe:
                        print(
                            f"WARNING: ModuleNotFoundError loading {filename}: {mnfe}. Skipping file."
                        )
                        continue
                    except Exception as e:
                        print(
                            f"Warning: Could not load or process {filename} for feature extraction: {e}"
                        )
                        traceback.print_exc()
                        continue

            if not found_file_processed:
                raise FileNotFoundError(
                    "No valid .joblib analysis files found or processed successfully."
                )

            # Convert sets to sorted lists
            self.all_labels = sorted(
                [str(lbl) for lbl in self.all_labels if lbl is not None]
            )

            # Update comboboxes
            self.feature_combo["values"] = self.all_features
            self.label_combo["values"] = self.all_labels

            # Set default selections
            if self.all_features:
                self.selected_feature.set(self.all_features[0])
            else:
                self.selected_feature.set("")
            if self.all_labels:
                self.selected_label.set(self.all_labels[0])
            else:
                self.selected_label.set("")

            if self.all_features and self.all_labels:
                self.status_var.set("Features and labels extracted. Ready to scan.")
            else:
                self.status_var.set(
                    "Warning: Could not extract all features or labels."
                )

        except Exception as e:
            self.status_var.set(f"Error extracting features/labels: {e}")
            messagebox.showerror(
                "Error", f"Could not extract features/labels from folder:\n{e}"
            )
            self.feature_combo["values"] = []
            self.label_combo["values"] = []
            self.all_features = []
            self.all_labels = []

    def run_scan(self):
        """Scans files in the selected folder based on user criteria.

        Loads each analysis file, runs feature extraction, checks sections
        matching the selected label and feature condition, and displays
        results in the treeview.
        """
        folder = self.folder_path.get()
        feature_to_check = self.selected_feature.get()  # Renamed variable for clarity
        label_to_check = self.selected_label.get()  # Renamed variable for clarity
        compare_type = self.comparison_type.get()
        threshold_str = self.threshold_value.get()

        # --- Validate Inputs ---
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder first.")
            return
        if not feature_to_check:
            messagebox.showerror("Error", "Please select a feature to analyze.")
            return
        if not label_to_check:
            messagebox.showerror("Error", "Please select a label to analyze.")
            return
        try:
            threshold = float(threshold_str)
        except ValueError:
            messagebox.showerror("Error", "Threshold must be a valid number.")
            return

        # --- Clear previous results ---
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.status_var.set(
            f"Scanning for '{label_to_check}' sections where '{feature_to_check}' {compare_type} {threshold}..."
        )
        self.master.update_idletasks()

        # --- Scan Files ---
        found_results = []
        files_scanned = 0
        errors = 0

        try:
            for filename in os.listdir(folder):
                if filename.endswith(".joblib"):
                    files_scanned += 1
                    file_path = os.path.join(folder, filename)
                    try:
                        track_data = joblib.load(file_path)
                        if not isinstance(track_data, dict):
                            continue

                        # <<< Call HMM feature extraction for EACH file >>>
                        calculated_features_list, original_labels = (
                            extract_section_features(track_data)
                        )

                        if (
                            not calculated_features_list
                            or not original_labels
                            or len(calculated_features_list) != len(original_labels)
                        ):
                            print(
                                f"Warning: Feature extraction failed or inconsistent for {filename}. Skipping."
                            )
                            continue  # Skip files if feature extraction fails

                        # <<< Iterate through CALCULATED features and ORIGINAL labels >>>
                        for i, sec_label in enumerate(original_labels):
                            if str(sec_label) == label_to_check:  # Compare as strings
                                if i < len(calculated_features_list) and isinstance(
                                    calculated_features_list[i], dict
                                ):
                                    # Use the dictionary from the calculated list
                                    feature_dict = calculated_features_list[i]
                                    value = feature_dict.get(
                                        feature_to_check
                                    )  # Check the selected feature

                                    if value is not None:
                                        try:
                                            value_float = float(value)
                                            if np.isfinite(value_float):
                                                # Perform comparison
                                                match = False
                                                if (
                                                    compare_type == "<"
                                                    and value_float < threshold
                                                ):
                                                    match = True
                                                elif (
                                                    compare_type == ">"
                                                    and value_float > threshold
                                                ):
                                                    match = True

                                                if match:
                                                    found_results.append(
                                                        (
                                                            filename,
                                                            i,
                                                            label_to_check,
                                                            feature_to_check,
                                                            f"{value_float:.4f}",  # Format value
                                                        )
                                                    )
                                        except (TypeError, ValueError):
                                            pass  # Ignore non-numeric or NaN

                    except ModuleNotFoundError as mnfe:
                        print(
                            f"Error loading {filename} during scan (ModuleNotFound): {mnfe}. Skipping file."
                        )
                        errors += 1
                    except Exception as load_err:
                        print(
                            f"Error loading/processing {filename} during scan: {load_err}"
                        )
                        # traceback.print_exc() # Uncomment for more detail
                        errors += 1

            # --- Display Results ---
            if found_results:
                # Sort results primarily by filename, then by section index
                found_results.sort(key=lambda x: (x[0], x[1]))
                for result_item in found_results:
                    self.results_tree.insert("", tk.END, values=result_item)
                self.status_var.set(
                    f"Scan complete. Found {len(found_results)} matching sections in {files_scanned} files."
                )
            else:
                self.status_var.set(
                    f"Scan complete. No matching sections found in {files_scanned} files."
                )

            if errors > 0:
                messagebox.showwarning(
                    "Scan Warning",
                    f"Encountered errors while processing {errors} files. Check console for details.",
                )

        except Exception as scan_err:
            self.status_var.set(f"Error during scan: {scan_err}")
            messagebox.showerror(
                "Scan Error",
                f"An unexpected error occurred during the scan:\n{scan_err}",
            )
            traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DataExplorerApp(root)
    root.mainloop()
