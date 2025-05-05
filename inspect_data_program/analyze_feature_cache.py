#!/usr/bin/env python3
# Feature Cache Analyzer with GUI
# Purpose: Analyze and output the structure of a feature cache file with a simple GUI

import os
import sys
import joblib
import json
import numpy as np
import traceback
from collections import Counter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime


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
    return str(obj)  # Default fallback for other types


def analyze_feature_cache(cache_path, output_path=None, status_callback=None):
    """
    Analyzes a feature cache file and outputs its structure.

    Args:
        cache_path (str): Path to the feature cache joblib file
        output_path (str, optional): Path to save the analysis JSON. If None, prints to console.
        status_callback (callable, optional): Function to call with status updates

    Returns:
        dict: Analysis results
    """
    if status_callback:
        status_callback(f"Analyzing feature cache: {cache_path}")
    else:
        print(f"Analyzing feature cache: {cache_path}")

    if not os.path.exists(cache_path):
        msg = f"ERROR: File not found at: {cache_path}"
        if status_callback:
            status_callback(msg)
        else:
            print(msg)
        return None

    try:
        # Load the cache file
        if status_callback:
            status_callback("Loading cache file...")
        cache_data = joblib.load(cache_path)
        if status_callback:
            status_callback("Successfully loaded cache file.")
        else:
            print("Successfully loaded cache file.")

        # Initialize analysis structure
        analysis = {
            "file_path": cache_path,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "top_level_keys": list(cache_data.keys()),
            "type": str(type(cache_data)),
            "structure": {},
            "statistics": {},
            "sample_data": {},
        }

        # Analyze top-level structure
        if status_callback:
            status_callback("Analyzing top-level structure...")
        else:
            print("Analyzing top-level structure...")

        for key in cache_data.keys():
            value = cache_data[key]
            value_type = type(value).__name__
            analysis["structure"][key] = {
                "type": value_type,
                "is_list": isinstance(value, list),
                "is_dict": isinstance(value, dict),
                "is_numpy": isinstance(value, np.ndarray),
            }

            # Add more details based on type
            if isinstance(value, list):
                analysis["structure"][key]["length"] = len(value)
                if value:
                    analysis["structure"][key]["first_item_type"] = type(
                        value[0]
                    ).__name__

                    # If list contains dictionaries, analyze their keys
                    if isinstance(value[0], dict):
                        all_dict_keys = set()
                        for item in value[:10]:  # Check first 10 items
                            all_dict_keys.update(item.keys())
                        analysis["structure"][key]["dict_keys"] = list(all_dict_keys)

                        # Count how many items have each key
                        key_counts = {k: 0 for k in all_dict_keys}
                        for item in value[:100]:  # Check up to 100 items
                            for k in item.keys():
                                if k in key_counts:
                                    key_counts[k] += 1
                        analysis["structure"][key]["key_presence"] = key_counts

                        # Check for feature data structure
                        if "section_features" in all_dict_keys and len(value) > 0:
                            # Sample the feature structure from the first item
                            if (
                                "section_features" in value[0]
                                and len(value[0]["section_features"]) > 0
                            ):
                                first_section = value[0]["section_features"][0]
                                analysis["sample_data"][
                                    "section_features_first_item"
                                ] = {
                                    "keys": list(first_section.keys()),
                                    "value_types": {
                                        k: type(v).__name__
                                        for k, v in first_section.items()
                                    },
                                }

            elif isinstance(value, dict):
                analysis["structure"][key]["keys"] = list(value.keys())
                if value:
                    # Sample a few values
                    sample_keys = list(value.keys())[:5]
                    analysis["structure"][key]["sample_values"] = {
                        k: str(type(value[k])) for k in sample_keys
                    }

            elif isinstance(value, np.ndarray):
                analysis["structure"][key]["shape"] = value.shape
                analysis["structure"][key]["dtype"] = str(value.dtype)

        # Special analysis for tracks if present
        if status_callback:
            status_callback("Analyzing track data...")

        if "tracks" in cache_data and isinstance(cache_data["tracks"], list):
            tracks = cache_data["tracks"]
            analysis["statistics"]["track_count"] = len(tracks)

            # Analyze section labels across all tracks
            all_labels = []
            section_counts = []
            feature_keys_found = set()

            for i, track in enumerate(tracks):
                if status_callback and i % 10 == 0:
                    status_callback(f"Processing track {i+1}/{len(tracks)}...")

                if i < 5:  # Store detailed info for the first 5 tracks
                    track_info = {
                        "track_name": track.get("track_name", f"Track_{i}"),
                        "keys": list(track.keys()),
                    }

                    # Check for section features and labels
                    if "section_features" in track and "section_labels" in track:
                        section_features = track["section_features"]
                        section_labels = track["section_labels"]

                        if len(section_features) == len(section_labels):
                            track_info["section_count"] = len(section_labels)
                            track_info["labels_present"] = list(set(section_labels))

                            # Collect all feature keys from first section
                            if section_features and len(section_features) > 0:
                                first_section = section_features[0]
                                feature_keys_found.update(first_section.keys())
                                track_info["first_section_keys"] = list(
                                    first_section.keys()
                                )

                                # Sample values for the first section
                                track_info["first_section_sample"] = {
                                    k: first_section[k]
                                    for k in list(first_section.keys())[:10]
                                }

                    analysis["sample_data"][f"track_{i}"] = track_info

                # Collect stats across all tracks
                if "section_labels" in track:
                    section_labels = track["section_labels"]
                    all_labels.extend(section_labels)
                    section_counts.append(len(section_labels))

            # Label statistics
            label_counts = Counter(all_labels)
            analysis["statistics"]["unique_labels"] = list(label_counts.keys())
            analysis["statistics"]["label_counts"] = dict(label_counts)
            analysis["statistics"]["total_sections"] = sum(section_counts)
            analysis["statistics"]["average_sections_per_track"] = (
                sum(section_counts) / len(tracks) if tracks else 0
            )
            analysis["statistics"]["all_feature_keys"] = list(feature_keys_found)

            # Additional feature analysis
            if feature_keys_found and tracks and "section_features" in tracks[0]:
                feature_stats = {}
                for feature_key in feature_keys_found:
                    feature_stats[feature_key] = {
                        "present_count": 0,
                        "min_value": float("inf"),
                        "max_value": float("-inf"),
                        "is_numeric": True,
                    }

                # Sample some tracks to get feature statistics
                sample_size = min(20, len(tracks))
                for i in range(sample_size):
                    track = tracks[i]
                    if "section_features" in track:
                        for section in track["section_features"]:
                            for feature_key in feature_keys_found:
                                if feature_key in section:
                                    value = section[feature_key]
                                    feature_stats[feature_key]["present_count"] += 1

                                    # Check if numeric
                                    if isinstance(
                                        value, (int, float, np.number)
                                    ) and np.isfinite(value):
                                        feature_stats[feature_key]["min_value"] = min(
                                            feature_stats[feature_key]["min_value"],
                                            float(value),
                                        )
                                        feature_stats[feature_key]["max_value"] = max(
                                            feature_stats[feature_key]["max_value"],
                                            float(value),
                                        )
                                    else:
                                        feature_stats[feature_key]["is_numeric"] = False
                                        if "min_value" in feature_stats[feature_key]:
                                            del feature_stats[feature_key]["min_value"]
                                        if "max_value" in feature_stats[feature_key]:
                                            del feature_stats[feature_key]["max_value"]

                # Clean up infinite values
                for feature_key in feature_stats:
                    if feature_stats[feature_key]["is_numeric"]:
                        if feature_stats[feature_key]["min_value"] == float("inf"):
                            feature_stats[feature_key]["min_value"] = "Not found"
                        if feature_stats[feature_key]["max_value"] == float("-inf"):
                            feature_stats[feature_key]["max_value"] = "Not found"

                analysis["statistics"]["feature_stats"] = feature_stats

        if status_callback:
            status_callback("Analysis complete.")
        else:
            print("Analysis complete.")

        # Output the results
        if output_path:
            if status_callback:
                status_callback(f"Saving analysis to: {output_path}")
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2, default=np_encoder)
            if status_callback:
                status_callback(f"Analysis saved to: {output_path}")
            else:
                print(f"Analysis saved to: {output_path}")
        else:
            # Print formatted JSON to console
            print("\n=== FEATURE CACHE ANALYSIS ===")
            print(json.dumps(analysis, indent=2, default=np_encoder))
            print("=============================")

        return analysis

    except Exception as e:
        error_msg = f"ERROR analyzing cache: {e}"
        if status_callback:
            status_callback(error_msg)
            status_callback(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return None


class FeatureCacheAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Cache Analyzer")
        self.root.geometry("600x500")
        self.root.minsize(500, 400)

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="Feature Cache Analyzer", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Description
        desc_text = "This tool analyzes a GMMHMM feature cache file (.joblib) and outputs its structure to help understand the data format."
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=550)
        desc_label.pack(pady=(0, 15))

        # Input file frame
        input_frame = ttk.LabelFrame(main_frame, text="Input File", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        # Input file path
        ttk.Label(input_frame, text="Cache File:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.input_path_var = tk.StringVar()
        input_path_entry = ttk.Entry(
            input_frame, textvariable=self.input_path_var, width=50
        )
        input_path_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Browse button
        browse_button = ttk.Button(
            input_frame, text="Browse...", command=self.browse_input
        )
        browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Output file frame
        output_frame = ttk.LabelFrame(main_frame, text="Output File", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))

        # Output file path
        ttk.Label(output_frame, text="Save Analysis To:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.output_path_var = tk.StringVar()
        output_path_entry = ttk.Entry(
            output_frame, textvariable=self.output_path_var, width=50
        )
        output_path_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Browse button for output
        save_button = ttk.Button(
            output_frame, text="Browse...", command=self.browse_output
        )
        save_button.grid(row=0, column=2, padx=5, pady=5)

        # Configure grid columns
        input_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(1, weight=1)

        # Analysis button
        self.analyze_button = ttk.Button(
            main_frame, text="Analyze Cache File", command=self.run_analysis
        )
        self.analyze_button.pack(pady=10)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Status log
        self.status_text = tk.Text(status_frame, height=10, width=70, wrap=tk.WORD)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for status
        status_scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=status_scrollbar.set)

        # Default paths for convenience
        default_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        default_cache_path = os.path.join(
            default_desktop,
            "song_analyzer_app_6",
            "inspect_data_program",
            "feature_inspection_cache.joblib",
        )
        if os.path.exists(default_cache_path):
            self.input_path_var.set(default_cache_path)

        default_output = os.path.join(default_desktop, "feature_cache_analysis.json")
        self.output_path_var.set(default_output)

        # Initial status
        self.update_status(
            "Ready. Select a cache file and output location, then click 'Analyze Cache File'."
        )

    def browse_input(self):
        """Browse for input joblib file"""
        filepath = filedialog.askopenfilename(
            title="Select Feature Cache File",
            filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")],
        )
        if filepath:
            self.input_path_var.set(filepath)
            # Auto-suggest output path based on input
            input_dir = os.path.dirname(filepath)
            input_name = os.path.splitext(os.path.basename(filepath))[0]
            suggested_output = os.path.join(input_dir, f"{input_name}_analysis.json")
            self.output_path_var.set(suggested_output)
            self.update_status(f"Selected cache file: {filepath}")

    def browse_output(self):
        """Browse for output JSON file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Analysis As",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if filepath:
            self.output_path_var.set(filepath)
            self.update_status(f"Analysis will be saved to: {filepath}")

    def update_status(self, message):
        """Update the status text box"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def run_analysis(self):
        """Run the analysis process"""
        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()

        if not input_path:
            messagebox.showerror("Error", "Please select an input cache file.")
            return

        if not os.path.exists(input_path):
            messagebox.showerror("Error", f"Input file not found: {input_path}")
            return

        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {e}")
                return

        # Disable the analyze button while running
        self.analyze_button.config(state="disabled")
        self.update_status(f"Starting analysis of {input_path}")

        try:
            # Run the analysis
            result = analyze_feature_cache(
                input_path, output_path, status_callback=self.update_status
            )

            if result:
                self.update_status("Analysis completed successfully!")
                # Ask if the user wants to open the output file
                if messagebox.askyesno(
                    "Success", "Analysis completed. Open the output file?"
                ):
                    self.open_file(output_path)
            else:
                self.update_status("Analysis failed. See above for details.")
                messagebox.showerror(
                    "Error", "Analysis failed. Check the status log for details."
                )
        except Exception as e:
            self.update_status(f"Error during analysis: {e}")
            self.update_status(traceback.format_exc())
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            # Re-enable the analyze button
            self.analyze_button.config(state="normal")

    def open_file(self, filepath):
        """Open a file with the default system application"""
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":  # macOS
            os.system(f'open "{filepath}"')
        else:  # Linux
            os.system(f'xdg-open "{filepath}"')


if __name__ == "__main__":
    # Run the GUI
    root = tk.Tk()
    app = FeatureCacheAnalyzerGUI(root)
    root.mainloop()
