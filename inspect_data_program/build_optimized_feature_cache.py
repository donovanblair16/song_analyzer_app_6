#!/usr/bin/env python3
# Feature Cache Optimizer Tool
# Purpose: Build an optimized feature cache from analysis files

import os
import sys
import joblib
import datetime
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    print(f"Adding project root to path: {PROJECT_ROOT}")
    sys.path.append(PROJECT_ROOT)

# Import from GMMHMM_modules
try:
    from GMMHMM_modules.data_utils import load_perfect_analyses
    from GMMHMM_modules.feature_extraction import extract_section_features
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    print("Please run this script from the project root directory.")
    sys.exit(1)


def build_optimized_feature_cache(input_dir, output_path, status_callback=None):
    """
    Builds an optimized feature cache from analysis files.

    Args:
        input_dir (str): Directory containing analysis files
        output_path (str): Path to save the optimized cache
        status_callback (callable, optional): Function to call with status updates

    Returns:
        bool: True if successful, False otherwise
    """
    if status_callback:
        status_callback(f"Building optimized feature cache from {input_dir}")
        status_callback(f"Output will be saved to {output_path}")
    else:
        print(f"Building optimized feature cache from {input_dir}")
        print(f"Output will be saved to {output_path}")

    # Load all analysis files
    if status_callback:
        status_callback("Loading analysis files...")

    all_data_with_filenames = load_perfect_analyses(input_dir)
    if not all_data_with_filenames:
        msg = "ERROR: No valid analysis files found."
        if status_callback:
            status_callback(msg)
        else:
            print(msg)
        return False

    file_count = len(all_data_with_filenames)
    if status_callback:
        status_callback(f"Loaded {file_count} analysis files.")
    else:
        print(f"Loaded {file_count} analysis files.")

    # Initialize the optimized cache structure
    optimized_cache = {
        "tracks": [],
        "metadata": {
            "created": datetime.datetime.now().isoformat(),
            "source_path": input_dir,
            "file_count": file_count,
        },
    }

    # Process each analysis file
    for i, (filename, track_data) in enumerate(all_data_with_filenames):
        msg = f"Processing file {i+1}/{file_count}: {filename}"
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

        try:
            # Extract features and labels
            section_feature_dicts, section_labels = extract_section_features(track_data)

            if (
                not section_feature_dicts
                or not section_labels
                or len(section_feature_dicts) != len(section_labels)
            ):
                msg = f" -> Skipping file: Invalid data from extract_section_features."
                if status_callback:
                    status_callback(msg)
                else:
                    print(msg)
                continue

            # Calculate some basic stats
            feature_keys = set()
            for section in section_feature_dicts:
                feature_keys.update(section.keys())

            # Create track entry
            track_entry = {
                "track_name": filename,
                "section_features": section_feature_dicts,
                "section_labels": section_labels,
                "sequence_length": len(section_labels),
                # You could also include other useful metadata here
                "duration": track_data.get("duration_processed", 0),
                "features_available": list(feature_keys),
            }

            # Add to tracks list
            optimized_cache["tracks"].append(track_entry)

        except Exception as e:
            msg = f" -> Error processing file: {e}"
            if status_callback:
                status_callback(msg)
                status_callback(traceback.format_exc())
            else:
                print(msg)
                traceback.print_exc()

    # Add statistics about the cached data
    track_count = len(optimized_cache["tracks"])
    if track_count == 0:
        msg = "ERROR: No valid tracks processed. Cache not created."
        if status_callback:
            status_callback(msg)
        else:
            print(msg)
        return False

    # Calculate feature stats
    all_features = set()
    section_count = 0
    label_counts = {}

    for track in optimized_cache["tracks"]:
        section_count += len(track["section_labels"])
        for feature_dict in track["section_features"]:
            all_features.update(feature_dict.keys())

        # Count labels
        for label in track["section_labels"]:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    # Add statistics to metadata
    optimized_cache["metadata"]["track_count"] = track_count
    optimized_cache["metadata"]["section_count"] = section_count
    optimized_cache["metadata"]["feature_keys"] = list(all_features)
    optimized_cache["metadata"]["labels"] = label_counts

    # Save the optimized cache
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if status_callback:
            status_callback(f"Saving optimized cache to {output_path}...")

        joblib.dump(optimized_cache, output_path)

        msg = f"Successfully created optimized feature cache with {track_count} tracks and {section_count} sections."
        msg2 = f"Saved to: {output_path}"
        if status_callback:
            status_callback(msg)
            status_callback(msg2)
        else:
            print(msg)
            print(msg2)
        return True
    except Exception as e:
        msg = f"ERROR saving optimized cache: {e}"
        if status_callback:
            status_callback(msg)
            status_callback(traceback.format_exc())
        else:
            print(msg)
            traceback.print_exc()
        return False


class FeatureCacheOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Cache Optimizer")
        self.root.geometry("600x500")
        self.root.minsize(500, 400)

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="Feature Cache Optimizer", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Description
        desc_text = "This tool creates an optimized feature cache from analysis files. The optimized cache contains pre-extracted features for faster GMMHMM training."
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=550)
        desc_label.pack(pady=(0, 15))

        # Input directory frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Directory", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        # Input directory path
        ttk.Label(input_frame, text="Analysis Files Directory:").grid(
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
        ttk.Label(output_frame, text="Save Cache To:").grid(
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

        # Build button
        self.build_button = ttk.Button(
            main_frame, text="Build Optimized Cache", command=self.run_build
        )
        self.build_button.pack(pady=10)

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
        default_perfect_folder = os.path.join(
            PROJECT_ROOT, "completed_analyses", "Perfect"
        )
        default_output_path = os.path.join(
            PROJECT_ROOT, "inspect_data_program", "optimized_feature_cache.joblib"
        )

        if os.path.exists(default_perfect_folder):
            self.input_path_var.set(default_perfect_folder)

        self.output_path_var.set(default_output_path)

        # Initial status
        self.update_status(
            "Ready. Select the directory containing analysis files and an output location, then click 'Build Optimized Cache'."
        )

    def browse_input(self):
        """Browse for input directory"""
        dirpath = filedialog.askdirectory(title="Select Directory with Analysis Files")
        if dirpath:
            self.input_path_var.set(dirpath)
            # Auto-suggest output path based on input
            input_dir_name = os.path.basename(dirpath)
            suggested_output = os.path.join(
                PROJECT_ROOT,
                "inspect_data_program",
                f"{input_dir_name}_optimized_cache.joblib",
            )
            self.output_path_var.set(suggested_output)
            self.update_status(f"Selected input directory: {dirpath}")

    def browse_output(self):
        """Browse for output file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Optimized Cache As",
            defaultextension=".joblib",
            filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")],
        )
        if filepath:
            self.output_path_var.set(filepath)
            self.update_status(f"Cache will be saved to: {filepath}")

    def update_status(self, message):
        """Update the status text box"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def run_build(self):
        """Run the build process"""
        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()

        if not input_path:
            messagebox.showerror("Error", "Please select an input directory.")
            return

        if not os.path.exists(input_path) or not os.path.isdir(input_path):
            messagebox.showerror(
                "Error", f"Input directory not found or not a directory: {input_path}"
            )
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

        # Disable the build button while running
        self.build_button.config(state="disabled")
        self.update_status(f"Starting build process...")

        try:
            # Run the build process
            result = build_optimized_feature_cache(
                input_path, output_path, status_callback=self.update_status
            )

            if result:
                self.update_status("Build process completed successfully!")
                messagebox.showinfo(
                    "Success", "Optimized feature cache created successfully."
                )
            else:
                self.update_status("Build process failed. See above for details.")
                messagebox.showerror(
                    "Error", "Build process failed. Check the status log for details."
                )
        except Exception as e:
            self.update_status(f"Error during build process: {e}")
            self.update_status(traceback.format_exc())
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            # Re-enable the build button
            self.build_button.config(state="normal")


if __name__ == "__main__":
    # Check if we should run the GUI or command line version
    if len(sys.argv) == 1:
        # Run the GUI
        root = tk.Tk()
        app = FeatureCacheOptimizerGUI(root)
        root.mainloop()
    else:
        # Command line version
        if len(sys.argv) < 3:
            print(
                "Usage: python build_optimized_cache.py input_directory output_file.joblib"
            )
            sys.exit(1)

        input_dir = sys.argv[1]
        output_path = sys.argv[2]

        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            print(f"ERROR: Input directory not found or not a directory: {input_dir}")
            sys.exit(1)

        # Run the build process
        success = build_optimized_feature_cache(input_dir, output_path)
        sys.exit(0 if success else 1)
