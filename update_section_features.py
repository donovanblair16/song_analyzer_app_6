# update_section_features.py

import os
import joblib
import numpy as np
import traceback
import argparse
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
from pathlib import Path

# Import the extract_section_features function
try:
    from audio_analysis_modules.feature_extraction import extract_section_features

    print("Successfully imported extract_section_features")
except ImportError as e:
    print(f"Error importing extract_section_features: {e}")
    print("Trying alternate import path...")
    try:
        from audio_analysis_wrapper import extract_section_features

        print("Successfully imported extract_section_features from wrapper")
    except ImportError:
        print("FATAL: Could not import extract_section_features function!")
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Import Error",
                "Could not import extract_section_features function. Please check your installation.",
            )
            root.destroy()
        except:
            pass
        exit(1)


class RedirectText:
    """Class to redirect stdout to a tkinter Text widget"""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


class SectionFeatureUpdater:
    def __init__(self, master):
        self.master = master
        master.title("Section Feature Updater")
        master.geometry("900x700")
        master.minsize(800, 600)

        # Variables - initialize BEFORE creating widgets
        self.file_path = None
        self.folder_path = None
        self.output_folder = None
        self.track_data = None
        self.original_features = None
        self.updated_features = None
        self.dry_run = tk.BooleanVar(value=False)
        self.force_update = tk.BooleanVar(value=True)

        # Create main frames
        self.create_widgets()

        # Redirect stdout to console widget
        self.stdout_backup = sys.stdout
        sys.stdout = RedirectText(self.console)

    def create_widgets(self):
        # Top frame for input options
        top_frame = ttk.Frame(self.master, padding=10)
        top_frame.pack(fill=tk.X)

        # File selection
        ttk.Label(top_frame, text="Select File:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.file_entry = ttk.Entry(top_frame, width=50)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(top_frame, text="Browse File...", command=self.browse_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Folder selection
        ttk.Label(top_frame, text="Select Folder:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.folder_entry = ttk.Entry(top_frame, width=50)
        self.folder_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(top_frame, text="Browse Folder...", command=self.browse_folder).grid(
            row=1, column=2, padx=5, pady=5
        )

        # Output folder
        ttk.Label(top_frame, text="Output Folder (optional):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.output_entry = ttk.Entry(top_frame, width=50)
        self.output_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(top_frame, text="Browse...", command=self.browse_output).grid(
            row=2, column=2, padx=5, pady=5
        )

        # Options
        options_frame = ttk.Frame(top_frame)
        options_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Checkbutton(
            options_frame, text="Dry Run (don't save changes)", variable=self.dry_run
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            options_frame,
            text="Force Update (recalculate all features)",
            variable=self.force_update,
        ).pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(
            button_frame, text="Analyze File", command=self.analyze_file, width=15
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Process Folder", command=self.process_folder, width=15
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Exit", command=self.cleanup_and_exit, width=10
        ).pack(side=tk.LEFT, padx=5)

        # Middle frame for comparison
        self.compare_frame = ttk.LabelFrame(
            self.master, text="Feature Comparison", padding=10
        )
        self.compare_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create a paned window for side-by-side comparison
        self.paned_window = ttk.PanedWindow(self.compare_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Original features frame (left side)
        original_frame = ttk.LabelFrame(
            self.paned_window, text="Original Features", padding=5
        )
        self.paned_window.add(original_frame, weight=1)

        # Original features tree
        self.original_tree = ttk.Treeview(
            original_frame, columns=("Value",), show="tree headings", height=20
        )
        self.original_tree.heading("#0", text="Feature")
        self.original_tree.heading("Value", text="Value")
        self.original_tree.column("#0", width=200)
        self.original_tree.column("Value", width=200)
        self.original_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        original_scrollbar = ttk.Scrollbar(
            original_frame, orient=tk.VERTICAL, command=self.original_tree.yview
        )
        original_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.original_tree.configure(yscrollcommand=original_scrollbar.set)

        # Updated features frame (right side)
        updated_frame = ttk.LabelFrame(
            self.paned_window, text="Updated Features", padding=5
        )
        self.paned_window.add(updated_frame, weight=1)

        # Updated features tree
        self.updated_tree = ttk.Treeview(
            updated_frame, columns=("Value",), show="tree headings", height=20
        )
        self.updated_tree.heading("#0", text="Feature")
        self.updated_tree.heading("Value", text="Value")
        self.updated_tree.column("#0", width=200)
        self.updated_tree.column("Value", width=200)
        self.updated_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        updated_scrollbar = ttk.Scrollbar(
            updated_frame, orient=tk.VERTICAL, command=self.updated_tree.yview
        )
        updated_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.updated_tree.configure(yscrollcommand=updated_scrollbar.set)

        # Selected section controls
        section_frame = ttk.Frame(self.compare_frame)
        section_frame.pack(fill=tk.X, pady=5)

        ttk.Label(section_frame, text="Section:").pack(side=tk.LEFT, padx=5)
        self.section_combo = ttk.Combobox(section_frame, width=10, state="readonly")
        self.section_combo.pack(side=tk.LEFT, padx=5)
        self.section_combo.bind("<<ComboboxSelected>>", self.on_section_selected)

        # Bottom frame for console output
        console_frame = ttk.LabelFrame(self.master, text="Console Output", padding=5)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Console widget
        self.console = scrolledtext.ScrolledText(console_frame, height=10, wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True)

        # Confirmation buttons hidden initially
        self.confirm_frame = ttk.Frame(self.master, padding=5)
        self.confirm_frame.pack(fill=tk.X, padx=10, pady=5)
        self.confirm_frame.pack_forget()  # Hide initially

        self.confirm_label = ttk.Label(
            self.confirm_frame, text="", font=("", 10, "bold")
        )
        self.confirm_label.pack(pady=5)

        button_subframe = ttk.Frame(self.confirm_frame)
        button_subframe.pack(pady=5)

        self.save_button = ttk.Button(
            button_subframe, text="Save Changes", command=self.confirm_save, width=15
        )
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.cancel_button = ttk.Button(
            button_subframe, text="Cancel", command=self.hide_confirm, width=10
        )
        self.cancel_button.pack(side=tk.LEFT, padx=10)

    def browse_file(self):
        """Browse for a file"""
        path = filedialog.askopenfilename(
            title="Select a joblib file",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
        )
        if not path:  # User cancelled
            return

        if path.endswith(".joblib"):
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)
            self.file_path = path
            # Clear folder path when file is selected
            self.folder_entry.delete(0, tk.END)
            self.folder_path = None
            print(f"Selected file: {path}")
        else:
            messagebox.showerror("Invalid Selection", "Please select a joblib file")

    def browse_folder(self):
        """Browse specifically for a folder"""
        folder = filedialog.askdirectory(title="Select Folder with Joblib Files")
        if not folder:  # User cancelled
            return

        self.folder_entry.delete(0, tk.END)
        self.folder_entry.insert(0, folder)
        self.folder_path = folder
        # Clear file path when folder is selected
        self.file_entry.delete(0, tk.END)
        self.file_path = None
        print(f"Selected folder: {folder}")

    def browse_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:  # User didn't cancel
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
            self.output_folder = folder
            print(f"Selected output folder: {folder}")

    def analyze_file(self):
        """Analyze a single file"""
        path = self.file_entry.get().strip()

        if not path:
            messagebox.showwarning(
                "No File Selected", "Please select a joblib file first."
            )
            return

        if not os.path.isfile(path) or not path.endswith(".joblib"):
            messagebox.showerror(
                "Invalid File", "The selected path is not a valid joblib file."
            )
            return

        print(f"Analyzing file: {path}")
        self.clear_comparison_view()
        self.hide_confirm()

        try:
            # Load the joblib file
            self.track_data = joblib.load(path)
            if not isinstance(self.track_data, dict):
                messagebox.showerror("Error", "Loaded data is not a dictionary")
                return

            # Check required keys
            required_keys = ["section_features", "semantic_labels", "rms", "rms_times"]
            missing_keys = [k for k in required_keys if k not in self.track_data]
            if missing_keys:
                messagebox.showerror(
                    "Error", f"File missing required keys: {missing_keys}"
                )
                return

            # Store original section features
            self.original_features = self.track_data.get("section_features", [])

            if not self.original_features or not self.track_data.get(
                "semantic_labels", []
            ):
                messagebox.showerror("Error", "No sections found in file")
                return

            # Calculate updated features
            print("Calculating updated section features...")
            updated_features, _ = extract_section_features(self.track_data)

            if not updated_features or len(updated_features) != len(
                self.original_features
            ):
                messagebox.showerror(
                    "Error",
                    f"Feature extraction returned {len(updated_features) if updated_features else 0} sections, expected {len(self.original_features)}",
                )
                return

            self.updated_features = updated_features

            # Update the section combo box
            self.section_combo["values"] = [
                f"Section {i+1}" for i in range(len(self.original_features))
            ]
            self.section_combo.current(0)  # Select first section

            # Show the first section comparison
            self.on_section_selected(None)

            # Show confirmation buttons
            self.show_confirm()

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error analyzing file: {str(e)}")
            traceback.print_exc()

    def process_folder(self):
        """Process all joblib files in a folder"""
        folder_path = self.folder_entry.get().strip()

        if not folder_path:
            messagebox.showwarning(
                "No Folder Selected", "Please select a folder first."
            )
            return

        if not os.path.isdir(folder_path):
            messagebox.showerror(
                "Invalid Folder", "The selected path is not a valid directory."
            )
            return

        # Get output folder
        output_folder = (
            self.output_entry.get().strip() if self.output_entry.get().strip() else None
        )
        dry_run = self.dry_run.get()
        force_update = self.force_update.get()

        # Confirm processing
        num_files = len([f for f in os.listdir(folder_path) if f.endswith(".joblib")])
        if num_files == 0:
            messagebox.showinfo(
                "No Files", "No joblib files found in the selected folder."
            )
            return

        if not messagebox.askyesno(
            "Confirm",
            f"Process {num_files} joblib files in the folder?\n\n"
            f"Dry Run: {dry_run}\n"
            f"Force Update: {force_update}\n"
            f"Output Folder: {output_folder if output_folder else 'Overwrite originals'}",
        ):
            return

        # Start processing in a separate thread
        self.clear_comparison_view()
        self.hide_confirm()

        # Call the process_folder function
        print(f"\nProcessing folder: {folder_path}")
        process_folder(folder_path, output_folder, dry_run, force_update)

        print("Folder processing complete!")

    def on_section_selected(self, event):
        """Update the comparison view when a section is selected"""
        if not self.original_features or not self.updated_features:
            return

        selected_idx = self.section_combo.current()
        if selected_idx < 0 or selected_idx >= len(self.original_features):
            return

        # Clear previous data
        for tree in [self.original_tree, self.updated_tree]:
            for item in tree.get_children():
                tree.delete(item)

        # Get original and updated features
        orig_section = self.original_features[selected_idx]
        updated_section = self.updated_features[selected_idx]

        # Insert into trees
        if isinstance(orig_section, dict) and isinstance(updated_section, dict):
            # All keys
            all_keys = sorted(
                set(list(orig_section.keys()) + list(updated_section.keys()))
            )

            for key in all_keys:
                # Original feature
                orig_value = orig_section.get(key, "N/A")
                if isinstance(orig_value, (float, np.float32, np.float64)):
                    orig_value = f"{orig_value:.4f}"
                elif isinstance(orig_value, np.ndarray):
                    orig_value = f"Array[{len(orig_value)}]"
                self.original_tree.insert("", "end", text=key, values=(orig_value,))

                # Updated feature
                updated_value = updated_section.get(key, "N/A")
                if isinstance(updated_value, (float, np.float32, np.float64)):
                    updated_value = f"{updated_value:.4f}"
                elif isinstance(updated_value, np.ndarray):
                    updated_value = f"Array[{len(updated_value)}]"

                # Highlight changed values
                tag = "changed" if orig_value != updated_value else ""
                item = self.updated_tree.insert(
                    "", "end", text=key, values=(updated_value,), tags=(tag,)
                )

                # New feature highlight
                if key not in orig_section:
                    self.updated_tree.item(item, tags=("new",))

            # Configure tags
            self.updated_tree.tag_configure("changed", background="#FFFF00")  # Yellow
            self.updated_tree.tag_configure("new", background="#90EE90")  # Light green
        else:
            # Non-dict sections, just show them as strings
            self.original_tree.insert(
                "", "end", text="Raw Value", values=(str(orig_section),)
            )
            self.updated_tree.insert(
                "", "end", text="Raw Value", values=(str(updated_section),)
            )

    def clear_comparison_view(self):
        """Clear the comparison trees"""
        for tree in [self.original_tree, self.updated_tree]:
            for item in tree.get_children():
                tree.delete(item)

        self.section_combo["values"] = []
        self.original_features = None
        self.updated_features = None

    def show_confirm(self):
        """Show the confirmation frame"""
        output_path = self.output_entry.get().strip()
        dry_run = self.dry_run.get()

        if dry_run:
            self.confirm_label.config(
                text="Dry Run enabled - changes will not be saved"
            )
            self.save_button.config(state=tk.DISABLED)
        else:
            if output_path:
                self.confirm_label.config(
                    text=f"Save changes to new location: {output_path}"
                )
            else:
                self.confirm_label.config(
                    text="Save changes by overwriting original file?"
                )
            self.save_button.config(state=tk.NORMAL)

        self.confirm_frame.pack(fill=tk.X, padx=10, pady=5)

    def hide_confirm(self):
        """Hide the confirmation frame"""
        self.confirm_frame.pack_forget()

    def confirm_save(self):
        """Save the updated features back to the file"""
        if not self.track_data or not self.updated_features:
            return

        file_path = self.file_entry.get().strip()
        output_folder = (
            self.output_entry.get().strip() if self.output_entry.get().strip() else None
        )

        try:
            # Store original labels and section data
            orig_semantic_labels = self.track_data.get("semantic_labels", [])
            orig_label_colors = self.track_data.get("label_colors", [])
            orig_section_starts = self.track_data.get("section_starts", [])

            # Update the track_data
            self.track_data["section_features"] = self.updated_features

            # Make sure we preserve original labels
            self.track_data["semantic_labels"] = orig_semantic_labels
            self.track_data["label_colors"] = orig_label_colors
            self.track_data["section_starts"] = orig_section_starts

            # Determine output path
            if output_folder:
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, os.path.basename(file_path))
                if os.path.normpath(output_path) == os.path.normpath(file_path):
                    output_path = output_path.replace(".joblib", "_updated.joblib")
            else:
                output_path = file_path

            # Save the file
            print(f"Saving to: {output_path}")
            joblib.dump(self.track_data, output_path, compress=3)
            print("Save successful!")

            messagebox.showinfo(
                "Success", f"Successfully saved updated features to {output_path}"
            )
            self.hide_confirm()

        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving file: {str(e)}")
            traceback.print_exc()

    def cleanup_and_exit(self):
        """Restore stdout and exit"""
        sys.stdout = self.stdout_backup
        self.master.destroy()


def update_features_in_file(
    file_path, output_folder=None, dry_run=False, force_update=True
):
    """
    Loads a joblib file, updates section features while preserving labels,
    and saves back to original file or a new location.

    Args:
        file_path: Path to the joblib file
        output_folder: If provided, saves to this folder instead of overwriting
        dry_run: If True, doesn't save changes (just for testing)
        force_update: If True, recalculate features even if relative_rms exists

    Returns:
        success: True if process completed successfully
    """
    print(f"\nProcessing: {os.path.basename(file_path)}")

    # Load the joblib file
    try:
        track_data = joblib.load(file_path)
        if not isinstance(track_data, dict):
            print("Error: Loaded data is not a dictionary")
            return False
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

    # Check required keys
    required_keys = ["section_features", "semantic_labels", "rms", "rms_times"]
    missing_keys = [k for k in required_keys if k not in track_data]
    if missing_keys:
        print(f"Error: File missing required keys: {missing_keys}")
        return False

    # Store the original labels, colors, and section starts
    original_section_features = track_data.get("section_features", [])
    original_semantic_labels = track_data.get("semantic_labels", [])
    original_label_colors = track_data.get("label_colors", [])
    original_section_starts = track_data.get("section_starts", [])

    # Check if features are present
    if not original_section_features or not original_semantic_labels:
        print("Error: No sections found in file")
        return False

    # Check for existing relative_rms (just for logging)
    has_relative_rms = False
    if (
        isinstance(original_section_features, list)
        and len(original_section_features) > 0
    ):
        if isinstance(original_section_features[0], dict):
            has_relative_rms = "relative_rms" in original_section_features[0]

    print(
        f"File has {len(original_section_features)} sections, relative_rms present: {has_relative_rms}"
    )

    # Always recalculate features if force_update is True
    if has_relative_rms and not force_update:
        print("Relative RMS feature already present - no update needed")
        return True

    # Run extract_section_features to get updated feature dictionaries
    try:
        print("Recalculating ALL section features...")
        updated_features, _ = extract_section_features(track_data)

        # Check if feature calculation was successful
        if not updated_features or len(updated_features) != len(
            original_section_features
        ):
            print(
                f"Error: Feature extraction returned {len(updated_features) if updated_features else 0} sections, expected {len(original_section_features)}"
            )
            return False

        # Verify that relative_rms was added (as a spot check)
        if updated_features and isinstance(updated_features[0], dict):
            if "relative_rms" not in updated_features[0]:
                print("Error: relative_rms still missing after feature extraction")
                return False
            else:
                print(
                    f"Successfully added relative_rms feature (first value: {updated_features[0]['relative_rms']})"
                )
                # Debug: Print all keys in the first section feature dict
                print(f"All feature keys: {sorted(updated_features[0].keys())}")

        # Update track_data with new features
        track_data["section_features"] = updated_features

        # Make sure we preserve the original labels
        track_data["semantic_labels"] = original_semantic_labels
        track_data["label_colors"] = original_label_colors
        track_data["section_starts"] = original_section_starts

        # If dry run, don't save
        if dry_run:
            print("Dry run - not saving changes")
            return True

        # Save the updated data
        if output_folder:
            # Save to new location
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            if os.path.normpath(output_path) == os.path.normpath(file_path):
                output_path = output_path.replace(".joblib", "_updated.joblib")
        else:
            # Overwrite existing file
            output_path = file_path

        print(f"Saving to: {output_path}")
        joblib.dump(track_data, output_path, compress=3)
        print("Save successful")
        return True

    except Exception as e:
        print(f"Error during feature calculation or save: {e}")
        traceback.print_exc()
        return False


def process_folder(folder_path, output_folder=None, dry_run=False, force_update=True):
    """Process all joblib files in a folder"""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return

    # Get all joblib files
    joblib_files = [f for f in os.listdir(folder_path) if f.endswith(".joblib")]

    if not joblib_files:
        print(f"No joblib files found in {folder_path}")
        return

    print(f"Found {len(joblib_files)} joblib files to process")

    # Process each file
    success_count = 0
    for file_name in tqdm(joblib_files):
        file_path = os.path.join(folder_path, file_name)
        if update_features_in_file(file_path, output_folder, dry_run, force_update):
            success_count += 1

    print(
        f"\nComplete! Successfully processed {success_count} of {len(joblib_files)} files"
    )


def browse_folder():
    """Opens a folder browser dialog and returns the selected path"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder")
    root.destroy()
    return folder_path


def browse_file():
    """Opens a file browser dialog and returns the selected path"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Joblib File",
        filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
    )
    root.destroy()
    return file_path


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Process command line arguments
        parser = argparse.ArgumentParser(
            description="Update feature values in joblib files"
        )
        parser.add_argument("--path", help="Path to joblib file or folder")
        parser.add_argument(
            "--output",
            help="Output folder (if not provided, will overwrite original files)",
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Run without saving changes"
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force recalculation of all features, even if relative_rms exists",
        )
        parser.add_argument(
            "--ui", action="store_true", help="Launch the graphical user interface"
        )
        args = parser.parse_args()

        # Launch UI if requested
        if args.ui:
            root = tk.Tk()
            app = SectionFeatureUpdater(root)
            root.mainloop()
        elif args.path:
            # Command line mode
            path = args.path
            force_update = args.force  # Defaults to False in argparse

            if os.path.isdir(path):
                print(f"Processing folder: {path}")
                process_folder(path, args.output, args.dry_run, force_update)
            elif os.path.isfile(path) and path.endswith(".joblib"):
                print(f"Processing single file: {path}")
                update_features_in_file(path, args.output, args.dry_run, force_update)
            else:
                print(f"Error: {path} is not a valid joblib file or folder")
        else:
            print(
                "Error: No path specified. Use --ui to launch the graphical interface."
            )
    else:
        # No command line args - launch UI
        root = tk.Tk()
        app = SectionFeatureUpdater(root)
        root.mainloop()
