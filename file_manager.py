# =============================================================================
# FILE: file_manager.py
# Purpose: Handles file selection, loading, and saving of analysis data.
# Added missing ttk import.
# FIXED: Convert defaultdict to dict after loading to prevent pickle error.
# =============================================================================

import os
import joblib
import tkinter as tk
from tkinter import ttk  # *** ADDED missing import ***
from tkinter import filedialog, messagebox
from collections import defaultdict
import traceback
import numpy as np  # Added for type checking during load validation

# Define constants directly or import if needed elsewhere
PERFECT_SUBFOLDER = "Perfect"
WIP_SUBFOLDER = "WIP"
# ANALYSIS_BASE_FOLDER needs to be passed or defined consistently
# For now, hardcode to match main_app.py - consider better config later
# *** Ensure this path is correct for your setup ***
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"  # Corrected path


class FileManager:
    """Handles file selection, loading, and saving operations."""

    def __init__(self, app_instance):
        """
        Initialize the FileManager.

        Args:
            app_instance (AudioAnalyzerApp): Reference to the main application instance.
        """
        self.app = app_instance
        self.analysis_base_folder = ANALYSIS_BASE_FOLDER  # Store base path

    def select_file(self, track_num):
        """Handles file selection for a given track number."""
        app = self.app  # Use local reference for clarity
        f_path = filedialog.askopenfilename(
            title=f"Select Track {track_num} Audio",  # Updated title to be more generic
            filetypes=[
                ("Audio files", "*.wav *.mp3"),  # Add MP3 to the main audio type
                ("WAV files", "*.wav"),  # Keep WAV option for backwards compatibility
                ("MP3 files", "*.mp3"),  # Add dedicated MP3 option
                ("All files", "*.*"),
            ],
        )
        if f_path:
            app.file_path[track_num] = f_path
            filename = os.path.basename(f_path)
            app.track_names[track_num] = filename
            label = app.file_label1 if track_num == 1 else app.file_label2
            if label:
                label.config(text=filename)

            app.playback_manager.stop()
            if hasattr(app, "status_label") and app.status_label:
                app.status_label.config(text="")
            # Use plot manager methods
            app.plot_manager.clear_plots()
            app.plot_manager.add_placeholder_labels()
            app.update_analyze_button_state()
            app.use_manual_bpm.set(False)
            if hasattr(app, "manual_bpm_check") and app.manual_bpm_check:
                app.manual_bpm_check.config(state=tk.DISABLED)
            app.update_manual_bpm_state()
            if app.toggle_labels_button:
                app.toggle_labels_button.config(state=tk.DISABLED)
            if app.save_button:
                app.save_button.config(state=tk.DISABLED)
            if app.section_editor:
                app.section_editor.clear()
            app.show_pre_cleanup_labels_var.set(False)
            app.show_hmm_var.set(False)  # Reset HMM view toggle

            # Clear previous data for this track, including HMM results
            app.track_data[track_num] = None
            if app.track_data.get(1):  # Check if track 1 still exists
                app.track_data[1].pop("hmm_semantic_labels", None)
                app.track_data[1].pop("hmm_label_colors", None)
                app.track_data[1].pop("hmm_section_starts", None)

            if app.mode.get() == "single" and track_num == 1:
                app.track_data[2] = None  # Clear track 2 if in single mode

            app._update_hmm_button_state()  # Ensure HMM buttons are disabled
            app._update_playback_buttons_state_from_manager("stopped")
        else:
            label = app.file_label1 if track_num == 1 else app.file_label2
            if label:
                label.config(text="Selection cancelled.")

    def _ask_save_status(self, parent):
        """Creates a modal dialog asking user to classify the save."""
        dialog = tk.Toplevel(parent)
        dialog.title("Save Analysis Status")
        dialog.transient(parent)
        dialog.grab_set()
        dialog.resizable(False, False)
        status_var = tk.StringVar(value="")
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        dialog.update_idletasks()
        dialog_w = dialog.winfo_width()
        dialog_h = dialog.winfo_height()
        center_x = parent_x + (parent_w // 2) - (dialog_w // 2)
        center_y = parent_y + (parent_h // 2) - (dialog_h // 2)
        dialog.geometry(f"+{center_x}+{center_y}")
        frame = ttk.Frame(dialog, padding="15")
        frame.pack(expand=True, fill="both")  # Uses ttk
        ttk.Label(frame, text="Save analysis as:", font="-weight bold").pack(
            pady=(0, 10)
        )  # Uses ttk
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=5)  # Uses ttk

        def set_status(status):
            status_var.set(status)
            dialog.destroy()

        # Uses ttk Button
        perfect_btn = ttk.Button(
            button_frame,
            text="Perfect (Training)",
            width=20,
            command=lambda: set_status(PERFECT_SUBFOLDER),
        )
        perfect_btn.pack(side=tk.LEFT, padx=10)
        wip_btn = ttk.Button(
            button_frame,
            text="WIP (Work in Progress)",
            width=20,
            command=lambda: set_status(WIP_SUBFOLDER),
        )
        wip_btn.pack(side=tk.LEFT, padx=10)
        parent.wait_window(dialog)
        return status_var.get()

    def save_analysis(self):
        """Saves analysis results, overwriting previous file for the same track in Perfect/WIP."""
        app = self.app
        print("DEBUG FileManager: Save Analysis called.")
        if app.mode.get() != "single" or not app.track_data.get(1):
            messagebox.showwarning(
                "Save Error",
                "Analysis data for Track 1 must exist and be in Single Track mode to save.",
            )
            return

        save_status = self._ask_save_status(app.master)  # Pass main window as parent
        if not save_status:
            print("DEBUG FileManager: Save status selection cancelled.")
            app.status_label.config(text="Save Cancelled", foreground="orange")
            return

        print(f"DEBUG FileManager: User selected save status: {save_status}")
        target_directory = os.path.join(self.analysis_base_folder, save_status)
        print(f"DEBUG FileManager: Target save directory: {target_directory}")
        try:
            os.makedirs(target_directory, exist_ok=True)
            print(f"DEBUG FileManager: Ensured directory '{target_directory}' exists.")
        except OSError as e:
            messagebox.showerror(
                "Save Error",
                f"Could not create save directory:\n{target_directory}\nError: {e}",
            )
            app.status_label.config(text="Save Error!", foreground="red")
            return

        try:
            base_audio_name = "track"
            if app.file_path.get(1):
                base_audio_name = os.path.splitext(os.path.basename(app.file_path[1]))[
                    0
                ]
            safe_base_name = "".join(
                c for c in base_audio_name if c.isalnum() or c in (" ", "_", "-")
            ).rstrip()
            fixed_filename = f"{safe_base_name}.analysis.joblib"
            save_path = os.path.join(target_directory, fixed_filename)
            print(f"DEBUG FileManager: Determined save path: {save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error creating filename: {e}")
            app.status_label.config(text="Save Error!", foreground="red")
            return

        confirm_overwrite = True
        if os.path.exists(save_path):
            print(f"DEBUG FileManager: File exists: {save_path}")
            confirm_overwrite = messagebox.askyesno(
                "Confirm Overwrite",
                f"Analysis file already exists:\n{save_path}\n\nOverwrite?",
            )
        if not confirm_overwrite:
            print("DEBUG FileManager: Overwrite cancelled by user.")
            app.status_label.config(text="Save Cancelled", foreground="orange")
            return

        try:
            print(f"DEBUG FileManager: Saving analysis data to: {save_path}")
            # *** IMPORTANT: Ensure data_to_save is a standard dict ***
            data_to_save = app.track_data[1]
            # Check if it's still a defaultdict (shouldn't be if load was fixed)
            if isinstance(data_to_save, defaultdict):
                print(
                    "DEBUG FileManager: Converting defaultdict to dict before saving."
                )
                data_to_save = dict(data_to_save)  # Convert just in case

            joblib.dump(data_to_save, save_path, compress=3)
            messagebox.showinfo(
                "Save Successful", f"Analysis saved successfully to:\n{save_path}"
            )
            app.status_label.config(text="Analysis Saved", foreground="green")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save analysis file:\n{e}")
            traceback.print_exc()
            app.status_label.config(text="Save Failed!", foreground="red")

    def load_analysis(self):
        """Loads previously saved analysis results."""
        app = self.app
        print("DEBUG FileManager: Load Analysis called.")
        initial_dir = (
            self.analysis_base_folder
            if os.path.isdir(self.analysis_base_folder)
            else "."
        )
        load_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Analysis File",
            filetypes=[("Joblib Analysis Files", "*.joblib"), ("All Files", "*.*")],
        )
        if not load_path:
            print("DEBUG FileManager: Load cancelled by user.")
            return

        app.playback_manager.stop()
        app.plot_manager.clear_plots()  # Use plot manager
        app.status_label.config(text="Loading...", foreground="orange")
        app.master.update_idletasks()
        try:
            print(f"DEBUG FileManager: Loading analysis data from: {load_path}")
            loaded_data_raw = joblib.load(load_path)  # Load raw data first
            if not isinstance(loaded_data_raw, dict):
                raise TypeError(
                    "Loaded file does not contain a valid analysis dictionary."
                )

            # *** FIX: Convert to standard dict immediately after loading ***
            loaded_data = dict(loaded_data_raw)
            print("DEBUG FileManager: Converted loaded data to standard dict.")

            # Basic check for essential keys
            required_keys = [
                "file_path",
                "sr",
                "y_processed",
                "semantic_labels",
                "section_features",
            ]
            if not all(key in loaded_data for key in required_keys):
                missing = [key for key in required_keys if key not in loaded_data]
                raise ValueError(f"Loaded data is missing essential keys: {missing}")
            print("DEBUG FileManager: Loaded data verified.")

            # Reset app state before loading
            app.track_data = {1: None, 2: None}
            app.file_path = {1: None, 2: None}
            app.track_data[1] = loaded_data  # Store the standard dict

            loaded_file_path = app.track_data[1].get("file_path")
            original_audio_found = False
            if loaded_file_path and os.path.exists(loaded_file_path):
                app.file_path[1] = loaded_file_path
                app.track_names[1] = os.path.basename(loaded_file_path)
                app.file_label1.config(text=app.track_names[1])
                print(
                    f"DEBUG FileManager: Original audio file found at: {loaded_file_path}"
                )
                original_audio_found = True
            else:
                print(
                    f"Warning: Original audio file not found at path stored in analysis: {loaded_file_path}"
                )
                messagebox.showwarning(
                    "Audio File Missing",
                    f"The original audio file path stored in the analysis was:\n{loaded_file_path}\n\nThis file could not be found. Playback requires the original audio. Please re-select the audio file if you wish to use playback.",
                )
                app.file_path[1] = None
                app.track_names[1] = (
                    f"Loaded: {os.path.basename(load_path)} (Audio Missing)"
                )
                app.file_label1.config(text=app.track_names[1])

            if original_audio_found:
                app.playback_manager.set_audio(
                    app.track_data[1].get("y_processed"), app.track_data[1].get("sr")
                )
            else:
                app.playback_manager.set_audio(None, None)

            # Update UI to reflect loaded state (single track mode)
            app.mode.set("single")
            app.update_ui_for_mode()  # This updates UI and clears plots again via plot_manager
            app.plot_manager.display_analysis_results()  # Use plot manager to display loaded results
            app.status_label.config(text="Analysis Loaded", foreground="green")

            # Update button states
            app._update_save_button_state()
            app._update_toggle_button_state()
            app._update_hmm_button_state()
            if app.track_data[1].get("bpm"):
                if hasattr(app, "manual_bpm_check") and app.manual_bpm_check:
                    app.manual_bpm_check.config(
                        state=tk.NORMAL
                    )  # Check if widget exists
                bpm_used = app.track_data[1].get("bpm", "")
                if hasattr(app, "manual_bpm_entry_var"):
                    app.manual_bpm_entry_var.set(
                        f"{bpm_used:.2f}"
                        if isinstance(bpm_used, (int, float, np.number))
                        else ""
                    )  # Check type
                app.update_manual_bpm_state()

            print("DEBUG FileManager: Load analysis finished successfully.")

        except Exception as e:
            messagebox.showerror(
                "Load Error", f"Failed to load or display analysis file:\n{e}"
            )
            traceback.print_exc()
            app.status_label.config(text="Load Failed!", foreground="red")
            # Reset state on error
            app.plot_manager.clear_plots()
            app.track_data = {1: None, 2: None}
            app.file_path = {1: None, 2: None}
            app.track_names = {1: "Track 1", 2: "Track 2"}
            app.file_label1.config(text="No file selected")
            app.file_label2.config(text="No file selected")
            app.update_ui_for_mode()
