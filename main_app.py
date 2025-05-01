# FILE: main_app.py
# Contains the Tkinter GUI Application class and the main execution block.
# Imports and uses functions from other modules.
# GUI building is handled by gui_builder.py
# HMM prediction handled by hmm_predictor.py
# Plot display handled by plot_manager.py
# File operations handled by file_manager.py
# Added section splitting functionality (Shift+Click on Waveform with confirmation).
# Fixed SyntaxErrors in _analyze_single_track and run_analysis.
# Fixed ImportError by defining _recalculate_section_features as a class method.
# Fixed AttributeError for checking Shift key in _on_waveform_click.
# =============================================================================

"""
Main application module for the Audio Analyzer Tool.

This module defines the main Tkinter application class `AudioAnalyzerApp` which
orchestrates the GUI, analysis processes, plotting, playback, and file operations.
It integrates functionalities from various other modules within the project.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt # Keep for plt.close('all')
import os
import traceback
from collections import defaultdict, Counter
import numpy as np
import scipy # For audio analysis, needed by merge recalc
import threading # For audio playback thread and lock
import datetime # For timestamp in saved filenames
import joblib # For saving/loading analysis data AND HMM model
import math # Added for isnan check
import librosa # For audio analysis, needed by merge recalc
import librosa.display # For audio analysis, needed by merge recalc
import copy # Needed for deepcopy during split

# --- Import functions/classes from other modules ---
try:
    from audio_analysis_wrapper import (
        load_and_preprocess,
        detect_sections,
        analyze_chroma_and_clusters,
        analyze_stereo_and_hpss,
        results_on_failure,
        calculate_bar_features,
        extract_section_features,
    )

    # import audio_plotting as ap # Now imported by PlotManager
    from playback_manager import PlaybackManager
    from section_editor import SectionEditor, ALLOWED_LABELS, COLOR_NAME_MAP, HEX_TO_COLOR_NAME
    # Import the corrected predictor class
    from hmm_predictor import HMMPredictor
    from gui_builder import build_gui
    from plot_manager import PlotManager
    # *** Import the new FileManager ***
    from file_manager import FileManager

except ImportError as e:
    print(f"Import Error: {e}\n"
          f"Could not import required modules.\n"
          f"Ensure all .py files are present and correct.")
    # Attempt to show GUI error box even if imports fail early
    try:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Import Error", f"Could not import required modules.\nEnsure all .py files are present.\n\nError: {e}")
        root.destroy()
    except tk.TclError:
        pass
    exit()

# --- Constants ---
# Base folder paths
# *** Ensure these paths are correct for your system ***
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"  # <-- CORRECTED PATH
PROJECT_BASE_FOLDER = os.path.dirname(
    ANALYSIS_BASE_FOLDER
)  # This should update correctly now
HMM_OUTPUT_FOLDER = os.path.join(
    PROJECT_BASE_FOLDER, "hmm_model"
)  # This should update correctly now

# Subfolder names (used by FileManager/SectionEditor)
PERFECT_SUBFOLDER = "Perfect"
WIP_SUBFOLDER = "WIP"

# *** Paths for the GMMHMM model and auxiliary data ***
# *** Make sure these match the output filenames in gmm_hmm_trainer.py ***
# Set N_FEATURES and N_MIXTURES based on the trainer script used
N_FEATURES_EXPECTED = 7 # Set based on the latest trainer (10 features)
N_MIXTURES_EXPECTED = 1 # Set based on the latest trainer (3 mixtures)
# *** ADD a variable for the cleaning flags used in the desired model ***
CLEANING_FLAGS_EXPECTED = "out_sh_con"  # Set to match the filename flags

# Construct the filename parts
feature_str = f"{N_FEATURES_EXPECTED}f"
mixture_str = f"{N_MIXTURES_EXPECTED}m"
# *** Construct the base filename including the cleaning flags ***
base_filename = f"gmmhmm_{feature_str}_{mixture_str}_{CLEANING_FLAGS_EXPECTED}"

# Construct the full paths using the base filename
GMMHMM_MODEL_PATH = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_model.joblib")
GMMHMM_AUX_PATH = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_aux.joblib")

# Minimum section duration allowed after a split (in seconds)
MIN_SPLIT_SECTION_DURATION_SEC = 1.0 # Adjust as needed


# --- Tkinter GUI Application Class ---
class AudioAnalyzerApp:
    """
    Main application class for the Audio Analyzer Tool GUI.

    This class integrates various components like audio analysis, HMM prediction,
    plotting, playback, file management, and section editing into a cohesive
    Tkinter-based graphical user interface. It manages the application state,
    orchestrates analysis workflows, and handles user interactions.

    Attributes:
        master (tk.Tk): The main Tkinter root window.
        file_path (dict): Stores the paths to the selected audio files for track 1 and 2.
        track_data (dict): Stores the analysis results dictionaries for track 1 and 2.
        track_names (dict): Display names for the tracks.
        plot_widgets (dict): Dictionary managed by PlotManager containing plot canvases/toolbars.
        playback_manager (PlaybackManager): Instance managing audio playback.
        plot_manager (PlotManager): Instance managing plot creation and updates.
        hmm_predictor (HMMPredictor): Instance managing HMM model loading and prediction.
        file_manager (FileManager): Instance managing file loading, saving, and selection.
        section_editor (SectionEditor): Instance managing the section editing interface.
        mode (tk.StringVar): Tkinter variable controlling 'single' or 'compare' mode.
        analysis_vars (dict): Dictionary of tk.BooleanVar for analysis option checkboxes.
        use_manual_bpm (tk.BooleanVar): Tkinter variable for manual BPM checkbox.
        manual_bpm_entry_var (tk.StringVar): Tkinter variable for manual BPM entry field.
        show_pre_cleanup_labels_var (tk.BooleanVar): Tkinter variable for toggling label view.
        show_hmm_var (tk.BooleanVar): Tkinter variable for toggling HMM result view.
        style (ttk.Style): Tkinter ttk style object.
        # Other attributes are GUI widgets initialized by build_gui
    """
    def __init__(self, master):
        """Initializes the AudioAnalyzerApp GUI application.

        Sets up the main window, initializes state variables, instantiates
        manager classes (Playback, Plot, HMM, File), creates Tkinter control
        variables, builds the GUI using `gui_builder.build_gui`, and sets
        the initial UI state.

        Args:
            master (tk.Tk): The root Tkinter window for the application.

        """
        self.master = master
        master.title("Audio Analyzer Tool")
        master.geometry("1100x900")
        master.protocol("WM_DELETE_WINDOW", self._on_closing) # Handle window close event

        # --- Application State Variables ---
        self.file_path = {1: None, 2: None} # Paths for track 1 and track 2 audio files
        self.track_data = {1: None, 2: None} # Stores analysis results dictionaries
        self.track_names = {1: "Track 1", 2: "Track 2"} # Display names
        self.plot_widgets = {} # Populated by PlotManager with canvas/toolbar widgets

        # --- GUI Widget Placeholders (initialized by build_gui) ---
        self.control_frame = None; self.notebook = None; self.tabs = {}
        self.status_label = None; self.toggle_labels_button = None
        self.show_hmm_button = None; self.play_pause_button = None
        self.stop_button = None; self.select_button1 = None; self.file_label1 = None
        self.select_button2 = None; self.file_label2 = None
        self.analyze_button = None; self.hmm_predict_button = None
        self.load_button = None; self.save_button = None
        self.manual_bpm_frame = None; self.manual_bpm_check = None
        self.manual_bpm_entry = None; self.update_bpm_button = None
        self.section_editor = None; self.waveform_summary_label = None
        self.playhead_line = None # Matplotlib line object for playback position

        # --- Instantiate Managers ---
        self.playback_manager = PlaybackManager(
            master=self.master,
            on_state_change=self._update_playback_buttons_state_from_manager,
            on_position_update=self._update_playhead_display
        )
        self.plot_manager = PlotManager(self) # Pass app instance for access
        self.hmm_predictor = HMMPredictor(GMMHMM_MODEL_PATH, GMMHMM_AUX_PATH)
        self.file_manager = FileManager(self) # Pass app instance

        # --- Tkinter Control Variables ---
        self.mode = tk.StringVar(value="single") # 'single' or 'compare'
        self.analysis_labels = {
            "sections": "Sections", "chroma": "Chroma/Labeling", "hpss": "HPSS",
            "stereo": "Stereo Width", "low_end": "Low-End Energy",
            "dyn_range": "Dynamic Range", "band_plot": "Band Analysis Plot"
        }
        # Default analysis options checked
        self.analysis_vars = {k: tk.BooleanVar(value=(k in ["sections", "chroma", "low_end"])) for k in self.analysis_labels.keys()}
        self.use_manual_bpm = tk.BooleanVar(value=False)
        self.manual_bpm_entry_var = tk.StringVar()
        self.show_pre_cleanup_labels_var = tk.BooleanVar(value=False) # Toggle between final/pre-cleanup labels
        self.show_hmm_var = tk.BooleanVar(value=False) # Toggle between original/HMM labels

        # --- Build GUI using the builder ---
        self.style = ttk.Style(); self.style.theme_use('clam') # Apply a theme
        build_gui(self) # Call the external function to populate the GUI

        # --- Initial UI State ---
        self.update_ui_for_mode()
        self.update_manual_bpm_state()

    # --- GUI Building Methods Removed (Moved to gui_builder.py) ---
    # --- Plot Management Methods Removed (Moved to plot_manager.py) ---
    # --- File Operation Methods Removed (Moved to file_manager.py) ---

    # --- UI Update Methods (Remain in main app) ---
    def update_manual_bpm_state(self):
        """Enables or disables the manual BPM entry and update button.

        The controls are enabled only if the 'Use Manual BPM' checkbox is checked
        AND analysis data exists for track 1.
        """
        analysis_done = bool(self.track_data.get(1)) # Check if track 1 data exists
        state = tk.NORMAL if self.use_manual_bpm.get() and analysis_done else tk.DISABLED
        # Check if widgets exist before configuring (robustness)
        if hasattr(self, 'manual_bpm_entry') and self.manual_bpm_entry:
            self.manual_bpm_entry.config(state=state)
        if hasattr(self, 'update_bpm_button') and self.update_bpm_button:
            self.update_bpm_button.config(state=state)

    def update_ui_for_mode(self):
        """Updates UI elements visibility and state based on Single/Compare mode.

        Shows/hides Track 2 controls, resets data, updates button labels,
        clears plots, and resets various UI states when the mode changes.
        """
        mode = self.mode.get(); is_compare = mode == "compare"
        analyze_btn_text = "Analyze & Compare" if is_compare else "Analyze Track"
        bpm_label_text = "Manual Tempo (Track 1)" if is_compare else "Manual Tempo"

        # Update button text and frame label
        if hasattr(self, 'analyze_button') and self.analyze_button:
            self.analyze_button.config(text=analyze_btn_text)
        if hasattr(self, 'manual_bpm_frame') and self.manual_bpm_frame:
            self.manual_bpm_frame.config(text=bpm_label_text)

        # Show/hide Track 2 controls
        if hasattr(self, 'select_button2') and self.select_button2 and hasattr(self, 'file_label2') and self.file_label2:
            if is_compare:
                self.select_button2.grid()
                self.file_label2.grid()
            else:
                # Hide and reset Track 2 data if switching away from compare mode
                self.select_button2.grid_remove()
                self.file_label2.grid_remove()
                self.file_path[2] = None
                self.track_data[2] = None
                if self.file_label2: self.file_label2.config(text="No file selected")

        # Reset states and plots
        self.playback_manager.stop()
        self.update_analyze_button_state()
        self.plot_manager.clear_plots()
        self.plot_manager.add_placeholder_labels()
        if hasattr(self, 'manual_bpm_check') and self.manual_bpm_check:
            self.manual_bpm_check.config(state=tk.DISABLED) # Disable until analysis
        self.update_manual_bpm_state()
        if self.toggle_labels_button:
            self.toggle_labels_button.config(state=tk.DISABLED)
        if self.section_editor:
            self.section_editor.clear()
            self.section_editor.update_button.config(state=tk.DISABLED)
        self.show_pre_cleanup_labels_var.set(False)
        self.show_hmm_var.set(False)
        self._update_playback_buttons_state_from_manager('stopped') # Reset playback buttons

    def update_analyze_button_state(self):
        """Enables or disables the 'Analyze' button based on file selection(s).

        Requires Track 1 file in single mode, or both Track 1 and Track 2 files
        in compare mode. Also updates save and HMM button states.
        """
        mode = self.mode.get(); state = tk.DISABLED
        if mode == "single" and self.file_path.get(1):
            state = tk.NORMAL
        elif mode == "compare" and self.file_path.get(1) and self.file_path.get(2):
            state = tk.NORMAL

        if hasattr(self, 'analyze_button') and self.analyze_button:
            self.analyze_button.config(state=state)

        # Update dependent buttons
        self._update_save_button_state()
        self._update_hmm_button_state()

    def _update_save_button_state(self):
        """Enables the 'Save Analysis' button.

        Enabled only when in 'single' mode and analysis data exists for Track 1.
        """
        state = tk.DISABLED
        if self.mode.get() == "single" and self.track_data.get(1):
            state = tk.NORMAL
        if self.save_button:
            self.save_button.config(state=state)

    def _update_hmm_button_state(self):
        """Enables/disables HMM Predict and Show HMM buttons.

        Predict button is enabled if in single mode, track data exists, and the
        required GMM-HMM model and auxiliary files are found.
        Show HMM button is enabled if HMM prediction results already exist in the
        current track data.
        """
        predict_state = tk.DISABLED
        show_state = tk.DISABLED
        print("DEBUG HMM Button Update: Checking state...") # DEBUG PRINT

        # Check mode and track data
        is_single_mode = self.mode.get() == 'single'
        has_track_data = bool(self.track_data.get(1))
        print(f" -> Single Mode: {is_single_mode}, Has Track 1 Data: {has_track_data}") # DEBUG PRINT

        if is_single_mode and has_track_data:
            # Check if the GMMHMM model files exist
            print(f" -> Checking Model Path: {GMMHMM_MODEL_PATH}") # DEBUG
            print(f" -> Checking Aux Path:   {GMMHMM_AUX_PATH}") # DEBUG
            model_exists = os.path.exists(GMMHMM_MODEL_PATH)
            aux_exists = os.path.exists(GMMHMM_AUX_PATH)
            print(f" -> Model Exists: {model_exists}, Aux Exists: {aux_exists}") # DEBUG PRINT

            if model_exists and aux_exists:
                print(" -> Enabling Predict Button.") # DEBUG PRINT
                predict_state = tk.NORMAL
            else:
                print(f"INFO: GMMHMM model/aux files not found. Predict button disabled.")
                # Keep predict_state as DISABLED

            # Check if HMM results already exist in track_data to enable Show HMM button
            if 'hmm_section_starts' in self.track_data[1]:
                show_state = tk.NORMAL
                print(" -> Enabling Show HMM Button (results found).") # DEBUG PRINT
            else:
                 print(" -> Disabling Show HMM Button (no results found).") # DEBUG PRINT
                 # Keep show_state as DISABLED

        # Apply states to buttons if they exist
        if hasattr(self, 'hmm_predict_button') and self.hmm_predict_button:
            self.hmm_predict_button.config(state=predict_state)
        if hasattr(self, 'show_hmm_button') and self.show_hmm_button:
            self.show_hmm_button.config(state=show_state)

        print("DEBUG HMM Button Update: Finished.") # DEBUG PRINT

    # --- Analysis Orchestration (Remains in main app) ---
    def run_analysis(self, is_update=False, manual_bpm_val=None):
        """Orchestrates the analysis and plotting process for selected tracks.

        Handles both initial analysis and re-analysis triggered by manual BPM updates.
        Manages UI state updates (status bar, button disabling), calls the
        core `_analyze_single_track` method, handles potential errors, updates
        playback data, and triggers plot display upon completion.

        Args:
            is_update (bool): True if called for a manual BPM update, False for
                              initial analysis. Defaults to False.
            manual_bpm_val (float, optional): The manual BPM value to use if
                                              `is_update` is True. Defaults to None.

        Raises:
            RuntimeError: If analysis fails critically for a required track.
                          (Caught internally and shown via messagebox).
        """
        mode = self.mode.get()
        # --- Pre-Analysis Checks and UI Updates ---
        if not is_update:
            # Initial analysis checks
            if mode == "single" and not self.file_path[1]:
                messagebox.showwarning("Missing File", "Please select Track 1.")
                return
            if mode == "compare" and not (self.file_path[1] and self.file_path[2]):
                messagebox.showwarning("Missing Files", "Please select both Track 1 and Track 2.")
                return
            # Reset UI for new analysis
            self.playback_manager.stop()
            self.status_label.config(text="Analyzing...", foreground="orange")
            self.master.update_idletasks()
            plt.close('all') # Close any previous matplotlib figures
            self.plot_manager.clear_plots()
            self.track_data = {1: None, 2: None} # Clear previous data
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False)
            self.show_hmm_var.set(False)
            self._update_hmm_button_state()
        else:
            # Re-analysis (BPM update) checks
            self.playback_manager.stop()
            self.status_label.config(text="Re-analyzing with new BPM...", foreground="orange")
            self.master.update_idletasks()
            plt.close('all')
            self.plot_manager.clear_plots()
            if manual_bpm_val is None:
                messagebox.showerror("Update Error", "Manual BPM value missing for update.")
                self.status_label.config(text="Update Error!", foreground="red")
                return
            # Reset relevant UI states for re-analysis
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False)
            self.show_hmm_var.set(False)
            self._update_hmm_button_state()

        # --- Run Analysis Core Logic ---
        try:
            print("--- Analyzing Track 1 ---")
            # Pass manual BPM only if it's an update
            bpm_override_t1 = manual_bpm_val if is_update else None
            results1 = self._analyze_single_track(1, bpm_override_t1)
            if results1 is None:
                # Error handled within _analyze_single_track via messagebox
                raise RuntimeError("Analysis failed for Track 1. Check console for details.")
            self.track_data[1] = results1
            # Set audio for playback immediately after T1 analysis
            self.playback_manager.set_audio(results1.get('y_processed'), results1.get('sr'))

            if mode == "compare":
                if not is_update:
                    # Analyze Track 2 only on initial compare analysis
                    print("\n--- Analyzing Track 2 ---")
                    results2 = self._analyze_single_track(2) # No BPM override for T2
                    if results2 is None:
                        raise RuntimeError("Analysis failed for Track 2. Check console for details.")
                    self.track_data[2] = results2
                else:
                    # Keep existing Track 2 data during BPM update for Track 1
                    print("--- Keeping existing Track 2 analysis ---" if self.track_data.get(2) else "--- Track 2 data missing ---")

        except Exception as e:
            error_msg = f"Analysis Error: {e}"
            self.status_label.config(text="Error!", foreground="red")
            print(error_msg)
            traceback.print_exc()
            messagebox.showerror("Analysis Error", f"{error_msg}\n\nCheck console output for more details.")
            # Reset UI on error
            self.plot_manager.clear_plots()
            self.plot_manager.add_placeholder_labels()
            plt.close('all')
            if hasattr(self.manual_bpm_check, 'config'): self.manual_bpm_check.config(state=tk.DISABLED)
            self.update_manual_bpm_state()
            self._update_playback_buttons_state_from_manager('stopped')
            self._update_save_button_state()
            self._update_hmm_button_state()
            return # Stop execution

        # --- Post-Analysis UI Updates ---
        self.plot_manager.display_analysis_results() # Display plots
        final_status = "BPM Update Complete" if is_update else "Analysis Complete"
        self.status_label.config(text=final_status, foreground="green")

        # Enable relevant controls now that analysis is done
        if self.track_data.get(1):
             if hasattr(self.manual_bpm_check, 'config'): self.manual_bpm_check.config(state=tk.NORMAL)
             # Set manual BPM entry to the detected value after initial analysis
             if not is_update:
                 bpm_used = self.track_data[1].get('bpm', '')
                 self.manual_bpm_entry_var.set(f"{bpm_used:.2f}" if isinstance(bpm_used, (int, float)) else "")
             # Enable label toggle if pre-cleanup labels exist
             if self.toggle_labels_button and self.track_data[1].get('labels_before_cleanup'):
                 self.toggle_labels_button.config(state=tk.NORMAL)
             elif self.toggle_labels_button:
                 self.toggle_labels_button.config(state=tk.DISABLED) # Disable if no pre-cleanup data
             # Enable section editor update button in single mode
             if self.section_editor and mode == "single":
                 self.section_editor.update_button.config(state=tk.NORMAL)

        self.update_manual_bpm_state() # Re-evaluate manual bpm entry state
        self.update_analyze_button_state() # Updates save/HMM buttons too
        print("Analysis/Update process finished.")

    def _analyze_single_track(self, track_num, manual_bpm_override=None):
        """Runs the complete analysis pipeline for a single track.

        This involves loading, preprocessing, tempo estimation (potentially overridden),
        section detection, chroma analysis, clustering, labeling, HPSS, stereo width,
        low-end energy, dynamic range, and calculation of various derived features.

        Args:
            track_num (int): The track number (1 or 2) to analyze.
            manual_bpm_override (float, optional): If provided, overrides the
                automatic BPM detection. Defaults to None.

        Returns:
            dict or None: A dictionary containing all analysis results for the track,
                          or None if a critical error occurs during processing.
                          Keys include 'y_processed', 'sr', 'bpm', 'section_starts',
                          'semantic_labels', 'section_features', etc.
        """
        results = defaultdict(lambda: None) # Use defaultdict for easier access
        try:
            file_to_analyze = self.file_path.get(track_num)
            if not file_to_analyze:
                print(f"Error: No file path set for Track {track_num}.")
                return None

            print(f" Step 1: Loading/Preprocessing/Tempo (T{track_num})...")
            preprocess_data = load_and_preprocess(file_to_analyze, track_num, manual_bpm_override)
            if preprocess_data is None: return None # Error handled in function
            results.update(preprocess_data)

            # --- Validate essential preprocessing results ---
            bpm = results.get('bpm'); sr = results.get('sr'); y_proc = results.get('y_processed')
            hop = results.get('hop_length'); dur = results.get('duration_processed')
            fp = results.get('file_path'); trim = results.get('trim_offset_sec', 0)
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [bpm, sr, hop, dur, fp]):
                missing = [k for k, v in {'bpm':bpm, 'sr':sr, 'hop_length':hop, 'duration':dur, 'file_path':fp}.items() if v is None or (isinstance(v, float) and math.isnan(v))]
                print(f"Error: Missing essential data after preprocessing T{track_num}: {missing}.")
                messagebox.showerror(f"Preprocessing Error (T{track_num})", f"Missing essential data: {missing}")
                return None
            if y_proc is None or y_proc.size == 0:
                print(f"Error: Processed audio is empty T{track_num}.")
                messagebox.showerror(f"Preprocessing Error (T{track_num})", "Processed audio data is empty.")
                return None

            # --- Calculate intermediate features needed for subsequent steps ---
            print(f" Step 1b: Calculating intermediate features (BPM: {bpm:.2f})...")
            spb = 60.0 / bpm if bpm > 0 else 0 # seconds per beat
            spbar = 4 * spb if spb > 0 else 0 # seconds per bar (assuming 4/4)
            results['seconds_per_bar'] = spbar
            # Estimate beat frame indices
            if dur > 0 and spb > 0:
                beat_times_est = np.arange(0, dur, spb) # Relative beat times
                results['beat_frames'] = librosa.time_to_frames(beat_times_est, sr=sr, hop_length=hop)
            else:
                results['beat_frames'] = np.array([], dtype=int)
            # Calculate frame-based RMS and times
            results['rms'] = np.nan_to_num(librosa.feature.rms(y=y_proc, hop_length=hop)[0])
            results['rms_times'] = librosa.frames_to_time(np.arange(len(results['rms'])), sr=sr, hop_length=hop) # Relative RMS times

            # Calculate average RMS per bar and absolute bar start times
            num_bars = int(np.ceil(dur / spbar)) if spbar > 0 else 0
            bar_rms_list = []; bar_starts_list = []
            rms_d = results.get('rms'); rms_t = results.get('rms_times') # Use relative times here
            if num_bars > 0 and rms_d is not None and rms_t is not None and rms_d.size == rms_t.size:
                for i in range(num_bars):
                    start_rel = i * spbar
                    end_rel = min((i + 1) * spbar, dur)
                    avg_rms_bar = 0.0
                    if start_rel < end_rel:
                        mask = (rms_t >= start_rel) & (rms_t < end_rel)
                        if np.any(mask):
                            valid_rms_in_bar = rms_d[mask][np.isfinite(rms_d[mask])]
                            avg_rms_bar = np.mean(valid_rms_in_bar) if valid_rms_in_bar.size > 0 else 0.0
                    bar_rms_list.append(avg_rms_bar)
                    bar_starts_list.append(start_rel + trim) # Store absolute bar start time
                results['bar_rms_data'] = bar_rms_list
                results['bar_starts_absolute'] = bar_starts_list
            else:
                results['bar_rms_data'] = None
                results['bar_starts_absolute'] = None
                print("Warning: Could not calculate bar RMS/Starts.")

            # --- Section Detection Step ---
            if self.analysis_vars['sections'].get():
                print(f" Step 2: Detecting Sections...")
                if results.get("bar_rms_data") is not None:
                    # Pass the results dict which now contains bar_rms_data
                    results["section_starts"], results["section_labels"] = detect_sections(results)
                else:
                    print(" -> Skipping Section Detection (bar RMS data missing).")
                    results["section_starts"], results["section_labels"] = None, None
            else:
                print(f" Step 2: Skipping Sections (Checkbox unchecked).")
                results["section_starts"]=None
                results["section_labels"]=None

            # --- Chroma/Cluster/Label Step ---
            # Run if checkbox is checked OR if sections were detected (needed for feature calc)
            run_chroma_step = self.analysis_vars['chroma'].get() or (results.get("section_starts") is not None)
            if run_chroma_step:
                 if results.get("section_starts") is not None:
                     print(f" Step 3: Running Chroma/Cluster/Label Analysis...")
                     results.update(analyze_chroma_and_clusters(results)) # Updates results dict in place
                 else:
                     print(" Step 3: Skipping Chroma/Labeling (Sections missing).")
                     results.update(results_on_failure(None)) # Ensure default keys exist
            else:
                print(f" Step 3: Skipping Chroma/Labeling (Checkbox unchecked).")
                results.update(results_on_failure(results.get("section_starts"))) # Ensure default keys exist

            # --- Spectrogram/HPSS/Stereo Step ---
            # Determine if spectrogram is needed based on checked options
            needs_spec = any(self.analysis_vars[k].get() for k in ['stereo','hpss','band_plot','low_end', 'chroma'])
            if needs_spec:
                print(f" Step 4: Running Spectrogram & HPSS Analysis...")
                results.update(analyze_stereo_and_hpss(results, self.analysis_vars['stereo'].get(), self.analysis_vars['hpss'].get()))
            else:
                print(f" Step 4: Skipping Spectrogram/HPSS Analysis.")
                # Ensure keys exist even if skipped
                results.setdefault("spec", None); results.setdefault("freqs", None)
                results.setdefault("times_absolute", None); results.setdefault("width_matrix", None)
                results.setdefault("rms_harm", None); results.setdefault("rms_perc", None)
                results.setdefault("rms_time_absolute", None);

            # --- Low-End Energy Step ---
            if self.analysis_vars['low_end'].get():
                print(f" Step 5: Calculating Low-End Energy...")
                spec=results.get("spec"); freqs=results.get("freqs")
                times_abs=results.get("times_absolute"); trim_offset=results.get("trim_offset_sec", 0)
                if spec is not None and freqs is not None and times_abs is not None and spec.shape[0]==freqs.size and spec.shape[1]==times_abs.size:
                    low_freq_mask = freqs < 150
                    if np.any(low_freq_mask):
                        low_end_energy = np.sum(spec[low_freq_mask,:], axis=0)
                        max_low_energy = np.max(low_end_energy)
                        # Normalize and store
                        results["low_energy_norm"] = (low_end_energy / max_low_energy if max_low_energy > 1e-9 else np.zeros_like(low_end_energy))
                        # Store RELATIVE times for low energy features
                        times_rel = times_abs - trim_offset
                        results["low_energy_times"] = times_rel
                        print(f" -> Stored low_energy_norm and RELATIVE low_energy_times (Length: {len(times_rel)})")
                    else:
                        print(" Warning: No frequencies below 150Hz found.")
                        results["low_energy_norm"]=None; results["low_energy_times"]=None
                else:
                    print(" Skipping Low-End Energy (Spectrogram data missing or inconsistent).")
                    results["low_energy_norm"]=None; results["low_energy_times"]=None
            else:
                print(" Step 5: Skipping Low-End Energy (Checkbox unchecked).")
                results["low_energy_norm"]=None; results["low_energy_times"]=None

            # --- Dynamic Range Step ---
            if self.analysis_vars['dyn_range'].get():
                print(f" Step 6: Calculating Dynamic Range...")
                y_dyn=results.get("y_processed"); sr_dyn=results.get("sr")
                hop_dyn=results.get("hop_length"); trim_dyn=results.get("trim_offset_sec",0)
                frame_len_dyn=2048 # Standard frame length for RMS/peak
                try:
                    if y_dyn is not None and len(y_dyn) >= frame_len_dyn and sr_dyn and hop_dyn and trim_dyn is not None:
                        # Calculate frame-based RMS and Peak
                        rms_f = librosa.feature.rms(y=y_dyn, frame_length=frame_len_dyn, hop_length=hop_dyn)[0]
                        # Calculate peak within each frame
                        y_frames = librosa.util.frame(y_dyn, frame_length=frame_len_dyn, hop_length=hop_dyn)
                        peak_f = np.max(np.abs(y_frames), axis=0)
                        # Ensure same length and calculate dynamic range (Peak/RMS)
                        min_len = min(len(rms_f), len(peak_f))
                        rms_f = rms_f[:min_len]; peak_f = peak_f[:min_len]
                        results["dyn_range"] = np.nan_to_num(peak_f / (rms_f + 1e-9)) # Avoid division by zero
                        # Calculate corresponding absolute times
                        dyn_times_rel = librosa.frames_to_time(np.arange(len(results["dyn_range"])), sr=sr_dyn, hop_length=hop_dyn)
                        results["dyn_times_absolute"] = dyn_times_rel + trim_dyn
                    else:
                        print(" Skipping Dynamic Range (Audio data missing or too short).")
                        results["dyn_range"]=None; results["dyn_times_absolute"]=None
                except Exception as de:
                    print(f" Error calculating Dynamic Range: {de}")
                    traceback.print_exc()
                    results["dyn_range"]=None; results["dyn_times_absolute"]=None
            else:
                # Ensure keys exist even if skipped
                results["dyn_range"]=None; results["dyn_times_absolute"]=None

            # --- Ensure default keys exist if analysis steps were skipped ---
            results.setdefault("semantic_labels", [])
            results.setdefault("label_colors", [])
            results.setdefault("section_features", []) # Crucial: ensure section_features list exists
            results.setdefault("labels_before_cleanup", [])

            print(f"--- Analysis completed for Track {track_num} ---")
            # Convert defaultdict back to a regular dict before returning
            return dict(results)

        except Exception as e:
            print(f"--- Unhandled Error during analysis of Track {track_num} ---")
            traceback.print_exc()
            messagebox.showerror(f"Error Analyzing Track {track_num}", f"Unexpected error during analysis:\n{e}")
            return None # Indicate failure

    # --- Method for Handling Manual BPM Update ---
    def update_plots_with_manual_bpm(self):
        """Validates the manual BPM input from the GUI and triggers re-analysis.

        Reads the value from the manual BPM entry field, checks if it's a valid
        number within a reasonable range (30-300), and then calls `run_analysis`
        with the `is_update` flag set and the provided BPM value. Disables
        buttons during re-analysis.
        """
        if not self.use_manual_bpm.get():
            messagebox.showinfo("Info", "Please check 'Use Manual BPM' first.")
            return
        try:
            bpm_str = self.manual_bpm_entry_var.get()
            bpm_val = float(bpm_str) if bpm_str else None
            # Validate BPM range
            assert bpm_val is not None and 30 <= bpm_val <= 300
        except (ValueError, AssertionError):
            messagebox.showerror("Invalid BPM", "Manual BPM must be a number between 30 and 300.")
            return

        # Disable buttons during update
        if hasattr(self, 'update_bpm_button'): self.update_bpm_button.config(state=tk.DISABLED)
        if hasattr(self, 'analyze_button'): self.analyze_button.config(state=tk.DISABLED)

        print(f"Triggering re-analysis with manual BPM: {bpm_val}")
        # Trigger analysis, passing the manual BPM value
        self.run_analysis(is_update=True, manual_bpm_val=bpm_val)

    # --- Playback GUI Update Callbacks ---
    def _update_playback_buttons_state_from_manager(self, state):
        """Updates the Play/Pause and Stop button states based on PlaybackManager state.

        This method is intended as a callback function passed to the PlaybackManager.
        It receives the new playback state and updates the GUI accordingly.

        Args:
            state (str): The new playback state ('playing', 'paused', 'stopped').
        """
        # print(f"DEBUG GUI: Playback state changed to: {state}") # Verbose
        is_playing = (state == 'playing')
        is_paused = (state == 'paused')
        # Determine if audio data is loaded and we are in single mode
        can_play = (self.playback_manager.audio_data is not None and
                    self.playback_manager.sample_rate is not None and
                    self.mode.get() == "single")

        play_pause_state = tk.NORMAL if can_play else tk.DISABLED
        stop_state = tk.NORMAL if (is_playing or is_paused) else tk.DISABLED

        # Update button text and state if the buttons exist
        if self.play_pause_button:
            self.play_pause_button.config(state=play_pause_state,
                                          text="Pause" if is_playing else "Play")
        if self.stop_button:
            self.stop_button.config(state=stop_state)

    def _update_playhead_display(self, frame):
        """Updates the position of the vertical playhead line on the waveform plot.

        This method is intended as a callback function passed to the PlaybackManager.
        It receives the current playback frame index, converts it to time, adjusts
        for any trim offset, and updates the x-data of the Matplotlib line object
        representing the playhead. It then requests a canvas redraw.

        Args:
            frame (int or None): The current audio frame index being played, or None
                                 if playback is stopped or position is unknown.
        """
        # Check if update is necessary or possible
        if frame is None or not self.playhead_line or not self.playback_manager.sample_rate:
            # Hide playhead if stopped or data missing
            if self.playhead_line and self.playhead_line.get_visible():
                self.playhead_line.set_visible(False)
                self._redraw_canvas()
            return

        # Calculate display time (absolute time on plot)
        current_time = frame / self.playback_manager.sample_rate # Time relative to start of loaded audio
        trim = self.track_data.get(1, {}).get('trim_offset_sec', 0) # Get trim offset for track 1
        display_time = current_time + trim # Add offset for absolute time axis

        # Update the playhead line position
        self.playhead_line.set_xdata([display_time, display_time])

        # Make sure it's visible and redraw
        if not self.playhead_line.get_visible():
            self.playhead_line.set_visible(True)
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Requests an idle redraw of the Matplotlib canvas for the waveform plot.

        Safely attempts to find the canvas widget associated with the 'Waveform'
        plot and schedules a redraw. Handles potential errors during redraw.
        """
        waveform_plot_widgets = self.plot_widgets.get('Waveform', {})
        canvas = waveform_plot_widgets.get('canvas')
        if canvas:
            try:
                canvas.draw_idle() # Schedule redraw, doesn't block
            except Exception as e:
                print(f"Error redrawing canvas: {e}") # Log error if redraw fails

    # --- Method to Toggle Label View ---
    def _toggle_label_view(self):
        """Switches the waveform plot labels between final and pre-cleanup versions.

        Reads the state of `show_pre_cleanup_labels_var`, triggers a replotting
        of the analysis results via PlotManager, and updates the state of the
        toggle button itself. Also ensures the save button state is updated,
        as saving might depend on the viewed labels.
        """
        print("DEBUG: _toggle_label_view called. Current state:", self.show_pre_cleanup_labels_var.get())
        # Re-display results; PlotManager checks the show_pre_cleanup_labels_var
        self.plot_manager.display_analysis_results()
        # Update button states based on the new view
        self._update_toggle_button_state()
        self._update_save_button_state()

    def _update_toggle_button_state(self):
         """Updates the state of the 'Toggle Labels' button.

         Enabled only if track 1 data exists, pre-cleanup labels are available,
         the mode is 'single', and the HMM results are not currently being shown.
         """
         state = tk.DISABLED
         if (self.track_data.get(1) and
             self.track_data[1].get('labels_before_cleanup') and
             self.mode.get() == 'single' and
             not self.show_hmm_var.get()): # Disable if showing HMM
             state = tk.NORMAL
         if self.toggle_labels_button:
             self.toggle_labels_button.config(state=state)

    # --- Waveform Click Handler (Seek, Edit, Split Functionality) ---
    def _on_waveform_click(self, event):
        """Handles mouse clicks on the waveform plot axes.

        Routes the click event to different actions based on the button pressed
        and modifier keys (Shift).

        - Left-Click: Seeks playback position (only in single mode).
        - Shift + Left-Click: Initiates the section split process (single mode, not HMM view).
        - Right-Click: Opens the section edit/merge pop-up (single mode, not HMM view).

        Args:
            event (matplotlib.backend_bases.MouseEvent): The Matplotlib mouse event.
        """
        # print(f"DEBUG CLICK HANDLER: Start. Button={event.button}, x={event.xdata}, y={event.ydata}, Axes={event.inaxes}, State={event.state}, Key={event.key}") # Verbose

        # Ignore clicks outside plot axes or if no data loaded for track 1
        if event.xdata is None or event.inaxes is None or not self.track_data.get(1):
            return

        # Check for Shift key modifier via event.key attribute
        shift_pressed = (event.key == 'shift') # Check if shift key was held during click

        # --- Shift + Left-Click (Button 1): Trigger Split ---
        if event.button == 1 and shift_pressed:
            print("DEBUG CLICK HANDLER: Split click detected (Shift + Left).")
            # Validate conditions for splitting
            if self.mode.get() != 'single':
                print("DEBUG CLICK HANDLER: Not in single mode. Split ignore.")
                return
            if self.show_hmm_var.get():
                messagebox.showinfo("Split Error", "Cannot split sections while viewing HMM results. Uncheck 'Show HMM'.")
                return

            clicked_time = event.xdata # Absolute time clicked on the plot axis
            section_starts = self.track_data[1].get("section_starts") # Use original starts for splitting
            if section_starts is None or not isinstance(section_starts, (list, np.ndarray)) or len(section_starts) == 0:
                print("DEBUG CLICK HANDLER: No valid section starts data found. Split ignore.")
                return

            # Find the index of the section containing the click
            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]
                is_last_section = (i == len(section_starts) - 1)
                if is_last_section:
                    # Allow clicking anywhere in the last section
                    if clicked_time >= start:
                        section_index = i
                        break
                else:
                    # Check if click is within the bounds of non-last sections
                    next_start = section_starts[i+1]
                    if clicked_time >= start and clicked_time < next_start:
                        section_index = i
                        break

            if section_index != -1:
                print(f"DEBUG CLICK HANDLER: Showing split popup for section index {section_index} at time {clicked_time:.2f}s.")
                # Call the popup to confirm/adjust split time based on bars
                self._show_split_section_popup(section_index, clicked_time)
            else:
                print("DEBUG CLICK HANDLER: Could not determine clicked section index for split.")

        # --- Right-Click (Button 3): Trigger Edit ---
        elif event.button == 3:
            print("DEBUG CLICK HANDLER: Edit click detected (Right-click).")
            # Validate conditions for editing
            if self.mode.get() != 'single' or not self.section_editor:
                print("DEBUG CLICK HANDLER: Not in single mode or no editor. Edit ignore.")
                return
            if self.show_hmm_var.get():
                messagebox.showinfo("Edit Info", "Please switch back to 'Original' view (uncheck 'Show HMM') to edit sections.")
                return

            clicked_time = event.xdata
            section_starts = self.track_data[1].get("section_starts") # Edit uses original starts
            if section_starts is None or not isinstance(section_starts, (list, np.ndarray)) or len(section_starts) == 0:
                print("DEBUG CLICK HANDLER: No valid section starts data found. Edit ignore.")
                return

            # Find the section index corresponding to the click time
            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]
                is_last_section = (i == len(section_starts) - 1)
                if is_last_section:
                    if clicked_time >= start:
                        section_index = i
                        break
                else:
                    next_start = section_starts[i+1]
                    if clicked_time >= start and clicked_time < next_start:
                        section_index = i
                        break

            if section_index != -1:
                print(f"DEBUG CLICK HANDLER: Click corresponds to section index {section_index} for edit.")
                # Show the edit/merge popup menu near the click event
                self._show_section_edit_popup(section_index, event)
            else:
                print("DEBUG CLICK HANDLER: Could not determine clicked section index for edit.")

        # --- Left-Click (Button 1, no shift): Seek Playback ---
        elif event.button == 1 and not shift_pressed:
            # Validate conditions for seeking
            if self.mode.get() != 'single': return # Only seek in single mode
            if self.playback_manager.audio_data is None or self.playback_manager.sample_rate is None: return # Need audio loaded

            # Convert click time (absolute plot time) to audio frame index
            clicked_time = event.xdata
            trim_offset = self.track_data.get(1, {}).get('trim_offset_sec', 0)
            sample_rate = self.playback_manager.sample_rate
            audio_length = len(self.playback_manager.audio_data)
            relative_time = clicked_time - trim_offset # Time relative to start of audio data
            target_frame = int(relative_time * sample_rate)
            # Clamp frame index within valid audio bounds
            target_frame = max(0, min(target_frame, audio_length - 1))

            # Tell PlaybackManager to seek
            self.playback_manager.seek(target_frame)
            # Immediately update playhead display for responsiveness
            if self.playhead_line:
                display_time = target_frame / sample_rate + trim_offset # Convert back to absolute plot time
                self.playhead_line.set_xdata([display_time, display_time])
                self.playhead_line.set_visible(True)
                self._redraw_canvas()
        # else: print(f"DEBUG CLICK HANDLER: Ignored button {event.button}.") # Verbose


    # --- Section Edit Pop-up Method ---
    def _show_section_edit_popup(self, section_index, event=None):
        """Creates and displays a modal pop-up dialog for editing or merging a section.

        Retrieves current label and color for the selected section index,
        creates a Toplevel window with Comboboxes for label/color selection,
        and buttons to apply edits or trigger merges with adjacent sections.
        Positions the popup near the mouse click event if provided.

        Args:
            section_index (int): The zero-based index of the section to edit/merge.
            event (matplotlib.backend_bases.MouseEvent, optional): The mouse click
                event that triggered the popup, used for positioning. Defaults to None.

        """
        print(f"--- DEBUG MainApp: _show_section_edit_popup for index: {section_index} ---")
        # Ensure necessary data and editor exist
        if not self.section_editor or not self.track_data.get(1) or 'section_starts' not in self.track_data[1]:
            print("DEBUG MainApp: Section editor or original track data not available for editing.")
            return

        item_id_str = str(section_index) # Treeview uses index as string ID

        try:
            # Get current values from the SectionEditor's Treeview
            current_values = self.section_editor.tree.item(item_id_str, 'values')
            if not current_values or len(current_values) < 6: # Check expected number of columns
                raise ValueError("Invalid values found in Treeview row.")
            section_num_display = current_values[1] # 1-based display number
            current_label = current_values[4]
            current_color_name = current_values[5]
            num_sections = len(self.track_data[1]['section_starts'])
        except Exception as e:
            messagebox.showerror("Edit Error", f"Could not retrieve current values for section {section_index + 1}.\nError: {e}")
            print(f"Error getting values for tree item {item_id_str}: {e}")
            return

        # --- Create Toplevel Popup Window ---
        editor_popup = tk.Toplevel(self.master)
        editor_popup.title(f"Edit/Merge Section {section_num_display}")
        editor_popup.transient(self.master) # Keep popup above main window
        editor_popup.resizable(False, False)
        popup_frame = ttk.Frame(editor_popup, padding="10")
        popup_frame.pack(expand=True, fill=tk.BOTH)

        # --- Edit Frame ---
        edit_frame = ttk.LabelFrame(popup_frame, text="Edit Label/Color", padding=5)
        edit_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        # Label selection
        ttk.Label(edit_frame, text="Label:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        label_combo = ttk.Combobox(edit_frame, values=ALLOWED_LABELS, state='readonly', width=15)
        label_combo.grid(row=0, column=1, padx=5, pady=5)
        label_combo.set(current_label if current_label in ALLOWED_LABELS else ALLOWED_LABELS[0])
        # Color selection
        ttk.Label(edit_frame, text="Color:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        color_combo = ttk.Combobox(edit_frame, values=list(COLOR_NAME_MAP.keys()), state='readonly', width=15)
        color_combo.grid(row=1, column=1, padx=5, pady=5)
        color_combo.set(current_color_name if current_color_name in COLOR_NAME_MAP else list(COLOR_NAME_MAP.keys())[0])
        # Apply button
        ok_button = ttk.Button(edit_frame, text="Apply Edit", width=12,
                               command=lambda p=editor_popup, item=item_id_str, lc=label_combo, cc=color_combo:
                               self.section_editor._commit_popup_edit(p, item, lc, cc))
        ok_button.grid(row=0, column=2, rowspan=2, padx=(10, 5), pady=5, sticky="ns")

        # --- Merge Frame ---
        merge_frame = ttk.LabelFrame(popup_frame, text="Merge Section", padding=5)
        merge_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        # Merge Previous button
        merge_prev_button = ttk.Button(merge_frame, text="Merge with Previous", width=20,
                                       command=lambda p=editor_popup, idx=section_index:
                                       self._trigger_merge(p, idx, 'prev'))
        merge_prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        if section_index == 0: # Disable if it's the first section
            merge_prev_button.config(state=tk.DISABLED)
        # Merge Next button
        merge_next_button = ttk.Button(merge_frame, text="Merge with Next", width=20,
                                       command=lambda p=editor_popup, idx=section_index:
                                       self._trigger_merge(p, idx, 'next'))
        merge_next_button.pack(side=tk.LEFT, padx=5, pady=5)
        if section_index >= num_sections - 1: # Disable if it's the last section
            merge_next_button.config(state=tk.DISABLED)

        # --- Cancel Button ---
        cancel_button = ttk.Button(popup_frame, text="Cancel", width=8, command=editor_popup.destroy)
        cancel_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # --- Position and Show Popup ---
        editor_popup.update_idletasks() # Ensure dimensions are calculated
        # Position near mouse click if event is provided
        if event and hasattr(event, 'x_root') and hasattr(event, 'y_root'):
            final_x = event.x_root + 10
            final_y = event.y_root + 10
            # Keep popup on screen
            screen_w = self.master.winfo_screenwidth(); screen_h = self.master.winfo_screenheight()
            popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height()
            final_x = max(0, min(final_x, screen_w - popup_w))
            final_y = max(0, min(final_y, screen_h - popup_h))
            editor_popup.geometry(f"+{final_x}+{final_y}")
        else:
            # Center relative to main window if no event
            main_win = self.master
            main_x = main_win.winfo_x(); main_y = main_win.winfo_y()
            main_w = main_win.winfo_width(); main_h = main_win.winfo_height()
            popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height()
            center_x = main_x + (main_w // 2) - (popup_w // 2)
            center_y = main_y + (main_h // 2) - (popup_h // 2)
            editor_popup.geometry(f"+{center_x}+{center_y}")

        editor_popup.grab_set() # Make popup modal
        editor_popup.wait_window() # Wait until popup is closed


    # --- Merge Logic ---
    def _trigger_merge(self, popup, section_index, direction):
        """Handles the click event from the merge buttons in the edit popup.

        Determines the correct boundary index to remove based on the direction
        ('prev' or 'next') and calls the main `_merge_section` method. Closes
        the popup window first.

        Args:
            popup (tk.Toplevel): The popup window widget to destroy.
            section_index (int): The index of the section that was right-clicked.
            direction (str): Either 'prev' or 'next', indicating merge direction.
        """
        popup.destroy() # Close the popup first

        if direction == 'prev':
            if section_index > 0:
                # To merge with previous, remove the boundary *at* section_index
                self._merge_section(section_index)
            else:
                messagebox.showerror("Merge Error", "Cannot merge the first section with previous.")
        elif direction == 'next':
            num_sections = len(self.track_data[1].get('section_starts', []))
            if section_index < num_sections - 1:
                # To merge with next, remove the boundary *after* section_index
                self._merge_section(section_index + 1)
            else:
                messagebox.showerror("Merge Error", "Cannot merge the last section with next.")

    def _merge_section(self, remove_boundary_index):
        """Merges two adjacent sections by removing the specified boundary.

        Removes the start time at `remove_boundary_index` and the corresponding
        label, color, and feature dictionary at that index. Updates the end time
        and duration of the preceding section. Recalculates features for the
        newly merged section. Clears any existing HMM results. Updates plots
        and UI state.

        Args:
            remove_boundary_index (int): The index of the section start time
                                         (boundary) to remove. This corresponds
                                         to the index of the *second* section
                                         in the pair being merged.

        Note:
            Operates directly on the `self.track_data[1]` dictionary.
        """
        print(f"DEBUG: Attempting to merge by removing boundary at index {remove_boundary_index}")
        if not self.track_data.get(1):
            messagebox.showerror("Merge Error", "No track data loaded.")
            return

        t_data = self.track_data[1]
        # Get references to the lists within track_data
        section_starts = t_data.get('section_starts')
        semantic_labels = t_data.get('semantic_labels')
        label_colors = t_data.get('label_colors')
        section_features = t_data.get('section_features') # List of dictionaries
        # Optional lists
        cluster_labels = t_data.get('cluster_labels')
        labels_before_cleanup = t_data.get('labels_before_cleanup')

        # Validate data structure
        if not all([isinstance(l, list) for l in [section_starts, semantic_labels, label_colors, section_features]]):
            messagebox.showerror("Merge Error", "Core section data lists are missing or invalid.")
            print("ERROR: Core section data lists missing for merge.")
            return

        num_sections_before_merge = len(section_starts)
        # Validate boundary index
        if remove_boundary_index <= 0 or remove_boundary_index >= num_sections_before_merge:
            messagebox.showerror("Merge Error", f"Invalid boundary index {remove_boundary_index} for merging.")
            print(f"ERROR: Invalid boundary index {remove_boundary_index} for merging {num_sections_before_merge} sections.")
            return

        # Indices for the sections involved
        keep_section_idx = remove_boundary_index - 1 # The section that will grow
        remove_section_idx = remove_boundary_index # The section being absorbed

        try:
            # Get times for duration calculation
            kept_start_time = section_features[keep_section_idx]['start_time']
            removed_end_time = section_features[remove_section_idx]['end_time']
            new_duration_sec = removed_end_time - kept_start_time
            new_duration_bars = round(new_duration_sec / t_data['seconds_per_bar']) if t_data.get('seconds_per_bar', 0) > 0 else 0

            # --- Remove the boundary and corresponding data ---
            print(f"DEBUG: Removing data for section index {remove_section_idx} (boundary index {remove_boundary_index})")
            # Remove the start time that defines the boundary
            del section_starts[remove_boundary_index]
            # Remove the data associated with the second section
            del semantic_labels[remove_section_idx]
            del label_colors[remove_section_idx]
            del section_features[remove_section_idx]
            # Remove from optional lists if they exist and have the correct length
            if cluster_labels and len(cluster_labels) == num_sections_before_merge:
                del cluster_labels[remove_section_idx]
            if labels_before_cleanup and len(labels_before_cleanup) == num_sections_before_merge:
                del labels_before_cleanup[remove_section_idx]

            # --- Update the kept section's data ---
            print(f"DEBUG: Updating kept section index {keep_section_idx} end time and duration.")
            section_features[keep_section_idx]['end_time'] = removed_end_time
            section_features[keep_section_idx]['duration_sec'] = new_duration_sec
            section_features[keep_section_idx]['duration_bars'] = new_duration_bars

            # --- Recalculate features for the merged section ---
            print(f"DEBUG: Recalculating features for merged section {keep_section_idx} ({kept_start_time:.2f} - {removed_end_time:.2f})")
            merged_features = self._recalculate_section_features(t_data, keep_section_idx)
            if merged_features:
                # Update all recalculated keys in the existing feature dict
                feature_keys_to_update = list(merged_features.keys())
                for key in feature_keys_to_update:
                    section_features[keep_section_idx][key] = merged_features[key]
                print("DEBUG: Features recalculated and updated in section_features list.")
            else:
                print("Warning: Feature recalculation failed for merged section.")

            # --- Update indices in subsequent feature dictionaries ---
            # Indices need to be decremented from the removed section onwards
            for i in range(keep_section_idx + 1, len(section_features)): # Start from the one after the kept section
                 if 'index' in section_features[i]:
                     section_features[i]['index'] -= 1 # Decrement index
                 else:
                     # This indicates a potential issue with data consistency
                     print(f"Warning: 'index' key missing in section_features at list index {i} during merge update.")


            # --- Update the main track_data dictionary (redundant but safe) ---
            t_data['section_starts'] = section_starts
            t_data['semantic_labels'] = semantic_labels
            t_data['label_colors'] = label_colors
            t_data['section_features'] = section_features
            if cluster_labels: t_data['cluster_labels'] = cluster_labels
            if labels_before_cleanup: t_data['labels_before_cleanup'] = labels_before_cleanup

            # --- Clear HMM results as they are now invalid ---
            if 'hmm_semantic_labels' in t_data: del t_data['hmm_semantic_labels']
            if 'hmm_label_colors' in t_data: del t_data['hmm_label_colors']
            if 'hmm_section_starts' in t_data: del t_data['hmm_section_starts']
            self.show_hmm_var.set(False) # Switch back to original view
            self._update_hmm_button_state() # Disable HMM buttons

            print(f"DEBUG: Merge successful. Removed section at original index {remove_section_idx}. Updated section at index {keep_section_idx}.")
            self.status_label.config(text="Sections Merged", foreground="blue")

            # --- Refresh plots and editor ---
            self.plot_manager.display_analysis_results() # Update plots
            self._update_save_button_state() # Enable save button

        except IndexError as e:
            messagebox.showerror("Merge Error", f"Index error during merge: {e}. Lists might be inconsistent.")
            print(f"ERROR: Index error during merge: {e}")
            traceback.print_exc()
        except Exception as e:
            messagebox.showerror("Merge Error", f"An unexpected error occurred during merge:\n{e}")
            print(f"ERROR: Unexpected error during merge: {e}")
            traceback.print_exc()


    # --- Split Logic ---
    def _show_split_section_popup(self, section_index, clicked_time):
        """Creates and displays a modal pop-up for splitting a section based on bar number.

        Calculates the bar number corresponding to the `clicked_time` and the
        bar range of the target section. Presents an entry field pre-filled with
        the clicked bar number, allowing the user to adjust it before confirming
        the split.

        Args:
            section_index (int): The zero-based index of the section to split.
            clicked_time (float): The absolute time (in seconds) where the user clicked.
        """
        print(f"--- DEBUG MainApp: _show_split_section_popup for index: {section_index} at time {clicked_time:.2f} ---")
        if not self.track_data.get(1) or not self.track_data[1].get('section_starts'):
            print("DEBUG MainApp: Track data or section starts not available for splitting.")
            return

        # --- Get section and timing information ---
        try:
            current_label = self.track_data[1]['semantic_labels'][section_index]
            section_num_display = section_index + 1 # 1-based index for display

            # Get bar timing info
            seconds_per_bar = self.track_data[1].get('seconds_per_bar', 0)
            if seconds_per_bar <= 0:
                messagebox.showerror("Split Error", "Cannot calculate bar position - missing tempo data (seconds_per_bar).")
                return

            # Convert clicked time to bar number (relative to audio start)
            trim_offset = self.track_data[1].get('trim_offset_sec', 0)
            clicked_time_rel = clicked_time - trim_offset
            clicked_bar = clicked_time_rel / seconds_per_bar

            # Get the boundary bars for this section (relative to audio start)
            section_start_time = self.track_data[1]['section_starts'][section_index]
            section_start_time_rel = section_start_time - trim_offset
            section_start_bar = section_start_time_rel / seconds_per_bar

            # Find end time (either next section start or end of track)
            if section_index + 1 < len(self.track_data[1]['section_starts']):
                section_end_time = self.track_data[1]['section_starts'][section_index + 1]
            else:
                # Use duration_processed (relative time) + trim_offset for absolute end
                section_end_time = self.track_data[1].get('duration_processed', 0) + trim_offset

            section_end_time_rel = section_end_time - trim_offset
            section_end_bar = section_end_time_rel / seconds_per_bar

        except IndexError:
            messagebox.showerror("Split Error", f"Cannot get label or time data for section index {section_index}.")
            return
        except KeyError as ke:
             messagebox.showerror("Split Error", f"Missing expected key in track data: {ke}")
             return

        # --- Create Pop-up Window ---
        split_popup = tk.Toplevel(self.master)
        split_popup.title(f"Split Section {section_num_display}")
        split_popup.transient(self.master)
        split_popup.resizable(False, False)
        popup_frame = ttk.Frame(split_popup, padding="15")
        popup_frame.pack(expand=True, fill=tk.BOTH)

        # --- Information Labels ---
        info_label = ttk.Label(popup_frame,
                              text=f"Split Section {section_num_display} ('{current_label}') at bar:")
        info_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)

        range_label = ttk.Label(popup_frame,
                               text=f"Section range: Bar {section_start_bar:.2f} to {section_end_bar:.2f}")
        range_label.grid(row=1, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)

        # --- Bar Entry ---
        bar_var = tk.StringVar(value=f"{clicked_bar:.2f}") # Pre-fill with clicked bar
        bar_entry = ttk.Entry(popup_frame, textvariable=bar_var, width=15)
        bar_entry.grid(row=2, column=0, padx=(0, 5), pady=5, sticky=tk.EW)
        ttk.Label(popup_frame, text="bar number").grid(row=2, column=1, padx=(0, 5), pady=5, sticky=tk.W)

        # --- Buttons Frame ---
        button_frame = ttk.Frame(popup_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # Split Button (calls commit function with bar number)
        split_button = ttk.Button(button_frame, text="Split", width=10,
                                 command=lambda p=split_popup, idx=section_index, bv=bar_var,
                                        spb=seconds_per_bar, to=trim_offset:
                                        self._commit_split_by_bar(p, idx, bv, spb, to))
        split_button.pack(side=tk.LEFT, padx=5)

        # Cancel Button
        cancel_button = ttk.Button(button_frame, text="Cancel", width=10, command=split_popup.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

        # --- Focus and Modality ---
        bar_entry.focus_set() # Set focus to the entry field
        split_popup.grab_set() # Make the popup modal
        split_popup.wait_window() # Wait for the popup to close
        print(f"DEBUG MainApp: Split pop-up closed for section index {section_index}.")

    def _commit_split(self, popup, section_index, time_var):
        """Validates the split time entered in the (older, time-based) split popup and calls _split_section.

        This method is likely deprecated if the bar-based popup is used, but kept
        for potential backward compatibility or alternative implementations.

        Args:
            popup (tk.Toplevel): The popup window widget.
            section_index (int): The zero-based index of the section to split.
            time_var (tk.StringVar): The Tkinter variable holding the entered split time string.

        """
        try:
            split_time_str = time_var.get()
            split_time = float(split_time_str)
        except ValueError:
            messagebox.showerror("Invalid Time", f"Split time must be a valid number.\nEntered: '{split_time_str}'", parent=popup)
            return

        # --- Perform validation (re-check bounds and min duration) ---
        t_data = self.track_data.get(1)
        if not t_data: return # Should not happen if popup opened
        section_starts = t_data.get('section_starts')
        num_sections_before = len(section_starts) if section_starts else 0

        if not (0 <= section_index < num_sections_before):
             messagebox.showerror("Split Error", "Invalid section index.", parent=popup); return

        original_start_time = section_starts[section_index]
        # Calculate original end time carefully
        original_end_time = section_starts[section_index + 1] if section_index + 1 < num_sections_before else t_data.get('duration_processed', float('inf')) + t_data.get('trim_offset_sec', 0)

        # Use a small epsilon for time comparisons to avoid floating point issues near boundaries
        epsilon = 1e-6
        # Check if split time is strictly within the section
        if not (original_start_time + epsilon < split_time < original_end_time - epsilon):
             messagebox.showerror("Split Error", f"Split time {split_time:.3f}s must be strictly within the section boundaries ({original_start_time:.3f}s - {original_end_time:.3f}s).", parent=popup)
             return

        # Check if resulting sections meet minimum duration
        if (split_time - original_start_time < MIN_SPLIT_SECTION_DURATION_SEC) or \
           (original_end_time - split_time < MIN_SPLIT_SECTION_DURATION_SEC):
            messagebox.showerror("Split Error", f"Resulting sections would be shorter than minimum duration ({MIN_SPLIT_SECTION_DURATION_SEC:.1f}s).", parent=popup)
            return

        # --- If all validations pass ---
        popup.destroy() # Close the popup
        self._split_section(section_index, split_time) # Perform the actual split

    def _commit_split_by_bar(self, popup, section_index, bar_var, seconds_per_bar, trim_offset):
        """Validates the split bar number from the popup, converts it to time, and calls _split_section.

        Args:
            popup (tk.Toplevel): The popup window widget.
            section_index (int): The zero-based index of the section to split.
            bar_var (tk.StringVar): The Tkinter variable holding the entered bar number string.
            seconds_per_bar (float): The duration of one bar in seconds.
            trim_offset (float): The trim offset of the audio file in seconds.
        """
        try:
            split_bar_str = bar_var.get()
            split_bar = float(split_bar_str)
        except ValueError:
            messagebox.showerror("Invalid Bar", f"Split bar must be a valid number.\nEntered: '{split_bar_str}'", parent=popup)
            return

        # Convert bar number to absolute time
        split_time = (split_bar * seconds_per_bar) + trim_offset

        # --- Perform validation (re-check bounds and min duration in seconds) ---
        t_data = self.track_data.get(1)
        if not t_data: return # Should not happen if popup opened
        section_starts = t_data.get('section_starts')
        num_sections_before = len(section_starts) if section_starts else 0

        if not (0 <= section_index < num_sections_before):
            messagebox.showerror("Split Error", "Invalid section index.", parent=popup)
            return

        original_start_time = section_starts[section_index]
        # Calculate original end time carefully
        original_end_time = section_starts[section_index + 1] if section_index + 1 < num_sections_before else t_data.get('duration_processed', float('inf')) + t_data.get('trim_offset_sec', 0)

        # Calculate the minimum duration in bars for display purposes
        min_split_section_duration_bars = MIN_SPLIT_SECTION_DURATION_SEC / seconds_per_bar if seconds_per_bar > 0 else float('inf')

        # Convert original boundaries to bars for display in error messages
        original_start_bar = (original_start_time - trim_offset) / seconds_per_bar if seconds_per_bar > 0 else 0
        original_end_bar = (original_end_time - trim_offset) / seconds_per_bar if seconds_per_bar > 0 else float('inf')

        # Use a small epsilon for time comparisons
        epsilon = 1e-6
        # Check if split time is strictly within the section
        if not (original_start_time + epsilon < split_time < original_end_time - epsilon):
            messagebox.showerror("Split Error",
                                f"Split bar {split_bar:.2f} (time {split_time:.3f}s) must be strictly within the section boundaries "
                                f"(bar {original_start_bar:.2f} - {original_end_bar:.2f}).",
                                parent=popup)
            return

        # Check if resulting sections meet minimum duration (in seconds)
        if ((split_time - original_start_time) < MIN_SPLIT_SECTION_DURATION_SEC) or \
           ((original_end_time - split_time) < MIN_SPLIT_SECTION_DURATION_SEC):
            messagebox.showerror("Split Error",
                                f"Resulting sections would be shorter than minimum duration "
                                f"({MIN_SPLIT_SECTION_DURATION_SEC:.1f}s or {min_split_section_duration_bars:.2f} bars).",
                                parent=popup)
            return

        # --- If all validations pass ---
        popup.destroy() # Close the popup
        self._split_section(section_index, split_time) # Perform the actual split using the calculated time

    def _split_section(self, section_index, split_time):
        """Splits a section at the given absolute time point.

        Modifies the `track_data[1]` dictionary by:
        1. Inserting the `split_time` into the `section_starts` list.
        2. Duplicating the label, color, and feature dictionary for the split section.
        3. Updating the end time/duration of the first part of the split section.
        4. Updating the start time/duration/index of the second part (newly inserted section).
        5. Updating the indices of all subsequent sections.
        6. Recalculating features for both newly formed sections using `_recalculate_section_features`.
        7. Clearing any existing HMM results.
        8. Refreshing plots and UI state.

        Args:
            section_index (int): The zero-based index of the section to split.
            split_time (float): The absolute time (in seconds) at which to split.

        Note:
            Operates directly on the `self.track_data[1]` dictionary.
        """
        print(f"DEBUG: Executing split for section {section_index} at time {split_time:.3f}")
        if not self.track_data.get(1): return # Should already be checked
        t_data = self.track_data[1]

        # --- Get original data lists ---
        section_starts = t_data.get('section_starts')
        semantic_labels = t_data.get('semantic_labels')
        label_colors = t_data.get('label_colors')
        section_features = t_data.get('section_features') # List of dictionaries
        # Optional lists that also need modification
        cluster_labels = t_data.get('cluster_labels')
        labels_before_cleanup = t_data.get('labels_before_cleanup')

        # Basic validation
        if not all([isinstance(l, list) for l in [section_starts, semantic_labels, label_colors, section_features]]):
             messagebox.showerror("Split Error", "Core section data lists missing or invalid.")
             return
        num_sections_before = len(section_starts)
        if not (0 <= section_index < num_sections_before):
             messagebox.showerror("Split Error", f"Invalid section index {section_index} for split.")
             return

        try:
            insert_index = section_index + 1 # New boundary/section goes after the current one
            original_start_time = section_starts[section_index]
            original_end_time = section_features[section_index]['end_time'] # Get original end time before modification

            # --- Insert new boundary and duplicate labels/colors ---
            print(f" -> Inserting boundary at {split_time:.3f}s (index {insert_index})")
            section_starts.insert(insert_index, split_time)
            original_label = semantic_labels[section_index] # Label of the section being split
            semantic_labels.insert(insert_index, original_label) # Duplicate label
            original_color = label_colors[section_index]
            label_colors.insert(insert_index, original_color) # Duplicate color

            # Duplicate optional list items if they exist and have the correct length
            if cluster_labels and len(cluster_labels) == num_sections_before:
                cluster_labels.insert(insert_index, cluster_labels[section_index])
            if labels_before_cleanup and len(labels_before_cleanup) == num_sections_before:
                labels_before_cleanup.insert(insert_index, labels_before_cleanup[section_index])

            # --- Handle section_features list ---
            original_feature_dict = section_features[section_index]
            # Create a deep copy for the second part to avoid modifying shared references later
            new_section_feature_dict = copy.deepcopy(original_feature_dict)

            # 1. Update first part (section_index) - times and durations
            print(f" -> Updating section {section_index} end time to {split_time:.3f}s")
            section_features[section_index]['end_time'] = split_time
            section_features[section_index]['duration_sec'] = split_time - original_start_time
            section_features[section_index]['duration_bars'] = round(section_features[section_index]['duration_sec'] / t_data['seconds_per_bar']) if t_data.get('seconds_per_bar', 0) > 0 else 0
            # Keep original start time and index ('index' key should already be correct)

            # 2. Update second part (newly inserted dict) - times, durations, and index
            print(f" -> Creating new section {insert_index} from {split_time:.3f}s to {original_end_time:.3f}s")
            new_section_feature_dict['start_time'] = split_time
            new_section_feature_dict['end_time'] = original_end_time # End time is the original end time
            new_section_feature_dict['duration_sec'] = original_end_time - split_time
            new_section_feature_dict['duration_bars'] = round(new_section_feature_dict['duration_sec'] / t_data['seconds_per_bar']) if t_data.get('seconds_per_bar', 0) > 0 else 0
            new_section_feature_dict['index'] = insert_index # Set correct index for the new section
            new_section_feature_dict['original_label'] = original_label # Ensure label consistency

            # Insert the new feature dictionary into the list at the correct position
            section_features.insert(insert_index, new_section_feature_dict)

            # 3. Update indices for all subsequent feature dictionaries
            print(f" -> Updating indices for sections {insert_index + 1} onwards...")
            for i in range(insert_index + 1, len(section_features)):
                 if 'index' in section_features[i]:
                     section_features[i]['index'] += 1 # Increment index
                 else:
                     # This indicates a potential issue with data consistency
                     print(f"Warning: 'index' key missing in section_features at list index {i} during split index update.")

            # --- Recalculate features for the two new sections ---
            # Recalculate for the first part (index section_index)
            print(f"DEBUG: Recalculating features for split section part 1 (index {section_index})")
            recalculated_part1 = self._recalculate_section_features(t_data, section_index)
            if recalculated_part1:
                # Update all recalculated keys in the existing feature dict
                feature_keys_to_update = list(recalculated_part1.keys())
                for key in feature_keys_to_update:
                    section_features[section_index][key] = recalculated_part1[key]
                print(" -> Part 1 features updated.")
            else:
                print(" -> Warning: Feature recalculation failed for part 1.")

            # Recalculate for the second part (index insert_index)
            print(f"DEBUG: Recalculating features for split section part 2 (index {insert_index})")
            recalculated_part2 = self._recalculate_section_features(t_data, insert_index)
            if recalculated_part2:
                # Update all recalculated keys in the newly inserted feature dict
                feature_keys_to_update = list(recalculated_part2.keys())
                for key in feature_keys_to_update:
                    section_features[insert_index][key] = recalculated_part2[key]
                print(" -> Part 2 features updated.")
            else:
                print(" -> Warning: Feature recalculation failed for part 2.")

            # --- Update the main track_data dictionary (redundant but safe) ---
            t_data['section_starts'] = section_starts
            t_data['semantic_labels'] = semantic_labels
            t_data['label_colors'] = label_colors
            t_data['section_features'] = section_features
            if cluster_labels: t_data['cluster_labels'] = cluster_labels
            if labels_before_cleanup: t_data['labels_before_cleanup'] = labels_before_cleanup

            # --- Clear any existing HMM results as they are now invalid ---
            if 'hmm_semantic_labels' in t_data: del t_data['hmm_semantic_labels']
            if 'hmm_label_colors' in t_data: del t_data['hmm_label_colors']
            if 'hmm_section_starts' in t_data: del t_data['hmm_section_starts']
            self.show_hmm_var.set(False) # Switch back to original view
            self._update_hmm_button_state() # Disable HMM buttons

            print(f"DEBUG: Split successful. Section {section_index} split into {section_index} and {insert_index}.")
            self.status_label.config(text="Section Split", foreground="blue")

            # --- Refresh plots and editor ---
            self.plot_manager.display_analysis_results() # Update plots
            self._update_save_button_state() # Enable save button

        except Exception as e:
            messagebox.showerror("Split Error", f"An unexpected error occurred during split:\n{e}")
            print(f"ERROR: Unexpected error during split: {e}")
            traceback.print_exc()


    # *** NEW METHOD: Recalculate features for a specific section ***
    def _recalculate_section_features(self, track_data, section_idx):
        """Recalculates various features for a specific section index.

        This helper method is used after a merge or split operation to update
        the feature dictionary for a section whose boundaries have changed.
        It extracts the necessary frame-level data (RMS, spectrogram, etc.)
        from the main `track_data` based on the section's updated start and
        end times and computes aggregate features like averages, standard
        deviations, ratios, and trends.

        Args:
            track_data (dict): The main track data dictionary containing frame-level
                               features and the `section_features` list.
            section_idx (int): The zero-based index within the `section_features`
                               list for which to recalculate features.

        Returns:
            dict or None: A dictionary containing the recalculated feature values
                          (e.g., 'avg_rms', 'spectral_centroid_avg', etc.). Returns
                          None if the section index is invalid or a critical error
                          occurs during calculation. Feature values will be NaN if
                          underlying data is missing or calculation fails for that
                          specific feature.
        """
        print(f"DEBUG App: Recalculating features for section index {section_idx}")
        new_features = {} # Dictionary to store recalculated features
        try:
            # --- Get section info and base data ---
            section_features_list = track_data.get('section_features', [])
            if not (0 <= section_idx < len(section_features_list)):
                print(f"ERROR Recalc: Invalid section index {section_idx}")
                return None
            section_info = section_features_list[section_idx]
            start_time_abs = section_info.get('start_time')
            end_time_abs = section_info.get('end_time')

            # Base data needed for calculations from the main track_data dict
            trim_offset = track_data.get("trim_offset_sec", 0)
            duration_processed = track_data.get("duration_processed")
            rms_frames = track_data.get("rms") # Frame-based RMS values
            rms_times = track_data.get("rms_times") # RELATIVE times for RMS frames
            spec = track_data.get("spec") # Spectrogram
            freqs = track_data.get("freqs") # Frequencies for spectrogram
            times_absolute = track_data.get("times_absolute") # ABSOLUTE times for spectral frames
            spectral_centroid_frames = track_data.get("spectral_centroid_frames") # Frame-based
            spectral_bandwidth_frames = track_data.get("spectral_bandwidth_frames") # Frame-based
            spectral_contrast_frames = track_data.get("spectral_contrast_frames") # Frame-based
            low_energy_norm = track_data.get("low_energy_norm") # Frame based low energy (normalized)
            low_energy_times = track_data.get("low_energy_times") # RELATIVE times for low energy

            # Validate essential timing info
            if start_time_abs is None or end_time_abs is None:
                print("ERROR Recalc: Missing start/end time for section.")
                return None

            # Calculate relative times for the section boundaries
            start_time_rel = start_time_abs - trim_offset
            end_time_rel = end_time_abs - trim_offset

            # --- Recalculate RMS-based features ---
            avg_rms, peak_rms, rms_std, rms_trend = np.nan, np.nan, np.nan, np.nan
            if rms_frames is not None and rms_times is not None:
                # Find RMS frames within the relative time boundaries
                rms_mask = (rms_times >= start_time_rel) & (rms_times < end_time_rel)
                # Get finite RMS values and corresponding times within the section
                section_rms_vals = rms_frames[rms_mask][np.isfinite(rms_frames[rms_mask])]
                section_time_vals = rms_times[rms_mask][np.isfinite(rms_frames[rms_mask])]

                if section_rms_vals.size > 0:
                    avg_rms = np.mean(section_rms_vals)
                    peak_rms = np.max(section_rms_vals)
                    # Standard deviation requires at least 2 points
                    if section_rms_vals.size >= 2: rms_std = np.std(section_rms_vals)
                    else: rms_std = 0.0 # Std dev is 0 for a single point
                    # Trend (slope) requires at least 2 points
                    if section_rms_vals.size > 1:
                        try:
                            # Calculate time relative to the start of the section's valid frames
                            relative_time_vals = section_time_vals - section_time_vals[0]
                            slope, _, _, _, _ = scipy.stats.linregress(relative_time_vals, section_rms_vals)
                            rms_trend = slope if np.isfinite(slope) else 0.0 # Use 0 if slope is NaN/Inf
                        except ValueError: # Handle potential linregress errors
                            rms_trend = 0.0
                    else: rms_trend = 0.0 # Slope is 0 for 0 or 1 point
                else: # If no valid RMS frames found in section
                    avg_rms=0.0; peak_rms=0.0; rms_std=0.0; rms_trend=0.0

            # Store recalculated RMS features
            new_features["avg_rms"] = avg_rms
            new_features["peak_rms"] = peak_rms
            new_features["rms_std_dev"] = rms_std # Overall RMS std dev (potentially deprecated?)
            new_features["rms_trend"] = rms_trend
            new_features["rms_std_dev_section"] = rms_std # Store as section-specific std dev

            # --- Recalculate Spectrogram-based features ---
            low_ratio, high_ratio, cent_avg, cent_std, bw_avg, cont_avg = (np.nan,) * 6
            spec_times_rel = None # Calculate relative spectral times if possible
            if times_absolute is not None:
                spec_times_rel = times_absolute - trim_offset

            if spec is not None and freqs is not None and spec_times_rel is not None:
                # Find spectral frame indices within the relative time boundaries
                spec_indices = np.where((spec_times_rel >= start_time_rel) & (spec_times_rel < end_time_rel))[0]
                if spec_indices.size > 0:
                    section_spec = spec[:, spec_indices] # Get spectrogram slice for the section
                    total_energy = np.sum(section_spec) + 1e-9 # Add epsilon to avoid division by zero

                    # Low/High Energy Ratios
                    low_freq_mask = freqs < 150
                    high_freq_mask = freqs > 5000
                    low_ratio = np.sum(section_spec[low_freq_mask,:]) / total_energy if np.any(low_freq_mask) else 0.0
                    high_ratio = np.sum(section_spec[high_freq_mask,:]) / total_energy if np.any(high_freq_mask) else 0.0

                    # Spectral Centroid Avg/Std
                    if spectral_centroid_frames is not None and len(spectral_centroid_frames) == spec.shape[1]:
                         section_centroid = spectral_centroid_frames[spec_indices][np.isfinite(spectral_centroid_frames[spec_indices])]
                         if section_centroid.size > 0: cent_avg = np.mean(section_centroid)
                         else: cent_avg = 0.0
                         if section_centroid.size >= 2: cent_std = np.std(section_centroid)
                         else: cent_std = 0.0
                    else: cent_avg=0.0; cent_std=0.0

                    # Spectral Bandwidth Avg
                    if spectral_bandwidth_frames is not None and len(spectral_bandwidth_frames) == spec.shape[1]:
                         section_bw = spectral_bandwidth_frames[spec_indices][np.isfinite(spectral_bandwidth_frames[spec_indices])]
                         if section_bw.size > 0: bw_avg = np.mean(section_bw)
                         else: bw_avg = 0.0
                    else: bw_avg=0.0

                    # Spectral Contrast Avg (simple mean across bands/frames)
                    if spectral_contrast_frames is not None and spectral_contrast_frames.shape[1] == spec.shape[1]:
                         section_cont = spectral_contrast_frames[:, spec_indices]
                         finite_cont = section_cont[np.isfinite(section_cont)]
                         if finite_cont.size > 0: cont_avg = np.mean(finite_cont)
                         else: cont_avg = 0.0
                    else: cont_avg=0.0
                else: # If no spectral frames found in section
                     low_ratio=0.0; high_ratio=0.0; cent_avg=0.0; cent_std=0.0; bw_avg=0.0; cont_avg=0.0

            # Store recalculated spectral features
            new_features["low_end_ratio"] = low_ratio
            new_features["high_end_ratio"] = high_ratio
            new_features["spectral_centroid_avg"] = cent_avg
            new_features["spectral_centroid_std_dev"] = cent_std # Overall centroid std dev (potentially deprecated?)
            new_features["spectral_bandwidth_avg"] = bw_avg
            new_features["spectral_contrast_avg"] = cont_avg
            new_features["centroid_std_dev_section"] = cent_std # Store as section-specific std dev

            # --- Recalculate Relative Position ---
            rel_pos = np.nan
            if start_time_abs is not None and duration_processed is not None and duration_processed > 0:
                # Clamp between 0 and 1
                rel_pos = max(0.0, min(start_time_abs / duration_processed, 1.0))
            new_features['relative_position'] = rel_pos

            # --- Recalculate Average Low-End Energy ---
            avg_low_e = np.nan
            if low_energy_norm is not None and low_energy_times is not None:
                 # Ensure low_energy_times is a numpy array before masking
                 if isinstance(low_energy_times, np.ndarray):
                     # Use relative times for low energy
                     le_indices = np.where((low_energy_times >= start_time_rel) & (low_energy_times < end_time_rel))[0]
                     if le_indices.size > 0:
                         finite_vals = low_energy_norm[le_indices][np.isfinite(low_energy_norm[le_indices])]
                         if finite_vals.size > 0: avg_low_e = np.mean(finite_vals)
                 else:
                     print(f" -> Recalc Warning: low_energy_times is not NumPy array for section {section_idx}.")
            new_features['low_energy_norm'] = avg_low_e

            # --- Recalculate Delta Features ---
            # Setting deltas to 0.0 after merge/split as recalculating based on
            # potentially changed neighbors is complex and might not be meaningful.
            new_features['delta_rms'] = 0.0
            new_features['delta_centroid'] = 0.0

            # --- Ensure all expected feature keys exist, assign NaN if calculation failed ---
            # Define the list of features that should ideally be present after calculation
            expected_keys = [
                "avg_rms", "peak_rms", "rms_std_dev", "rms_trend",
                "low_end_ratio", "high_end_ratio", "spectral_centroid_avg",
                "spectral_centroid_std_dev", "spectral_bandwidth_avg", "spectral_contrast_avg",
                "relative_position", "low_energy_norm",
                "rms_std_dev_section", "centroid_std_dev_section",
                "delta_rms", "delta_centroid"
                # Add any other features expected to be calculated here
            ]
            for key in expected_keys:
                if key not in new_features or not np.isfinite(new_features.get(key, np.nan)):
                    # Set to NaN if missing or non-finite, except for deltas which we default to 0
                    if key not in ['delta_rms', 'delta_centroid']:
                         new_features[key] = np.nan

            # Debug print the recalculated features
            print(f" -> Recalculated features for index {section_idx}: { {k: f'{v:.2f}' if isinstance(v, float) else v for k,v in new_features.items()} }")
            return new_features # Return the dictionary of recalculated features

        except Exception as e:
            print(f"ERROR recalculating features for section {section_idx}: {e}")
            traceback.print_exc()
            return None # Indicate failure


    # --- Save/Load Methods --- # Now Handled by FileManager
    def _save_analysis(self):
        """Saves the current analysis results for Track 1 via FileManager."""
        self.file_manager.save_analysis()

    def _load_analysis(self):
        """Loads a previously saved analysis file via FileManager."""
        self.file_manager.load_analysis()

    def _ask_save_status(self, parent):
        """Asks the user if they want to save changes via FileManager."""
        return self.file_manager.ask_save_status(parent)

    def select_file(self, track_num):
        """Opens a file dialog to select an audio file via FileManager."""
        self.file_manager.select_file(track_num)


    # --- Manual Section Editing Callback ---
    def _apply_section_edits_from_editor(self, new_labels, new_colors_hex):
        """Applies label and color edits made in the SectionEditor to the track data.

        This method is called by the SectionEditor instance when the user confirms
        their edits. It updates the `semantic_labels`, `label_colors`, and the
        `original_label` within the `section_features` list in `track_data[1]`.
        It also clears any existing HMM results, refreshes plots, and updates UI state.

        Args:
            new_labels (list[str]): The list of updated section labels.
            new_colors_hex (list[str]): The list of updated section color hex codes.
        """
        print("DEBUG MainApp: Applying section edits received from editor...")
        if not self.track_data.get(1):
            messagebox.showerror("Update Error", "No track data loaded to apply edits to.")
            return

        current_labels = self.track_data[1].get('semantic_labels', [])
        # Validate data consistency
        if len(new_labels) != len(current_labels) or len(new_colors_hex) != len(current_labels):
            messagebox.showerror("Update Error", f"Data length mismatch when applying edits. Expected {len(current_labels)}, got {len(new_labels)} labels, {len(new_colors_hex)} colors.")
            return

        try:
            # Update main data lists
            self.track_data[1]['semantic_labels'] = new_labels
            self.track_data[1]['label_colors'] = new_colors_hex

            # Sync the 'original_label' field in the section_features list
            section_features = self.track_data[1].get('section_features', [])
            print(f"DEBUG: Syncing 'original_label' in section_features. Features count: {len(section_features)}, New labels count: {len(new_labels)}")
            if len(section_features) == len(new_labels):
                mismatches_found = False
                for i in range(len(section_features)):
                    # Ensure the key exists before assigning
                    section_features[i]['original_label'] = new_labels[i]
                # Verify sync (optional debug check)
                for i in range(len(section_features)):
                    if section_features[i].get('original_label') != new_labels[i]:
                        print(f"WARNING: Mismatch persists at Section {i} after update!")
                        mismatches_found = True
                if not mismatches_found:
                    print(" -> Sync successful: 'original_label' in section_features matches semantic_labels.")
            else:
                print("WARNING: Cannot sync 'original_label' due to length mismatch between section_features and new_labels.")

            # Clear HMM results as edits invalidate them
            if 'hmm_semantic_labels' in self.track_data[1]: del self.track_data[1]['hmm_semantic_labels']
            if 'hmm_label_colors' in self.track_data[1]: del self.track_data[1]['hmm_label_colors']
            if 'hmm_section_starts' in self.track_data[1]: del self.track_data[1]['hmm_section_starts']
            self.show_hmm_var.set(False) # Switch view back
            self._update_hmm_button_state() # Disable HMM buttons

            print("DEBUG MainApp: track_data updated with edits. Cleared HMM results.")
            print("DEBUG MainApp: Replotting waveform with updated labels/colors...")
            # Refresh plots and UI
            self.plot_manager.display_analysis_results()
            messagebox.showinfo("Update Complete", "Section labels and colors updated.")
            self.status_label.config(text="Sections Updated", foreground="blue")
            self._update_save_button_state() # Enable save button

        except Exception as e:
            messagebox.showerror("Update Error", f"Failed to apply section edits:\n{e}")
            traceback.print_exc()

    # *** HMM Prediction Trigger Method ***
    def _trigger_hmm_prediction(self):
        """Initiates HMM-based section prediction for the analyzed track.

        Called when the 'Run HMM Prediction' button is clicked. It ensures
        track data is loaded, loads the HMM model via HMMPredictor, runs the
        prediction, stores the results in separate keys within `track_data[1]`,
        updates the UI state (status bar, buttons), and refreshes the plots
        to show the HMM results.
        """
        print("--- Triggering HMM Prediction ---")
        self.status_label.config(text="Running HMM...", foreground="orange")
        self.master.update_idletasks()

        if not self.track_data.get(1):
            messagebox.showerror("HMM Error", "Please analyze Track 1 first.")
            self.status_label.config(text="HMM Error!", foreground="red")
            return

        # Attempt to load the HMM model and auxiliary data
        if not self.hmm_predictor.load_model():
            # Error message shown by load_model
            self.status_label.config(text="HMM Load Failed!", foreground="red")
            return

        # Run prediction using the loaded model
        hmm_results = self.hmm_predictor.predict(self.track_data[1])

        if hmm_results is not None:
            hmm_section_starts, hmm_semantic_labels, hmm_label_colors = hmm_results
            if not hmm_semantic_labels:
                # Handle case where HMM predicts no sections (e.g., very short track)
                print("HMM Prediction returned no labels.")
                self.status_label.config(text="HMM: No Sections Predicted", foreground="orange")
                # Ensure HMM keys are removed if prediction yields nothing
                self.track_data[1].pop('hmm_section_starts', None)
                self.track_data[1].pop('hmm_semantic_labels', None)
                self.track_data[1].pop('hmm_label_colors', None)
            else:
                # Store HMM results in separate keys to avoid overwriting original analysis
                self.track_data[1]['hmm_section_starts'] = hmm_section_starts
                self.track_data[1]['hmm_semantic_labels'] = hmm_semantic_labels
                self.track_data[1]['hmm_label_colors'] = hmm_label_colors
                print(f"HMM Prediction successful, stored {len(hmm_semantic_labels)} predicted sections.")
                self.status_label.config(text="HMM Prediction Complete", foreground="green")
                self.show_hmm_var.set(True) # Automatically switch view to HMM results
            self.plot_manager.display_analysis_results() # Refresh plots
        else:
            # Prediction failed (error message shown by predictor)
            self.status_label.config(text="HMM Prediction Failed!", foreground="red")
            # Ensure HMM keys are removed on failure
            self.track_data[1].pop('hmm_section_starts', None)
            self.track_data[1].pop('hmm_semantic_labels', None)
            self.track_data[1].pop('hmm_label_colors', None)
            self.show_hmm_var.set(False) # Ensure view is not HMM
            self.plot_manager.display_analysis_results() # Refresh plots

        # Update button states after prediction attempt
        self._update_hmm_button_state()

    def _on_closing(self):
        """Handles the window close event (clicking the 'X' button).

        Ensures the audio playback stream is stopped and cleaned up properly
        before destroying the main Tkinter window.
        """
        print("Window closing...")
        self.playback_manager.cleanup() # Stop audio stream and release resources
        self.master.destroy() # Close the Tkinter window

# --- Main Execution Block ---
if __name__ == "__main__":
    """Main entry point when the script is executed directly."""
    root = tk.Tk() # Create the main Tkinter window
    app = AudioAnalyzerApp(root) # Instantiate the application class
    root.mainloop() # Start the Tkinter event loop
