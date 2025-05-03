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
# IMPLEMENTED: Section shifting functionality via dialog box.
# =============================================================================

"""
Main application module for the Audio Analyzer Tool.

This module defines the main Tkinter application class `AudioAnalyzerApp` which
orchestrates the GUI, analysis processes, plotting, playback, and file operations.
It integrates functionalities from various other modules within the project.
"""

import tkinter as tk
# Added simpledialog, though using Toplevel for custom layout
from tkinter import ttk, filedialog, messagebox, simpledialog
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
from debug_utils import debug_print, debug_timing, function_trace
from main_app_config import (
    ANALYSIS_BASE_FOLDER,
    PROJECT_BASE_FOLDER,
    HMM_OUTPUT_FOLDER,
    PERFECT_SUBFOLDER,
    WIP_SUBFOLDER,
    GMMHMM_MODEL_PATH,
    GMMHMM_AUX_PATH,
    N_FEATURES_EXPECTED,
    N_MIXTURES_EXPECTED,
    CLEANING_FLAGS_EXPECTED,
    MIN_SPLIT_SECTION_DURATION_SEC,
)

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
    from section_manager import SectionManager
    from ui_state_manager import UIStateManager


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
        self.shift_sections_button = None # Placeholder for the new button

        # --- Initialize UI State Manager BEFORE creating other managers ---
        from ui_state_manager import UIStateManager
        self.ui_manager = UIStateManager(self)

        # --- Instantiate Managers ---
        self.playback_manager = PlaybackManager(
            master=self.master,
            on_state_change=self.ui_manager.update_playback_buttons_state,
            on_position_update=self._update_playhead_display,
        )
        self.plot_manager = PlotManager(self) # Pass app instance for access
        self.hmm_predictor = HMMPredictor(GMMHMM_MODEL_PATH, GMMHMM_AUX_PATH)
        self.file_manager = FileManager(self) # Pass app instance
        self.section_manager = SectionManager(self) # Pass app instance

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
        self.ui_manager.update_ui_for_mode()
        self.ui_manager.update_manual_bpm_state()

    # --- GUI Building Methods Removed (Moved to gui_builder.py) ---
    # --- Plot Management Methods Removed (Moved to plot_manager.py) ---
    # --- File Operation Methods Removed (Moved to file_manager.py) ---
    # --- UI Update Methods (Remain in main app) --- moved to ui_state_manager.py

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
            if self.shift_sections_button: self.shift_sections_button.config(state=tk.DISABLED) # Disable shift button
            self.show_pre_cleanup_labels_var.set(False)
            self.show_hmm_var.set(False)
            self.ui_manager.update_hmm_button_state()
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
            if self.shift_sections_button: self.shift_sections_button.config(state=tk.DISABLED) # Disable shift button
            self.show_pre_cleanup_labels_var.set(False)
            self.show_hmm_var.set(False)
            self.ui_manager.update_hmm_button_state()

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
            self.ui_manager.update_manual_bpm_state()
            self.ui_manager.update_playback_buttons_state('stopped')
            self.ui_manager.update_save_button_state()
            self.ui_manager.update_hmm_button_state()
            if self.shift_sections_button: self.shift_sections_button.config(state=tk.DISABLED) # Ensure shift button disabled
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
            # Enable shift button in single mode
            if self.shift_sections_button and mode == "single":
                self.shift_sections_button.config(state=tk.NORMAL)

        self.ui_manager.update_manual_bpm_state() # Re-evaluate manual bpm entry state
        self.ui_manager.update_analyze_button_state() # Updates save/HMM buttons too
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

    # --- Method to Update Playhead Display ---

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
        self.ui_manager.update_toggle_button_state()
        self.ui_manager.update_save_button_state()

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

    # --- section_manager methods | Merge/Split/Shift ---

    def _trigger_merge(self, popup, section_index, direction):
        """Handles merge button clicks from the edit popup."""
        popup.destroy()

        if direction == 'prev':
            if section_index > 0:
                self.section_manager.merge_section(section_index)
            else:
                messagebox.showerror("Merge Error", "Cannot merge the first section with previous.")
        elif direction == 'next':
            num_sections = len(self.track_data[1].get('section_starts', []))
            if section_index < num_sections - 1:
                self.section_manager.merge_section(section_index + 1)
            else:
                messagebox.showerror("Merge Error", "Cannot merge the last section with next.")

    def _commit_split_by_bar(self, popup, section_index, bar_var, seconds_per_bar, trim_offset):
        """Validates and processes split operations."""
        try:
            split_bar = float(bar_var.get())
        except ValueError:
            messagebox.showerror("Invalid Bar", f"Split bar must be a valid number.", parent=popup)
            return

        # Convert bar to time
        split_time = (split_bar * seconds_per_bar) + trim_offset

        # Validate split via SectionManager
        is_valid, error_message = self.section_manager.validate_split_time(section_index, split_time)
        if not is_valid:
            messagebox.showerror("Split Error", error_message, parent=popup)
            return

        popup.destroy()
        self.section_manager.split_section(section_index, split_time)

    def _commit_section_shift(self, popup, start_bar_var, shift_amount_var):
        """Processes section shifting operations."""
        try:
            start_bar = int(start_bar_var.get())
            shift_bars = int(shift_amount_var.get())

            if start_bar < 1:
                raise ValueError("Start bar must be 1 or greater.")

            popup.destroy()
            self.section_manager.shift_sections(start_bar, shift_bars)

        except ValueError as ve:
            messagebox.showerror("Invalid Input", f"Please enter valid integer numbers.\nError: {ve}", parent=popup)
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=popup)
            traceback.print_exc()

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

    # --- Section Shift Methods ---
    def _trigger_shift_sections_popup(self):
        """
        Creates and displays a modal dialog box for shifting section boundaries.
        Called when the 'Shift Sections...' button is clicked.
        """
        # --- Pre-checks ---
        track_num = 1 # Assuming shift only works in single track mode for now
        if self.mode.get() != "single" or not self.track_data.get(track_num):
            messagebox.showwarning("Shift Sections", "Please load and analyze a single track first.")
            return

        td = self.track_data[track_num]
        if not td.get('section_starts') or not td.get('seconds_per_bar'):
            messagebox.showerror("Shift Sections Error", "Missing necessary section or timing data (seconds_per_bar) for shifting.")
            return
        if td['seconds_per_bar'] <= 0:
            messagebox.showerror("Shift Sections Error", "Invalid timing data (seconds_per_bar <= 0). Cannot shift by bars.")
            return

        # --- Create Dialog Window ---
        shift_popup = tk.Toplevel(self.master)
        shift_popup.title("Shift Section Boundaries")
        shift_popup.transient(self.master)
        shift_popup.resizable(False, False)
        popup_frame = ttk.Frame(shift_popup, padding="15")
        popup_frame.pack(expand=True, fill=tk.BOTH)

        # --- Input Fields ---
        start_bar_var = tk.StringVar(value="1") # Default start bar
        shift_amount_var = tk.StringVar(value="0") # Default shift amount

        # Start Bar
        ttk.Label(popup_frame, text="Shift sections starting FROM Bar #:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        start_bar_entry = ttk.Entry(popup_frame, textvariable=start_bar_var, width=8)
        start_bar_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Shift Amount
        ttk.Label(popup_frame, text="Shift Amount (bars, +/-):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        shift_amount_entry = ttk.Entry(popup_frame, textvariable=shift_amount_var, width=8)
        shift_amount_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # --- Buttons ---
        button_frame = ttk.Frame(popup_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(15, 0))

        ok_button = ttk.Button(button_frame, text="Apply Shift", width=12,
                              command=lambda p=shift_popup, sbv=start_bar_var, sav=shift_amount_var:
                              self._commit_section_shift(p, sbv, sav))
        ok_button.pack(side=tk.LEFT, padx=10)

        cancel_button = ttk.Button(button_frame, text="Cancel", width=10, command=shift_popup.destroy)
        cancel_button.pack(side=tk.LEFT, padx=10)

        # --- Focus and Modality ---
        start_bar_entry.focus_set()
        shift_popup.grab_set()
        shift_popup.wait_window()

    def _commit_section_shift(self, popup, start_bar_var, shift_amount_var):
        """
        Validates input from the shift dialog and calls the apply logic.
        Called by the 'Apply Shift' button in the shift popup.
        """
        try:
            start_bar = int(start_bar_var.get())
            shift_bars = int(shift_amount_var.get())

            # Basic validation
            if start_bar < 1:
                raise ValueError("Start bar must be 1 or greater.")
            # Allow zero shift (useful for testing maybe?)
            # if shift_bars == 0:
            #     raise ValueError("Shift amount cannot be zero.")

            # Further validation might be needed in _apply_section_shift
            # based on the number of sections, etc.

            popup.destroy() # Close the dialog first
            self.section_manager.shift_sections(
                start_bar, shift_bars
            )  # Call the SectionManager

        except ValueError as ve:
            messagebox.showerror("Invalid Input", f"Please enter valid integer numbers.\nError: {ve}", parent=popup)
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=popup)
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
        self.ui_manager.update_hmm_button_state()

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
