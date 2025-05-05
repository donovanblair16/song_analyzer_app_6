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
# FIXED: Added _apply_section_edits callback method.
# ADDED: Functionality to manually select HMM model files.
# FIXED: Crash on macOS when selecting HMM model due to potentially invalid initialdir.
# FIXED: Removed filetypes argument from askopenfilename for HMM selection to prevent potential macOS crash.
# ADDED: Default HMM model loading on startup.
# FIXED: Added missing 'import glob'.
# ADDED: More debug prints to _load_default_hmm_model.
# =============================================================================

"""
Main application module for the Audio Analyzer Tool.

This module defines the main Tkinter application class `AudioAnalyzerApp` which
orchestrates the GUI, analysis processes, plotting, playback, and file operations.
It integrates functionalities from various other modules within the project.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import os
import traceback
from collections import defaultdict, Counter
import numpy as np
import scipy
import threading
import datetime
import joblib
import math
import librosa
import librosa.display
import copy
import glob # *** ADDED MISSING IMPORT ***
from debug_utils import debug_print, debug_timing, function_trace
from main_app_config import (
    ANALYSIS_BASE_FOLDER,
    PROJECT_BASE_FOLDER,
    HMM_OUTPUT_FOLDER, # Keep this for HMM model selection/loading
    PERFECT_SUBFOLDER,
    WIP_SUBFOLDER,
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
    from playback_manager import PlaybackManager
    from section_editor import SectionEditor, ALLOWED_LABELS, COLOR_NAME_MAP, HEX_TO_COLOR_NAME
    from hmm_predictor import HMMPredictor
    from gui_builder import build_gui
    from plot_manager import PlotManager
    from file_manager import FileManager
    from section_manager import SectionManager
    from ui_state_manager import UIStateManager

except ImportError as e:
    print(f"Import Error: {e}\n"
          f"Could not import required modules.\n"
          f"Ensure all .py files are present and correct.")
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
    (Docstring attributes list remains largely the same)
    """
    def __init__(self, master):
        """Initializes the AudioAnalyzerApp GUI application."""
        self.master = master
        master.title("Audio Analyzer Tool")
        master.geometry("1100x900")
        master.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Application State Variables ---
        self.file_path = {1: None, 2: None}
        self.track_data = {1: None, 2: None}
        self.track_names = {1: "Track 1", 2: "Track 2"}
        self.plot_widgets = {}

        # --- GUI Widget Placeholders ---
        self.control_frame = None; self.notebook = None; self.tabs = {}
        self.status_label = None; self.toggle_labels_button = None
        self.show_hmm_button = None; self.play_pause_button = None
        self.stop_button = None; self.select_button1 = None; self.file_label1 = None
        self.select_button2 = None; self.file_label2 = None
        self.analyze_button = None; self.hmm_predict_button = None
        self.select_hmm_button = None
        self.load_button = None; self.save_button = None
        self.manual_bpm_frame = None; self.manual_bpm_check = None
        self.manual_bpm_entry = None; self.update_bpm_button = None
        self.section_editor = None; self.waveform_summary_label = None
        self.playhead_line = None
        self.shift_sections_button = None


        # --- Initialize UI State Manager ---
        self.ui_manager = UIStateManager(self)

        # --- Instantiate Managers ---
        self.playback_manager = PlaybackManager(
            master=self.master,
            on_state_change=self.ui_manager.update_playback_buttons_state,
            on_position_update=self._update_playhead_display,
        )
        self.plot_manager = PlotManager(self)
        # Pass only the output folder path
        self.hmm_predictor = HMMPredictor(HMM_OUTPUT_FOLDER) # Instantiated here
        self.file_manager = FileManager(self)
        self.section_manager = SectionManager(self)

        # --- Tkinter Control Variables ---
        self.mode = tk.StringVar(value="single")
        self.analysis_labels = {
            "sections": "Sections", "chroma": "Chroma/Labeling", "hpss": "HPSS",
            "stereo": "Stereo Width", "low_end": "Low-End Energy",
            "dyn_range": "Dynamic Range", "band_plot": "Band Analysis Plot"
        }
        self.analysis_vars = {k: tk.BooleanVar(value=(k in ["sections", "chroma", "low_end"])) for k in self.analysis_labels.keys()}
        self.use_manual_bpm = tk.BooleanVar(value=False)
        self.manual_bpm_entry_var = tk.StringVar()
        self.show_pre_cleanup_labels_var = tk.BooleanVar(value=False)
        self.show_hmm_var = tk.BooleanVar(value=False)


        # --- Build GUI using the builder ---
        self.style = ttk.Style(); self.style.theme_use('clam')
        build_gui(self) # Call the external function to populate the GUI

        # --- Attempt to load the most recent HMM model by default ---
        self._load_default_hmm_model() # NEW METHOD CALL

        # --- Initial UI State ---
        self.ui_manager.update_ui_for_mode()
        self.ui_manager.update_manual_bpm_state()
        # Update HMM button state based on whether default load succeeded
        self.ui_manager.update_hmm_button_state()


    # --- NEW Method to Load Default HMM ---
    def _load_default_hmm_model(self):
        """Attempts to find and load the most recent HMM model on startup."""
        # --- ADDED PRINT ---
        print("DEBUG MainApp: Entering _load_default_hmm_model method.")
        # --- END PRINT ---
        default_model_path = None
        default_aux_path = None
        latest_time = 0

        try:
            # --- ADDED PRINT ---
            print(f"DEBUG MainApp: Checking HMM_OUTPUT_FOLDER: {HMM_OUTPUT_FOLDER}")
            # --- END PRINT ---
            if not HMM_OUTPUT_FOLDER or not os.path.isdir(HMM_OUTPUT_FOLDER):
                print(f"Warning: HMM_OUTPUT_FOLDER ('{HMM_OUTPUT_FOLDER}') not found or invalid. Cannot load default model.")
                return # Exit if folder is invalid

            # Find all model files
            model_pattern = os.path.join(HMM_OUTPUT_FOLDER, "*_model.joblib")
            # --- ADDED PRINT ---
            print(f"DEBUG MainApp: Searching for models with pattern: {model_pattern}")
            # --- END PRINT ---
            model_files = glob.glob(model_pattern) # Uses glob
            # --- ADDED PRINT ---
            print(f"DEBUG MainApp: Found model files: {model_files}")
            # --- END PRINT ---

            if not model_files:
                print("No HMM model files found in the output folder.")
                return # Exit if no models found

            # Find the most recent model file
            print("DEBUG MainApp: Finding the most recent model file...")
            for model_file in model_files:
                try:
                    mod_time = os.path.getmtime(model_file)
                    # print(f"  - Checking {os.path.basename(model_file)}: Time {mod_time}") # Optional verbose print
                    if mod_time > latest_time:
                        latest_time = mod_time
                        default_model_path = model_file
                except OSError as e:
                    print(f"Warning: Could not get modification time for {model_file}. Error: {e}")
                    continue # Skip this file

            if not default_model_path:
                print("Could not determine the most recent model file (perhaps due to access errors).")
                return # Exit if no recent model determined

            print(f"DEBUG MainApp: Most recent model determined: {os.path.basename(default_model_path)}")

            # Find the corresponding aux file
            base_name = os.path.basename(default_model_path).replace('_model.joblib', '')
            expected_aux_path = os.path.join(os.path.dirname(default_model_path), f"{base_name}_aux.joblib")
            print(f"DEBUG MainApp: Expecting aux file: {os.path.basename(expected_aux_path)}")

            if not os.path.exists(expected_aux_path):
                print(f"Error: Matching aux file not found for the most recent model: {os.path.basename(expected_aux_path)}")
                return # Exit if aux file missing

            default_aux_path = expected_aux_path
            print(f"DEBUG MainApp: Found matching aux file.")

            # Set paths in the predictor
            self.hmm_predictor.model_path = default_model_path
            self.hmm_predictor.aux_path = default_aux_path
            print(f"DEBUG MainApp: Set predictor paths: model='{os.path.basename(self.hmm_predictor.model_path)}', aux='{os.path.basename(self.hmm_predictor.aux_path)}'")

            # Attempt to load the model
            print(f"DEBUG MainApp: Calling hmm_predictor.load_model()...")
            if self.hmm_predictor.load_model():
                # Check if status_label exists before configuring
                if self.status_label:
                    self.status_label.config(text=f"Default HMM Loaded: {base_name}", foreground="blue")
                print("Default HMM model loaded successfully.")
            else:
                # load_model will show an error messagebox if it fails
                if self.status_label:
                    self.status_label.config(text="Default HMM Load Failed!", foreground="red")
                # Clear paths if load failed
                self.hmm_predictor.model_path = None
                self.hmm_predictor.aux_path = None
                print("Default HMM model loading failed (check predictor logs/messagebox).")


        except Exception as e:
            print(f"Error during default HMM model loading: {e}")
            traceback.print_exc()
            if self.status_label: # Check if status label exists yet
                 self.status_label.config(text="Error loading default HMM!", foreground="red")
            # Ensure paths are cleared on error
            if self.hmm_predictor:
                 self.hmm_predictor.model_path = None
                 self.hmm_predictor.aux_path = None
        finally:
             # --- ADDED PRINT ---
             print("DEBUG MainApp: Exiting _load_default_hmm_model method.")
             # --- END PRINT ---


    # --- Analysis Orchestration ---
    def run_analysis(self, is_update=False, manual_bpm_val=None):
        """Orchestrates the analysis and plotting process for selected tracks."""
        # (Code remains largely the same, ensure it calls ui_manager updates)
        print("\nDEBUG: Analysis checkbox states at start of run_analysis:")
        for k, v in self.analysis_vars.items():
            print(f"  {k}: {v.get()}")
        print(f"DEBUG: Low-End Energy checkbox specifically: {self.analysis_vars['low_end'].get()}")

        mode = self.mode.get()
        # --- Pre-Analysis Checks and UI Updates ---
        if not is_update:
            if mode == "single" and not self.file_path[1]:
                messagebox.showwarning("Missing File", "Please select Track 1.")
                return
            if mode == "compare" and not (self.file_path[1] and self.file_path[2]):
                messagebox.showwarning("Missing Files", "Please select both Track 1 and Track 2.")
                return
            self.playback_manager.stop()
            self.status_label.config(text="Analyzing...", foreground="orange")
            self.master.update_idletasks()
            plt.close('all')
            self.plot_manager.clear_plots()
            self.track_data = {1: None, 2: None}
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            if self.shift_sections_button: self.shift_sections_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False)
            self.show_hmm_var.set(False)
            self.ui_manager.update_hmm_button_state() # Update HMM buttons
        else:
            self.playback_manager.stop()
            self.status_label.config(text="Re-analyzing with new BPM...", foreground="orange")
            self.master.update_idletasks()
            plt.close('all')
            self.plot_manager.clear_plots()
            if manual_bpm_val is None:
                messagebox.showerror("Update Error", "Manual BPM value missing for update.")
                self.status_label.config(text="Update Error!", foreground="red")
                return
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            if self.shift_sections_button: self.shift_sections_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False)
            self.show_hmm_var.set(False)
            self.ui_manager.update_hmm_button_state() # Update HMM buttons

        # --- Run Analysis Core Logic ---
        try:
            print("--- Analyzing Track 1 ---")
            bpm_override_t1 = manual_bpm_val if is_update else None
            results1 = self._analyze_single_track(1, bpm_override_t1)
            if results1 is None:
                raise RuntimeError("Analysis failed for Track 1. Check console for details.")
            self.track_data[1] = results1

            self.playback_manager.set_audio(results1.get('y_processed'), results1.get('sr'))

            if mode == "compare":
                if not is_update:
                    print("\n--- Analyzing Track 2 ---")
                    results2 = self._analyze_single_track(2)
                    if results2 is None:
                        raise RuntimeError("Analysis failed for Track 2. Check console for details.")
                    self.track_data[2] = results2
                else:
                    print("--- Keeping existing Track 2 analysis ---" if self.track_data.get(2) else "--- Track 2 data missing ---")

        except Exception as e:
            error_msg = f"Analysis Error: {e}"
            self.status_label.config(text="Error!", foreground="red")
            print(error_msg)
            traceback.print_exc()
            messagebox.showerror("Analysis Error", f"{error_msg}\n\nCheck console output for more details.")
            self.plot_manager.clear_plots()
            self.plot_manager.add_placeholder_labels()
            plt.close('all')
            if hasattr(self.manual_bpm_check, 'config'): self.manual_bpm_check.config(state=tk.DISABLED)
            self.ui_manager.update_manual_bpm_state()
            self.ui_manager.update_playback_buttons_state('stopped')
            self.ui_manager.update_save_button_state()
            self.ui_manager.update_hmm_button_state() # Update HMM buttons
            if self.shift_sections_button: self.shift_sections_button.config(state=tk.DISABLED)
            return

        # --- Post-Analysis UI Updates ---
        self.plot_manager.display_analysis_results()
        final_status = "BPM Update Complete" if is_update else "Analysis Complete"
        self.status_label.config(text=final_status, foreground="green")

        if self.track_data.get(1):
            if hasattr(self.manual_bpm_check, 'config'): self.manual_bpm_check.config(state=tk.NORMAL)
            if not is_update:
                bpm_used = self.track_data[1].get('bpm', '')
                self.manual_bpm_entry_var.set(f"{bpm_used:.2f}" if isinstance(bpm_used, (int, float)) else "")
            if self.toggle_labels_button and self.track_data[1].get('labels_before_cleanup'):
                self.toggle_labels_button.config(state=tk.NORMAL)
            elif self.toggle_labels_button:
                self.toggle_labels_button.config(state=tk.DISABLED)
            if self.section_editor and mode == "single":
                self.section_editor.populate(self.track_data[1])
                self.section_editor.update_button.config(state=tk.NORMAL)
            if self.shift_sections_button and mode == "single":
                self.shift_sections_button.config(state=tk.NORMAL)

        self.ui_manager.update_manual_bpm_state()
        self.ui_manager.update_analyze_button_state() # Updates save/HMM buttons too
        print("Analysis/Update process finished.")


    def _analyze_single_track(self, track_num, manual_bpm_override=None):
        """Runs the complete analysis pipeline for a single track."""
        # (Code remains largely the same)
        results = defaultdict(lambda: None)
        try:
            file_to_analyze = self.file_path.get(track_num)
            if not file_to_analyze:
                print(f"Error: No file path set for Track {track_num}.")
                return None

            print(f" Step 1: Loading/Preprocessing/Tempo (T{track_num})...")
            preprocess_data = load_and_preprocess(file_to_analyze, track_num, manual_bpm_override)
            if preprocess_data is None: return None
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
            spb = 60.0 / bpm if bpm > 0 else 0
            spbar = 4 * spb if spb > 0 else 0
            results['seconds_per_bar'] = spbar
            if dur > 0 and spb > 0:
                beat_times_est = np.arange(0, dur, spb)
                results['beat_frames'] = librosa.time_to_frames(beat_times_est, sr=sr, hop_length=hop)
            else:
                results['beat_frames'] = np.array([], dtype=int)
            results['rms'] = np.nan_to_num(librosa.feature.rms(y=y_proc, hop_length=hop)[0])
            results['rms_times'] = librosa.frames_to_time(np.arange(len(results['rms'])), sr=sr, hop_length=hop)

            num_bars = int(np.ceil(dur / spbar)) if spbar > 0 else 0
            bar_rms_list = []; bar_starts_list = []
            rms_d = results.get('rms'); rms_t = results.get('rms_times')
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
                    bar_starts_list.append(start_rel + trim)
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
                    results["section_starts"], results["section_labels"] = detect_sections(results)
                else:
                    print(" -> Skipping Section Detection (bar RMS data missing).")
                    results["section_starts"], results["section_labels"] = None, None
            else:
                print(f" Step 2: Skipping Sections (Checkbox unchecked).")
                results["section_starts"]=None
                results["section_labels"]=None

            # --- Chroma/Cluster/Label Step ---
            run_chroma_step = self.analysis_vars['chroma'].get() or (results.get("section_starts") is not None)
            if run_chroma_step:
                if results.get("section_starts") is not None:
                    print(f" Step 3: Running Chroma/Cluster/Label Analysis...")
                    results.update(analyze_chroma_and_clusters(results))
                else:
                    print(" Step 3: Skipping Chroma/Labeling (Sections missing).")
                    results.update(results_on_failure(None))
            else:
                print(f" Step 3: Skipping Chroma/Labeling (Checkbox unchecked).")
                results.update(results_on_failure(results.get("section_starts")))

            # --- Spectrogram/HPSS/Stereo Step ---
            needs_spec = any(self.analysis_vars[k].get() for k in ['stereo','hpss','band_plot','low_end', 'chroma'])
            if needs_spec:
                print(f" Step 4: Running Spectrogram & HPSS Analysis...")
                results.update(analyze_stereo_and_hpss(results, self.analysis_vars['stereo'].get(), self.analysis_vars['hpss'].get()))
            else:
                print(f" Step 4: Skipping Spectrogram/HPSS Analysis.")
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
                        results["low_energy_norm"] = (low_end_energy / max_low_energy if max_low_energy > 1e-9 else np.zeros_like(low_end_energy))
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
                frame_len_dyn=2048
                try:
                    if y_dyn is not None and len(y_dyn) >= frame_len_dyn and sr_dyn and hop_dyn and trim_dyn is not None:
                        rms_f = librosa.feature.rms(y=y_dyn, frame_length=frame_len_dyn, hop_length=hop_dyn)[0]
                        y_frames = librosa.util.frame(y_dyn, frame_length=frame_len_dyn, hop_length=hop_dyn)
                        peak_f = np.max(np.abs(y_frames), axis=0)
                        min_len = min(len(rms_f), len(peak_f))
                        rms_f = rms_f[:min_len]; peak_f = peak_f[:min_len]
                        results["dyn_range"] = np.nan_to_num(peak_f / (rms_f + 1e-9))
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
                results["dyn_range"]=None; results["dyn_times_absolute"]=None

            # --- Extract Features Step ---
            print(f" Step X: Extracting Section Features...")
            if results.get("section_starts") and results.get("semantic_labels"):
                try:
                    section_features_list, semantic_labels_list = extract_section_features(results)
                    results["section_features"] = section_features_list
                    results["semantic_labels"] = semantic_labels_list
                    print(f" -> Successfully extracted/calculated features for {len(section_features_list)} sections")
                except Exception as e:
                    print(f" -> ERROR in section feature extraction: {e}")
                    traceback.print_exc()
                    results["section_features"] = [] # Clear on error
            else:
                print(f" -> Skipping Section Feature Extraction (no section data available)")
                results["section_features"] = []

            # --- Ensure default keys exist ---
            results.setdefault("semantic_labels", [])
            results.setdefault("label_colors", [])
            results.setdefault("labels_before_cleanup", [])

            print(f"--- Analysis completed for Track {track_num} ---")
            return dict(results)

        except Exception as e:
            print(f"--- Unhandled Error during analysis of Track {track_num} ---")
            traceback.print_exc()
            messagebox.showerror(f"Error Analyzing Track {track_num}", f"Unexpected error during analysis:\n{e}")
            return None

    # --- Method for Handling Manual BPM Update ---
    def update_plots_with_manual_bpm(self):
        """Validates the manual BPM input and triggers re-analysis."""
        # (Code remains the same)
        if not self.use_manual_bpm.get(): messagebox.showinfo("Info", "Please check 'Use Manual BPM' first."); return
        try:
            bpm_str = self.manual_bpm_entry_var.get(); bpm_val = float(bpm_str) if bpm_str else None
            assert bpm_val is not None and 30 <= bpm_val <= 300
        except (ValueError, AssertionError): messagebox.showerror("Invalid BPM", "Manual BPM must be a number between 30 and 300."); return
        if hasattr(self, 'update_bpm_button'): self.update_bpm_button.config(state=tk.DISABLED)
        if hasattr(self, 'analyze_button'): self.analyze_button.config(state=tk.DISABLED)
        print(f"Triggering re-analysis with manual BPM: {bpm_val}")
        self.run_analysis(is_update=True, manual_bpm_val=bpm_val)

    # --- Method to Update Playhead Display ---
    def _update_playhead_display(self, frame):
        """Updates the position of the vertical playhead line on the waveform plot."""
        # (Code remains the same)
        if frame is None or not self.playhead_line or not self.playback_manager.sample_rate:
            if self.playhead_line and self.playhead_line.get_visible(): self.playhead_line.set_visible(False); self._redraw_canvas()
            return
        current_time = frame / self.playback_manager.sample_rate; trim = self.track_data.get(1, {}).get('trim_offset_sec', 0)
        display_time = current_time + trim; self.playhead_line.set_xdata([display_time, display_time])
        if not self.playhead_line.get_visible(): self.playhead_line.set_visible(True)
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Requests an idle redraw of the Matplotlib canvas for the waveform plot."""
        # (Code remains the same)
        canvas = self.plot_widgets.get('Waveform', {}).get('canvas')
        if canvas:
            try: canvas.draw_idle()
            except Exception as e: print(f"Error redrawing canvas: {e}")

    # --- Method to Toggle Label View ---
    def _toggle_label_view(self):
        """Switches the waveform plot labels between final and pre-cleanup versions."""
        # (Code remains the same)
        print("DEBUG: _toggle_label_view called. Current state:", self.show_pre_cleanup_labels_var.get())
        self.plot_manager.display_analysis_results()
        self.ui_manager.update_toggle_button_state()
        self.ui_manager.update_save_button_state()

    # --- Waveform Click Handler ---
    def _on_waveform_click(self, event):
        """Handles mouse clicks on the waveform plot axes."""
        # (Code remains largely the same)
        if event.xdata is None or event.inaxes is None or not self.track_data.get(1): return
        shift_pressed = (event.key == 'shift')
        if event.button == 1 and shift_pressed: # Split
            print("DEBUG CLICK HANDLER: Split click detected (Shift + Left).")
            if self.mode.get() != 'single': return
            if self.show_hmm_var.get(): messagebox.showinfo("Split Error", "Cannot split sections while viewing HMM results."); return
            clicked_time = event.xdata; section_starts = self.track_data[1].get("section_starts")
            if section_starts is None or len(section_starts) == 0: return
            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]; is_last = (i == len(section_starts) - 1); next_start = section_starts[i+1] if not is_last else float('inf')
                if (is_last and clicked_time >= start) or (clicked_time >= start and clicked_time < next_start): section_index = i; break
            if section_index != -1: self._show_split_section_popup(section_index, clicked_time)
        elif event.button == 3: # Edit/Merge
            print("DEBUG CLICK HANDLER: Edit click detected (Right-click).")
            if self.mode.get() != 'single' or not self.section_editor: return
            if self.show_hmm_var.get(): messagebox.showinfo("Edit Info", "Please switch back to 'Original' view to edit sections."); return
            clicked_time = event.xdata; section_starts = self.track_data[1].get("section_starts")
            if section_starts is None or len(section_starts) == 0: return
            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]; is_last = (i == len(section_starts) - 1); next_start = section_starts[i+1] if not is_last else float('inf')
                if (is_last and clicked_time >= start) or (clicked_time >= start and clicked_time < next_start): section_index = i; break
            if section_index != -1: self._show_section_edit_popup(section_index, event)
        elif event.button == 1 and not shift_pressed: # Seek
            if self.mode.get() != 'single': return
            if self.playback_manager.audio_data is None or self.playback_manager.sample_rate is None: return
            clicked_time = event.xdata; trim_offset = self.track_data.get(1, {}).get('trim_offset_sec', 0)
            sample_rate = self.playback_manager.sample_rate; audio_length = len(self.playback_manager.audio_data)
            relative_time = clicked_time - trim_offset; target_frame = int(relative_time * sample_rate)
            target_frame = max(0, min(target_frame, audio_length - 1)); self.playback_manager.seek(target_frame)
            if self.playhead_line:
                display_time = target_frame / sample_rate + trim_offset; self.playhead_line.set_xdata([display_time, display_time])
                self.playhead_line.set_visible(True); self._redraw_canvas()

    # --- Section Edit Pop-up Method ---
    def _show_section_edit_popup(self, section_index, event=None):
        """Creates and displays a modal pop-up dialog for editing or merging a section."""
        # (Code remains the same)
        print(f"--- DEBUG MainApp: _show_section_edit_popup for index: {section_index} ---")
        if not self.section_editor or not self.track_data.get(1) or 'section_starts' not in self.track_data[1]: print("DEBUG MainApp: Section editor or data not available."); return
        item_id_str = str(section_index)
        try:
            current_values = self.section_editor.tree.item(item_id_str, 'values')
            if not current_values or len(current_values) < 6: raise ValueError("Invalid values in Treeview row.")
            section_num_display = current_values[1]; current_label = current_values[4]; current_color_name = current_values[5]
            num_sections = len(self.track_data[1]['section_starts'])
        except Exception as e: messagebox.showerror("Edit Error", f"Could not retrieve current values for section {section_index + 1}.\nError: {e}"); return
        editor_popup = tk.Toplevel(self.master); editor_popup.title(f"Edit/Merge Section {section_num_display}"); editor_popup.transient(self.master); editor_popup.resizable(False, False)
        popup_frame = ttk.Frame(editor_popup, padding="10"); popup_frame.pack(expand=True, fill=tk.BOTH)
        edit_frame = ttk.LabelFrame(popup_frame, text="Edit Label/Color", padding=5); edit_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(edit_frame, text="Label:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W); label_combo = ttk.Combobox(edit_frame, values=ALLOWED_LABELS, state='readonly', width=15); label_combo.grid(row=0, column=1, padx=5, pady=5); label_combo.set(current_label if current_label in ALLOWED_LABELS else ALLOWED_LABELS[0])
        ttk.Label(edit_frame, text="Color:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W); color_combo = ttk.Combobox(edit_frame, values=list(COLOR_NAME_MAP.keys()), state='readonly', width=15); color_combo.grid(row=1, column=1, padx=5, pady=5); color_combo.set(current_color_name if current_color_name in COLOR_NAME_MAP else list(COLOR_NAME_MAP.keys())[0])
        ok_button = ttk.Button(edit_frame, text="Apply Edit", width=12, command=lambda p=editor_popup, item=item_id_str, lc=label_combo, cc=color_combo: self.section_editor._commit_popup_edit(p, item, lc, cc)); ok_button.grid(row=0, column=2, rowspan=2, padx=(10, 5), pady=5, sticky="ns")
        merge_frame = ttk.LabelFrame(popup_frame, text="Merge Section", padding=5); merge_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        merge_prev_button = ttk.Button(merge_frame, text="Merge with Previous", width=20, command=lambda p=editor_popup, idx=section_index: self._trigger_merge(p, idx, 'prev')); merge_prev_button.pack(side=tk.LEFT, padx=5, pady=5);
        if section_index == 0: merge_prev_button.config(state=tk.DISABLED)
        merge_next_button = ttk.Button(merge_frame, text="Merge with Next", width=20, command=lambda p=editor_popup, idx=section_index: self._trigger_merge(p, idx, 'next')); merge_next_button.pack(side=tk.LEFT, padx=5, pady=5);
        if section_index >= num_sections - 1: merge_next_button.config(state=tk.DISABLED)
        cancel_button = ttk.Button(popup_frame, text="Cancel", width=8, command=editor_popup.destroy); cancel_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        editor_popup.update_idletasks()
        if event and hasattr(event, 'x_root') and hasattr(event, 'y_root'):
            final_x = event.x_root + 10; final_y = event.y_root + 10; screen_w = self.master.winfo_screenwidth(); screen_h = self.master.winfo_screenheight(); popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height()
            final_x = max(0, min(final_x, screen_w - popup_w)); final_y = max(0, min(final_y, screen_h - popup_h)); editor_popup.geometry(f"+{final_x}+{final_y}")
        else:
            main_win = self.master; main_x = main_win.winfo_x(); main_y = main_win.winfo_y(); main_w = main_win.winfo_width(); main_h = main_win.winfo_height(); popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height()
            center_x = main_x + (main_w // 2) - (popup_w // 2); center_y = main_y + (main_h // 2) - (popup_h // 2); editor_popup.geometry(f"+{center_x}+{center_y}")
        editor_popup.grab_set(); editor_popup.wait_window()

    # --- Merge/Split/Shift Logic (delegated to SectionManager) ---
    def _trigger_merge(self, popup, section_index, direction):
        """Handles merge button clicks via SectionManager."""
        # (Code remains the same)
        popup.destroy()
        if direction == 'prev':
            if section_index > 0: self.section_manager.merge_section(section_index)
            else: messagebox.showerror("Merge Error", "Cannot merge the first section with previous.")
        elif direction == 'next':
            num_sections = len(self.track_data[1].get('section_starts', []))
            if section_index < num_sections - 1: self.section_manager.merge_section(section_index + 1)
            else: messagebox.showerror("Merge Error", "Cannot merge the last section with next.")

    def _show_split_section_popup(self, section_index, clicked_time):
        """Creates and displays a modal pop-up for splitting a section based on bar number."""
        # (Code remains the same)
        print(f"--- DEBUG MainApp: _show_split_section_popup for index: {section_index} at time {clicked_time:.2f} ---")
        if not self.track_data.get(1) or not self.track_data[1].get('section_starts'): return
        try:
            current_label = self.track_data[1]['semantic_labels'][section_index]; section_num_display = section_index + 1
            seconds_per_bar = self.track_data[1].get('seconds_per_bar', 0)
            if seconds_per_bar <= 0: messagebox.showerror("Split Error", "Cannot calculate bar position - missing tempo data."); return
            trim_offset = self.track_data[1].get('trim_offset_sec', 0); clicked_time_rel = clicked_time - trim_offset
            clicked_bar = clicked_time_rel / seconds_per_bar
            section_start_time = self.track_data[1]['section_starts'][section_index]; section_start_time_rel = section_start_time - trim_offset
            section_start_bar = section_start_time_rel / seconds_per_bar
            if section_index + 1 < len(self.track_data[1]['section_starts']): section_end_time = self.track_data[1]['section_starts'][section_index + 1]
            else: section_end_time = self.track_data[1].get('duration_processed', 0) + trim_offset
            section_end_time_rel = section_end_time - trim_offset; section_end_bar = section_end_time_rel / seconds_per_bar
        except (IndexError, KeyError) as e: messagebox.showerror("Split Error", f"Cannot get data for section {section_index}: {e}"); return
        split_popup = tk.Toplevel(self.master); split_popup.title(f"Split Section {section_num_display}"); split_popup.transient(self.master); split_popup.resizable(False, False)
        popup_frame = ttk.Frame(split_popup, padding="15"); popup_frame.pack(expand=True, fill=tk.BOTH)
        info_label = ttk.Label(popup_frame, text=f"Split Section {section_num_display} ('{current_label}') at bar:"); info_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        range_label = ttk.Label(popup_frame, text=f"Section range: Bar {section_start_bar:.2f} to {section_end_bar:.2f}"); range_label.grid(row=1, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        bar_var = tk.StringVar(value=f"{clicked_bar:.2f}"); bar_entry = ttk.Entry(popup_frame, textvariable=bar_var, width=15); bar_entry.grid(row=2, column=0, padx=(0, 5), pady=5, sticky=tk.EW)
        ttk.Label(popup_frame, text="bar number").grid(row=2, column=1, padx=(0, 5), pady=5, sticky=tk.W)
        button_frame = ttk.Frame(popup_frame); button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        split_button = ttk.Button(button_frame, text="Split", width=10, command=lambda p=split_popup, idx=section_index, bv=bar_var, spb=seconds_per_bar, to=trim_offset: self._commit_split_by_bar(p, idx, bv, spb, to)); split_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", width=10, command=split_popup.destroy); cancel_button.pack(side=tk.LEFT, padx=5)
        bar_entry.focus_set(); split_popup.grab_set(); split_popup.wait_window()
        print(f"DEBUG MainApp: Split pop-up closed for section index {section_index}.")

    def _commit_split_by_bar(self, popup, section_index, bar_var, seconds_per_bar, trim_offset):
        """Validates and processes split operations via SectionManager."""
        # (Code remains the same)
        try: split_bar = float(bar_var.get())
        except ValueError: messagebox.showerror("Invalid Bar", f"Split bar must be a valid number.", parent=popup); return
        split_time = (split_bar * seconds_per_bar) + trim_offset
        is_valid, error_message = self.section_manager.validate_split_time(section_index, split_time)
        if not is_valid: messagebox.showerror("Split Error", error_message, parent=popup); return
        popup.destroy()
        self.section_manager.split_section(section_index, split_time)

    def _trigger_shift_sections_popup(self):
        """Creates and displays a modal dialog box for shifting section boundaries."""
        # (Code remains the same)
        track_num = 1
        if self.mode.get() != "single" or not self.track_data.get(track_num): messagebox.showwarning("Shift Sections", "Please load and analyze a single track first."); return
        td = self.track_data[track_num]
        if not td.get('section_starts') or not td.get('seconds_per_bar'): messagebox.showerror("Shift Sections Error", "Missing necessary section or timing data."); return
        if td['seconds_per_bar'] <= 0: messagebox.showerror("Shift Sections Error", "Invalid timing data (seconds_per_bar <= 0)."); return
        shift_popup = tk.Toplevel(self.master); shift_popup.title("Shift Section Boundaries"); shift_popup.transient(self.master); shift_popup.resizable(False, False)
        popup_frame = ttk.Frame(shift_popup, padding="15"); popup_frame.pack(expand=True, fill=tk.BOTH)
        start_bar_var = tk.StringVar(value="1"); shift_amount_var = tk.StringVar(value="0")
        ttk.Label(popup_frame, text="Shift sections starting FROM Bar #:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        start_bar_entry = ttk.Entry(popup_frame, textvariable=start_bar_var, width=8); start_bar_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(popup_frame, text="Shift Amount (bars, +/-):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        shift_amount_entry = ttk.Entry(popup_frame, textvariable=shift_amount_var, width=8); shift_amount_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        button_frame = ttk.Frame(popup_frame); button_frame.grid(row=2, column=0, columnspan=2, pady=(15, 0))
        ok_button = ttk.Button(button_frame, text="Apply Shift", width=12, command=lambda p=shift_popup, sbv=start_bar_var, sav=shift_amount_var: self._commit_section_shift(p, sbv, sav)); ok_button.pack(side=tk.LEFT, padx=10)
        cancel_button = ttk.Button(button_frame, text="Cancel", width=10, command=shift_popup.destroy); cancel_button.pack(side=tk.LEFT, padx=10)
        start_bar_entry.focus_set(); shift_popup.grab_set(); shift_popup.wait_window()

    def _commit_section_shift(self, popup, start_bar_var, shift_amount_var):
        """Validates input and processes section shifting via SectionManager."""
        # (Code remains the same)
        try:
            start_bar = int(start_bar_var.get()); shift_bars = int(shift_amount_var.get())
            if start_bar < 1: raise ValueError("Start bar must be 1 or greater.")
            popup.destroy()
            self.section_manager.shift_sections(start_bar, shift_bars)
        except ValueError as ve: messagebox.showerror("Invalid Input", f"Please enter valid integer numbers.\nError: {ve}", parent=popup)
        except Exception as e: messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=popup); traceback.print_exc()

    # --- Callback for Section Editor ---
    def _apply_section_edits(self, new_labels, new_colors_hex):
        """Callback function passed to SectionEditor to apply changes."""
        # (Code remains the same)
        print(f"--- DEBUG MainApp: Applying {len(new_labels)} section edits ---")
        track_num = 1
        if self.mode.get() != "single": print("Warning: Section edits applied unexpectedly while not in single mode."); return
        if not self.track_data.get(track_num): messagebox.showerror("Error", "Cannot apply edits: Track data not loaded."); return
        current_num_sections = len(self.track_data[track_num].get('section_starts', []))
        if len(new_labels) != current_num_sections or len(new_colors_hex) != current_num_sections:
            messagebox.showerror("Error", f"Edit mismatch: Expected {current_num_sections} sections, but received {len(new_labels)} edits.")
            print(f"ERROR: Edit count ({len(new_labels)}) doesn't match current section count ({current_num_sections}).")
            if self.section_editor: self.section_editor.update_button.config(state=tk.DISABLED)
            return
        try:
            self.track_data[track_num]['semantic_labels'] = list(new_labels)
            self.track_data[track_num]['label_colors'] = list(new_colors_hex)
            self.file_manager.mark_as_modified() # Use the file manager method
            self.plot_manager.display_analysis_results()
            self.status_label.config(text=f"Applied {len(new_labels)} section edits.", foreground="blue")
            print(f"--- DEBUG MainApp: Finished applying edits. ---")
        except Exception as e:
             messagebox.showerror("Error Applying Edits", f"An unexpected error occurred:\n{e}")
             print(f"ERROR applying section edits: {e}")
             traceback.print_exc()
             self.status_label.config(text="Error Applying Edits!", foreground="red")

    # --- File Operations (delegated to FileManager) ---
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

    # --- HMM Model Selection and Prediction ---
    def _select_hmm_model(self):
        """Opens a dialog to select an HMM model file."""
        print("--- Selecting HMM Model ---")

        # --- FIX: Validate HMM_OUTPUT_FOLDER and set initial_dir safely ---
        try:
            # Ensure HMM_OUTPUT_FOLDER is defined and is a string
            if HMM_OUTPUT_FOLDER is None or not isinstance(HMM_OUTPUT_FOLDER, str):
                 print(f"Warning: HMM_OUTPUT_FOLDER is not a valid string path ({HMM_OUTPUT_FOLDER}). Defaulting initial directory.")
                 initial_dir = os.path.expanduser("~") # Default to home directory
            elif not os.path.isdir(HMM_OUTPUT_FOLDER):
                 print(f"Warning: HMM_OUTPUT_FOLDER is not a valid directory: {HMM_OUTPUT_FOLDER}. Defaulting initial directory.")
                 # Optionally try to create it? Or just default.
                 # os.makedirs(HMM_OUTPUT_FOLDER, exist_ok=True) # Uncomment to try creating
                 initial_dir = os.path.expanduser("~") # Default to home directory
            else:
                 initial_dir = HMM_OUTPUT_FOLDER # Use the valid path

            print(f"DEBUG: Using initial directory for file dialog: {initial_dir}")
        except Exception as e:
             print(f"Error determining initial directory: {e}. Defaulting.")
             initial_dir = os.path.expanduser("~") # Safe fallback
        # --- End FIX ---

        # --- Call the file dialog ---
        try:
            model_path = filedialog.askopenfilename(
                initialdir=initial_dir, # Use the validated initial_dir
                title="Select HMM Model File",
                # --- FIX: Remove filetypes to avoid potential macOS crash ---
                # filetypes=[("HMM Model Files", "*_model.joblib")],
            )
        except Exception as e:
             # Catch potential errors during the dialog call itself
             messagebox.showerror("File Dialog Error", f"Could not open the file selection dialog:\n{e}")
             print(f"ERROR: Failed to open file dialog: {e}")
             traceback.print_exc()
             return # Abort if dialog fails

        # --- Rest of the function (processing model_path) ---
        if not model_path:
            print("HMM Model selection cancelled.")
            # Clear paths if selection is cancelled
            self.hmm_predictor.model_path = None
            self.hmm_predictor.aux_path = None
            self.ui_manager.update_hmm_button_state() # Update button state
            return

        # --- Check if selected file actually ends with _model.joblib ---
        # This is important because we removed the filetypes filter
        if not model_path.endswith("_model.joblib"):
             messagebox.showerror("Invalid File", "Please select a valid HMM model file (ending in '_model.joblib').")
             self.hmm_predictor.model_path = None
             self.hmm_predictor.aux_path = None
             self.ui_manager.update_hmm_button_state()
             return
        # --- End Check ---


        base_name = os.path.basename(model_path).replace('_model.joblib', '')
        expected_aux_path = os.path.join(os.path.dirname(model_path), f"{base_name}_aux.joblib")

        if not os.path.exists(expected_aux_path):
            messagebox.showerror("HMM Load Error",
                                 f"Matching auxiliary file not found for the selected model:\n"
                                 f"Expected: {os.path.basename(expected_aux_path)}\n"
                                 f"In folder: {os.path.dirname(expected_aux_path)}")
            self.hmm_predictor.model_path = None
            self.hmm_predictor.aux_path = None
            self.status_label.config(text="HMM Aux File Missing!", foreground="red")
        else:
            self.hmm_predictor.model_path = model_path
            self.hmm_predictor.aux_path = expected_aux_path
            print(f"HMM Model selected: {os.path.basename(model_path)}")
            print(f"HMM Aux file found: {os.path.basename(expected_aux_path)}")
            self.status_label.config(text=f"HMM Model Loaded: {base_name}", foreground="blue")

        self.ui_manager.update_hmm_button_state()

    def _trigger_hmm_prediction(self):
        """Initiates HMM-based section prediction using the selected model."""
        print("--- Triggering HMM Prediction ---")
        self.status_label.config(text="Running HMM...", foreground="orange")
        self.master.update_idletasks()

        if not self.track_data.get(1):
            messagebox.showerror("HMM Error", "Please analyze Track 1 first.")
            self.status_label.config(text="HMM Error!", foreground="red")
            return

        # Attempt to load the SELECTED HMM model via predictor
        if not self.hmm_predictor.load_model():
            # Error message shown by load_model if paths aren't set or loading fails
            self.status_label.config(text="HMM Load Failed!", foreground="red")
            self.ui_manager.update_hmm_button_state()
            return  # Stop if model didn't load

        # Run prediction using the loaded model
        hmm_results = self.hmm_predictor.predict(self.track_data[1])

        if hmm_results is not None:
            # Unpack the results including posteriors (which may be None)
            if len(hmm_results) >= 4:  # Check if posteriors are included
                hmm_section_starts, hmm_semantic_labels, hmm_label_colors, hmm_posteriors = hmm_results
            else:
                hmm_section_starts, hmm_semantic_labels, hmm_label_colors = hmm_results
                hmm_posteriors = None
                
            if not hmm_semantic_labels:
                print("HMM Prediction returned no labels.")
                self.status_label.config(text="HMM: No Sections Predicted", foreground="orange")
                self.track_data[1].pop('hmm_section_starts', None)
                self.track_data[1].pop('hmm_semantic_labels', None)
                self.track_data[1].pop('hmm_label_colors', None)
                self.track_data[1].pop('hmm_posteriors', None)  # Also clear posteriors
                self.track_data[1].pop('hmm_feature_importance', None)  # Clear feature importance data
            else:
                self.track_data[1]['hmm_section_starts'] = hmm_section_starts
                self.track_data[1]['hmm_semantic_labels'] = hmm_semantic_labels
                self.track_data[1]['hmm_label_colors'] = hmm_label_colors
                self.track_data[1]['hmm_posteriors'] = hmm_posteriors  # Store posteriors
                
                # Calculate feature importance metrics (in a separate step to reduce compute in critical path)
                print("Calculating feature importance metrics...")
                try:
                    feature_importance = self.hmm_predictor.calculate_feature_importance(self.track_data[1])
                    if feature_importance:
                        self.track_data[1]['hmm_feature_importance'] = feature_importance
                        # Also store the indices of kept sections for reference (if available)
                        if 'hmm_kept_indices' in feature_importance:
                            self.track_data[1]['hmm_kept_indices'] = feature_importance['hmm_kept_indices']
                        
                        print(f"Feature importance calculation successful with {len(feature_importance['feature_names'])} features.")
                    else:
                        print("Feature importance calculation returned None.")
                except Exception as e:
                    print(f"Error calculating feature importance: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Log completion
                if hmm_posteriors is not None:
                    print(f"DEBUG: Stored HMM posteriors with shape {hmm_posteriors.shape}")
                    print(f"DEBUG: First few values: {hmm_posteriors[0][:3]}...")  # Print sample values
                else:
                    print("DEBUG: No HMM posteriors available to store")
                print(f"HMM Prediction successful, stored {len(hmm_semantic_labels)} predicted sections.")
                
                self.status_label.config(text="HMM Prediction Complete", foreground="green")
                self.show_hmm_var.set(True)
            self.plot_manager.display_analysis_results()
        else:
            self.status_label.config(text="HMM Prediction Failed!", foreground="red")
            self.track_data[1].pop('hmm_section_starts', None)
            self.track_data[1].pop('hmm_semantic_labels', None)
            self.track_data[1].pop('hmm_label_colors', None)
            self.track_data[1].pop('hmm_posteriors', None)  # Also clear posteriors
            self.track_data[1].pop('hmm_feature_importance', None)  # Clear feature importance data
            self.show_hmm_var.set(False)
            self.plot_manager.display_analysis_results()

        # Update button states after prediction attempt
        self.ui_manager.update_hmm_button_state()

    # --- Window Closing ---
    def _on_closing(self):
        """Handles the window close event."""
        # (Code remains the same)
        print("Window closing...")
        if self.file_manager.is_modified:
             save_choice = messagebox.askyesnocancel("Unsaved Changes", "You have unsaved changes. Save them before quitting?")
             if save_choice is True:
                 self.file_manager.save_analysis()
                 if self.file_manager.is_modified: print("DEBUG MainApp: Quit cancelled because save was cancelled."); return
             elif save_choice is None: print("DEBUG MainApp: Quit cancelled by user."); return
        self.playback_manager.cleanup()
        self.master.destroy()

# --- Main Execution Block ---
if __name__ == "__main__":
    """Main entry point when the script is executed directly."""
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
