# main_app_backup.py


"""
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
    Manages UI, state, analysis orchestration, plot embedding, and audio playback.
    """
    def __init__(self, master):
        """Initialize the GUI application."""
        self.master = master
        master.title("Audio Analyzer Tool")
        master.geometry("1100x900")
        master.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Application State Variables ---
        self.file_path = {1: None, 2: None}
        self.track_data = {1: None, 2: None}
        self.track_names = {1: "Track 1", 2: "Track 2"}
        self.plot_widgets = {} # PlotManager will manage this dict's content
        # GUI elements will be created by gui_builder and assigned to self
        # (Initialize to None, set by build_gui)
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
        self.playhead_line = None

        # --- Instantiate Managers ---
        self.playback_manager = PlaybackManager(
            master=self.master,
            on_state_change=self._update_playback_buttons_state_from_manager,
            on_position_update=self._update_playhead_display
        )
        self.plot_manager = PlotManager(self)
        # *** Initialize HMMPredictor with GMMHMM paths ***
        self.hmm_predictor = HMMPredictor(GMMHMM_MODEL_PATH, GMMHMM_AUX_PATH)
        self.file_manager = FileManager(self) # Instantiate FileManager

        # --- Tkinter Control Variables ---
        self.mode = tk.StringVar(value="single")
        self.analysis_labels = {
            "sections": "Sections", "chroma": "Chroma/Labeling", "hpss": "HPSS",
            "stereo": "Stereo Width", "low_end": "Low-End Energy",
            "dyn_range": "Dynamic Range", "band_plot": "Band Analysis Plot"
        }
        # Default analysis options
        self.analysis_vars = {k: tk.BooleanVar(value=(k in ["sections", "chroma", "low_end"])) for k in self.analysis_labels.keys()} # Added low_end default
        self.use_manual_bpm = tk.BooleanVar(value=False)
        self.manual_bpm_entry_var = tk.StringVar()
        self.show_pre_cleanup_labels_var = tk.BooleanVar(value=False)
        self.show_hmm_var = tk.BooleanVar(value=False) # Controls HMM vs Original view

        # --- Build GUI using the builder ---
        self.style = ttk.Style(); self.style.theme_use('clam') # Or another theme like 'alt', 'default'
        build_gui(self) # Pass self (the app instance) to the builder function

        # --- Initial UI State ---
        self.update_ui_for_mode()
        self.update_manual_bpm_state()

    # --- GUI Building Methods Removed (Moved to gui_builder.py) ---
    # --- Plot Management Methods Removed (Moved to plot_manager.py) ---
    # --- File Operation Methods Removed (Moved to file_manager.py) ---

    # --- UI Update Methods (Remain in main app) ---
    def update_manual_bpm_state(self):
        """Enables/disables manual BPM controls based on checkbox and data."""
        analysis_done = bool(self.track_data.get(1)) # Check if track 1 data exists
        state = tk.NORMAL if self.use_manual_bpm.get() and analysis_done else tk.DISABLED
        if hasattr(self, 'manual_bpm_entry') and self.manual_bpm_entry: self.manual_bpm_entry.config(state=state)
        if hasattr(self, 'update_bpm_button') and self.update_bpm_button: self.update_bpm_button.config(state=state)

    def update_ui_for_mode(self):
        """Updates UI elements based on the selected mode (Single/Compare)."""
        mode = self.mode.get(); is_compare = mode == "compare"
        analyze_btn_text = "Analyze & Compare" if is_compare else "Analyze Track"
        bpm_label_text = "Manual Tempo (Track 1)" if is_compare else "Manual Tempo"
        if hasattr(self, 'analyze_button') and self.analyze_button: self.analyze_button.config(text=analyze_btn_text)
        if hasattr(self, 'manual_bpm_frame') and self.manual_bpm_frame: self.manual_bpm_frame.config(text=bpm_label_text)
        if hasattr(self, 'select_button2') and self.select_button2 and hasattr(self, 'file_label2') and self.file_label2:
            if is_compare: self.select_button2.grid(); self.file_label2.grid()
            else: self.select_button2.grid_remove(); self.file_label2.grid_remove(); self.file_path[2] = None; self.track_data[2] = None;
            if self.file_label2: self.file_label2.config(text="No file selected")
        self.playback_manager.stop()
        self.update_analyze_button_state(); self.plot_manager.clear_plots(); self.plot_manager.add_placeholder_labels()
        if hasattr(self, 'manual_bpm_check') and self.manual_bpm_check: self.manual_bpm_check.config(state=tk.DISABLED)
        self.update_manual_bpm_state()
        if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
        if self.section_editor: self.section_editor.clear(); self.section_editor.update_button.config(state=tk.DISABLED)
        self.show_pre_cleanup_labels_var.set(False); self.show_hmm_var.set(False)
        self._update_playback_buttons_state_from_manager('stopped')

    def update_analyze_button_state(self):
        """Enables or disables the 'Analyze' button based on file selection."""
        mode = self.mode.get(); state = tk.DISABLED
        if mode == "single" and self.file_path.get(1): state = tk.NORMAL
        elif mode == "compare" and self.file_path.get(1) and self.file_path.get(2): state = tk.NORMAL
        if hasattr(self, 'analyze_button') and self.analyze_button: self.analyze_button.config(state=state)
        self._update_save_button_state(); self._update_hmm_button_state()

    def _update_save_button_state(self):
        """Enables Save button only if in single mode and track 1 data exists."""
        state = tk.DISABLED
        if self.mode.get() == "single" and self.track_data.get(1): state = tk.NORMAL
        if self.save_button: self.save_button.config(state=state)

    def _update_hmm_button_state(self):
        """Enable HMM buttons only if single track analysis exists and model files found."""
        predict_state = tk.DISABLED
        show_state = tk.DISABLED
        print("DEBUG HMM Button Update: Checking state...") # DEBUG PRINT

        # Check mode and track data
        is_single_mode = self.mode.get() == 'single'
        has_track_data = bool(self.track_data.get(1))
        print(f" -> Single Mode: {is_single_mode}, Has Track 1 Data: {has_track_data}") # DEBUG PRINT

        if is_single_mode and has_track_data:
            # *** Check if the GMMHMM model files exist ***
            # --- DEBUG PRINT THE PATHS BEING CHECKED ---
            print(f" -> Checking Model Path: {GMMHMM_MODEL_PATH}")
            print(f" -> Checking Aux Path:   {GMMHMM_AUX_PATH}")
            # --- END DEBUG PRINT ---

            model_exists = os.path.exists(GMMHMM_MODEL_PATH)
            aux_exists = os.path.exists(GMMHMM_AUX_PATH)
            print(f" -> Model Exists: {model_exists}, Aux Exists: {aux_exists}") # DEBUG PRINT

            if model_exists and aux_exists:
                print(" -> Enabling Predict Button.") # DEBUG PRINT
                predict_state = tk.NORMAL
            else:
                print(f"INFO: GMMHMM model/aux files not found. Predict button disabled.")
                predict_state = tk.DISABLED

            # Check if HMM results already exist in track_data to enable Show HMM button
            if 'hmm_section_starts' in self.track_data[1]:
                show_state = tk.NORMAL
                print(" -> Enabling Show HMM Button (results found).") # DEBUG PRINT
            else:
                 print(" -> Disabling Show HMM Button (no results found).") # DEBUG PRINT


        # Apply states to buttons
        if hasattr(self, 'hmm_predict_button') and self.hmm_predict_button:
            self.hmm_predict_button.config(state=predict_state)
        if hasattr(self, 'show_hmm_button') and self.show_hmm_button:
            self.show_hmm_button.config(state=show_state)

        print("DEBUG HMM Button Update: Finished.") # DEBUG PRINT

    # --- Analysis Orchestration (Remains in main app) ---
    def run_analysis(self, is_update=False, manual_bpm_val=None):
        """Orchestrates the analysis and plotting process for selected tracks."""
        mode = self.mode.get()
        # --- Pre-Analysis Checks and UI Updates ---
        if not is_update:
            if mode == "single" and not self.file_path[1]: messagebox.showwarning("Missing File", "Please select Track 1."); return
            if mode == "compare" and not (self.file_path[1] and self.file_path[2]): messagebox.showwarning("Missing Files", "Please select both Track 1 and Track 2."); return
            self.playback_manager.stop(); self.status_label.config(text="Analyzing...", foreground="orange"); self.master.update_idletasks(); plt.close('all'); self.plot_manager.clear_plots(); self.track_data = {1: None, 2: None}
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False); self.show_hmm_var.set(False); self._update_hmm_button_state()
        else:
            self.playback_manager.stop(); self.status_label.config(text="Re-analyzing with new BPM...", foreground="orange"); self.master.update_idletasks(); plt.close('all'); self.plot_manager.clear_plots()
            if manual_bpm_val is None: messagebox.showerror("Update Error", "Manual BPM value missing for update."); self.status_label.config(text="Update Error!", foreground="red"); return
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False); self.show_hmm_var.set(False); self._update_hmm_button_state()
        # --- Run Analysis Core Logic ---
        try:
            print("--- Analyzing Track 1 ---"); bpm_override_t1 = manual_bpm_val if is_update else None; results1 = self._analyze_single_track(1, bpm_override_t1)
            if results1 is None: raise RuntimeError("Analysis failed for Track 1. Check console for details.")
            self.track_data[1] = results1
            self.playback_manager.set_audio(results1.get('y_processed'), results1.get('sr'))
            if mode == "compare":
                if not is_update:
                    print("\n--- Analyzing Track 2 ---"); results2 = self._analyze_single_track(2);
                    if results2 is None: raise RuntimeError("Analysis failed for Track 2. Check console for details.")
                    self.track_data[2] = results2
                # *** CORRECTED SYNTAX FOR ELSE BLOCK ***
                else:
                    # Keep existing Track 2 data during BPM update for Track 1
                    print("--- Keeping existing Track 2 analysis ---" if self.track_data.get(2) else "--- Track 2 data missing ---")
        except Exception as e:
            error_msg = f"Analysis Error: {e}"; self.status_label.config(text="Error!", foreground="red"); print(error_msg); traceback.print_exc(); messagebox.showerror("Analysis Error", f"{error_msg}\n\nCheck console output for more details.");
            self.plot_manager.clear_plots(); self.plot_manager.add_placeholder_labels(); plt.close('all');
            if hasattr(self.manual_bpm_check, 'config'): self.manual_bpm_check.config(state=tk.DISABLED)
            self.update_manual_bpm_state(); self._update_playback_buttons_state_from_manager('stopped'); self._update_save_button_state(); self._update_hmm_button_state(); return
        # --- Post-Analysis UI Updates ---
        self.plot_manager.display_analysis_results(); final_status = "BPM Update Complete" if is_update else "Analysis Complete"; self.status_label.config(text=final_status, foreground="green")
        if self.track_data.get(1):
             if hasattr(self.manual_bpm_check, 'config'): self.manual_bpm_check.config(state=tk.NORMAL)
             if not is_update: bpm_used = self.track_data[1].get('bpm', ''); self.manual_bpm_entry_var.set(f"{bpm_used:.2f}" if isinstance(bpm_used, (int, float)) else "")
             if self.toggle_labels_button and self.track_data[1].get('labels_before_cleanup'): self.toggle_labels_button.config(state=tk.NORMAL)
             elif self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
             if self.section_editor and mode == "single": self.section_editor.update_button.config(state=tk.NORMAL)
        self.update_manual_bpm_state(); self.update_analyze_button_state()
        print("Analysis/Update process finished.")

    def _analyze_single_track(self, track_num, manual_bpm_override=None):
        """Runs all analysis steps for a single specified track number."""
        results = defaultdict(lambda: None);
        try:
            print(f" Step 1: Loading/Preprocessing/Tempo (T{track_num})...")
            preprocess_data = load_and_preprocess(self.file_path[track_num], track_num, manual_bpm_override)
            if preprocess_data is None: return None
            results.update(preprocess_data)
            bpm = results.get('bpm'); sr = results.get('sr'); y_proc = results.get('y_processed'); hop = results.get('hop_length'); dur = results.get('duration_processed'); fp = results.get('file_path'); trim = results.get('trim_offset_sec', 0)
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [bpm, sr, hop, dur, fp]): missing = [k for k, v in {'bpm':bpm, 'sr':sr, 'hop_length':hop, 'duration':dur, 'file_path':fp}.items() if v is None or (isinstance(v, float) and math.isnan(v))]; print(f"Error: Missing essential data after preprocessing T{track_num}: {missing}."); return None
            if y_proc is None or y_proc.size == 0: print(f"Error: Processed audio is empty T{track_num}."); return None
            print(f" Step 1b: Calculating intermediate features (BPM: {bpm:.2f})...")
            spb = 60.0 / bpm if bpm > 0 else 0; spbar = 4 * spb if spb > 0 else 0; results['seconds_per_bar'] = spbar
            if dur > 0 and spb > 0: beat_times_est = np.arange(0, dur, spb); results['beat_frames'] = librosa.time_to_frames(beat_times_est, sr=sr, hop_length=hop)
            else: results['beat_frames'] = np.array([], dtype=int)
            results['rms'] = np.nan_to_num(librosa.feature.rms(y=y_proc, hop_length=hop)[0]); results['rms_times'] = librosa.frames_to_time(np.arange(len(results['rms'])), sr=sr, hop_length=hop)
            num_bars = int(np.ceil(dur / spbar)) if spbar > 0 else 0; bar_rms_list = []; bar_starts_list = []; rms_d = results.get('rms'); rms_t = results.get('rms_times')
            if num_bars > 0 and rms_d is not None and rms_t is not None and rms_d.size == rms_t.size:
                for i in range(num_bars):
                    start_rel = i * spbar; end_rel = min((i + 1) * spbar, dur); avg_rms_bar = 0.0
                    if start_rel < end_rel: mask = (rms_t >= start_rel) & (rms_t < end_rel)
                    if np.any(mask): valid_rms_in_bar = rms_d[mask][np.isfinite(rms_d[mask])]; avg_rms_bar = np.mean(valid_rms_in_bar) if valid_rms_in_bar.size > 0 else 0.0
                    bar_rms_list.append(avg_rms_bar); bar_starts_list.append(start_rel + trim)
                results['bar_rms_data'] = bar_rms_list; results['bar_starts_absolute'] = bar_starts_list
            else: results['bar_rms_data'] = None; results['bar_starts_absolute'] = None; print("Warning: Could not calculate bar RMS/Starts.")

            # --- Section Detection Step ---
            if self.analysis_vars['sections'].get():
                print(f" Step 2: Detecting Sections...")
                if results.get("bar_rms_data") is not None:
                    results["section_starts"], results["section_labels"] = detect_sections(results)
                else:
                    print(" -> Skipping Section Detection (bar RMS data missing).")
                    results["section_starts"], results["section_labels"] = None, None
            # *** CORRECTED SYNTAX FOR ELSE BLOCK ***
            else:
                # Skip if checkbox unchecked
                print(f" Step 2: Skipping Sections (Checkbox unchecked).")
                results["section_starts"]=None
                results["section_labels"]=None

            # --- Chroma/Cluster/Label Step ---
            run_chroma_step = self.analysis_vars['chroma'].get() or (results.get("section_starts") is not None)
            if run_chroma_step:
                 if results.get("section_starts") is not None: print(f" Step 3: Running Chroma/Cluster/Label Analysis..."); results.update(analyze_chroma_and_clusters(results)) # Returns dict
                 else: print(" Step 3: Skipping Chroma/Labeling (Sections missing)."); results.update(results_on_failure(None)) # Returns dict
            else: print(f" Step 3: Skipping Chroma/Labeling (Checkbox unchecked)."); results.update(results_on_failure(results.get("section_starts"))) # Returns dict

            # --- Spectrogram/HPSS/Stereo Step ---
            needs_spec = any(self.analysis_vars[k].get() for k in ['stereo','hpss','band_plot','low_end', 'chroma'])
            if needs_spec: print(f" Step 4: Running Spectrogram & HPSS Analysis..."); results.update(analyze_stereo_and_hpss(results, self.analysis_vars['stereo'].get(), self.analysis_vars['hpss'].get())) # Returns dict
            else: print(f" Step 4: Skipping Spectrogram/HPSS Analysis."); results.setdefault("spec", None); results.setdefault("freqs", None); results.setdefault("times_absolute", None); results.setdefault("width_matrix", None); results.setdefault("rms_harm", None); results.setdefault("rms_perc", None); results.setdefault("rms_time_absolute", None);

            # --- Low-End Energy Step ---
            if self.analysis_vars['low_end'].get():
                print(f" Step 5: Calculating Low-End Energy..."); spec=results.get("spec"); freqs=results.get("freqs"); times_abs=results.get("times_absolute"); trim_offset=results.get("trim_offset_sec", 0)
                if spec is not None and freqs is not None and times_abs is not None and spec.shape[0]==freqs.size and spec.shape[1]==times_abs.size:
                    low_freq_mask = freqs < 150
                    if np.any(low_freq_mask):
                        low_end_energy = np.sum(spec[low_freq_mask,:], axis=0); max_low_energy = np.max(low_end_energy)
                        results["low_energy_norm"] = (low_end_energy / max_low_energy if max_low_energy > 1e-9 else np.zeros_like(low_end_energy))
                        times_rel = times_abs - trim_offset; results["low_energy_times"] = times_rel
                        print(f" -> Stored low_energy_norm and RELATIVE low_energy_times (Length: {len(times_rel)})")
                    else: print(" Warning: No frequencies below 150Hz found."); results["low_energy_norm"]=None; results["low_energy_times"]=None
                else: print(" Skipping Low-End Energy (Spectrogram data missing or inconsistent)."); results["low_energy_norm"]=None; results["low_energy_times"]=None
            else: print(" Step 5: Skipping Low-End Energy (Checkbox unchecked)."); results["low_energy_norm"]=None; results["low_energy_times"]=None

            # --- Dynamic Range Step ---
            if self.analysis_vars['dyn_range'].get():
                print(f" Step 6: Calculating Dynamic Range..."); y_dyn=results.get("y_processed"); sr_dyn=results.get("sr"); hop_dyn=results.get("hop_length"); trim_dyn=results.get("trim_offset_sec",0); frame_len_dyn=2048
                try:
                    if y_dyn is not None and len(y_dyn) >= frame_len_dyn and sr_dyn and hop_dyn and trim_dyn is not None:
                        rms_f = librosa.feature.rms(y=y_dyn, frame_length=frame_len_dyn, hop_length=hop_dyn)[0]; y_frames = librosa.util.frame(y_dyn, frame_length=frame_len_dyn, hop_length=hop_dyn); peak_f = np.max(np.abs(y_frames), axis=0)
                        min_len = min(len(rms_f), len(peak_f)); rms_f = rms_f[:min_len]; peak_f = peak_f[:min_len]; results["dyn_range"] = np.nan_to_num(peak_f / (rms_f + 1e-9))
                        dyn_times_rel = librosa.frames_to_time(np.arange(len(results["dyn_range"])), sr=sr_dyn, hop_length=hop_dyn); results["dyn_times_absolute"] = dyn_times_rel + trim_dyn
                    else: print(" Skipping Dynamic Range (Audio data missing or too short)."); results["dyn_range"]=None; results["dyn_times_absolute"]=None
                except Exception as de: print(f" Error calculating Dynamic Range: {de}"); traceback.print_exc(); results["dyn_range"]=None; results["dyn_times_absolute"]=None
            else: results["dyn_range"]=None; results["dyn_times_absolute"]=None

            # --- Ensure default keys exist ---
            results.setdefault("semantic_labels", []); results.setdefault("label_colors", []); results.setdefault("section_features", []); results.setdefault("labels_before_cleanup", [])
            print(f"--- Analysis completed for Track {track_num} ---")
            # *** Convert defaultdict to dict before returning ***
            return dict(results)
        except Exception as e: print(f"--- Unhandled Error during analysis of Track {track_num} ---"); traceback.print_exc(); messagebox.showerror(f"Error Analyzing Track {track_num}", f"Unexpected error during analysis:\n{e}"); return None

    # --- Method for Handling Manual BPM Update ---
    def update_plots_with_manual_bpm(self):
        """Validates manual BPM input and triggers re-analysis for Track 1."""
        if not self.use_manual_bpm.get(): messagebox.showinfo("Info", "Please check 'Use Manual BPM' first."); return
        try: bpm_str = self.manual_bpm_entry_var.get(); bpm_val = float(bpm_str) if bpm_str else None; assert bpm_val is not None and 30 <= bpm_val <= 300
        except (ValueError, AssertionError): messagebox.showerror("Invalid BPM", "Manual BPM must be a number between 30 and 300."); return
        if hasattr(self, 'update_bpm_button'): self.update_bpm_button.config(state=tk.DISABLED)
        if hasattr(self, 'analyze_button'): self.analyze_button.config(state=tk.DISABLED)
        print(f"Triggering re-analysis with manual BPM: {bpm_val}"); self.run_analysis(is_update=True, manual_bpm_val=bpm_val)

    # --- Playback GUI Update Callbacks ---
    def _update_playback_buttons_state_from_manager(self, state):
        """Callback from PlaybackManager to update GUI button states (Play/Pause/Stop)."""
        # print(f"DEBUG GUI: Playback state changed to: {state}") # Verbose
        is_playing = (state == 'playing'); is_paused = (state == 'paused')
        can_play = (self.playback_manager.audio_data is not None and self.playback_manager.sample_rate is not None and self.mode.get() == "single")
        play_pause_state = tk.NORMAL if can_play else tk.DISABLED; stop_state = tk.NORMAL if (is_playing or is_paused) else tk.DISABLED
        if self.play_pause_button: self.play_pause_button.config(state=play_pause_state, text="Pause" if is_playing else "Play")
        if self.stop_button: self.stop_button.config(state=stop_state)

    def _update_playhead_display(self, frame):
        """Callback from PlaybackManager to update the visual playhead line on plots."""
        if frame is None or not self.playhead_line or not self.playback_manager.sample_rate:
            if self.playhead_line and self.playhead_line.get_visible(): self.playhead_line.set_visible(False); self._redraw_canvas()
            return
        current_time = frame / self.playback_manager.sample_rate; trim = self.track_data.get(1, {}).get('trim_offset_sec', 0); display_time = current_time + trim
        self.playhead_line.set_xdata([display_time, display_time]);
        if not self.playhead_line.get_visible(): self.playhead_line.set_visible(True)
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Requests a redraw of the Matplotlib canvas embedded in the Tkinter window."""
        waveform_plot_widgets = self.plot_widgets.get('Waveform', {}); canvas = waveform_plot_widgets.get('canvas')
        if canvas:
            try: canvas.draw_idle()
            except Exception as e: print(f"Error redrawing canvas: {e}")

    # --- Method to Toggle Label View ---
    def _toggle_label_view(self):
        """Switches the waveform plot labels between final and pre-cleanup."""
        print("DEBUG: _toggle_label_view called. Current state:", self.show_pre_cleanup_labels_var.get())
        self.plot_manager.display_analysis_results(); self._update_toggle_button_state(); self._update_save_button_state()

    def _update_toggle_button_state(self):
         """ Enable toggle button only if analysis is done, pre-cleanup data exists, and not currently showing HMM results. """
         state = tk.DISABLED
         if (self.track_data.get(1) and self.track_data[1].get('labels_before_cleanup') and self.mode.get() == 'single' and not self.show_hmm_var.get()): state = tk.NORMAL
         if self.toggle_labels_button: self.toggle_labels_button.config(state=state)

    # --- Waveform Click Handler (Seek, Edit, Split Functionality) ---
    def _on_waveform_click(self, event):
        """Handles mouse clicks on the waveform plot for seeking, editing, or splitting."""
        # print(f"DEBUG CLICK HANDLER: Start. Button={event.button}, x={event.xdata}, y={event.ydata}, Axes={event.inaxes}, State={event.state}, Key={event.key}") # Verbose

        # Ignore clicks outside plot axes
        if event.xdata is None or event.inaxes is None: return

        # Check for Shift key modifier via event.key attribute
        shift_pressed = (event.key == 'shift') # CORRECTED check

        # --- Shift + Left-Click (Button 1): Trigger Split ---
        if event.button == 1 and shift_pressed:
            print("DEBUG CLICK HANDLER: Split click detected (Shift + Left).")
            if self.mode.get() != 'single' or not self.track_data.get(1): print("DEBUG CLICK HANDLER: Not in single mode or no data. Split ignore."); return
            if self.show_hmm_var.get(): messagebox.showinfo("Split Error", "Cannot split sections while viewing HMM results. Uncheck 'Show HMM'."); return

            clicked_time = event.xdata
            section_starts = self.track_data[1].get("section_starts")
            if section_starts is None or not isinstance(section_starts, (list, np.ndarray)) or len(section_starts) == 0: print("DEBUG CLICK HANDLER: No valid section starts data found. Split ignore."); return

            # Find the index of the section containing the click
            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]; is_last_section = (i == len(section_starts) - 1)
                if is_last_section:
                    # Allow clicking anywhere in the last section
                    if clicked_time >= start: section_index = i; break
                else:
                    next_start = section_starts[i+1]
                    if clicked_time >= start and clicked_time < next_start: section_index = i; break

            if section_index != -1:
                print(f"DEBUG CLICK HANDLER: Showing split popup for section index {section_index} at time {clicked_time:.2f}s.")
                # Call the popup to confirm/adjust split time
                self._show_split_section_popup(section_index, clicked_time)
            else: print("DEBUG CLICK HANDLER: Could not determine clicked section index for split.")

        # --- Right-Click (Button 3): Trigger Edit ---
        elif event.button == 3:
            print("DEBUG CLICK HANDLER: Edit click detected (Right-click).")
            if self.mode.get() != 'single' or not self.track_data.get(1) or not self.section_editor: print("DEBUG CLICK HANDLER: Not in single mode or no data/editor. Edit ignore."); return
            if self.show_hmm_var.get(): messagebox.showinfo("Edit Info", "Please switch back to 'Original' view to edit sections."); return

            clicked_time = event.xdata
            section_starts = self.track_data[1].get("section_starts") # Edit uses original starts
            if section_starts is None or not isinstance(section_starts, (list, np.ndarray)) or len(section_starts) == 0: print("DEBUG CLICK HANDLER: No valid section starts data found. Edit ignore."); return

            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]; is_last_section = (i == len(section_starts) - 1)
                if is_last_section:
                    if clicked_time >= start: section_index = i; break
                else: next_start = section_starts[i+1];
                if clicked_time >= start and clicked_time < next_start: section_index = i; break

            if section_index != -1: print(f"DEBUG CLICK HANDLER: Click corresponds to section index {section_index} for edit."); self._show_section_edit_popup(section_index, event)
            else: print("DEBUG CLICK HANDLER: Could not determine clicked section index for edit.")

        # --- Left-Click (Button 1, no shift): Seek Playback ---
        elif event.button == 1 and not shift_pressed:
            if self.mode.get() != 'single': return
            if self.playback_manager.audio_data is None or self.playback_manager.sample_rate is None: return

            clicked_time = event.xdata; trim_offset = self.track_data.get(1, {}).get('trim_offset_sec', 0); sample_rate = self.playback_manager.sample_rate; audio_length = len(self.playback_manager.audio_data)
            relative_time = clicked_time - trim_offset; target_frame = int(relative_time * sample_rate); target_frame = max(0, min(target_frame, audio_length - 1))
            self.playback_manager.seek(target_frame)
            if self.playhead_line: display_time = target_frame / sample_rate + trim_offset; self.playhead_line.set_xdata([display_time, display_time]); self.playhead_line.set_visible(True); self._redraw_canvas();
        # else: print(f"DEBUG CLICK HANDLER: Ignored button {event.button}.") # Verbose


    # --- Section Edit Pop-up Method ---
    def _show_section_edit_popup(self, section_index, event=None):
        """Creates and shows the modal pop-up dialog for editing or merging a section."""
        # ... (code for this function remains the same as previous version) ...
        print(f"--- DEBUG MainApp: _show_section_edit_popup for index: {section_index} ---")
        if not self.section_editor or not self.track_data.get(1) or 'section_starts' not in self.track_data[1]: print("DEBUG MainApp: Section editor or original track data not available for editing."); return
        item_id_str = str(section_index)
        try:
            current_values = self.section_editor.tree.item(item_id_str, 'values')
            if not current_values or len(current_values) < 6: raise ValueError("Invalid values found in Treeview row.")
            section_num_display = current_values[1]; current_label = current_values[4]; current_color_name = current_values[5]
            num_sections = len(self.track_data[1]['section_starts'])
        except Exception as e: messagebox.showerror("Edit Error", f"Could not retrieve current values for section {section_index + 1}.\nError: {e}"); print(f"Error getting values for tree item {item_id_str}: {e}"); return
        editor_popup = tk.Toplevel(self.master); editor_popup.title(f"Edit/Merge Section {section_num_display}")
        editor_popup.transient(self.master); editor_popup.resizable(False, False)
        popup_frame = ttk.Frame(editor_popup, padding="10"); popup_frame.pack(expand=True, fill=tk.BOTH)
        edit_frame = ttk.LabelFrame(popup_frame, text="Edit Label/Color", padding=5); edit_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(edit_frame, text="Label:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W); label_combo = ttk.Combobox(edit_frame, values=ALLOWED_LABELS, state='readonly', width=15); label_combo.grid(row=0, column=1, padx=5, pady=5)
        label_combo.set(current_label if current_label in ALLOWED_LABELS else ALLOWED_LABELS[0])
        ttk.Label(edit_frame, text="Color:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W); color_combo = ttk.Combobox(edit_frame, values=list(COLOR_NAME_MAP.keys()), state='readonly', width=15); color_combo.grid(row=1, column=1, padx=5, pady=5)
        color_combo.set(current_color_name if current_color_name in COLOR_NAME_MAP else list(COLOR_NAME_MAP.keys())[0])
        ok_button = ttk.Button(edit_frame, text="Apply Edit", width=12, command=lambda p=editor_popup, item=item_id_str, lc=label_combo, cc=color_combo: self.section_editor._commit_popup_edit(p, item, lc, cc)); ok_button.grid(row=0, column=2, rowspan=2, padx=(10, 5), pady=5, sticky="ns")
        merge_frame = ttk.LabelFrame(popup_frame, text="Merge Section", padding=5); merge_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        merge_prev_button = ttk.Button(merge_frame, text="Merge with Previous", width=20, command=lambda p=editor_popup, idx=section_index: self._trigger_merge(p, idx, 'prev')); merge_prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        if section_index == 0: merge_prev_button.config(state=tk.DISABLED)
        merge_next_button = ttk.Button(merge_frame, text="Merge with Next", width=20, command=lambda p=editor_popup, idx=section_index: self._trigger_merge(p, idx, 'next')); merge_next_button.pack(side=tk.LEFT, padx=5, pady=5)
        if section_index >= num_sections - 1: merge_next_button.config(state=tk.DISABLED)
        cancel_button = ttk.Button(popup_frame, text="Cancel", width=8, command=editor_popup.destroy); cancel_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        editor_popup.update_idletasks()
        if event and hasattr(event, 'x_root') and hasattr(event, 'y_root'): final_x = event.x_root + 10; final_y = event.y_root + 10; screen_w = self.master.winfo_screenwidth(); screen_h = self.master.winfo_screenheight(); popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height(); final_x = max(0, min(final_x, screen_w - popup_w)); final_y = max(0, min(final_y, screen_h - popup_h)); editor_popup.geometry(f"+{final_x}+{final_y}")
        else: main_win = self.master; main_x = main_win.winfo_x(); main_y = main_win.winfo_y(); main_w = main_win.winfo_width(); main_h = main_win.winfo_height(); popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height(); center_x = main_x + (main_w // 2) - (popup_w // 2); center_y = main_y + (main_h // 2) - (popup_h // 2); editor_popup.geometry(f"+{center_x}+{center_y}")
        editor_popup.grab_set(); editor_popup.wait_window()


    # --- Merge Logic ---
    def _trigger_merge(self, popup, section_index, direction):
        """Called by merge buttons, determines indices and calls the main merge function."""
        # ... (code for this function remains the same as previous version) ...
        popup.destroy();
        if direction == 'prev':
            if section_index > 0: self._merge_section(section_index)
            else: messagebox.showerror("Merge Error", "Cannot merge the first section with previous.")
        elif direction == 'next':
            num_sections = len(self.track_data[1].get('section_starts', []));
            if section_index < num_sections - 1: self._merge_section(section_index + 1)
            else: messagebox.showerror("Merge Error", "Cannot merge the last section with next.")

    def _merge_section(self, remove_boundary_index):
        """
        Merges two adjacent sections by removing the boundary between them.
        Recalculates features for the merged section. Operates on ORIGINAL track_data.
        """
        # ... (code for this function remains the same as previous version,
        #      including call to self._recalculate_section_features) ...
        print(f"DEBUG: Attempting to merge by removing boundary at index {remove_boundary_index}")
        if not self.track_data.get(1): messagebox.showerror("Merge Error", "No track data loaded."); return
        t_data = self.track_data[1]
        section_starts = t_data.get('section_starts'); semantic_labels = t_data.get('semantic_labels'); label_colors = t_data.get('label_colors'); section_features = t_data.get('section_features')
        cluster_labels = t_data.get('cluster_labels'); labels_before_cleanup = t_data.get('labels_before_cleanup')
        if not all([isinstance(l, list) for l in [section_starts, semantic_labels, label_colors, section_features]]): messagebox.showerror("Merge Error", "Core section data lists are missing or invalid."); print("ERROR: Core section data lists missing for merge."); return
        num_sections_before_merge = len(section_starts)
        if remove_boundary_index <= 0 or remove_boundary_index >= num_sections_before_merge: messagebox.showerror("Merge Error", f"Invalid boundary index {remove_boundary_index} for merging."); print(f"ERROR: Invalid boundary index {remove_boundary_index} for merging {num_sections_before_merge} sections."); return
        keep_section_idx = remove_boundary_index - 1; remove_section_idx = remove_boundary_index
        try:
            kept_start_time = section_features[keep_section_idx]['start_time']; removed_end_time = section_features[remove_section_idx]['end_time']; new_duration_sec = removed_end_time - kept_start_time; new_duration_bars = round(new_duration_sec / t_data['seconds_per_bar']) if t_data.get('seconds_per_bar', 0) > 0 else 0
            print(f"DEBUG: Removing data for section index {remove_section_idx} (boundary index {remove_boundary_index})")
            del section_starts[remove_boundary_index]; del semantic_labels[remove_section_idx]; del label_colors[remove_section_idx]; del section_features[remove_section_idx]
            if cluster_labels and len(cluster_labels) == num_sections_before_merge: del cluster_labels[remove_section_idx]
            if labels_before_cleanup and len(labels_before_cleanup) == num_sections_before_merge: del labels_before_cleanup[remove_section_idx]
            print(f"DEBUG: Updating kept section index {keep_section_idx} end time and duration.")
            section_features[keep_section_idx]['end_time'] = removed_end_time; section_features[keep_section_idx]['duration_sec'] = new_duration_sec; section_features[keep_section_idx]['duration_bars'] = new_duration_bars
            print(f"DEBUG: Recalculating features for merged section {keep_section_idx} ({kept_start_time:.2f} - {removed_end_time:.2f})")
            # *** Use self._recalculate_section_features ***
            merged_features = self._recalculate_section_features(t_data, keep_section_idx)
            if merged_features:
                feature_keys_to_update = list(merged_features.keys()) # Update all recalculated keys
                for key in feature_keys_to_update: section_features[keep_section_idx][key] = merged_features[key]
                print("DEBUG: Features recalculated and updated in section_features list.")
            else: print("Warning: Feature recalculation failed for merged section.")
            for i in range(keep_section_idx + 1, len(section_features)):
                 if 'index' in section_features[i]: section_features[i]['index'] -= 1
                 else: print(f"Warning: 'index' key missing in section_features at list index {i}")
            t_data['section_starts'] = section_starts; t_data['semantic_labels'] = semantic_labels; t_data['label_colors'] = label_colors; t_data['section_features'] = section_features
            if cluster_labels: t_data['cluster_labels'] = cluster_labels
            if labels_before_cleanup: t_data['labels_before_cleanup'] = labels_before_cleanup
            if 'hmm_semantic_labels' in t_data: del t_data['hmm_semantic_labels']
            if 'hmm_label_colors' in t_data: del t_data['hmm_label_colors']
            if 'hmm_section_starts' in t_data: del t_data['hmm_section_starts']
            self.show_hmm_var.set(False); self._update_hmm_button_state()
            print(f"DEBUG: Merge successful. Removed section at original index {remove_section_idx}. Updated section at index {keep_section_idx}."); self.status_label.config(text="Sections Merged", foreground="blue")
            self.plot_manager.display_analysis_results(); self._update_save_button_state()
        except IndexError as e: messagebox.showerror("Merge Error", f"Index error during merge: {e}. Lists might be inconsistent."); print(f"ERROR: Index error during merge: {e}"); traceback.print_exc()
        except Exception as e: messagebox.showerror("Merge Error", f"An unexpected error occurred during merge:\n{e}"); print(f"ERROR: Unexpected error during merge: {e}"); traceback.print_exc()

    # --- Split Logic ---
    def _show_split_section_popup(self, section_index, clicked_time):
        """Creates and shows the modal pop-up dialog for splitting a section using bar numbers."""
        print(f"--- DEBUG MainApp: _show_split_section_popup for index: {section_index} at time {clicked_time:.2f} ---")
        if not self.track_data.get(1) or not self.track_data[1].get('section_starts'):
            print("DEBUG MainApp: Track data or section starts not available for splitting.")
            return

        # Get current section label for display and bar information
        try:
            current_label = self.track_data[1]['semantic_labels'][section_index]
            section_num_display = section_index + 1  # Use 1-based index for display
            
            # Get bar information
            seconds_per_bar = self.track_data[1].get('seconds_per_bar', 0)
            if seconds_per_bar <= 0:
                messagebox.showerror("Split Error", "Cannot calculate bar position - missing tempo data.")
                return
                
            # Convert clicked time to bar number (fractional)
            trim_offset = self.track_data[1].get('trim_offset_sec', 0)
            clicked_time_rel = clicked_time - trim_offset
            clicked_bar = clicked_time_rel / seconds_per_bar
            
            # Get the boundary bars for this section
            section_start_time = self.track_data[1]['section_starts'][section_index]
            section_start_time_rel = section_start_time - trim_offset
            section_start_bar = section_start_time_rel / seconds_per_bar
            
            # Find end time (either next section start or end of track)
            if section_index + 1 < len(self.track_data[1]['section_starts']):
                section_end_time = self.track_data[1]['section_starts'][section_index + 1]
            else:
                section_end_time = self.track_data[1].get('duration_processed', 0) + trim_offset
            
            section_end_time_rel = section_end_time - trim_offset
            section_end_bar = section_end_time_rel / seconds_per_bar
            
        except IndexError:
            messagebox.showerror("Split Error", f"Cannot get label for section index {section_index}.")
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
        bar_var = tk.StringVar(value=f"{clicked_bar:.2f}")  # Use 2 decimal places for precision
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

        # Focus and Modality
        bar_entry.focus_set()
        split_popup.grab_set()
        split_popup.wait_window()
        print(f"DEBUG MainApp: Split pop-up closed for section index {section_index}.")

    def _commit_split(self, popup, section_index, time_var):
        """Validates the split time from the popup and calls _split_section."""
        try:
            split_time_str = time_var.get()
            split_time = float(split_time_str)
        except ValueError:
            messagebox.showerror("Invalid Time", f"Split time must be a valid number.\nEntered: '{split_time_str}'", parent=popup)
            return

        # Perform validation (re-check bounds and min duration)
        t_data = self.track_data.get(1)
        if not t_data: return # Should not happen if popup opened
        section_starts = t_data.get('section_starts')
        num_sections_before = len(section_starts) if section_starts else 0

        if not (0 <= section_index < num_sections_before):
             messagebox.showerror("Split Error", "Invalid section index.", parent=popup); return

        original_start_time = section_starts[section_index]
        original_end_time = section_starts[section_index + 1] if section_index + 1 < num_sections_before else t_data.get('duration_processed', float('inf')) + t_data.get('trim_offset_sec', 0)

        # Use a small epsilon for time comparisons to avoid floating point issues near boundaries
        epsilon = 1e-6
        if not (original_start_time + epsilon < split_time < original_end_time - epsilon):
             messagebox.showerror("Split Error", f"Split time {split_time:.3f}s must be strictly within the section boundaries ({original_start_time:.3f}s - {original_end_time:.3f}s).", parent=popup)
             return

        if (split_time - original_start_time < MIN_SPLIT_SECTION_DURATION_SEC) or \
           (original_end_time - split_time < MIN_SPLIT_SECTION_DURATION_SEC):
            messagebox.showerror("Split Error", f"Resulting sections would be shorter than minimum duration ({MIN_SPLIT_SECTION_DURATION_SEC:.1f}s).", parent=popup)
            return

        # If all validations pass
        popup.destroy() # Close the popup
        self._split_section(section_index, split_time) # Perform the actual split

    def _commit_split_by_bar(self, popup, section_index, bar_var, seconds_per_bar, trim_offset):
        """Validates the split bar from the popup and converts to time for _split_section."""
        try:
            split_bar_str = bar_var.get()
            split_bar = float(split_bar_str)
        except ValueError:
            messagebox.showerror("Invalid Bar", f"Split bar must be a valid number.\nEntered: '{split_bar_str}'", parent=popup)
            return

        # Convert bar to absolute time
        split_time = (split_bar * seconds_per_bar) + trim_offset
        
        # Perform validation (re-check bounds and min duration)
        t_data = self.track_data.get(1)
        if not t_data: return  # Should not happen if popup opened
        section_starts = t_data.get('section_starts')
        num_sections_before = len(section_starts) if section_starts else 0

        if not (0 <= section_index < num_sections_before):
            messagebox.showerror("Split Error", "Invalid section index.", parent=popup)
            return

        original_start_time = section_starts[section_index]
        original_end_time = section_starts[section_index + 1] if section_index + 1 < num_sections_before else t_data.get('duration_processed', float('inf')) + t_data.get('trim_offset_sec', 0)

        # Calculate the minimum duration in bars and convert to seconds for validation
        min_split_section_duration_bars = MIN_SPLIT_SECTION_DURATION_SEC / seconds_per_bar
        
        # Convert original boundaries to bars for display in error messages
        original_start_bar = (original_start_time - trim_offset) / seconds_per_bar
        original_end_bar = (original_end_time - trim_offset) / seconds_per_bar

        # Use a small epsilon for time comparisons to avoid floating point issues near boundaries
        epsilon = 1e-6
        if not (original_start_time + epsilon < split_time < original_end_time - epsilon):
            messagebox.showerror("Split Error", 
                                f"Split bar {split_bar:.2f} must be strictly within the section boundaries "
                                f"(bar {original_start_bar:.2f} - {original_end_bar:.2f}).", 
                                parent=popup)
            return

        if ((split_time - original_start_time) < MIN_SPLIT_SECTION_DURATION_SEC) or \
           ((original_end_time - split_time) < MIN_SPLIT_SECTION_DURATION_SEC):
            messagebox.showerror("Split Error", 
                                f"Resulting sections would be shorter than minimum duration "
                                f"({MIN_SPLIT_SECTION_DURATION_SEC:.1f}s or {min_split_section_duration_bars:.2f} bars).", 
                                parent=popup)
            return

        # If all validations pass
        popup.destroy()  # Close the popup
        self._split_section(section_index, split_time)  # Perform the actual split with the time value

    def _split_section(self, section_index, split_time):
        """
        Splits a section at the given time point. Inserts new boundary,
        duplicates label/color, recalculates features for both new sections.
        Operates on ORIGINAL track_data.

        Args:
            section_index (int): The index of the section to split.
            split_time (float): The absolute time at which to split the section.
        """
        print(f"DEBUG: Executing split for section {section_index} at time {split_time:.3f}")
        if not self.track_data.get(1): return # Should already be checked
        t_data = self.track_data[1]

        # --- Get original data lists ---
        section_starts = t_data.get('section_starts')
        semantic_labels = t_data.get('semantic_labels')
        label_colors = t_data.get('label_colors')
        section_features = t_data.get('section_features') # List of dictionaries
        # Optional lists
        cluster_labels = t_data.get('cluster_labels')
        labels_before_cleanup = t_data.get('labels_before_cleanup')

        # Basic check (should be guaranteed by caller)
        if not all([isinstance(l, list) for l in [section_starts, semantic_labels, label_colors, section_features]]): return
        num_sections_before = len(section_starts)
        if not (0 <= section_index < num_sections_before): return

        try:
            insert_index = section_index + 1
            original_start_time = section_starts[section_index]
            original_end_time = section_features[section_index]['end_time'] # Get original end time

            # --- Insert new boundary and duplicate labels/colors ---
            print(f" -> Inserting boundary at {split_time:.3f}s (index {insert_index})")
            section_starts.insert(insert_index, split_time)
            original_label = semantic_labels[section_index] # Label of the section being split
            semantic_labels.insert(insert_index, original_label)
            original_color = label_colors[section_index]
            label_colors.insert(insert_index, original_color)

            # Duplicate optional list items if they exist
            if cluster_labels and len(cluster_labels) == num_sections_before:
                cluster_labels.insert(insert_index, cluster_labels[section_index])
            if labels_before_cleanup and len(labels_before_cleanup) == num_sections_before:
                labels_before_cleanup.insert(insert_index, labels_before_cleanup[section_index])

            # --- Handle section_features list ---
            original_feature_dict = section_features[section_index]
            # Create a deep copy for the second part to avoid modifying shared references
            new_section_feature_dict = copy.deepcopy(original_feature_dict)

            # Update first part (section_index) - times and durations
            print(f" -> Updating section {section_index} end time to {split_time:.3f}s")
            section_features[section_index]['end_time'] = split_time
            section_features[section_index]['duration_sec'] = split_time - original_start_time
            section_features[section_index]['duration_bars'] = round(section_features[section_index]['duration_sec'] / t_data['seconds_per_bar']) if t_data.get('seconds_per_bar', 0) > 0 else 0
            # Keep original start time and index for the first part

            # Update second part (insert_index) - times, durations, and index
            print(f" -> Creating new section {insert_index} from {split_time:.3f}s to {original_end_time:.3f}s")
            new_section_feature_dict['start_time'] = split_time
            new_section_feature_dict['end_time'] = original_end_time # End time is the original end time
            new_section_feature_dict['duration_sec'] = original_end_time - split_time
            new_section_feature_dict['duration_bars'] = round(new_section_feature_dict['duration_sec'] / t_data['seconds_per_bar']) if t_data.get('seconds_per_bar', 0) > 0 else 0
            new_section_feature_dict['index'] = insert_index # Set correct index for the new section
            new_section_feature_dict['original_label'] = original_label # Inherit original label

            # Insert the new feature dictionary into the list
            section_features.insert(insert_index, new_section_feature_dict)

            # Update indices for all subsequent feature dictionaries
            print(f" -> Updating indices for sections {insert_index + 1} onwards...")
            for i in range(insert_index + 1, len(section_features)):
                 if 'index' in section_features[i]:
                     section_features[i]['index'] += 1
                 else:
                     print(f"Warning: 'index' key missing in section_features at list index {i} during index update.")

            # --- Recalculate features for the two new sections ---
            # *** Use self._recalculate_section_features ***
            print(f"DEBUG: Recalculating features for split section part 1 (index {section_index})")
            recalculated_part1 = self._recalculate_section_features(t_data, section_index)
            if recalculated_part1:
                feature_keys_to_update = list(recalculated_part1.keys())
                for key in feature_keys_to_update: section_features[section_index][key] = recalculated_part1[key]
                print(" -> Part 1 features updated.")
            else: print(" -> Warning: Feature recalculation failed for part 1.")

            print(f"DEBUG: Recalculating features for split section part 2 (index {insert_index})")
            recalculated_part2 = self._recalculate_section_features(t_data, insert_index)
            if recalculated_part2:
                feature_keys_to_update = list(recalculated_part2.keys())
                for key in feature_keys_to_update: section_features[insert_index][key] = recalculated_part2[key]
                print(" -> Part 2 features updated.")
            else: print(" -> Warning: Feature recalculation failed for part 2.")

            # --- Update the main track_data dictionary ---
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
            self.show_hmm_var.set(False); self._update_hmm_button_state()

            print(f"DEBUG: Split successful. Section {section_index} split into {section_index} and {insert_index}.")
            self.status_label.config(text="Section Split", foreground="blue")

            # --- Refresh plots and editor ---
            self.plot_manager.display_analysis_results()
            self._update_save_button_state() # Enable save button

        except Exception as e:
            messagebox.showerror("Split Error", f"An unexpected error occurred during split:\n{e}")
            print(f"ERROR: Unexpected error during split: {e}")
            traceback.print_exc()


    # *** NEW METHOD: Recalculate features for a specific section ***
    def _recalculate_section_features(self, track_data, section_idx):
        """
        Helper method to recalculate various features for a specific section
        index after its boundaries have been updated (e.g., after merge or split).
        This is defined within the App class to easily access track_data.

        Args:
            track_data (dict): The main track data dictionary.
            section_idx (int): The index of the section to recalculate features for.

        Returns:
            dict or None: A dictionary containing the recalculated feature values,
                          or None if recalculation fails.
        """
        print(f"DEBUG App: Recalculating features for section index {section_idx}")
        new_features = {}
        try:
            # Get section info and base data
            section_features_list = track_data.get('section_features', [])
            if not (0 <= section_idx < len(section_features_list)):
                print(f"ERROR Recalc: Invalid section index {section_idx}")
                return None
            section_info = section_features_list[section_idx]
            start_time_abs = section_info.get('start_time')
            end_time_abs = section_info.get('end_time')

            # Base data needed for calculations
            trim_offset = track_data.get("trim_offset_sec", 0)
            duration_processed = track_data.get("duration_processed")
            rms_frames = track_data.get("rms")
            rms_times = track_data.get("rms_times") # Relative
            spec = track_data.get("spec")
            freqs = track_data.get("freqs")
            times_absolute = track_data.get("times_absolute") # Absolute
            spectral_centroid_frames = track_data.get("spectral_centroid_frames") # Frame-based
            spectral_bandwidth_frames = track_data.get("spectral_bandwidth_frames") # Frame-based
            spectral_contrast_frames = track_data.get("spectral_contrast_frames") # Frame-based
            low_energy_norm = track_data.get("low_energy_norm") # Frame based low energy
            low_energy_times = track_data.get("low_energy_times") # RELATIVE times for low energy

            if start_time_abs is None or end_time_abs is None:
                print("ERROR Recalc: Missing start/end time for section.")
                return None

            start_time_rel = start_time_abs - trim_offset
            end_time_rel = end_time_abs - trim_offset

            # --- Recalculate RMS-based features ---
            avg_rms, peak_rms, rms_std, rms_trend = np.nan, np.nan, np.nan, np.nan
            if rms_frames is not None and rms_times is not None:
                rms_mask = (rms_times >= start_time_rel) & (rms_times < end_time_rel)
                section_rms_vals = rms_frames[rms_mask][np.isfinite(rms_frames[rms_mask])]
                section_time_vals = rms_times[rms_mask][np.isfinite(rms_frames[rms_mask])]
                if section_rms_vals.size > 0:
                    avg_rms = np.mean(section_rms_vals)
                    peak_rms = np.max(section_rms_vals)
                    if section_rms_vals.size >= 2: rms_std = np.std(section_rms_vals)
                    else: rms_std = 0.0
                    if section_rms_vals.size > 1:
                        try: relative_time_vals = section_time_vals - section_time_vals[0]; slope, _, _, _, _ = scipy.stats.linregress(relative_time_vals, section_rms_vals); rms_trend = slope if np.isfinite(slope) else 0.0
                        except ValueError: rms_trend = 0.0
                    else: rms_trend = 0.0
                else: avg_rms=0.0; peak_rms=0.0; rms_std=0.0; rms_trend=0.0
            new_features["avg_rms"] = avg_rms; new_features["peak_rms"] = peak_rms; new_features["rms_std_dev"] = rms_std; new_features["rms_trend"] = rms_trend
            new_features["rms_std_dev_section"] = rms_std # Add the section-specific std dev

            # --- Recalculate Spectrogram-based features ---
            low_ratio, high_ratio, cent_avg, cent_std, bw_avg, cont_avg = (np.nan,) * 6
            spec_times_rel = None
            if times_absolute is not None: spec_times_rel = times_absolute - trim_offset

            if spec is not None and freqs is not None and spec_times_rel is not None:
                spec_indices = np.where((spec_times_rel >= start_time_rel) & (spec_times_rel < end_time_rel))[0]
                if spec_indices.size > 0:
                    section_spec = spec[:, spec_indices]; total_energy = np.sum(section_spec) + 1e-9
                    low_freq_mask = freqs < 150; high_freq_mask = freqs > 5000
                    low_ratio = np.sum(section_spec[low_freq_mask,:]) / total_energy if np.any(low_freq_mask) else 0.0
                    high_ratio = np.sum(section_spec[high_freq_mask,:]) / total_energy if np.any(high_freq_mask) else 0.0
                    if spectral_centroid_frames is not None and len(spectral_centroid_frames) == spec.shape[1]:
                         section_centroid = spectral_centroid_frames[spec_indices][np.isfinite(spectral_centroid_frames[spec_indices])]
                         if section_centroid.size > 0: cent_avg = np.mean(section_centroid)
                         else: cent_avg = 0.0
                         if section_centroid.size >= 2: cent_std = np.std(section_centroid)
                         else: cent_std = 0.0
                    else: cent_avg=0.0; cent_std=0.0
                    if spectral_bandwidth_frames is not None and len(spectral_bandwidth_frames) == spec.shape[1]:
                         section_bw = spectral_bandwidth_frames[spec_indices][np.isfinite(spectral_bandwidth_frames[spec_indices])]
                         if section_bw.size > 0: bw_avg = np.mean(section_bw)
                         else: bw_avg = 0.0
                    else: bw_avg=0.0
                    if spectral_contrast_frames is not None and spectral_contrast_frames.shape[1] == spec.shape[1]:
                         section_cont = spectral_contrast_frames[:, spec_indices]
                         finite_cont = section_cont[np.isfinite(section_cont)]
                         if finite_cont.size > 0: cont_avg = np.mean(finite_cont) # Simple mean across all bands/frames
                         else: cont_avg = 0.0
                    else: cont_avg=0.0
                else: low_ratio=0.0; high_ratio=0.0; cent_avg=0.0; cent_std=0.0; bw_avg=0.0; cont_avg=0.0
            new_features["low_end_ratio"] = low_ratio; new_features["high_end_ratio"] = high_ratio; new_features["spectral_centroid_avg"] = cent_avg; new_features["spectral_centroid_std_dev"] = cent_std; new_features["spectral_bandwidth_avg"] = bw_avg; new_features["spectral_contrast_avg"] = cont_avg
            new_features["centroid_std_dev_section"] = cent_std # Add the section-specific std dev

            # --- Recalculate Relative Position ---
            rel_pos = np.nan
            if start_time_abs is not None and duration_processed is not None and duration_processed > 0:
                rel_pos = max(0.0, min(start_time_abs / duration_processed, 1.0))
            new_features['relative_position'] = rel_pos

            # --- Recalculate Average Low-End Energy ---
            avg_low_e = np.nan
            if low_energy_norm is not None and low_energy_times is not None:
                 if isinstance(low_energy_times, np.ndarray):
                     le_indices = np.where((low_energy_times >= start_time_rel) & (low_energy_times < end_time_rel))[0]
                     if le_indices.size > 0:
                         finite_vals = low_energy_norm[le_indices][np.isfinite(low_energy_norm[le_indices])]
                         if finite_vals.size > 0: avg_low_e = np.mean(finite_vals)
                 else: print(f" -> Recalc Warning: low_energy_times is not NumPy array for section {section_idx}.")
            new_features['low_energy_norm'] = avg_low_e

            # --- Recalculate Delta Features ---
            # For recalculation after split/merge, delta features are tricky.
            # The 'previous' section might have changed.
            # Simplest approach: Set deltas to 0 or NaN after a split/merge.
            # More complex: Find the actual previous section and recalculate.
            # Setting to 0.0 for now.
            new_features['delta_rms'] = 0.0
            new_features['delta_centroid'] = 0.0

            # Ensure all expected keys exist, assign NaN if calculation failed
            expected_keys = [
                "avg_rms", "peak_rms", "rms_std_dev", "rms_trend",
                "low_end_ratio", "high_end_ratio", "spectral_centroid_avg",
                "spectral_centroid_std_dev", "spectral_bandwidth_avg", "spectral_contrast_avg",
                "relative_position", "low_energy_norm",
                "rms_std_dev_section", "centroid_std_dev_section",
                "delta_rms", "delta_centroid"
            ]
            for key in expected_keys:
                if key not in new_features or not np.isfinite(new_features[key]):
                    # Set to NaN if missing or non-finite, except for deltas which we default to 0 here
                    if key not in ['delta_rms', 'delta_centroid']:
                         new_features[key] = np.nan

            print(f" -> Recalculated features for index {section_idx}: { {k: f'{v:.2f}' if isinstance(v, float) else v for k,v in new_features.items()} }")
            return new_features

        except Exception as e:
            print(f"ERROR recalculating features for section {section_idx}: {e}")
            traceback.print_exc()
            return None # Indicate failure


    # --- Save/Load Methods --- # Now Handled by FileManager
    def _save_analysis(self): self.file_manager.save_analysis()
    def _load_analysis(self): self.file_manager.load_analysis()
    def _ask_save_status(self, parent): return self.file_manager.ask_save_status(parent)
    def select_file(self, track_num): self.file_manager.select_file(track_num)


    # --- Manual Section Editing Callback ---
    def _apply_section_edits_from_editor(self, new_labels, new_colors_hex):
        """Callback function executed by SectionEditor when 'Update' is clicked."""
        # ... (code for this function remains the same as previous version) ...
        print("DEBUG MainApp: Applying section edits received from editor...")
        if not self.track_data.get(1): messagebox.showerror("Update Error", "No track data loaded to apply edits to."); return
        current_labels = self.track_data[1].get('semantic_labels', [])
        if len(new_labels) != len(current_labels) or len(new_colors_hex) != len(current_labels): messagebox.showerror("Update Error", f"Data length mismatch when applying edits. Expected {len(current_labels)}, got {len(new_labels)} labels, {len(new_colors_hex)} colors."); return
        try:
            self.track_data[1]['semantic_labels'] = new_labels; self.track_data[1]['label_colors'] = new_colors_hex
            section_features = self.track_data[1].get('section_features', [])
            print(f"DEBUG: Syncing 'original_label' in section_features. Features count: {len(section_features)}, New labels count: {len(new_labels)}")
            if len(section_features) == len(new_labels):
                mismatches_found = False
                for i in range(len(section_features)):
                    if 'original_label' in section_features[i]:
                        if section_features[i]['original_label'] != new_labels[i]: section_features[i]['original_label'] = new_labels[i]
                    else: section_features[i]['original_label'] = new_labels[i]
                for i in range(len(section_features)):
                    if section_features[i].get('original_label') != new_labels[i]: print(f"WARNING: Mismatch persists at Section {i} after update!"); mismatches_found = True
                if not mismatches_found: print(" -> Sync successful: 'original_label' in section_features matches semantic_labels.")
            else: print("WARNING: Cannot sync 'original_label' due to length mismatch between section_features and new_labels.")
            if 'hmm_semantic_labels' in self.track_data[1]: del self.track_data[1]['hmm_semantic_labels']
            if 'hmm_label_colors' in self.track_data[1]: del self.track_data[1]['hmm_label_colors']
            if 'hmm_section_starts' in self.track_data[1]: del self.track_data[1]['hmm_section_starts']
            self.show_hmm_var.set(False); self._update_hmm_button_state()
            print("DEBUG MainApp: track_data updated with edits. Cleared HMM results."); print("DEBUG MainApp: Replotting waveform with updated labels/colors...")
            self.plot_manager.display_analysis_results(); messagebox.showinfo("Update Complete", "Section labels and colors updated."); self.status_label.config(text="Sections Updated", foreground="blue"); self._update_save_button_state()
        except Exception as e: messagebox.showerror("Update Error", f"Failed to apply section edits:\n{e}"); traceback.print_exc()

    # *** HMM Prediction Trigger Method ***
    def _trigger_hmm_prediction(self):
        """Callback for the 'Run HMM Prediction' button."""
        print("--- Triggering HMM Prediction ---")
        self.status_label.config(text="Running HMM...", foreground="orange")
        self.master.update_idletasks()
        
        if not self.track_data.get(1):
            messagebox.showerror("HMM Error", "Please analyze Track 1 first.")
            self.status_label.config(text="HMM Error!", foreground="red")
            return
            
        if not self.hmm_predictor.load_model():
            self.status_label.config(text="HMM Load Failed!", foreground="red")
            return
            
        hmm_results = self.hmm_predictor.predict(self.track_data[1])
        if hmm_results is not None:
            hmm_section_starts, hmm_semantic_labels, hmm_label_colors = hmm_results
            if not hmm_semantic_labels:
                print("HMM Prediction returned no labels.")
                self.status_label.config(text="HMM: No Sections Predicted", foreground="orange")
                self.track_data[1].pop('hmm_section_starts', None)
                self.track_data[1].pop('hmm_semantic_labels', None)
                self.track_data[1].pop('hmm_label_colors', None)
            else:
                # Store HMM results in separate keys (not overwriting original semantic_labels)
                self.track_data[1]['hmm_section_starts'] = hmm_section_starts
                self.track_data[1]['hmm_semantic_labels'] = hmm_semantic_labels  # FIXED: Store in hmm_semantic_labels
                self.track_data[1]['hmm_label_colors'] = hmm_label_colors
                print(f"HMM Prediction successful, stored {len(hmm_semantic_labels)} predicted sections.")
                self.status_label.config(text="HMM Prediction Complete", foreground="green")
                self.show_hmm_var.set(True)
            self.plot_manager.display_analysis_results()
        else:
            self.status_label.config(text="HMM Prediction Failed!", foreground="red")
            self.track_data[1].pop('hmm_section_starts', None)
            self.track_data[1].pop('hmm_semantic_labels', None)
            self.track_data[1].pop('hmm_label_colors', None)
            self.show_hmm_var.set(False)
            self.plot_manager.display_analysis_results()
        
        self._update_hmm_button_state()

    def _on_closing(self):
        """Handles window closing: stops audio stream and destroys window."""
        print("Window closing..."); self.playback_manager.cleanup(); self.master.destroy()

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk(); app = AudioAnalyzerApp(root); root.mainloop()
