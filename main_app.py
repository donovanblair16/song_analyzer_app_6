# =============================================================================
# FILE: main_app.py
# Contains the Tkinter GUI Application class and the main execution block.
# Imports and uses functions from audio_analysis.py and audio_plotting.py.
# Uses PlaybackManager for audio playback.
# Uses SectionEditor for manual editing.
# Includes Save/Load with Perfect/WIP subfolders and overwrite confirmation.
# Added right-click on waveform to edit sections.
# Added Merge Section functionality to edit pop-up.
# =============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import traceback
from collections import defaultdict
import numpy as np
import threading # For audio playback thread and lock
# import queue # No longer needed directly here
# import sounddevice as sd # No longer needed directly here
import datetime # For timestamp in saved filenames
import joblib # For saving/loading analysis data
import math # Added for isnan check

# --- Import functions from other modules ---
try:
    # *** Ensure audio_analysis is imported correctly ***
    from audio_analysis import (
        load_and_preprocess, detect_sections, analyze_chroma_and_clusters,
        analyze_stereo_and_hpss, results_on_failure # Removed calculate_bar_features import
    )
    # *** UPDATED: Import plotting module correctly ***
    import audio_plotting as ap # Use alias to avoid potential name clashes

    # *** Import the new PlaybackManager and SectionEditor ***
    from playback_manager import PlaybackManager
    # *** UPDATED: Import SectionEditor correctly ***
    from section_editor import SectionEditor, ALLOWED_LABELS, COLOR_NAME_MAP, HEX_TO_COLOR_NAME

except ImportError as e:
    print(f"Import Error: {e}\n"
          f"Could not import analysis/plotting/playback/editor modules.\n"
          f"Ensure all required .py files are in the same folder as 'main_app.py'.")
    try:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Import Error", f"Could not import required modules.\nEnsure all .py files are present.\n\nError: {e}")
        root.destroy()
    except tk.TclError: pass
    exit()

import librosa
import librosa.display

# --- Constants ---
# WARNING: Hardcoding the base path like this makes the application
# less portable. Consider making this configurable later.
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/Song Analyzer App 6/completed_analyses"
PERFECT_SUBFOLDER = "Perfect"
WIP_SUBFOLDER = "WIP"


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
        master.geometry("1100x900") # Set initial window size
        master.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Application State Variables ---
        self.file_path = {1: None, 2: None}
        self.track_data = {1: None, 2: None}
        self.track_names = {1: "Track 1", 2: "Track 2"}
        self.plot_widgets = {}
        self.waveform_summary_label = None
        self.toggle_labels_button = None
        self.save_button = None
        self.load_button = None
        self.section_editor = None
        self.playhead_line = None

        # --- Playback Manager ---
        self.playback_manager = PlaybackManager(
            master=self.master,
            on_state_change=self._update_playback_buttons_state_from_manager,
            on_position_update=self._update_playhead_display
        )

        # --- Tkinter Control Variables ---
        self.mode = tk.StringVar(value="single")
        self.analysis_labels = {
            "sections": "Sections", "chroma": "Chroma/Labeling", "hpss": "HPSS",
            "stereo": "Stereo Width", "low_end": "Low-End Energy",
            "dyn_range": "Dynamic Range", "band_plot": "Band Analysis Plot"
        }
        self.analysis_vars = {k: tk.BooleanVar(value=(k in ["sections", "chroma"])) for k in self.analysis_labels.keys()}
        self.use_manual_bpm = tk.BooleanVar(value=False)
        self.manual_bpm_entry_var = tk.StringVar()
        self.show_pre_cleanup_labels_var = tk.BooleanVar(value=False)

        # --- Build GUI ---
        self.style = ttk.Style(); self.style.theme_use('clam')
        self._create_control_panel()
        self._create_plot_notebook()

        # --- Initial UI State ---
        self.update_ui_for_mode()
        self.update_manual_bpm_state()

    def _create_control_panel(self):
        """Creates the top frame containing controls."""
        self.control_frame = ttk.Frame(self.master, padding="5")
        self.control_frame.pack(side=tk.TOP, fill=tk.X, pady=(5,0))

        # Action/Status/Toggle/Playback Frame
        action_status_frame = ttk.Frame(self.control_frame, padding="5")
        action_status_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
        self.status_label = ttk.Label(action_status_frame, text="", foreground="blue", width=20, anchor=tk.CENTER)
        self.status_label.pack(pady=5)
        self.toggle_labels_button = ttk.Checkbutton(action_status_frame, text="Show Pre-Cleanup Labels", variable=self.show_pre_cleanup_labels_var, command=self._toggle_label_view, state=tk.DISABLED)
        self.toggle_labels_button.pack(pady=5)
        playback_frame = ttk.Frame(action_status_frame); playback_frame.pack(pady=10)
        self.play_pause_button = ttk.Button(playback_frame, text="Play", command=self.playback_manager.toggle_play_pause, state=tk.DISABLED, width=6)
        self.play_pause_button.grid(row=0, column=0, padx=2)
        self.stop_button = ttk.Button(playback_frame, text="Stop", command=self.playback_manager.stop, state=tk.DISABLED, width=6)
        self.stop_button.grid(row=0, column=1, padx=2)

        # File Selection & Actions Area
        file_frame = ttk.LabelFrame(self.control_frame, text="Files & Actions", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
        self.select_button1 = ttk.Button(file_frame, text="Select Track 1 (.wav)", command=lambda: self.select_file(1)); self.select_button1.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.file_label1 = ttk.Label(file_frame, text="No file selected", relief=tk.SUNKEN, padding=3, width=40, anchor=tk.W); self.file_label1.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        self.select_button2 = ttk.Button(file_frame, text="Select Track 2 (.wav)", command=lambda: self.select_file(2)); self.select_button2.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.file_label2 = ttk.Label(file_frame, text="No file selected", relief=tk.SUNKEN, padding=3, width=40, anchor=tk.W); self.file_label2.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        self.analyze_button = ttk.Button(file_frame, text="Analyze Track", command=self.run_analysis, state=tk.DISABLED); self.analyze_button.grid(row=2, column=0, columnspan=2, padx=5, pady=(8,2), sticky=tk.EW)
        save_load_frame = ttk.Frame(file_frame); save_load_frame.grid(row=3, column=0, columnspan=2, pady=(5,2), sticky=tk.EW)
        self.load_button = ttk.Button(save_load_frame, text="Load Analysis", command=self._load_analysis); self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        self.save_button = ttk.Button(save_load_frame, text="Save Analysis", command=self._save_analysis, state=tk.DISABLED); self.save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        file_frame.grid_columnconfigure(1, weight=1)

        # Mode Selection
        mode_frame = ttk.LabelFrame(self.control_frame, text="Mode", padding=5); mode_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
        ttk.Radiobutton(mode_frame, text="Single Track", variable=self.mode, value="single", command=self.update_ui_for_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Compare Tracks", variable=self.mode, value="compare", command=self.update_ui_for_mode).pack(anchor=tk.W)

        # Manual BPM Input
        self.manual_bpm_frame = ttk.LabelFrame(self.control_frame, text="Manual Tempo", padding=5); self.manual_bpm_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
        self.manual_bpm_check = ttk.Checkbutton(self.manual_bpm_frame, text="Use Manual BPM", variable=self.use_manual_bpm, command=self.update_manual_bpm_state, state=tk.DISABLED); self.manual_bpm_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.manual_bpm_entry = ttk.Entry(self.manual_bpm_frame, textvariable=self.manual_bpm_entry_var, width=6, state=tk.DISABLED); self.manual_bpm_entry.grid(row=1, column=0, sticky=tk.W, pady=(2,0))
        self.update_bpm_button = ttk.Button(self.manual_bpm_frame, text="Update", command=self.update_plots_with_manual_bpm, state=tk.DISABLED, width=6); self.update_bpm_button.grid(row=1, column=1, sticky=tk.W, padx=(2,0), pady=(2,0))

        # Analysis Options Checkboxes
        options_frame = ttk.LabelFrame(self.control_frame, text="Analyses To Perform", padding=5); options_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
        max_rows = 4
        for i, (key, var) in enumerate(self.analysis_vars.items()):
             row = i % max_rows; col = i // max_rows
             cb = ttk.Checkbutton(options_frame, text=self.analysis_labels[key], variable=var); cb.grid(row=row, column=col, sticky=tk.W, padx=3, pady=1)

    def _create_plot_notebook(self):
        """Creates the ttk.Notebook widget to hold plot tabs."""
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.tab_names = ['Waveform', 'Energy/Balance', 'Timbre/Texture', 'Low-End', 'Dynamic Range', 'Stereo Width', 'HPSS', 'Root Note', 'Band Analysis']
        self.tabs = {}
        for name in self.tab_names:
            tab = ttk.Frame(self.notebook)
            self.tabs[name] = tab
            self.notebook.add(tab, text=name)
            self.plot_widgets[name] = {}
            if name == 'Waveform':
                editor_outer_frame = ttk.Frame(tab)
                editor_outer_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(10,0))
                self.section_editor = SectionEditor(editor_outer_frame,
                                                    apply_callback=self._apply_section_edits_from_editor,
                                                    app_ref=self)
                self.section_editor.pack(anchor=tk.CENTER, pady=0)
            else:
                 self._add_placeholder_label(name, f"Select file(s) and click Analyze.")

    def update_manual_bpm_state(self):
        """Enables/disables manual BPM controls."""
        analysis_done = bool(self.track_data.get(1))
        state = tk.NORMAL if self.use_manual_bpm.get() and analysis_done else tk.DISABLED
        if hasattr(self, 'manual_bpm_entry'): self.manual_bpm_entry.config(state=state)
        if hasattr(self, 'update_bpm_button'): self.update_bpm_button.config(state=state)

    def update_ui_for_mode(self):
        """Updates UI elements based on the selected mode."""
        mode = self.mode.get(); is_compare = mode == "compare"
        btn_text = "Analyze & Compare" if is_compare else "Analyze Track"
        bpm_label = "Manual Tempo (Track 1)" if is_compare else "Manual Tempo"
        if hasattr(self, 'analyze_button'): self.analyze_button.config(text=btn_text)
        if hasattr(self, 'manual_bpm_frame'): self.manual_bpm_frame.config(text=bpm_label)
        if hasattr(self, 'select_button2') and hasattr(self, 'file_label2'):
            if is_compare: self.select_button2.grid(); self.file_label2.grid()
            else: self.select_button2.grid_remove(); self.file_label2.grid_remove(); self.file_path[2] = None; self.track_data[2] = None
        self.playback_manager.stop()
        self.update_analyze_button_state(); self._clear_plots(); self._add_placeholder_labels()
        if hasattr(self, 'manual_bpm_check'): self.manual_bpm_check.config(state=tk.DISABLED)
        self.update_manual_bpm_state()
        if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
        if self.save_button: self.save_button.config(state=tk.DISABLED)
        if self.section_editor: self.section_editor.clear()
        self.show_pre_cleanup_labels_var.set(False)
        self._update_playback_buttons_state_from_manager('stopped')

    def update_analyze_button_state(self):
        """Enables or disables the 'Analyze' button."""
        mode = self.mode.get()
        state = tk.DISABLED
        if mode == "single" and self.file_path.get(1): state = tk.NORMAL
        elif mode == "compare" and self.file_path.get(1) and self.file_path.get(2): state = tk.NORMAL
        if hasattr(self, 'analyze_button'): self.analyze_button.config(state=state)
        self._update_save_button_state()

    def _update_save_button_state(self):
        """Enables Save button only if in single mode and track 1 data exists."""
        state = tk.DISABLED
        if self.mode.get() == "single" and self.track_data.get(1): state = tk.NORMAL
        if self.save_button: self.save_button.config(state=state)

    def select_file(self, track_num):
        """Handles file selection."""
        f_path = filedialog.askopenfilename(title=f"Select Track {track_num} WAV", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if f_path:
            self.file_path[track_num] = f_path; filename = os.path.basename(f_path)
            self.track_names[track_num] = filename
            label = self.file_label1 if track_num == 1 else self.file_label2
            if label: label.config(text=filename)
            self.playback_manager.stop()
            if hasattr(self, 'status_label'): self.status_label.config(text="")
            self._clear_plots(); self._add_placeholder_labels(); self.update_analyze_button_state()
            self.use_manual_bpm.set(False);
            if hasattr(self, 'manual_bpm_check'): self.manual_bpm_check.config(state=tk.DISABLED)
            self.update_manual_bpm_state()
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            if self.section_editor: self.section_editor.clear()
            self.show_pre_cleanup_labels_var.set(False)
            self.track_data[track_num] = None
            if self.mode.get() == "single" and track_num == 1: self.track_data[2] = None
            self._update_playback_buttons_state_from_manager('stopped')
        else:
            label = self.file_label1 if track_num == 1 else self.file_label2
            if label: label.config(text="Selection cancelled.")

    def _add_placeholder_label(self, tab_name, message):
        """Adds placeholder label to a tab, ensuring editor frame is ignored."""
        if tab_name in self.tabs:
            tab = self.tabs[tab_name]
            if not tab.winfo_exists(): return
            has_other_content = False
            widgets_to_destroy = []
            for w in tab.winfo_children():
                 if w.winfo_exists():
                     if isinstance(w, ttk.Label) and 'placeholder' in str(w.winfo_name()): widgets_to_destroy.append(w)
                     elif tab_name != 'Waveform' or not (hasattr(self, 'section_editor') and self.section_editor and (w == self.section_editor or w == self.section_editor.master)): has_other_content = True
            for w in widgets_to_destroy: w.destroy()
            if not has_other_content and tab_name != 'Waveform':
                ph = ttk.Label(tab, text=message, padding=20, anchor=tk.CENTER, name='placeholder')
                ph.pack(expand=True, fill=tk.BOTH)
                if tab_name not in self.plot_widgets: self.plot_widgets[tab_name] = {}
                self.plot_widgets[tab_name]['placeholder'] = ph

    def _add_placeholder_labels(self):
        """Adds placeholders to all tabs."""
        for name in self.tab_names:
            if name == 'Waveform':
                 if name in self.plot_widgets and 'placeholder' in self.plot_widgets[name]:
                     if self.plot_widgets[name]['placeholder'].winfo_exists(): self.plot_widgets[name]['placeholder'].destroy()
                     del self.plot_widgets[name]['placeholder']
                 continue
            is_analysis_selected = True; msg = f"Select file(s) and click Analyze."
            map_key = {'Low-End':'low_end', 'Dynamic Range':'dyn_range', 'Stereo Width':'stereo','HPSS':'hpss','Root Note':'chroma', 'Band Analysis':'band_plot', 'Energy/Balance':'chroma', 'Timbre/Texture':'chroma'}.get(name)
            if map_key and not self.analysis_vars[map_key].get(): is_analysis_selected = False; msg = "Analysis not selected."
            elif name == 'Band Analysis' and not (self.analysis_vars['stereo'].get() or self.analysis_vars['hpss'].get()): is_analysis_selected = False; msg = "Requires Stereo or HPSS Analysis."
            files_ok = (self.mode.get() == "single" and self.file_path.get(1)) or (self.mode.get() == "compare" and self.file_path.get(1) and self.file_path.get(2))
            if not is_analysis_selected or not files_ok: self._add_placeholder_label(name, msg if not is_analysis_selected else "Select file(s)...")
            else: self._add_placeholder_label(name, f"Ready for {name} analysis...")

    def _clear_plots(self):
        """Removes existing plots, toolbars, placeholders, and summary. Clears editor."""
        print("Clearing plots and summary...")
        for name, widgets in self.plot_widgets.items():
            if name in self.tabs:
                tab = self.tabs[name]
                if tab.winfo_exists():
                    widgets_to_destroy = []
                    widgets_to_keep = []
                    if name == 'Waveform' and hasattr(self, 'section_editor') and self.section_editor:
                        if self.section_editor.master and self.section_editor.master.winfo_exists(): widgets_to_keep.append(self.section_editor.master)
                    for widget_type, widget in widgets.items():
                         if widget_type in ['canvas', 'toolbar', 'placeholder', 'figure']: widgets_to_destroy.append(widget)
                    for child in tab.winfo_children():
                         if child not in widgets_to_keep:
                              is_plot_widget = False
                              for widget_dict in self.plot_widgets.values():
                                   if ('canvas' in widget_dict and widget_dict['canvas'] and hasattr(widget_dict['canvas'], 'get_tk_widget') and widget_dict['canvas'].get_tk_widget() and hasattr(widget_dict['canvas'].get_tk_widget(), 'master') and child == widget_dict['canvas'].get_tk_widget().master): is_plot_widget = True; break
                                   if 'placeholder' in widget_dict and widget_dict['placeholder'] and child == widget_dict['placeholder']: is_plot_widget = True; break
                              if is_plot_widget:
                                   if child not in widgets_to_keep: widgets_to_destroy.append(child)
                    for widget in widgets_to_destroy:
                         if widget:
                             try:
                                 if isinstance(widget, (FigureCanvasTkAgg)): tk_widget = widget.get_tk_widget(); tk_widget.master.destroy() if tk_widget and tk_widget.master.winfo_exists() else None
                                 elif hasattr(widget, 'winfo_exists') and widget.winfo_exists(): widget.destroy()
                             except Exception as e: print(f" Minor error destroying widget: {e}")
            if name == 'Waveform' and hasattr(self, 'waveform_summary_label') and self.waveform_summary_label:
                 try: self.waveform_summary_label.destroy() if self.waveform_summary_label.winfo_exists() else None
                 except Exception: pass
                 self.waveform_summary_label = None
        if self.section_editor: self.section_editor.clear()
        self.plot_widgets = {name: {} for name in self.tab_names}
        self.waveform_summary_label = None; self.playhead_line = None
        if self.track_data.get(1):
            self.track_data[1].pop('hmm_semantic_labels', None)
            self.track_data[1].pop('hmm_label_colors', None)

    # --- Main Analysis Runner ---
    def run_analysis(self, is_update=False, manual_bpm_val=None):
        """Orchestrates the analysis and plotting process."""
        mode = self.mode.get()
        if not is_update:
            if mode == "single" and not self.file_path[1]: messagebox.showwarning("Missing File", "Please select Track 1."); return
            if mode == "compare" and not (self.file_path[1] and self.file_path[2]): messagebox.showwarning("Missing Files", "Please select both Track 1 and Track 2."); return
            self.playback_manager.stop()
            self.status_label.config(text="Analyzing...", foreground="orange"); self.master.update_idletasks(); plt.close('all'); self._clear_plots(); self.track_data = {1: None, 2: None}
            if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
            if self.save_button: self.save_button.config(state=tk.DISABLED)
            self.show_pre_cleanup_labels_var.set(False)
        else:
             self.playback_manager.stop()
             self.status_label.config(text="Re-analyzing...", foreground="orange"); self.master.update_idletasks(); plt.close('all'); self._clear_plots()
             if manual_bpm_val is None: messagebox.showerror("Update Error", "Manual BPM value missing for update."); self.status_label.config(text="Update Error!", foreground="red"); return
             if self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
             if self.save_button: self.save_button.config(state=tk.DISABLED)
             self.show_pre_cleanup_labels_var.set(False)
        try:
            print("--- Analyzing Track 1 ---"); results1 = self._analyze_single_track(1, manual_bpm_val if is_update else None);
            if results1 is None: raise RuntimeError("Analysis failed for Track 1. Check console for details.")
            self.track_data[1] = results1
            self.playback_manager.set_audio(results1.get('y_processed'), results1.get('sr'))
            if mode == "compare":
                if not is_update:
                    print("\n--- Analyzing Track 2 ---"); results2 = self._analyze_single_track(2);
                    if results2 is None: raise RuntimeError("Analysis failed for Track 2. Check console for details.")
                    self.track_data[2] = results2
                else: print("--- Keeping existing Track 2 analysis ---" if self.track_data.get(2) else "--- Track 2 data missing ---")
        except Exception as e:
            error_msg = f"Analysis Error: {e}"; self.status_label.config(text="Error!", foreground="red"); print(error_msg); traceback.print_exc(); messagebox.showerror("Analysis Error", f"{error_msg}\n\nCheck console output for more details."); self._clear_plots(); self._add_placeholder_labels(); plt.close('all'); self.manual_bpm_check.config(state=tk.DISABLED); self.update_manual_bpm_state(); self._update_playback_buttons_state_from_manager('stopped'); self._update_save_button_state(); return
        self._display_analysis_results()
        final_status = "Update Complete" if is_update else "Analysis Complete"; self.status_label.config(text=final_status, foreground="green")
        if self.track_data.get(1):
             self.manual_bpm_check.config(state=tk.NORMAL)
             if not is_update: bpm_used = self.track_data[1].get('bpm', ''); self.manual_bpm_entry_var.set(f"{bpm_used:.2f}" if isinstance(bpm_used, (int, float)) else "")
             if self.toggle_labels_button and self.track_data[1].get('labels_before_cleanup'): self.toggle_labels_button.config(state=tk.NORMAL)
             elif self.toggle_labels_button: self.toggle_labels_button.config(state=tk.DISABLED)
             if self.section_editor and mode == "single": self.section_editor.update_button.config(state=tk.NORMAL)
        self.update_manual_bpm_state(); self.update_analyze_button_state()
        print("Analysis/Update process finished.")

    # --- Helper to Display Results ---
    def _display_analysis_results(self):
        """Generates and embeds plots and populates editor based on current mode and data."""
        mode = self.mode.get()
        print(f"\n--- Generating Plots & Populating Editor (Mode: {mode}) ---")
        plot_configs = [
            ('Waveform', ap.create_single_track_plot, ap.create_comparison_plot, ap.plot_waveform, ["sections", "chroma"], {}),
            ('Energy/Balance', ap.create_single_track_plot, None, ap.plot_energy_balance_features, ["chroma"], {'num_rows': 5}),
            ('Timbre/Texture', ap.create_single_track_plot, None, ap.plot_timbre_texture_features, ["chroma"], {'num_rows': 3}),
            ('Low-End', ap.create_single_track_plot, ap.create_comparison_plot, ap.plot_low_energy, ['low_end'], {}),
            ('Dynamic Range', ap.create_single_track_plot, ap.create_comparison_plot, ap.plot_dynamic_range, ['dyn_range'], {}),
            ('Stereo Width', ap.create_single_track_plot, ap.create_comparison_plot, ap.plot_stereo_width_overlay, ['stereo'], {}),
            ('HPSS', ap.create_single_track_plot, ap.create_comparison_plot, ap.plot_hpss, ['hpss'], {}),
            ('Root Note', ap.create_single_track_plot, ap.create_comparison_plot, ap.plot_root_note, ['chroma'], {}),
            ('Band Analysis', ap.create_band_analysis_plot, ap.create_band_comparison_plot, None, ['band_plot', 'stereo', 'hpss'], {})
        ]
        t1_data = self.track_data.get(1); t2_data = self.track_data.get(2) if mode == "compare" else None
        display_data1 = t1_data.copy() if t1_data else {}
        display_data2 = t2_data.copy() if t2_data else {}

        for tab_name, s_func, c_func, p_func_comp, req_keys, s_func_kwargs in plot_configs:
            if tab_name not in self.tabs: continue
            analysis_enabled = True; core_analysis = req_keys[0] if req_keys else None
            if core_analysis and core_analysis in self.analysis_vars and not self.analysis_vars[core_analysis].get(): analysis_enabled = False
            elif tab_name == 'Band Analysis' and not (self.analysis_vars['stereo'].get() or self.analysis_vars['hpss'].get()): analysis_enabled = False; print(f" Skipping {tab_name}: Needs Stereo or HPSS Analysis.")
            elif tab_name in ['Energy/Balance', 'Timbre/Texture'] and not self.analysis_vars['chroma'].get(): analysis_enabled = False; print(f" Skipping {tab_name}: Needs Chroma/Labeling Analysis for features.")
            data_available = (display_data1) if mode == "single" else (display_data1 or display_data2)

            if analysis_enabled and data_available:
                print(f" Plotting {tab_name}..."); fig = None
                try:
                    plot_data1 = display_data1 if display_data1 else {}; plot_data2 = display_data2 if display_data2 else {}
                    if mode=="single":
                        if s_func == ap.create_band_analysis_plot: fig = s_func(plot_data1, self.track_names[1])
                        elif s_func == ap.create_single_track_plot: fig = s_func(plot_data1, p_func_comp, tab_name, self.track_names[1], **s_func_kwargs)
                    else: # Compare mode
                         if c_func == ap.create_band_comparison_plot: fig = c_func(plot_data1, plot_data2, self.track_names[1], self.track_names[2])
                         elif c_func is not None: fig = c_func(plot_data1, plot_data2, p_func_comp, f"{tab_name} Comparison", self.track_names[1], self.track_names[2])
                         else: print(f" Skipping comparison for {tab_name}."); self._add_placeholder_label(tab_name, "Comparison not available.")
                    if fig:
                        self._embed_plot(fig, tab_name)
                        if tab_name == 'Waveform' and mode == 'single' and self.waveform_summary_label is not None and plot_data1:
                            labels_for_summary = plot_data1.get('semantic_labels'); features_for_summary = plot_data1.get('section_features')
                            summary_string = "Structure (T1): Error generating summary."
                            if labels_for_summary and features_for_summary:
                                try: summary_string = "Structure (T1): " + ap.generate_structure_summary(labels_for_summary, features_for_summary) # Use ap alias
                                except Exception as summary_e: print(f"Error generating summary string: {summary_e}")
                            else: summary_string = "Structure (T1): Data missing for summary."
                            self.waveform_summary_label.config(text=summary_string); self.master.update_idletasks()
                            wrap_w = self.tabs[tab_name].winfo_width() - 20; self.waveform_summary_label.config(wraplength=max(100, wrap_w))
                    elif analysis_enabled: self._add_placeholder_label(tab_name, "Plot skipped (figure generation failed).")
                except Exception as plot_e: print(f"--- Error plot {tab_name} ---"); traceback.print_exc(); self._add_placeholder_label(tab_name, f"Error:\n{plot_e}"); plt.close(fig) if fig else None
            elif not data_available: print(f" Skipping {tab_name} (analysis data missing)."); self._add_placeholder_label(tab_name, "Analysis data missing.")
            else: print(f" Skipping {tab_name} (analysis not selected / dependencies missing)."); self._add_placeholder_label(tab_name, "Analysis not selected / dependencies missing.")
        if mode == "single" and self.section_editor and t1_data:
            self.section_editor.populate(t1_data)


    # --- Analysis Orchestration Method for a Single Track ---
    def _analyze_single_track(self, track_num, manual_bpm_override=None):
        """Runs analysis steps for one track."""
        results = defaultdict(lambda: None)
        try:
            print(f" Step 1: Loading/Preprocessing/Tempo (T{track_num})...")
            preprocess_data = load_and_preprocess(self.file_path[track_num], track_num, manual_bpm_override)
            if preprocess_data is None: return None
            results.update(preprocess_data)
            bpm=results.get('bpm'); sr=results.get('sr'); y_proc=results.get('y_processed'); hop=results.get('hop_length'); dur=results.get('duration_processed'); fp=results.get('file_path'); trim=results.get('trim_offset_sec',0)
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [bpm, sr, hop, dur, fp]): missing = [k for k, v in {'bpm':bpm, 'sr':sr, 'hop_length':hop, 'duration':dur, 'file_path':fp}.items() if v is None or (isinstance(v, float) and math.isnan(v))]; print(f"Error: Missing essential data T{track_num}: {missing}."); return None
            if y_proc is None or y_proc.size == 0: print(f"Error: Processed audio empty T{track_num}."); return None
            print(f" Step 1b: Calculating intermediate features (BPM: {bpm})...")
            spb = 60.0 / bpm if bpm > 0 else 0; spbar = 4 * spb if spb > 0 else 0; results['seconds_per_bar'] = spbar
            if dur > 0 and spb > 0: beat_times_est = np.arange(0, dur, spb); results['beat_frames'] = librosa.time_to_frames(beat_times_est, sr=sr, hop_length=hop)
            else: results['beat_frames'] = np.array([], dtype=int)
            results['rms'] = np.nan_to_num(librosa.feature.rms(y=y_proc, hop_length=hop)[0]); results['rms_times'] = librosa.frames_to_time(np.arange(len(results['rms'])), sr=sr, hop_length=hop)
            num_bars = int(np.ceil(dur / spbar)) if spbar > 0 else 0; bar_rms_list = []; bar_starts_list = []; rms_d = results.get('rms'); rms_t = results.get('rms_times')
            if num_bars > 0 and rms_d is not None and rms_t is not None and rms_d.size == rms_t.size:
                for i in range(num_bars):
                    start_rel = i * spbar; end_rel = min((i + 1) * spbar, dur); avg_rms_bar = 0
                    if start_rel < end_rel: mask = (rms_t >= start_rel) & (rms_t < end_rel)
                    if np.any(mask): valid_rms_in_bar = rms_d[mask][np.isfinite(rms_d[mask])]; avg_rms_bar = np.mean(valid_rms_in_bar) if valid_rms_in_bar.size > 0 else 0
                    bar_rms_list.append(avg_rms_bar); bar_starts_list.append(start_rel + trim)
                results['bar_rms_data'] = bar_rms_list; results['bar_starts_absolute'] = bar_starts_list
            else: results['bar_rms_data'] = None; results['bar_starts_absolute'] = None; print("Warning: Could not calculate bar RMS/Starts.")
            if self.analysis_vars['sections'].get(): print(f" Step 2: Detecting Sections..."); results["section_starts"], results["section_labels"] = detect_sections(results) if results.get("bar_rms_data") is not None else (None, None)
            else: print(f" Step 2: Skipping Sections."); results["section_starts"]=None; results["section_labels"]=None
            if self.analysis_vars['chroma'].get() or self.analysis_vars['sections'].get():
                 if results.get("section_starts") is not None: print(f" Step 3: Running Chroma/Cluster/Label Analysis..."); results.update(analyze_chroma_and_clusters(results))
                 else: print(" Step 3: Skipping Chroma/Labeling (Sections missing)."); results.update(results_on_failure(None))
            else: print(f" Step 3: Skipping Chroma/Labeling."); results.update(results_on_failure(results.get("section_starts")))
            needs_spec = any(self.analysis_vars[k].get() for k in ['stereo','hpss','band_plot','low_end', 'chroma'])
            if needs_spec: print(f" Step 4: Running Stereo/HPSS Analysis..."); results.update(analyze_stereo_and_hpss(results, self.analysis_vars['stereo'].get(), self.analysis_vars['hpss'].get()))
            else: print(f" Step 4: Skipping Stereo/HPSS."); results.setdefault("width_matrix", None); results.setdefault("rms_harm", None); results.setdefault("rms_perc", None); results.setdefault("rms_time_absolute", None);
            if self.analysis_vars['low_end'].get():
                print(f" Step 5: Calculating Low-End Energy..."); spec=results.get("spec"); freqs=results.get("freqs"); times=results.get("times_absolute")
                if spec is not None and freqs is not None and times is not None and spec.shape[0]==freqs.size and spec.shape[1]==times.size:
                    lm=freqs<150;
                    if np.any(lm): le=np.sum(spec[lm,:],axis=0); n=np.max(le); results["low_energy_norm"]=(le/n if n>1e-9 else np.zeros_like(le)); results["low_energy_times"]=times
                    else: print(" Warning: No frequencies below 150Hz found."); results["low_energy_norm"]=None; results["low_energy_times"]=None
                else: print(" Skipping Low-End Energy (Spectrogram data missing)."); results["low_energy_norm"]=None; results["low_energy_times"]=None
            else: results["low_energy_norm"]=None; results["low_energy_times"]=None
            if self.analysis_vars['dyn_range'].get():
                print(f" Step 6: Calculating Dynamic Range..."); y_dyn=results.get("y_processed"); sr_dyn=results.get("sr"); hop_dyn=results.get("hop_length"); trim_dyn=results.get("trim_offset_sec",0); fdyn=2048
                try:
                    if y_dyn is not None and len(y_dyn)>=fdyn and sr_dyn and hop_dyn and trim_dyn is not None: rms_f=librosa.feature.rms(y=y_dyn,frame_length=fdyn,hop_length=hop_dyn)[0]; yf=librosa.util.frame(y_dyn,frame_length=fdyn,hop_length=hop_dyn); peak_f=np.max(np.abs(yf),axis=0); minl=min(len(rms_f),len(peak_f)); rms_f=rms_f[:minl]; peak_f=peak_f[:minl]; results["dyn_range"]=np.nan_to_num(rms_f/(peak_f+1e-9)); dtimes=librosa.frames_to_time(np.arange(len(results["dyn_range"])),sr=sr_dyn,hop_length=hop_dyn); results["dyn_times_absolute"]=dtimes+trim_dyn
                    else: print(" Skipping Dynamic Range."); results["dyn_range"]=None; results["dyn_times_absolute"]=None
                except Exception as de: print(f" Error calculating Dynamic Range: {de}"); results["dyn_range"]=None; results["dyn_times_absolute"]=None
            else: results["dyn_range"]=None; results["dyn_times_absolute"]=None
            results.setdefault("semantic_labels", []); results.setdefault("label_colors", []); results.setdefault("section_features", []); results.setdefault("labels_before_cleanup", [])
            return dict(results)
        except Exception as e: print(f"--- Unhandled Error T{track_num} ---"); traceback.print_exc(); messagebox.showerror(f"Error T{track_num}", f"Unexpected error during analysis:\n{e}"); return None

    # --- Method for Handling Manual BPM Update ---
    def update_plots_with_manual_bpm(self):
        """Validates manual BPM input and triggers re-analysis."""
        if not self.use_manual_bpm.get(): messagebox.showinfo("Info", "Please check 'Use Manual BPM' first."); return
        try: bpm_str = self.manual_bpm_entry_var.get(); bpm_val = float(bpm_str) if bpm_str else None; assert 30 <= bpm_val <= 300
        except (ValueError, AssertionError): messagebox.showerror("Invalid BPM", "BPM must be between 30 and 300."); return
        self.update_bpm_button.config(state=tk.DISABLED); self.analyze_button.config(state=tk.DISABLED)
        self.run_analysis(is_update=True, manual_bpm_val=bpm_val)

    # --- Method for Embedding Matplotlib Plots in Tkinter ---
    def _embed_plot(self, fig, tab_name):
        """Clears the specified tab (except editor) and embeds the plot."""
        print(f"DEBUG: _embed_plot called for tab: {tab_name}")
        if tab_name not in self.tabs: print(f"Error: Tab '{tab_name}' not found."); plt.close(fig); return
        tab=self.tabs[tab_name];
        print(f"DEBUG: Clearing plot widgets in tab {tab_name}")
        widgets_to_keep = []
        if tab_name == 'Waveform' and hasattr(self, 'section_editor') and self.section_editor:
             if self.section_editor.master and self.section_editor.master.winfo_exists():
                 widgets_to_keep.append(self.section_editor.master) # Keep the outer frame
        for widget in list(tab.winfo_children()):
            if widget not in widgets_to_keep: 
                try: widget.destroy(); 
                except Exception: pass
        if tab_name in self.plot_widgets: self.plot_widgets[tab_name] = {}
        if tab_name == 'Waveform': self.waveform_summary_label = None; self.playhead_line = None
        try:
            plot_frame = ttk.Frame(tab); plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            print(f"DEBUG: Creating canvas for {tab_name}"); canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas_widget = canvas.get_tk_widget()
            print(f"DEBUG: Creating toolbar for {tab_name}"); toolbar = NavigationToolbar2Tk(canvas, plot_frame); toolbar.update()
            if tab_name == 'Waveform':
                print(f"DEBUG: Creating waveform summary label"); self.waveform_summary_label = ttk.Label(plot_frame, text="Structure: Calculating...", padding=(5, 2), anchor=tk.W, justify=tk.LEFT, wraplength=800); self.waveform_summary_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0)); print(f"DEBUG: Waveform summary label packed")
                axes_list = fig.get_axes()
                if axes_list:
                    ax = axes_list[0]; self.playhead_line, = ax.plot([0, 0], ax.get_ylim(), color='white', lw=1.0, alpha=0.8, visible=False, zorder=10); print(f"DEBUG: Initial playhead line created (visible=False)")
                else: print("Warning: Could not get axes from figure for playhead line."); self.playhead_line = None
                print(f"DEBUG: Connecting click event for Waveform canvas"); canvas.mpl_connect('button_press_event', self._on_waveform_click)
            print(f"DEBUG: Packing toolbar for {tab_name}"); toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            print(f"DEBUG: Packing canvas for {tab_name}"); canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.plot_widgets[tab_name]={'canvas':canvas,'toolbar':toolbar,'figure':fig}; print(f"DEBUG: Widgets stored for {tab_name}")
        except Exception as ee: print(f"--- Error embed {tab_name} ---"); traceback.print_exc(); ttk.Label(tab, text=f"Plot Error:\n{ee}", foreground='red').pack(expand=True)
        finally: plt.close(fig); print(f"DEBUG: _embed_plot finished for tab: {tab_name}")

    # --- Playback GUI Update Callbacks ---
    def _update_playback_buttons_state_from_manager(self, state):
        """Callback from PlaybackManager to update GUI button states."""
        print(f"DEBUG GUI: Playback state changed to: {state}")
        is_playing = (state == 'playing'); is_paused = (state == 'paused')
        can_play = (self.playback_manager.audio_data is not None and self.playback_manager.sample_rate is not None and self.mode.get() == "single")
        play_pause_state = tk.NORMAL if can_play else tk.DISABLED; stop_state = tk.NORMAL if (is_playing or is_paused) else tk.DISABLED
        self.play_pause_button.config(state=play_pause_state, text="Pause" if is_playing else "Play"); self.stop_button.config(state=stop_state)

    def _update_playhead_display(self, frame):
        """Callback from PlaybackManager to update the visual playhead."""
        if frame is None or not self.playhead_line or not self.playback_manager.sample_rate:
            if self.playhead_line: self.playhead_line.set_visible(False); self._redraw_canvas()
            return
        current_time = frame / self.playback_manager.sample_rate; trim = self.track_data.get(1, {}).get('trim_offset_sec', 0); display_time = current_time + trim
        self.playhead_line.set_xdata([display_time, display_time]);
        if not self.playhead_line.get_visible(): self.playhead_line.set_visible(True)
        self._redraw_canvas()

    def _redraw_canvas(self):
        """Requests a redraw of the waveform canvas if it exists."""
        if 'Waveform' in self.plot_widgets and self.plot_widgets['Waveform'].get('canvas'):
            try: self.plot_widgets['Waveform']['canvas'].draw_idle()
            except Exception as e: print(f"Error redrawing canvas: {e}")

    # --- Method to Toggle Label View ---
    def _toggle_label_view(self):
        """Switches the waveform plot labels between final and pre-cleanup."""
        print("DEBUG: _toggle_label_view called.")
        if self.mode.get() == "compare": print("DEBUG: In compare mode, toggle disabled."); messagebox.showinfo("Info", "Label toggle only available in Single Track mode."); self.show_pre_cleanup_labels_var.set(False); self._update_toggle_button_state(); return
        t1_data = self.track_data.get(1)
        if not t1_data: print("DEBUG: No track 1 data found for toggle."); self.show_pre_cleanup_labels_var.set(False); self._update_toggle_button_state(); return
        self._update_toggle_button_state()
        show_raw = self.show_pre_cleanup_labels_var.get(); final_labels = t1_data.get('semantic_labels'); pre_cleanup_labels = t1_data.get('labels_before_cleanup')
        if final_labels is None or pre_cleanup_labels is None: print("DEBUG: Label data missing for toggle."); messagebox.showwarning("Missing Data", "Label data missing."); self.show_pre_cleanup_labels_var.set(False); self._update_toggle_button_state(); return
        labels_to_show = pre_cleanup_labels if show_raw else final_labels; label_type_str = 'Pre-Cleanup' if show_raw else 'Final'; print(f"DEBUG: Selected labels: {label_type_str}")
        temp_data = t1_data.copy(); temp_data['semantic_labels'] = labels_to_show; print(f"DEBUG: Created temp_data with {label_type_str} labels.")
        try:
            print(f"DEBUG: Calling create_single_track_plot for Waveform..."); fig = ap.create_single_track_plot(temp_data, ap.plot_waveform, 'Waveform', self.track_names[1]) # Use ap alias
            print(f"DEBUG: create_single_track_plot returned figure: {fig is not None}")
            if fig:
                print(f"DEBUG: Calling _embed_plot for Waveform..."); self._embed_plot(fig, 'Waveform')
                print(f"DEBUG: _embed_plot finished for Waveform.")
                if self.waveform_summary_label is not None:
                    print(f"DEBUG: Updating summary label..."); summary_string = "Structure: Error generating summary."
                    features_for_summary = temp_data.get('section_features')
                    if labels_to_show and features_for_summary:
                         try: summary_string = "Structure: " + ap.generate_structure_summary(labels_to_show, features_for_summary); print(f"DEBUG: Generated summary: {summary_string[:100]}...") # Use ap alias
                         except Exception as summary_e: print(f"Error generating summary string: {summary_e}")
                    else: summary_string = "Structure: Data missing for summary."; print(f"DEBUG: Summary data missing.")
                    self.waveform_summary_label.config(text=summary_string); self.master.update_idletasks()
                    wrap_w = self.tabs['Waveform'].winfo_width() - 20; self.waveform_summary_label.config(wraplength=max(100, wrap_w)); print(f"DEBUG: Summary label updated.")
            else: print(f"DEBUG: Figure generation failed in _toggle_label_view."); messagebox.showerror("Plot Error", "Failed to regenerate waveform plot.")
        except Exception as e: print(f"DEBUG: Error during plot regeneration/embedding in toggle: {e}"); messagebox.showerror("Plot Error", f"Error updating waveform plot:\n{e}"); traceback.print_exc()
        print("DEBUG: _toggle_label_view finished.")

    def _update_toggle_button_state(self):
         """ Enable toggle button only if analysis is done and data exists """
         state = tk.DISABLED
         if self.track_data.get(1) and self.track_data[1].get('labels_before_cleanup') and self.mode.get() == 'single': state = tk.NORMAL
         if self.toggle_labels_button: self.toggle_labels_button.config(state=state)

    # --- Waveform Click Handler (Seek & Edit Functionality) ---
    def _on_waveform_click(self, event):
        """Handles mouse clicks on the waveform plot to seek playback or trigger edit."""
        print(f"DEBUG CLICK HANDLER: Start. Button={event.button}, x={event.xdata}, y={event.ydata}")

        # --- Right-Click: Trigger Edit ---
        if event.button == 3 or event.button == 2: # Button 2 for macOS trackpad right-click
            print("DEBUG CLICK HANDLER: Right-click detected.")
            if self.mode.get() != 'single' or not self.track_data.get(1) or not self.section_editor: print("DEBUG CLICK HANDLER: Not in single mode or no data/editor. Edit ignore."); return
            if event.xdata is None or event.inaxes is None: print("DEBUG CLICK HANDLER: Click outside axes. Edit ignore."); return

            clicked_time = event.xdata
            section_starts = self.track_data[1].get("section_starts")
            if section_starts is None: print("DEBUG CLICK HANDLER: No section starts data. Edit ignore."); return

            section_index = -1
            for i in range(len(section_starts)):
                start = section_starts[i]; is_last_section = (i == len(section_starts) - 1)
                if is_last_section:
                    if clicked_time >= start: section_index = i; break
                else:
                    next_start = section_starts[i+1]
                    if clicked_time >= start and clicked_time < next_start: section_index = i; break

            if section_index != -1:
                print(f"DEBUG CLICK HANDLER: Click corresponds to section index {section_index}.")
                self._show_section_edit_popup(section_index, event)
            else: print("DEBUG CLICK HANDLER: Could not determine clicked section index.")

        # --- Left-Click: Seek Playback ---
        elif event.button == 1:
            print("DEBUG CLICK HANDLER: Left-click detected (Seek).")
            if self.mode.get() != 'single': print("DEBUG CLICK HANDLER: Not single mode, exiting seek."); return
            waveform_widgets = self.plot_widgets.get('Waveform', {}); fig = waveform_widgets.get('figure')
            if not fig or event.inaxes not in fig.get_axes(): print("DEBUG CLICK HANDLER: Click not in figure axes, exiting seek."); return
            if self.playback_manager.audio_data is None or self.playback_manager.sample_rate is None: print("DEBUG CLICK HANDLER: No audio data in playback manager, exiting seek."); return
            if event.xdata is None: print("DEBUG CLICK HANDLER: Click xdata is None, exiting seek."); return

            clicked_time = event.xdata; trim_offset = self.track_data[1].get('trim_offset_sec', 0); sample_rate = self.playback_manager.sample_rate; audio_length = len(self.playback_manager.audio_data)
            relative_time = clicked_time - trim_offset; target_frame = int(relative_time * sample_rate); target_frame = max(0, min(target_frame, audio_length - 1))
            print(f"DEBUG CLICK HANDLER: Seek to time {clicked_time:.2f}s -> Target frame {target_frame}")
            self.playback_manager.seek(target_frame)
            if self.playhead_line:
                 display_time = target_frame / sample_rate + trim_offset; self.playhead_line.set_xdata([display_time, display_time]); self.playhead_line.set_visible(True); self._redraw_canvas(); print(f"DEBUG CLICK HANDLER: Playhead moved visually to {display_time:.2f}s")
        else: print(f"DEBUG CLICK HANDLER: Ignored button {event.button}.")


    # --- Section Edit Pop-up Method (Moved from SectionEditor) ---
    def _show_section_edit_popup(self, section_index, event=None):
        """Creates and shows the modal pop-up dialog for editing or merging a section."""
        print(f"--- DEBUG MainApp: _show_section_edit_popup for index: {section_index} ---")
        if not self.section_editor or not self.track_data.get(1): print("DEBUG MainApp: Section editor or track data not available."); return

        item_id_str = str(section_index)
        try:
            current_values = self.section_editor.tree.item(item_id_str, 'values')
            if not current_values or len(current_values) < 6: raise ValueError("Invalid values found in Treeview row.")
            section_num_display = current_values[1]; current_label = current_values[4]; current_color_name = current_values[5]
            num_sections = len(self.track_data[1]['section_starts']) # Get total number of sections
        except Exception as e: messagebox.showerror("Edit Error", f"Could not retrieve current values for section {section_index + 1}.\nError: {e}"); print(f"Error getting values for tree item {item_id_str}: {e}"); return

        editor_popup = tk.Toplevel(self.master); editor_popup.title(f"Edit/Merge Section {section_num_display}")
        editor_popup.transient(self.master); editor_popup.resizable(False, False)
        popup_frame = ttk.Frame(editor_popup, padding="10"); popup_frame.pack(expand=True, fill=tk.BOTH)

        # --- Edit Controls ---
        edit_frame = ttk.LabelFrame(popup_frame, text="Edit Label/Color", padding=5)
        edit_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(edit_frame, text="Label:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        label_combo = ttk.Combobox(edit_frame, values=ALLOWED_LABELS, state='readonly', width=15); label_combo.grid(row=0, column=1, padx=5, pady=5)
        if current_label in ALLOWED_LABELS: label_combo.set(current_label)
        else: label_combo.set(ALLOWED_LABELS[0])

        ttk.Label(edit_frame, text="Color:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        color_combo = ttk.Combobox(edit_frame, values=list(COLOR_NAME_MAP.keys()), state='readonly', width=15); color_combo.grid(row=1, column=1, padx=5, pady=5)
        if current_color_name in COLOR_NAME_MAP: color_combo.set(current_color_name)
        else: color_combo.set(list(COLOR_NAME_MAP.keys())[0])

        ok_button = ttk.Button(edit_frame, text="Apply Edit", width=12, command=lambda p=editor_popup, item=item_id_str, lc=label_combo, cc=color_combo: self.section_editor._commit_popup_edit(p, item, lc, cc));
        ok_button.grid(row=0, column=2, rowspan=2, padx=(10, 5), pady=5, sticky="ns")

        # --- Merge Controls ---
        merge_frame = ttk.LabelFrame(popup_frame, text="Merge Section", padding=5)
        merge_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        merge_prev_button = ttk.Button(merge_frame, text="Merge with Previous", width=20,
                                       command=lambda p=editor_popup, idx=section_index: self._trigger_merge(p, idx, 'prev'))
        merge_prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        if section_index == 0: merge_prev_button.config(state=tk.DISABLED) # Disable if first section

        merge_next_button = ttk.Button(merge_frame, text="Merge with Next", width=20,
                                       command=lambda p=editor_popup, idx=section_index: self._trigger_merge(p, idx, 'next'))
        merge_next_button.pack(side=tk.LEFT, padx=5, pady=5)
        if section_index >= num_sections - 1: merge_next_button.config(state=tk.DISABLED) # Disable if last section

        # --- Cancel Button ---
        cancel_button = ttk.Button(popup_frame, text="Cancel", width=8, command=editor_popup.destroy)
        cancel_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # Position Pop-up
        editor_popup.update_idletasks()
        if event and hasattr(event, 'x_root') and hasattr(event, 'y_root'): # Position near click event if available
            final_x = event.x_root + 10; final_y = event.y_root + 10
            screen_w = self.master.winfo_screenwidth(); screen_h = self.master.winfo_screenheight()
            popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height()
            final_x = max(0, min(final_x, screen_w - popup_w)); final_y = max(0, min(final_y, screen_h - popup_h))
            print(f"DEBUG MainApp: Positioning pop-up near click at (+{final_x},{final_y})"); editor_popup.geometry(f"+{final_x}+{final_y}")
        else: # Center on main window
            main_win = self.master; main_x = main_win.winfo_x(); main_y = main_win.winfo_y(); main_w = main_win.winfo_width(); main_h = main_win.winfo_height()
            popup_w = editor_popup.winfo_width(); popup_h = editor_popup.winfo_height()
            center_x = main_x + (main_w // 2) - (popup_w // 2); center_y = main_y + (main_h // 2) - (popup_h // 2)
            print(f"DEBUG MainApp: Centering pop-up at (+{center_x},{center_y})"); editor_popup.geometry(f"+{center_x}+{center_y}")

        print("DEBUG MainApp: Making pop-up modal and waiting..."); editor_popup.grab_set(); editor_popup.wait_window()
        print(f"DEBUG MainApp: Pop-up closed for section index {section_index}.")


    # --- Merge Logic ---
    def _trigger_merge(self, popup, section_index, direction):
        """Called by merge buttons, determines indices and calls merge function."""
        popup.destroy() # Close the popup first
        if direction == 'prev':
            if section_index > 0:
                index_to_remove = section_index
                # Merge into the previous section (index - 1)
                self._merge_section(index_to_remove)
            else:
                messagebox.showerror("Merge Error", "Cannot merge the first section with previous.")
        elif direction == 'next':
            num_sections = len(self.track_data[1]['section_starts'])
            if section_index < num_sections - 1:
                index_to_remove = section_index + 1
                # Merge the next section (index + 1) into the current one (index)
                self._merge_section(index_to_remove)
            else:
                 messagebox.showerror("Merge Error", "Cannot merge the last section with next.")

    def _merge_section(self, remove_boundary_index):
        """
        Merges two adjacent sections by removing the boundary between them.

        Args:
            remove_boundary_index (int): The index of the boundary *start time*
                                         in section_starts to remove. This effectively
                                         merges section `remove_boundary_index - 1`
                                         and section `remove_boundary_index`.
                                         The resulting section retains the properties
                                         of section `remove_boundary_index - 1`.
        """
        print(f"DEBUG: Attempting to merge by removing boundary at index {remove_boundary_index}")
        if not self.track_data.get(1): messagebox.showerror("Merge Error", "No track data loaded."); return
        t_data = self.track_data[1]
        num_sections = len(t_data['section_starts'])

        # Validate index
        if remove_boundary_index <= 0 or remove_boundary_index >= num_sections:
            messagebox.showerror("Merge Error", f"Invalid boundary index {remove_boundary_index} for merging.")
            print(f"ERROR: Invalid boundary index {remove_boundary_index} for merging {num_sections} sections.")
            return

        # Index of the section *before* the boundary (this one remains)
        keep_section_idx = remove_boundary_index - 1
        # Index of the section *after* the boundary (this one gets removed)
        remove_section_idx = remove_boundary_index

        try:
            # Get durations to add
            removed_duration_sec = t_data['section_features'][remove_section_idx].get('duration_sec', 0)
            removed_duration_bars = t_data['section_features'][remove_section_idx].get('duration_bars', 0)

            # --- Update Data Lists ---
            # Remove boundary start time
            del t_data['section_starts'][remove_boundary_index]
            # Remove label, color, features of the removed section
            del t_data['semantic_labels'][remove_section_idx]
            del t_data['label_colors'][remove_section_idx]
            del t_data['section_features'][remove_section_idx]
            # Also remove from cluster labels and labels_before_cleanup if they exist and match length
            if 'cluster_labels' in t_data and len(t_data['cluster_labels']) == num_sections:
                del t_data['cluster_labels'][remove_section_idx]
            if 'labels_before_cleanup' in t_data and len(t_data['labels_before_cleanup']) == num_sections:
                del t_data['labels_before_cleanup'][remove_section_idx]

            # --- Update Remaining Section's Features ---
            # Update end time, duration_sec, duration_bars
            # The end time of the kept section is now the end time of the removed section
            new_end_time = t_data['section_features'][keep_section_idx]['start_time'] + \
                           t_data['section_features'][keep_section_idx]['duration_sec'] + \
                           removed_duration_sec
            t_data['section_features'][keep_section_idx]['end_time'] = new_end_time
            t_data['section_features'][keep_section_idx]['duration_sec'] += removed_duration_sec
            t_data['section_features'][keep_section_idx]['duration_bars'] += removed_duration_bars
            # Update index for all subsequent features
            for i in range(keep_section_idx + 1, len(t_data['section_features'])):
                 t_data['section_features'][i]['index'] -= 1

            # Note: We are NOT recalculating avg_rms, spectral features etc. for the merged section here.
            # The merged section keeps the label, color, and calculated features of the *first* section involved.

            print(f"DEBUG: Merge successful. Removed section {remove_section_idx}. Updated section {keep_section_idx}.")

            # --- Refresh UI ---
            self.status_label.config(text="Sections Merged", foreground="blue")
            self._display_analysis_results() # Redraw plots and repopulate editor
            self._update_save_button_state() # Ensure save is enabled

        except IndexError as e:
            messagebox.showerror("Merge Error", f"Index error during merge: {e}. Lists might be inconsistent.")
            print(f"ERROR: Index error during merge: {e}"); traceback.print_exc()
        except Exception as e:
            messagebox.showerror("Merge Error", f"An unexpected error occurred during merge:\n{e}")
            print(f"ERROR: Unexpected error during merge: {e}"); traceback.print_exc()


    # --- Save/Load Methods ---

    def _ask_save_status(self, parent):
        """Creates a modal dialog asking user to classify the save."""
        dialog = tk.Toplevel(parent); dialog.title("Save Analysis Status"); dialog.transient(parent); dialog.grab_set(); dialog.resizable(False, False)
        status_var = tk.StringVar(value="")
        parent_x = parent.winfo_rootx(); parent_y = parent.winfo_rooty(); parent_w = parent.winfo_width(); parent_h = parent.winfo_height()
        dialog.update_idletasks(); dialog_w = dialog.winfo_width(); dialog_h = dialog.winfo_height()
        center_x = parent_x + (parent_w // 2) - (dialog_w // 2); center_y = parent_y + (parent_h // 2) - (dialog_h // 2)
        dialog.geometry(f"+{center_x}+{center_y}")
        frame = ttk.Frame(dialog, padding="15"); frame.pack(expand=True, fill="both")
        ttk.Label(frame, text="Save analysis as:", font="-weight bold").pack(pady=(0, 10))
        button_frame = ttk.Frame(frame); button_frame.pack(pady=5)
        def set_status(status): status_var.set(status); dialog.destroy()
        perfect_btn = ttk.Button(button_frame, text="Perfect (Training)", width=20, command=lambda: set_status(PERFECT_SUBFOLDER)); perfect_btn.pack(side=tk.LEFT, padx=10)
        wip_btn = ttk.Button(button_frame, text="WIP (Work in Progress)", width=20, command=lambda: set_status(WIP_SUBFOLDER)); wip_btn.pack(side=tk.LEFT, padx=10)
        parent.wait_window(dialog)
        return status_var.get()

    def _save_analysis(self):
        """Saves analysis results, overwriting previous file for the same track in Perfect/WIP."""
        print("DEBUG: Save Analysis called.")
        if self.mode.get() != "single" or not self.track_data.get(1): messagebox.showwarning("Save Error", "Analysis data for Track 1 must exist and be in Single Track mode to save."); return
        save_status = self._ask_save_status(self.master)
        if not save_status: print("DEBUG: Save status selection cancelled."); self.status_label.config(text="Save Cancelled", foreground="orange"); return
        print(f"DEBUG: User selected save status: {save_status}")
        target_directory = os.path.join(ANALYSIS_BASE_FOLDER, save_status); print(f"DEBUG: Target save directory: {target_directory}")
        try: os.makedirs(target_directory, exist_ok=True); print(f"DEBUG: Ensured directory '{target_directory}' exists.")
        except OSError as e: messagebox.showerror("Save Error", f"Could not create save directory:\n{target_directory}\nError: {e}"); self.status_label.config(text="Save Error!", foreground="red"); return
        try:
            base_audio_name = "track";
            if self.file_path.get(1): base_audio_name = os.path.splitext(os.path.basename(self.file_path[1]))[0]
            safe_base_name = "".join(c for c in base_audio_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            fixed_filename = f"{safe_base_name}.analysis.joblib"; save_path = os.path.join(target_directory, fixed_filename); print(f"DEBUG: Determined save path: {save_path}")
        except Exception as e: messagebox.showerror("Save Error", f"Error creating filename: {e}"); self.status_label.config(text="Save Error!", foreground="red"); return
        confirm_overwrite = True
        if os.path.exists(save_path):
            print(f"DEBUG: File exists: {save_path}"); confirm_overwrite = messagebox.askyesno("Confirm Overwrite", f"Analysis file already exists:\n{save_path}\n\nOverwrite?")
        if not confirm_overwrite: print("DEBUG: Overwrite cancelled by user."); self.status_label.config(text="Save Cancelled", foreground="orange"); return
        try:
            print(f"DEBUG: Saving analysis data to: {save_path}"); data_to_save = self.track_data[1]; joblib.dump(data_to_save, save_path, compress=3)
            messagebox.showinfo("Save Successful", f"Analysis saved successfully to:\n{save_path}"); self.status_label.config(text="Analysis Saved", foreground="green")
        except Exception as e: messagebox.showerror("Save Error", f"Failed to save analysis file:\n{e}"); traceback.print_exc(); self.status_label.config(text="Save Failed!", foreground="red")

    def _load_analysis(self):
        """Loads previously saved analysis results."""
        print("DEBUG: Load Analysis called.")
        initial_dir = ANALYSIS_BASE_FOLDER if os.path.isdir(ANALYSIS_BASE_FOLDER) else "."
        load_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select Analysis File", filetypes=[("Joblib Analysis Files", "*.joblib"), ("All Files", "*.*")])
        if not load_path: print("DEBUG: Load cancelled by user."); return
        self.playback_manager.stop(); self._clear_plots()
        self.status_label.config(text="Loading...", foreground="orange"); self.master.update_idletasks()
        try:
            print(f"DEBUG: Loading analysis data from: {load_path}"); loaded_data = joblib.load(load_path)
            if not isinstance(loaded_data, dict): raise TypeError("Loaded file does not contain a valid analysis dictionary.")
            required_keys = ['file_path', 'sr', 'y_processed', 'semantic_labels', 'section_features']
            if not all(key in loaded_data for key in required_keys): missing = [key for key in required_keys if key not in loaded_data]; raise ValueError(f"Loaded data is missing essential keys: {missing}")
            print("DEBUG: Loaded data verified.")
            self.track_data = {1: None, 2: None}; self.track_data[1] = defaultdict(lambda: None, loaded_data)
            loaded_file_path = self.track_data[1].get('file_path'); original_audio_found = False
            if loaded_file_path and os.path.exists(loaded_file_path):
                 self.file_path[1] = loaded_file_path; self.track_names[1] = os.path.basename(loaded_file_path); self.file_label1.config(text=self.track_names[1]); print(f"DEBUG: Original audio file found at: {loaded_file_path}"); original_audio_found = True
            else:
                 print(f"Warning: Original audio file not found at path stored in analysis: {loaded_file_path}"); messagebox.showwarning("Audio File Missing", f"The original audio file path stored in the analysis was:\n{loaded_file_path}\n\nThis file could not be found. Playback requires the original audio. Please re-select the audio file if you wish to use playback.")
                 self.file_path[1] = None; self.track_names[1] = f"Loaded: {os.path.basename(load_path)} (Audio Missing)"; self.file_label1.config(text=self.track_names[1])
            if original_audio_found: self.playback_manager.set_audio(self.track_data[1].get('y_processed'), self.track_data[1].get('sr'))
            else: self.playback_manager.set_audio(None, None)
            self.mode.set("single"); self.update_ui_for_mode() # This updates UI and clears plots again
            self._display_analysis_results() # Re-display results after UI update
            self.status_label.config(text="Analysis Loaded", foreground="green")
            self._update_save_button_state(); self._update_toggle_button_state()
            if self.track_data[1].get('bpm'):
                 self.manual_bpm_check.config(state=tk.NORMAL); bpm_used = self.track_data[1].get('bpm', ''); self.manual_bpm_entry_var.set(f"{bpm_used:.2f}" if isinstance(bpm_used, (int, float)) else ""); self.update_manual_bpm_state()
            print("DEBUG: Load analysis finished successfully.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load or display analysis file:\n{e}"); traceback.print_exc(); self.status_label.config(text="Load Failed!", foreground="red")
            self._clear_plots(); self.track_data = {1: None, 2: None}; self.file_path = {1: None, 2: None}; self.track_names = {1: "Track 1", 2: "Track 2"}; self.file_label1.config(text="No file selected"); self.file_label2.config(text="No file selected"); self.update_ui_for_mode()

    # --- Manual Section Editing Callback ---
    def _apply_section_edits_from_editor(self, new_labels, new_colors_hex):
        """Callback function executed by SectionEditor when 'Update' is clicked."""
        print("DEBUG MainApp: Applying section edits received from editor...")
        if not self.track_data.get(1): messagebox.showerror("Update Error", "No track data loaded to apply edits to."); return
        # Check length against semantic_labels which should be the correct length
        current_labels = self.track_data[1].get('semantic_labels', [])
        if len(new_labels) != len(current_labels) or len(new_colors_hex) != len(current_labels):
             messagebox.showerror("Update Error", f"Data length mismatch when applying edits. Expected {len(current_labels)}, got {len(new_labels)} labels, {len(new_colors_hex)} colors."); return
        try:
            self.track_data[1]['semantic_labels'] = new_labels; self.track_data[1]['label_colors'] = new_colors_hex
            print("DEBUG MainApp: track_data updated with edits.")
            print("DEBUG MainApp: Replotting waveform with updated labels/colors...")
            temp_data = self.track_data[1].copy()
            fig = ap.create_single_track_plot(temp_data, ap.plot_waveform, 'Waveform', self.track_names[1]) # Use ap alias
            if fig:
                self._embed_plot(fig, 'Waveform')
                if self.waveform_summary_label is not None:
                    summary_string = "Structure: " + ap.generate_structure_summary(new_labels, self.track_data[1]['section_features']) # Use ap alias
                    self.waveform_summary_label.config(text=summary_string); self.master.update_idletasks()
                    wrap_w = self.tabs['Waveform'].winfo_width() - 20; self.waveform_summary_label.config(wraplength=max(100, wrap_w))
                messagebox.showinfo("Update Complete", "Section labels and colors updated."); self.status_label.config(text="Sections Updated", foreground="blue"); self._update_save_button_state()
            else: messagebox.showerror("Plot Error", "Failed to regenerate waveform plot after edits.")
        except Exception as e: messagebox.showerror("Update Error", f"Failed to apply section edits:\n{e}"); traceback.print_exc()

    def _on_closing(self):
        """Handles window closing: stops audio stream."""
        print("Window closing...")
        self.playback_manager.cleanup()
        self.master.destroy()

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()