# ui_state_manager.py
import tkinter as tk
from tkinter import ttk


class UIStateManager:
    """
    Manages the UI state of the Audio Analyzer application.

    This class encapsulates all UI update and state management logic,
    maintaining the enabled/disabled states of buttons, entries, and other
    controls based on the application's current state.

    Attributes:
        app (AudioAnalyzerApp): Reference to the main application instance.
    """

    def __init__(self, app):
        """
        Initialize with a reference to the main application.

        Args:
            app (AudioAnalyzerApp): Reference to the main application instance.
        """
        self.app = app

    def update_manual_bpm_state(self):
        """
        Enables or disables the manual BPM entry and update button.

        The controls are enabled only if the 'Use Manual BPM' checkbox is checked
        AND analysis data exists for track 1.
        """
        analysis_done = bool(self.app.track_data.get(1))  # Check if track 1 data exists
        state = (
            tk.NORMAL
            if self.app.use_manual_bpm.get() and analysis_done
            else tk.DISABLED
        )

        # Check if widgets exist before configuring (robustness)
        if hasattr(self.app, "manual_bpm_entry") and self.app.manual_bpm_entry:
            self.app.manual_bpm_entry.config(state=state)
        if hasattr(self.app, "update_bpm_button") and self.app.update_bpm_button:
            self.app.update_bpm_button.config(state=state)

    def update_ui_for_mode(self):
        """
        Updates UI elements visibility and state based on Single/Compare mode.

        Shows/hides Track 2 controls, resets data, updates button labels,
        clears plots, and resets various UI states when the mode changes.
        """
        mode = self.app.mode.get()
        is_compare = mode == "compare"
        analyze_btn_text = "Analyze & Compare" if is_compare else "Analyze Track"
        bpm_label_text = "Manual Tempo (Track 1)" if is_compare else "Manual Tempo"

        # Update button text and frame label
        if hasattr(self.app, "analyze_button") and self.app.analyze_button:
            self.app.analyze_button.config(text=analyze_btn_text)
        if hasattr(self.app, "manual_bpm_frame") and self.app.manual_bpm_frame:
            self.app.manual_bpm_frame.config(text=bpm_label_text)

        # Show/hide Track 2 controls
        if (
            hasattr(self.app, "select_button2")
            and self.app.select_button2
            and hasattr(self.app, "file_label2")
            and self.app.file_label2
        ):
            if is_compare:
                self.app.select_button2.grid()
                self.app.file_label2.grid()
            else:
                # Hide and reset Track 2 data if switching away from compare mode
                self.app.select_button2.grid_remove()
                self.app.file_label2.grid_remove()
                self.app.file_path[2] = None
                self.app.track_data[2] = None
                if self.app.file_label2:
                    self.app.file_label2.config(text="No file selected")

        # Reset states and plots
        self.app.playback_manager.stop()
        self.update_analyze_button_state()
        self.app.plot_manager.clear_plots()
        self.app.plot_manager.add_placeholder_labels()

        if hasattr(self.app, "manual_bpm_check") and self.app.manual_bpm_check:
            self.app.manual_bpm_check.config(
                state=tk.DISABLED
            )  # Disable until analysis

        self.update_manual_bpm_state()

        if self.app.toggle_labels_button:
            self.app.toggle_labels_button.config(state=tk.DISABLED)

        if self.app.section_editor:
            self.app.section_editor.clear()
            self.app.section_editor.update_button.config(state=tk.DISABLED)

        # Enable/disable shift button based on mode and data
        editor_state = (
            tk.NORMAL
            if mode == "single" and self.app.track_data.get(1)
            else tk.DISABLED
        )
        if (
            hasattr(self.app, "shift_sections_button")
            and self.app.shift_sections_button
        ):
            self.app.shift_sections_button.config(state=editor_state)

        self.app.show_pre_cleanup_labels_var.set(False)
        self.app.show_hmm_var.set(False)
        self.update_playback_buttons_state("stopped")  # Reset playback buttons

    def update_analyze_button_state(self):
        """
        Enables or disables the 'Analyze' button based on file selection(s).

        Requires Track 1 file in single mode, or both Track 1 and Track 2 files
        in compare mode. Also updates save and HMM button states.
        """
        mode = self.app.mode.get()
        state = tk.DISABLED

        if mode == "single" and self.app.file_path.get(1):
            state = tk.NORMAL
        elif (
            mode == "compare"
            and self.app.file_path.get(1)
            and self.app.file_path.get(2)
        ):
            state = tk.NORMAL

        if hasattr(self.app, "analyze_button") and self.app.analyze_button:
            self.app.analyze_button.config(state=state)

        # Update dependent buttons
        self.update_save_button_state()
        self.update_hmm_button_state()

    def update_save_button_state(self):
        """
        Enables the 'Save Analysis' button.

        Enabled only when in 'single' mode and analysis data exists for Track 1.
        """
        state = tk.DISABLED
        if self.app.mode.get() == "single" and self.app.track_data.get(1):
            state = tk.NORMAL

        if self.app.save_button:
            self.app.save_button.config(state=state)

    def update_hmm_button_state(self):
        """
        Enables/disables HMM Predict and Show HMM buttons.

        Predict button is enabled if in single mode, track data exists, and the
        required GMM-HMM model and auxiliary files are found.
        Show HMM button is enabled if HMM prediction results already exist in the
        current track data.
        """
        predict_state = tk.DISABLED
        show_state = tk.DISABLED

        # Check mode and track data
        is_single_mode = self.app.mode.get() == "single"
        has_track_data = bool(self.app.track_data.get(1))

        if is_single_mode and has_track_data:
            # Check if the GMMHMM model files exist
            import os
            from main_app_config import GMMHMM_MODEL_PATH, GMMHMM_AUX_PATH

            model_exists = os.path.exists(GMMHMM_MODEL_PATH)
            aux_exists = os.path.exists(GMMHMM_AUX_PATH)

            if model_exists and aux_exists:
                predict_state = tk.NORMAL

            # Check if HMM results already exist in track_data to enable Show HMM button
            if "hmm_section_starts" in self.app.track_data[1]:
                show_state = tk.NORMAL

        # Apply states to buttons if they exist
        if hasattr(self.app, "hmm_predict_button") and self.app.hmm_predict_button:
            self.app.hmm_predict_button.config(state=predict_state)
        if hasattr(self.app, "show_hmm_button") and self.app.show_hmm_button:
            self.app.show_hmm_button.config(state=show_state)

    def update_toggle_button_state(self):
        """
        Updates the state of the 'Toggle Labels' button.

        Enabled only if track 1 data exists, pre-cleanup labels are available,
        the mode is 'single', and the HMM results are not currently being shown.
        """
        state = tk.DISABLED
        if (
            self.app.track_data.get(1)
            and self.app.track_data[1].get("labels_before_cleanup")
            and self.app.mode.get() == "single"
            and not self.app.show_hmm_var.get()
        ):  # Disable if showing HMM
            state = tk.NORMAL

        if self.app.toggle_labels_button:
            self.app.toggle_labels_button.config(state=state)

    def update_playback_buttons_state(self, state):
        """
        Updates the Play/Pause and Stop button states based on playback state.

        Args:
            state (str): The playback state ('playing', 'paused', 'stopped').
        """
        is_playing = state == "playing"
        is_paused = state == "paused"

        # Determine if audio data is loaded and we are in single mode
        can_play = (
            self.app.playback_manager.audio_data is not None
            and self.app.playback_manager.sample_rate is not None
            and self.app.mode.get() == "single"
        )

        play_pause_state = tk.NORMAL if can_play else tk.DISABLED
        stop_state = tk.NORMAL if (is_playing or is_paused) else tk.DISABLED

        # Update button text and state if the buttons exist
        if self.app.play_pause_button:
            self.app.play_pause_button.config(
                state=play_pause_state, text="Pause" if is_playing else "Play"
            )
        if self.app.stop_button:
            self.app.stop_button.config(state=stop_state)
