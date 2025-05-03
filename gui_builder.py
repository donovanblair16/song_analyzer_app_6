# =============================================================================
# FILE: gui_builder.py
# Purpose: Contains functions to build the GUI components for the main app.
# Includes:
# - Shift Sections button.
# - Feature Table tab with Treeview.
# =============================================================================

import tkinter as tk
from tkinter import ttk

# Import SectionEditor components needed to instantiate it
try:
    # Assuming section_editor.py is accessible
    from section_editor import SectionEditor, ALLOWED_LABELS, COLOR_NAME_MAP
except ImportError:
    print(
        "ERROR in gui_builder: Could not import from section_editor. Ensure file exists."
    )
    # Define fallbacks or raise error if critical
    SectionEditor = None  # Or a dummy class/function
    ALLOWED_LABELS = []
    COLOR_NAME_MAP = {}


def build_gui(app):
    """Builds the main GUI elements by calling helper functions."""
    print("DEBUG GUI Builder: Building GUI...")
    _build_control_panel(app)
    _build_plot_notebook(app)  # This now builds the feature table tab too
    print("DEBUG GUI Builder: GUI build complete.")


# --- Control Panel Builder ---
def _build_control_panel(app):
    """Creates the top frame containing controls."""
    print("DEBUG GUI Builder: Building control panel...")
    # Use 'app' instead of 'self' to access main application attributes/methods
    app.control_frame = ttk.Frame(app.master, padding="5")
    app.control_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

    # Action/Status/Toggle/Playback Frame
    action_status_frame = ttk.Frame(app.control_frame, padding="5")
    action_status_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
    app.status_label = ttk.Label(
        action_status_frame, text="", foreground="blue", width=20, anchor=tk.CENTER
    )
    app.status_label.pack(pady=5)

    # Toggles Frame
    toggles_frame = ttk.Frame(action_status_frame)
    toggles_frame.pack(pady=5, anchor=tk.W)
    app.toggle_labels_button = ttk.Checkbutton(
        toggles_frame,
        text="Show Pre-Cleanup Labels",
        variable=app.show_pre_cleanup_labels_var,
        command=app._toggle_label_view,
        state=tk.DISABLED,
    )
    app.toggle_labels_button.pack(anchor=tk.W)
    # HMM Toggle
    # Use plot_manager method for command
    app.show_hmm_button = ttk.Checkbutton(
        toggles_frame,
        text="Show HMM Prediction",
        variable=app.show_hmm_var,
        command=app.plot_manager.display_analysis_results,
        state=tk.DISABLED,
    )
    app.show_hmm_button.pack(anchor=tk.W)

    # Playback Frame (Now includes Shift Button)
    playback_frame = ttk.Frame(action_status_frame)
    playback_frame.pack(pady=10)
    app.play_pause_button = ttk.Button(
        playback_frame,
        text="Play",
        command=app.playback_manager.toggle_play_pause,
        state=tk.DISABLED,
        width=6,
    )
    app.play_pause_button.grid(row=0, column=0, padx=2)
    app.stop_button = ttk.Button(
        playback_frame,
        text="Stop",
        command=app.playback_manager.stop,
        state=tk.DISABLED,
        width=6,
    )
    app.stop_button.grid(row=0, column=1, padx=2)

    # --- Shift Sections Button ---
    # Added the Shift Sections button, linked to a method in main_app.py
    # It starts disabled and will be enabled when analysis data is ready.
    app.shift_sections_button = ttk.Button(
        playback_frame,
        text="Shift Sections...",
        command=app._trigger_shift_sections_popup,
        state=tk.DISABLED,
        width=15,
    )  # Increased width
    app.shift_sections_button.grid(
        row=0, column=2, padx=(10, 2)
    )  # Place it next to Stop button with padding
    # --- End Shift Sections Button ---

    # File Selection & Actions Area
    file_frame = ttk.LabelFrame(app.control_frame, text="Files & Actions", padding=5)
    file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
    # File selection buttons and labels
    app.select_button1 = ttk.Button(
        file_frame,
        text="Select Track 1 (.wav)",
        command=lambda: app.file_manager.select_file(1),
    )
    app.select_button1.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
    app.file_label1 = ttk.Label(
        file_frame,
        text="No file selected",
        relief=tk.SUNKEN,
        padding=3,
        width=40,
        anchor=tk.W,
    )
    app.file_label1.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
    app.select_button2 = ttk.Button(
        file_frame,
        text="Select Track 2 (.wav)",
        command=lambda: app.file_manager.select_file(2),
    )
    app.select_button2.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
    app.file_label2 = ttk.Label(
        file_frame,
        text="No file selected",
        relief=tk.SUNKEN,
        padding=3,
        width=40,
        anchor=tk.W,
    )
    app.file_label2.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
    # Analysis and HMM buttons frame
    analysis_buttons_frame = ttk.Frame(file_frame)
    analysis_buttons_frame.grid(
        row=2, column=0, columnspan=2, sticky=tk.EW, pady=(8, 2)
    )
    app.analyze_button = ttk.Button(
        analysis_buttons_frame,
        text="Analyze Track",
        command=app.run_analysis,
        state=tk.DISABLED,
    )
    app.analyze_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
    # HMM Button
    app.hmm_predict_button = ttk.Button(
        analysis_buttons_frame,
        text="Run HMM Prediction",
        command=app._trigger_hmm_prediction,
        state=tk.DISABLED,
    )
    app.hmm_predict_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
    # Save/Load frame
    save_load_frame = ttk.Frame(file_frame)
    save_load_frame.grid(row=3, column=0, columnspan=2, pady=(5, 2), sticky=tk.EW)
    # File operation buttons
    app.load_button = ttk.Button(
        save_load_frame, text="Load Analysis", command=app.file_manager.load_analysis
    )
    app.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
    app.save_button = ttk.Button(
        save_load_frame,
        text="Save Analysis",
        command=app.file_manager.save_analysis,
        state=tk.DISABLED,
    )
    app.save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
    file_frame.grid_columnconfigure(1, weight=1)  # Allow file label column to expand

    # Mode Selection
    mode_frame = ttk.LabelFrame(app.control_frame, text="Mode", padding=5)
    mode_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
    ttk.Radiobutton(
        mode_frame,
        text="Single Track",
        variable=app.mode,
        value="single",
        command=app.ui_manager.update_ui_for_mode,
    ).pack(anchor=tk.W)
    ttk.Radiobutton(
        mode_frame,
        text="Compare Tracks",
        variable=app.mode,
        value="compare",
        command=app.ui_manager.update_ui_for_mode,
    ).pack(anchor=tk.W)

    # Manual BPM Input
    app.manual_bpm_frame = ttk.LabelFrame(
        app.control_frame, text="Manual Tempo", padding=5
    )
    app.manual_bpm_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
    app.manual_bpm_check = ttk.Checkbutton(
        app.manual_bpm_frame,
        text="Use Manual BPM",
        variable=app.use_manual_bpm,
        command=app.ui_manager.update_manual_bpm_state,
        state=tk.DISABLED,
    )
    app.manual_bpm_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)
    app.manual_bpm_entry = ttk.Entry(
        app.manual_bpm_frame,
        textvariable=app.manual_bpm_entry_var,
        width=6,
        state=tk.DISABLED,
    )
    app.manual_bpm_entry.grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
    app.update_bpm_button = ttk.Button(
        app.manual_bpm_frame,
        text="Update",
        command=app.update_plots_with_manual_bpm,
        state=tk.DISABLED,
        width=6,
    )
    app.update_bpm_button.grid(row=1, column=1, sticky=tk.W, padx=(2, 0), pady=(2, 0))

    # Analysis Options Checkboxes
    options_frame = ttk.LabelFrame(
        app.control_frame, text="Analyses To Perform", padding=5
    )
    options_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y, anchor=tk.N)
    max_rows = 4  # Adjust layout if more options are added
    # Ensure analysis_labels exists on app instance before accessing
    if hasattr(app, "analysis_labels") and app.analysis_labels:
        for i, (key, var) in enumerate(app.analysis_vars.items()):
            # Check if the key from analysis_vars exists in analysis_labels
            if key in app.analysis_labels:
                row = i % max_rows
                col = i // max_rows
                cb = ttk.Checkbutton(
                    options_frame, text=app.analysis_labels[key], variable=var
                )
                cb.grid(row=row, column=col, sticky=tk.W, padx=3, pady=1)
            else:
                print(
                    f"Warning: Key '{key}' found in analysis_vars but not in analysis_labels."
                )
    else:
        print(
            "Warning: app.analysis_labels not found or is empty. Cannot create analysis option checkboxes."
        )


# --- Plot Notebook Builder ---
def _build_plot_notebook(app):
    """Creates the ttk.Notebook widget to hold plot tabs and the feature table."""
    print("DEBUG GUI Builder: Building plot notebook...")
    app.notebook = ttk.Notebook(app.master)
    app.notebook.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

    # Define tab names - ADDED 'Feature Table'
    app.tab_names = [
        "Waveform",
        "Feature Table",
        "Energy/Balance",
        "Timbre/Texture",
        "Low-End",
        "Dynamic Range",
        "Stereo Width",
        "HPSS",
        "Root Note",
        "Band Analysis",
    ]
    app.tabs = {}

    # Initialize plot_widgets dictionary and feature_table_tree attribute
    if not hasattr(app, "plot_widgets"):
        app.plot_widgets = {}
    app.feature_table_tree = None  # Initialize attribute

    for name in app.tab_names:
        tab = ttk.Frame(app.notebook, padding="5")  # Use regular Frame for tab content
        app.tabs[name] = tab
        app.notebook.add(tab, text=name)

        if name == "Waveform":
            app.plot_widgets[name] = {}  # Initialize plot widgets dict for this tab
            # Create an outer frame that fills horizontally at the bottom
            editor_outer_frame = ttk.Frame(tab)
            editor_outer_frame.pack(
                side=tk.BOTTOM, fill=tk.X, expand=False, pady=(10, 0)
            )
            # Instantiate the editor inside the outer frame
            if SectionEditor:
                apply_callback_func = getattr(
                    app, "_apply_section_edits_from_editor", None
                )
                if not callable(apply_callback_func):
                    apply_callback_func = lambda *args: print(
                        "ERROR: Apply callback missing!"
                    )
                app.section_editor = SectionEditor(
                    editor_outer_frame, apply_callback=apply_callback_func, app_ref=app
                )
                app.section_editor.pack(anchor=tk.CENTER, pady=0)
            else:
                ttk.Label(
                    editor_outer_frame,
                    text="Error: SectionEditor could not be loaded.",
                    foreground="red",
                ).pack()

        elif name == "Feature Table":
            # Create a frame for the Treeview and Scrollbars
            table_frame = ttk.Frame(tab)
            table_frame.pack(fill=tk.BOTH, expand=True)

            # Create Scrollbars
            vsb = ttk.Scrollbar(table_frame, orient="vertical")
            hsb = ttk.Scrollbar(table_frame, orient="horizontal")

            # Create Treeview - Columns set dynamically in main_app._populate_feature_table
            app.feature_table_tree = ttk.Treeview(
                table_frame,
                columns=(),  # Empty initially
                show="headings",
                yscrollcommand=vsb.set,
                xscrollcommand=hsb.set,
            )

            # Configure Scrollbars
            vsb.config(command=app.feature_table_tree.yview)
            hsb.config(command=app.feature_table_tree.xview)

            # Pack widgets
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            app.feature_table_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Add placeholder label (will be managed by main_app)
            placeholder = ttk.Label(
                app.feature_table_tree,
                text="Analyze Track 1 to view features.",
                padding=10,
            )
            # Note: Packing placeholder directly into Treeview frame might interfere
            # It's better managed by the _populate_feature_table logic in main_app

        else:  # Other Plot Tabs
            app.plot_widgets[name] = {}  # Initialize plot widgets dict for plot tabs
            # Call placeholder method using app instance's plot manager
            if hasattr(app, "plot_manager") and app.plot_manager:
                app.plot_manager.add_placeholder_label(
                    name, f"Select file(s) and click Analyze."
                )
            else:
                ttk.Label(tab, text="Plot Manager not initialized.").pack(pady=20)
