# =============================================================================
# FILE: plot_manager.py
# Purpose: Manages displaying plots within the Tkinter notebook tabs.
# ADDED: Debug print before section editor populate call.
# =============================================================================

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import traceback

# Import plotting functions from audio_plotting
try:
    # Ensure this import path matches your project structure
    # If plot_manager.py is inside a subfolder, adjust accordingly
    import audio_plotting as ap

    print("DEBUG PlotManager: Successfully imported audio_plotting.")
except ImportError:
    print("ERROR in plot_manager: Could not import audio_plotting. Ensure file exists.")

    # Define dummy functions if needed, though plotting will fail
    class DummyPlotter:
        def __getattr__(self, name):
            def _missing_plot(*args, **kwargs):
                print(f"ERROR: audio_plotting.{name} not available.")
                return None

            return _missing_plot

    ap = DummyPlotter()


class PlotManager:
    """Handles clearing, creating placeholders, embedding, and updating plots."""

    def __init__(self, app_instance):
        """
        Initialize the PlotManager.

        Args:
            app_instance (AudioAnalyzerApp): Reference to the main application instance.
        """
        self.app = app_instance  # Store reference to main app
        print("DEBUG PlotManager: Initialized.")

    def add_placeholder_label(self, tab_name, message):
        """Adds placeholder label to a tab, ensuring editor frame is ignored."""
        app = self.app  # Use local variable for clarity
        if tab_name in app.tabs:
            tab = app.tabs[tab_name]
            if not tab.winfo_exists():
                return
            has_other_content = False
            widgets_to_destroy = []
            for w in tab.winfo_children():
                if w.winfo_exists():
                    if isinstance(w, ttk.Label) and "placeholder" in str(
                        w.winfo_name()
                    ):
                        widgets_to_destroy.append(w)
                    # Check if it's NOT the editor frame or its outer frame
                    elif tab_name != "Waveform" or not (
                        hasattr(app, "section_editor")
                        and app.section_editor
                        and (w == app.section_editor or w == app.section_editor.master)
                    ):
                        has_other_content = True
            for w in widgets_to_destroy:
                w.destroy()
            if not has_other_content and tab_name != "Waveform":
                ph = ttk.Label(
                    tab, text=message, padding=20, anchor=tk.CENTER, name="placeholder"
                )
                ph.pack(expand=True, fill=tk.BOTH)
                if tab_name not in app.plot_widgets:
                    app.plot_widgets[tab_name] = {}
                app.plot_widgets[tab_name]["placeholder"] = ph

    def add_placeholder_labels(self):
        """Adds placeholders to all tabs."""
        app = self.app
        for name in app.tab_names:
            if name == "Waveform":
                if name in app.plot_widgets and "placeholder" in app.plot_widgets[name]:
                    if app.plot_widgets[name]["placeholder"].winfo_exists():
                        app.plot_widgets[name]["placeholder"].destroy()
                    del app.plot_widgets[name]["placeholder"]
                continue
            is_analysis_selected = True
            msg = f"Select file(s) and click Analyze."
            map_key = {
                "Low-End": "low_end",
                "Dynamic Range": "dyn_range",
                "Stereo Width": "stereo",
                "HPSS": "hpss",
                "Root Note": "chroma",
                "Band Analysis": "band_plot",
                "Energy/Balance": "chroma",
                "Timbre/Texture": "chroma",
                "HMM Posteriors": "chroma",
                "Feature Importance": "chroma",  # Added
                "Emission Probabilities": "chroma",  # Added
            }.get(name)
            if map_key and not app.analysis_vars[map_key].get():
                is_analysis_selected = False
                msg = "Analysis not selected."
            elif name == "Band Analysis" and not (
                app.analysis_vars["stereo"].get() or app.analysis_vars["hpss"].get()
            ):
                is_analysis_selected = False
                msg = "Requires Stereo or HPSS Analysis."
            elif name in [
                "HMM Posteriors",
                "Feature Importance",
                "Emission Probabilities",
            ]:
                is_analysis_selected = (
                    True  # These depend on prediction, not initial analysis selection
                )
                msg = "Run HMM prediction to see HMM-related results."
            files_ok = (app.mode.get() == "single" and app.file_path.get(1)) or (
                app.mode.get() == "compare"
                and app.file_path.get(1)
                and app.file_path.get(2)
            )
            if not is_analysis_selected or not files_ok:
                self.add_placeholder_label(
                    name, msg if not is_analysis_selected else "Select file(s)..."
                )
            else:
                self.add_placeholder_label(name, f"Ready for {name} analysis...")

    def clear_plots(self, keep_editor=False):
        """Removes existing plots, toolbars, placeholders, and summary. Clears editor unless keep_editor is True."""
        app = self.app
        print("Clearing plots and summary...")
        for name, widgets in app.plot_widgets.items():
            if name in app.tabs:
                tab = app.tabs[name]
                if tab.winfo_exists():
                    widgets_to_destroy = []
                    widgets_to_keep = []
                    # Keep editor frame only if flag is set
                    if (
                        name == "Waveform"
                        and keep_editor
                        and hasattr(app, "section_editor")
                        and app.section_editor
                    ):
                        if (
                            app.section_editor.master
                            and app.section_editor.master.winfo_exists()
                        ):
                            widgets_to_keep.append(app.section_editor.master)

                    for widget_type, widget in widgets.items():
                        if widget_type in [
                            "canvas",
                            "toolbar",
                            "placeholder",
                            "figure",
                        ]:
                            widgets_to_destroy.append(widget)
                    for child in tab.winfo_children():
                        if child not in widgets_to_keep:
                            is_plot_widget = False
                            for widget_dict in app.plot_widgets.values():
                                if (
                                    "canvas" in widget_dict
                                    and widget_dict["canvas"]
                                    and hasattr(widget_dict["canvas"], "get_tk_widget")
                                    and widget_dict["canvas"].get_tk_widget()
                                    and hasattr(
                                        widget_dict["canvas"].get_tk_widget(), "master"
                                    )
                                    and child
                                    == widget_dict["canvas"].get_tk_widget().master
                                ):
                                    is_plot_widget = True
                                    break
                                if (
                                    "placeholder" in widget_dict
                                    and widget_dict["placeholder"]
                                    and child == widget_dict["placeholder"]
                                ):
                                    is_plot_widget = True
                                    break
                            if is_plot_widget:
                                if child not in widgets_to_keep:
                                    widgets_to_destroy.append(child)
                    for widget in widgets_to_destroy:
                        if widget:
                            try:
                                if isinstance(widget, (FigureCanvasTkAgg)):
                                    if (
                                        widget.get_tk_widget()
                                        and widget.get_tk_widget().master
                                        and widget.get_tk_widget().master.winfo_exists()
                                    ):
                                        widget.get_tk_widget().master.destroy()
                                elif (
                                    hasattr(widget, "winfo_exists")
                                    and widget.winfo_exists()
                                ):
                                    widget.destroy()
                            except Exception as e:
                                print(f" Minor error destroying widget: {e}")
            if (
                name == "Waveform"
                and hasattr(app, "waveform_summary_label")
                and app.waveform_summary_label
            ):
                try:
                    (
                        app.waveform_summary_label.destroy()
                        if app.waveform_summary_label.winfo_exists()
                        else None
                    )
                except Exception:
                    pass
                app.waveform_summary_label = None
        # Clear editor content only if not keeping it
        if not keep_editor and hasattr(app, "section_editor") and app.section_editor:
            app.section_editor.clear()
        app.plot_widgets = {name: {} for name in app.tab_names}
        app.waveform_summary_label = None
        app.playhead_line = None

    def embed_plot(self, fig, tab_name):
        """Clears the specified tab (except editor) and embeds the plot."""
        app = self.app
        print(f"DEBUG PlotManager: _embed_plot called for tab: {tab_name}")
        if tab_name not in app.tabs:
            print(f"Error: Tab '{tab_name}' not found.")
            plt.close(fig)
            return
        tab = app.tabs[tab_name]
        print(f"DEBUG PlotManager: Clearing plot widgets in tab {tab_name}")
        widgets_to_keep = []
        if (
            tab_name == "Waveform"
            and hasattr(app, "section_editor")
            and app.section_editor
        ):
            if app.section_editor.master and app.section_editor.master.winfo_exists():
                widgets_to_keep.append(
                    app.section_editor.master
                )  # Keep the outer frame
        for widget in list(tab.winfo_children()):
            if widget not in widgets_to_keep:
                try:
                    widget.destroy()
                except Exception:
                    pass
        if tab_name in app.plot_widgets:
            app.plot_widgets[tab_name] = {}
        if tab_name == "Waveform":
            app.waveform_summary_label = None
            app.playhead_line = None
        try:
            plot_frame = ttk.Frame(tab)
            plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            print(f"DEBUG PlotManager: Creating canvas for {tab_name}")
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_widget = canvas.get_tk_widget()
            print(f"DEBUG PlotManager: Creating toolbar for {tab_name}")
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
            if tab_name == "Waveform":
                print(f"DEBUG PlotManager: Creating waveform summary label")
                app.waveform_summary_label = ttk.Label(
                    plot_frame,
                    text="Structure: Calculating...",
                    padding=(5, 2),
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=800,
                )
                app.waveform_summary_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
                print(f"DEBUG PlotManager: Waveform summary label packed")
                axes_list = fig.get_axes()
                if axes_list:
                    ax = axes_list[0]
                    (app.playhead_line,) = ax.plot(
                        [0, 0],
                        ax.get_ylim(),
                        color="white",
                        lw=1.0,
                        alpha=0.8,
                        visible=False,
                        zorder=10,
                    )
                    print(
                        f"DEBUG PlotManager: Initial playhead line created (visible=False)"
                    )
                else:
                    print("Warning: Could not get axes from figure for playhead line.")
                    app.playhead_line = None
                print(f"DEBUG PlotManager: Connecting click event for Waveform canvas")
                canvas.mpl_connect(
                    "button_press_event", app._on_waveform_click
                )  # Call main app's handler
            print(f"DEBUG PlotManager: Packing toolbar for {tab_name}")
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            print(f"DEBUG PlotManager: Packing canvas for {tab_name}")
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            app.plot_widgets[tab_name] = {
                "canvas": canvas,
                "toolbar": toolbar,
                "figure": fig,
            }
            print(f"DEBUG PlotManager: Widgets stored for {tab_name}")
        except Exception as ee:
            print(f"--- Error embed {tab_name} ---")
            traceback.print_exc()
            ttk.Label(tab, text=f"Plot Error:\n{ee}", foreground="red").pack(
                expand=True
            )
        finally:
            plt.close(fig)
            print(f"DEBUG PlotManager: _embed_plot finished for tab: {tab_name}")

    def display_analysis_results(self):
        """Generates and embeds plots and populates editor based on current mode and data."""
        app = self.app
        mode = app.mode.get()
        display_mode = "HMM" if app.show_hmm_var.get() else "Original"
        print(
            f"\n--- PlotManager: Generating Plots & Populating Editor (Mode: {mode}, View: {display_mode}) ---"
        )

        t1_data_source = app.track_data.get(1)
        t2_data_source = app.track_data.get(2) if mode == "compare" else None

        # Create a copy of the data to potentially modify for display
        display_data1 = {}
        if t1_data_source:
            display_data1 = t1_data_source.copy()
            # Check if HMM view is selected AND HMM data exists
            if (
                app.show_hmm_var.get()
                and "hmm_semantic_labels" in t1_data_source
                and t1_data_source["hmm_semantic_labels"]
            ):
                print("DEBUG PlotManager: Using HMM labels and colors for display.")
                display_data1["semantic_labels"] = t1_data_source.get(
                    "hmm_semantic_labels", []
                )
                display_data1["label_colors"] = t1_data_source.get(
                    "hmm_label_colors", []
                )
                display_data1["section_starts"] = t1_data_source.get(
                    "hmm_section_starts", []
                )
                # Pass flag to waveform plot
                display_data1["use_hmm_view"] = True
                # Disable pre-cleanup toggle when showing HMM results
                if hasattr(app, "toggle_labels_button"):
                    app.toggle_labels_button.config(state=tk.DISABLED)
                if hasattr(app, "show_pre_cleanup_labels_var"):
                    app.show_pre_cleanup_labels_var.set(False)
            else:
                print(
                    "DEBUG PlotManager: Using Original analysis labels and colors for display."
                )
                display_data1.setdefault("semantic_labels", [])
                display_data1.setdefault("label_colors", [])
                display_data1.setdefault("section_starts", [])
                display_data1["use_hmm_view"] = (
                    False  # Ensure flag is set for original view
                )
                # Re-enable pre-cleanup toggle button if original data exists
                app.ui_manager.update_toggle_button_state()  # Call main app's method

        display_data2 = t2_data_source.copy() if t2_data_source else {}
        # HMM comparison not implemented, display_data2 always uses original

        # Clear existing plots before drawing new ones based on selected view
        self.clear_plots(keep_editor=True)  # Keep editor populated with original data

        plot_configs = [
            (
                "Waveform",
                ap.create_single_track_plot,
                ap.create_comparison_plot,
                ap.plot_waveform,
                ["sections", "chroma"],  # Dependency for spectral centroid overlay
                {},
            ),
            (
                "HMM Posteriors",
                ap.create_single_track_plot,
                None,
                ap.plot_hmm_posteriors,
                ["sections", "chroma"],  # Needs original sections for context
                {},
            ),
            (  # Added Feature Importance
                "Feature Importance",
                ap.create_single_track_plot,
                None,  # No comparison plot defined
                ap.plot_feature_importance,
                ["chroma"],  # Depends on having features calculated
                {"num_rows": 2},  # Requires 2 axes
            ),
            (  # Added Emission Probabilities
                "Emission Probabilities",
                ap.create_single_track_plot,
                None,  # No comparison plot defined
                ap.plot_emission_probabilities,
                ["chroma"],  # Depends on having features calculated
                {"num_rows": 2},  # Requires 2 axes
            ),
            (
                "Energy/Balance",
                ap.create_single_track_plot,
                None,
                ap.plot_energy_balance_features,
                ["chroma"],
                {"num_rows": 5},
            ),
            (
                "Timbre/Texture",
                ap.create_single_track_plot,
                None,
                ap.plot_timbre_texture_features,
                ["chroma"],
                {"num_rows": 3},
            ),
            (
                "Low-End",
                ap.create_single_track_plot,
                ap.create_comparison_plot,
                ap.plot_low_energy,
                ["low_end"],
                {},
            ),
            (
                "Dynamic Range",
                ap.create_single_track_plot,
                ap.create_comparison_plot,
                ap.plot_dynamic_range,
                ["dyn_range"],
                {},
            ),
            (
                "Stereo Width",
                ap.create_single_track_plot,
                ap.create_comparison_plot,
                ap.plot_stereo_width_overlay,
                ["stereo"],
                {},
            ),
            (
                "HPSS",
                ap.create_single_track_plot,
                ap.create_comparison_plot,
                ap.plot_hpss,
                ["hpss"],
                {},
            ),
            (
                "Root Note",
                ap.create_single_track_plot,
                ap.create_comparison_plot,
                ap.plot_root_note,
                ["chroma"],
                {},
            ),
            (
                "Band Analysis",
                ap.create_band_analysis_plot,
                ap.create_band_comparison_plot,
                None,
                ["band_plot", "stereo", "hpss"],
                {},
            ),
        ]

        for (
            tab_name,
            s_func,
            c_func,
            p_func_comp,
            req_keys,
            s_func_kwargs,
        ) in plot_configs:
            if tab_name not in app.tabs:
                continue

            # Special handling for HMM-related tabs
            is_hmm_tab = tab_name in [
                "HMM Posteriors",
                "Feature Importance",
                "Emission Probabilities",
            ]
            if is_hmm_tab:
                print(f"DEBUG: Checking {tab_name} tab conditions:")
                print(f"  - Mode is single: {mode == 'single'}")
                print(f"  - HMM view active: {app.show_hmm_var.get()}")
                print(f"  - track_data available: {bool(t1_data_source)}")
                data_key = {
                    "HMM Posteriors": "hmm_posteriors",
                    "Feature Importance": "hmm_feature_importance",
                    "Emission Probabilities": "hmm_feature_importance",  # Uses same source data
                }[tab_name]

                if t1_data_source:
                    print(f"  - {data_key} key exists: {data_key in t1_data_source}")
                    if data_key in t1_data_source:
                        print(
                            f"  - {data_key} not None: {t1_data_source[data_key] is not None}"
                        )
                        if t1_data_source[data_key] is not None and hasattr(
                            t1_data_source[data_key], "shape"
                        ):
                            print(
                                f"  - {data_key} shape: {t1_data_source[data_key].shape}"
                            )
                        elif t1_data_source[data_key] is not None and isinstance(
                            t1_data_source[data_key], dict
                        ):
                            print(
                                f"  - {data_key} keys: {list(t1_data_source[data_key].keys())}"
                            )

                should_show_hmm_plot = (
                    mode == "single"
                    and app.show_hmm_var.get()
                    and t1_data_source
                    and data_key in t1_data_source
                    and t1_data_source[data_key] is not None
                )

                if should_show_hmm_plot:
                    print(f" PlotManager: Plotting {tab_name}...")
                    try:
                        # Ensure we have all necessary data for plotting
                        plot_data = display_data1.copy()  # Start with HMM view data
                        # Add original data needed for context by some plots
                        plot_data["section_starts"] = t1_data_source.get(
                            "section_starts"
                        )
                        plot_data["seconds_per_bar"] = t1_data_source.get(
                            "seconds_per_bar"
                        )
                        plot_data["trim_offset_sec"] = t1_data_source.get(
                            "trim_offset_sec"
                        )
                        plot_data["duration_processed"] = t1_data_source.get(
                            "duration_processed"
                        )

                        # Add int_to_label mapping if available (needed for posteriors/emission)
                        if (
                            hasattr(app.hmm_predictor, "int_to_label")
                            and app.hmm_predictor.int_to_label
                        ):
                            plot_data["hmm_int_to_label"] = (
                                app.hmm_predictor.int_to_label
                            )
                            print(
                                f"DEBUG: Added int_to_label mapping to plot_data: {app.hmm_predictor.int_to_label}"
                            )

                        # Add feature importance data if needed
                        if tab_name in ["Feature Importance", "Emission Probabilities"]:
                            plot_data["hmm_feature_importance"] = t1_data_source.get(
                                "hmm_feature_importance"
                            )

                        fig = s_func(
                            plot_data,
                            p_func_comp,
                            tab_name,  # Pass tab name as title for multi-plot figures
                            app.track_names[1],
                            **s_func_kwargs,  # Pass num_rows etc.
                        )

                        if fig:
                            self.embed_plot(fig, tab_name)
                        else:
                            self.add_placeholder_label(
                                tab_name, f"Could not generate {tab_name} plot."
                            )
                    except Exception as post_e:
                        print(f"--- Error plot {tab_name} ---")
                        traceback.print_exc()
                        self.add_placeholder_label(tab_name, f"Error:\n{post_e}")
                else:
                    msg = f"{tab_name} not available."
                    if not app.show_hmm_var.get():
                        msg = f"Switch to HMM view to display {tab_name}."
                    elif not (
                        t1_data_source
                        and data_key in t1_data_source
                        and t1_data_source[data_key] is not None
                    ):
                        msg = f"Run HMM prediction first or required data ('{data_key}') missing."

                    self.add_placeholder_label(tab_name, msg)

                # Skip normal processing for these special tabs
                continue

            # Normal tab processing for other tabs
            # Check if original analysis was selected for this plot type
            analysis_enabled = True
            core_analysis = req_keys[0] if req_keys else None
            if (
                core_analysis
                and core_analysis in app.analysis_vars
                and not app.analysis_vars[core_analysis].get()
            ):
                analysis_enabled = False
            elif tab_name == "Band Analysis" and not (
                app.analysis_vars["stereo"].get() or app.analysis_vars["hpss"].get()
            ):
                analysis_enabled = False
                print(f" Skipping {tab_name}: Needs Stereo or HPSS Analysis.")
            elif (
                tab_name in ["Energy/Balance", "Timbre/Texture"]
                and not app.analysis_vars["chroma"].get()
            ):
                analysis_enabled = False
                print(
                    f" Skipping {tab_name}: Needs Chroma/Labeling Analysis for features."
                )

            # Use display_data1/2 which contain the selected labels/colors/starts
            data_available = (
                bool(display_data1)
                if mode == "single"
                else (bool(display_data1) or bool(display_data2))
            )

            if analysis_enabled and data_available:
                print(f" PlotManager: Plotting {tab_name} using {display_mode} data...")
                fig = None
                try:
                    # Pass the potentially modified display_data to plotting functions
                    plot_data1 = display_data1 if display_data1 else {}
                    plot_data2 = display_data2 if display_data2 else {}
                    if mode == "single":
                        # Pass the correct display data to the plotting function
                        if s_func == ap.create_band_analysis_plot:
                            fig = s_func(plot_data1, app.track_names[1])
                        elif s_func == ap.create_single_track_plot:
                            fig = s_func(
                                plot_data1,
                                p_func_comp,
                                tab_name,
                                app.track_names[1],
                                **s_func_kwargs,
                            )
                    else:  # Compare mode (HMM comparison not implemented, uses original)
                        plot_data1_comp = (
                            t1_data_source.copy() if t1_data_source else {}
                        )  # Use original for T1 in compare
                        plot_data2_comp = (
                            t2_data_source.copy() if t2_data_source else {}
                        )  # Use original for T2
                        if c_func == ap.create_band_comparison_plot:
                            fig = c_func(
                                plot_data1_comp,
                                plot_data2_comp,
                                app.track_names[1],
                                app.track_names[2],
                            )
                        elif c_func is not None:
                            fig = c_func(
                                plot_data1_comp,
                                plot_data2_comp,
                                p_func_comp,
                                f"{tab_name} Comparison",
                                app.track_names[1],
                                app.track_names[2],
                            )
                        else:
                            print(f" Skipping comparison for {tab_name}.")
                            self.add_placeholder_label(
                                tab_name, "Comparison not available."
                            )

                    if fig:
                        self.embed_plot(fig, tab_name)  # Call self.embed_plot
                        # Update summary label (only for original view on waveform tab)
                        if (
                            tab_name == "Waveform"
                            and mode == "single"
                            and app.waveform_summary_label is not None
                        ):
                            if (
                                not app.show_hmm_var.get()
                            ):  # Only show summary for original view
                                labels_for_summary = plot_data1.get("semantic_labels")
                                features_for_summary = plot_data1.get(
                                    "section_features"
                                )
                                summary_string = (
                                    "Structure (T1): Error generating summary."
                                )
                                if labels_for_summary and features_for_summary:
                                    try:
                                        summary_string = (
                                            "Structure (T1): "
                                            + ap.generate_structure_summary(
                                                labels_for_summary, features_for_summary
                                            )
                                        )
                                    except Exception as summary_e:
                                        print(
                                            f"Error generating summary string: {summary_e}"
                                        )
                                else:
                                    summary_string = (
                                        "Structure (T1): Data missing for summary."
                                    )
                                app.waveform_summary_label.config(text=summary_string)
                                app.master.update_idletasks()
                                wrap_w = app.tabs[tab_name].winfo_width() - 20
                                app.waveform_summary_label.config(
                                    wraplength=max(100, wrap_w)
                                )
                            else:  # Clear summary for HMM view
                                app.waveform_summary_label.config(
                                    text="Structure Summary (N/A for HMM view)"
                                )

                    elif analysis_enabled:
                        self.add_placeholder_label(
                            tab_name, "Plot skipped (figure generation failed)."
                        )
                except Exception as plot_e:
                    print(f"--- Error plot {tab_name} ---")
                    traceback.print_exc()
                    self.add_placeholder_label(tab_name, f"Error:\n{plot_e}")
                    plt.close(fig) if fig else None
            elif not data_available:
                print(f" Skipping {tab_name} (analysis data missing).")
                self.add_placeholder_label(tab_name, "Analysis data missing.")
            else:
                print(
                    f" Skipping {tab_name} (analysis not selected / dependencies missing)."
                )
                self.add_placeholder_label(
                    tab_name, "Analysis not selected / dependencies missing."
                )

        # --- Section editor population ---
        # <<< ADDED DEBUG PRINT >>>
        print(
            f"DEBUG PlotManager: Checking conditions for populate: mode='{mode}', editor_exists={app.section_editor is not None}, data_exists={bool(t1_data_source)}"
        )
        # <<< END DEBUG PRINT >>>
        if mode == "single" and app.section_editor and t1_data_source:
            print(
                "DEBUG PlotManager: Populating section editor with original analysis data."
            )
            try:
                app.section_editor.populate(t1_data_source)  # Use original data source
            except Exception as pop_e:
                print(f"ERROR PlotManager: Failed to populate section editor: {pop_e}")
                traceback.print_exc()
                # Optionally show an error to the user or disable the editor
                if app.section_editor:
                    app.section_editor.clear()
                    # Maybe add a placeholder label to the editor frame itself?
