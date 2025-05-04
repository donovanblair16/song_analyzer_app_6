# =============================================================================
# FILE: inspect_perfect_features_gui.py
# PURPOSE: Provides a GUI to load and inspect section features from
#          analysis files stored in the 'completed_analyses/Perfect' folder.
#          Includes file list, feature display, and caching.
# USAGE: Run this script from the main project directory
#        (e.g., song_analyzer_app_6) or its subdirectory.
# MODIFIED:
# - Added print statements for debugging path issues.
# - Fixed cache file path to be relative to the script's directory.
# - Added "Show NaN Only" filter checkbox.
# - Added display area for unique feature sets found across files.
# - Data loading now pre-calculates NaN presence and unique feature sets.
# - Listbox population applies the NaN filter.
# - Implemented PERSISTENT caching using a .joblib cache file.
# - Loads from cache on startup if available.
# - Refresh button updates cache file.
# - Loads ALL analysis files into cache at startup/refresh (background).
# - File selection retrieves directly from cache.
# - Increased font size in file listbox.
# - Removed '.analysis.joblib' suffix from displayed filenames.
# - FIXED: Added explicit listbox update after initial load
# - FIXED: Better error handling for corrupted cache
# - FIXED: Addressed filter application issues
# - FIXED: Improved GUI updates during loading
# - ADDED: Feature set filtering functionality
# - ADDED: Robust column visibility toggle and column reordering functionality
# - REMOVED: Limitation to first 5 files for processing entire folder
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd  # Using pandas for DataFrame creation before display
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font as tkFont  # Added tkFont
import traceback
import time  # For checking modification times
import threading  # To load data in background thread
from collections import defaultdict  # For feature set analysis

# --- Configuration ---
# Determine the script's own directory to make paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory to find the main project root
PROJECT_BASE_FOLDER = os.path.dirname(SCRIPT_DIR)

ANALYSIS_BASE_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "completed_analyses")
PERFECT_SUBFOLDER = "Perfect"
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, PERFECT_SUBFOLDER)
FILENAME_SUFFIX = ".analysis.joblib"  # Define suffix for easy removal

# --- Persistent Cache File ---
CACHE_FILENAME = "feature_inspector_cache.joblib"
# Save cache in the SAME directory as this script
CACHE_FILE_PATH = os.path.join(SCRIPT_DIR, CACHE_FILENAME)

# --- DEBUG: Print calculated paths ---
print(f"--- DEBUG PATHS ---")
print(f"SCRIPT_DIR: {SCRIPT_DIR}")
print(f"PROJECT_BASE_FOLDER: {PROJECT_BASE_FOLDER}")
print(f"ANALYSIS_BASE_FOLDER: {ANALYSIS_BASE_FOLDER}")
print(f"PERFECT_FOLDER_PATH: {PERFECT_FOLDER_PATH}")
print(f"CACHE_FILE_PATH: {CACHE_FILE_PATH}")
print(f"--- END DEBUG PATHS ---")


# --- Main Application Class ---


class FeatureInspectorApp:
    """
    Tkinter GUI application for inspecting section features in analysis files
    with persistent caching and filtering.
    """

    def __init__(self, master):
        """Initialize the application."""
        self.master = master
        master.title("Perfect Analysis Feature Inspector (Cached)")
        master.geometry(
            "1300x750"
        )  # Increased width to accommodate column control pane

        # Data Caching & State
        self.data_cache = {}  # {file_path: track_data}
        self.cache_timestamps = {}  # {file_path: mtime}
        self.full_file_paths = []  # Full paths corresponding to listbox order
        self.file_has_nan = {}  # {file_path: bool} - Calculated during load
        self.unique_feature_sets = []  # List of tuples representing unique sets of keys
        self.loading_in_progress = False
        self.selected_feature_set = None  # Currently selected feature set for filtering
        self.file_to_feature_sets = {}  # Maps file paths to their feature sets

        # Column visibility and order tracking
        self.all_columns = []  # All possible columns in original order
        self.column_visibility = (
            {}
        )  # Dictionary to track visibility {column_name: is_visible}
        self.column_order = []  # Current display order of columns
        self.current_data = []  # Store current row data for redisplaying
        self.dragging = False  # Flag for column drag operation
        self.drag_column = None  # Column being dragged

        # --- Tkinter Variables ---
        self.show_nan_only_var = tk.BooleanVar(
            value=False
        )  # Make sure this is False by default
        self.status_var = tk.StringVar(value="Initializing...")

        # --- Create Custom Font ---
        default_font = tkFont.nametofont("TkDefaultFont")
        listbox_font_size = default_font.actual()["size"] + 4
        self.listbox_font = tkFont.Font(size=listbox_font_size)
        self.small_font = tkFont.Font(
            size=default_font.actual()["size"] - 1
        )  # For feature set display

        # --- Create Main UI Structure ---
        self.main_paned_window = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Pane: File List, Filters & Controls ---
        self.left_frame = ttk.Frame(self.main_paned_window, width=250)
        self.main_paned_window.add(self.left_frame, weight=1)

        # --- File List Area ---
        file_list_frame = ttk.LabelFrame(self.left_frame, text="Analysis Files")
        file_list_frame.pack(pady=(0, 5), padx=5, fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(file_list_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.file_list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=self.file_list_scrollbar.set,
            exportselection=False,
            font=self.listbox_font,
        )
        self.file_list_scrollbar.config(command=self.file_listbox.yview)
        self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # --- Filter & Control Area ---
        control_frame = ttk.Frame(self.left_frame)
        control_frame.pack(fill=tk.X, pady=5, padx=5)

        # NaN Filter Checkbox
        self.nan_filter_check = ttk.Checkbutton(
            control_frame,
            text="Show Files with NaN Features Only",
            variable=self.show_nan_only_var,
            command=self._apply_filters,
        )
        self.nan_filter_check.pack(anchor=tk.W)

        # Refresh Button
        self.refresh_button = ttk.Button(
            control_frame, text="Refresh Data", command=self._refresh_data
        )
        self.refresh_button.pack(pady=5)

        # Reset Filters Button
        self.reset_button = ttk.Button(
            control_frame, text="Reset Filters", command=self._reset_filters
        )
        self.reset_button.pack(pady=5)

        # --- Feature Set Display Area ---
        feature_set_frame = ttk.LabelFrame(
            self.left_frame, text="Unique Feature Sets (Click to Filter)"
        )
        feature_set_frame.pack(
            pady=(5, 0), padx=5, fill=tk.BOTH, expand=False
        )  # Don't expand vertically much

        self.feature_set_listbox = tk.Listbox(
            feature_set_frame, height=6, font=self.small_font
        )  # Limit height
        feature_set_scrollbar = ttk.Scrollbar(
            feature_set_frame,
            orient=tk.VERTICAL,
            command=self.feature_set_listbox.yview,
        )
        self.feature_set_listbox.config(yscrollcommand=feature_set_scrollbar.set)
        feature_set_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_set_listbox.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5
        )
        # Bind selection event for feature set filtering
        self.feature_set_listbox.bind("<<ListboxSelect>>", self.on_feature_set_select)

        # --- Middle Pane: Column Visibility Controls ---
        self.column_control_frame = ttk.Frame(self.main_paned_window, width=250)
        self.main_paned_window.add(self.column_control_frame, weight=1)

        # Column control area
        column_control_label_frame = ttk.LabelFrame(
            self.column_control_frame, text="Column Visibility"
        )
        column_control_label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas with scrollbar for column checkboxes
        column_canvas_frame = ttk.Frame(column_control_label_frame)
        column_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.column_scroll = ttk.Scrollbar(column_canvas_frame, orient=tk.VERTICAL)
        self.column_canvas = tk.Canvas(
            column_canvas_frame,
            yscrollcommand=self.column_scroll.set,
            highlightthickness=0,
        )
        self.column_scroll.config(command=self.column_canvas.yview)
        self.column_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.column_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for checkboxes inside canvas
        self.column_checkbox_frame = ttk.Frame(self.column_canvas)
        self.column_canvas_window = self.column_canvas.create_window(
            (0, 0),
            window=self.column_checkbox_frame,
            anchor=tk.NW,
            tags="self.column_checkbox_frame",
        )

        # Quick selection buttons for columns
        column_button_frame = ttk.Frame(column_control_label_frame)
        column_button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            column_button_frame, text="Select All", command=self._select_all_columns
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            column_button_frame, text="Deselect All", command=self._deselect_all_columns
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            column_button_frame, text="Reset Order", command=self._reset_column_order
        ).pack(side=tk.LEFT, padx=2)

        # Instructions for column reordering
        ttk.Label(
            column_control_label_frame,
            text="Drag column headers to reorder",
            font=self.small_font,
            wraplength=180,
        ).pack(pady=(0, 5))

        # Configure the canvas for proper scrolling
        self.column_checkbox_frame.bind(
            "<Configure>", self._on_checkbox_frame_configure
        )
        self.column_canvas.bind("<Configure>", self._on_column_canvas_configure)

        # --- Right Pane: Feature Display Treeview ---
        self.right_frame = ttk.Frame(self.main_paned_window, width=650)
        self.main_paned_window.add(self.right_frame, weight=3)

        ttk.Label(self.right_frame, text="Section Features:").pack(
            pady=(0, 5), anchor=tk.W
        )

        tree_frame = ttk.Frame(self.right_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.feature_tree = ttk.Treeview(
            tree_frame,
            columns=(),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self.feature_tree.yview)
        hsb.config(command=self.feature_tree.xview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.feature_tree.pack(fill=tk.BOTH, expand=True)

        # Bind events for column drag-and-drop reordering
        self.feature_tree.bind("<ButtonPress-1>", self._start_column_drag)
        self.feature_tree.bind("<B1-Motion>", self._column_drag_motion)
        self.feature_tree.bind("<ButtonRelease-1>", self._end_column_drag)

        # --- Status Bar ---
        status_bar = ttk.Label(
            master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Load Cache or Start Initial Load ---
        self.master.after(100, self._load_cache_or_start_load)

    # --- Canvas Configuration Handlers ---

    def _on_checkbox_frame_configure(self, event):
        """Update the scroll region when the checkbox frame changes size."""
        self.column_canvas.configure(scrollregion=self.column_canvas.bbox("all"))

    def _on_column_canvas_configure(self, event):
        """Adjust the inner frame width when the canvas is resized."""
        canvas_width = event.width
        self.column_canvas.itemconfig(self.column_canvas_window, width=canvas_width)

    # --- Column Control Functions ---

    def _setup_column_checkboxes(self, columns):
        """Set up the column visibility checkboxes."""
        # Clear existing checkboxes
        for widget in self.column_checkbox_frame.winfo_children():
            widget.destroy()

        # Set up the initial state of all columns
        for col in columns:
            if col not in self.column_visibility:
                # By default, all columns are visible
                self.column_visibility[col] = True

        # Create the checkboxes in the original column order
        for i, col in enumerate(columns):
            # Create a variable for this checkbox
            var = tk.BooleanVar(value=self.column_visibility.get(col, True))

            # Create a checkbox for this column
            checkbox = ttk.Checkbutton(
                self.column_checkbox_frame,
                text=col,
                variable=var,
                command=lambda c=col, v=var: self._toggle_column_visibility(c, v),
            )
            checkbox.pack(anchor=tk.W, pady=1, padx=5, fill=tk.X)

            # Store the variable for later access
            self.column_visibility[col] = var.get()

    def _toggle_column_visibility(self, column, var=None):
        """Toggle visibility of a specific column."""
        # Update the visibility state
        if var is not None:
            self.column_visibility[column] = var.get()
        else:
            current = self.column_visibility.get(column, True)
            self.column_visibility[column] = not current

        # Update the treeview display
        self._refresh_treeview()

    def _select_all_columns(self):
        """Select all column checkboxes."""
        # Update visibility state for all columns
        for col in self.all_columns:
            self.column_visibility[col] = True

        # Update checkbox states
        for widget in self.column_checkbox_frame.winfo_children():
            if isinstance(widget, ttk.Checkbutton):
                # This is a hack to set the variable value and update the checkbox display
                widget.state(["selected"])

        # Refresh the treeview
        self._refresh_treeview()
        self._update_status("All columns selected")

    def _deselect_all_columns(self):
        """Deselect all column checkboxes except essential ones."""
        # Standard columns that should always be visible
        standard_cols = ["#", "Label", "Start (s)", "End (s)", "Dur (s)"]

        # Update visibility state for all columns
        for col in self.all_columns:
            self.column_visibility[col] = col in standard_cols

        # Update checkbox states in the UI
        for widget in self.column_checkbox_frame.winfo_children():
            if isinstance(widget, ttk.Checkbutton):
                # Get the column name from the checkbox text
                col_name = widget.cget("text")
                if col_name in standard_cols:
                    widget.state(["selected"])
                else:
                    widget.state(["!selected"])

        # Refresh the treeview
        self._refresh_treeview()
        self._update_status("Deselected all columns except standard ones")

    def _reset_column_order(self):
        """Reset columns to their original order."""
        # Restore original column order
        self.column_order = self.all_columns.copy()

        # Refresh the treeview with the new order
        self._refresh_treeview()
        self._update_status("Column order reset to original")

    def _get_visible_columns_in_order(self):
        """Get the list of visible columns in the current display order."""
        # If column_order is empty, initialize it with all columns
        if not self.column_order:
            self.column_order = self.all_columns.copy()

        # Filter visible columns based on current order
        visible_columns = []
        for col in self.column_order:
            if self.column_visibility.get(col, True):
                visible_columns.append(col)

        # Handle any visible columns not in the current order
        for col in self.all_columns:
            if self.column_visibility.get(col, True) and col not in visible_columns:
                visible_columns.append(col)

        return visible_columns

    def _refresh_treeview(self):
        """Refresh the treeview with the current column visibility and order settings."""
        # Get visible columns in current order
        visible_columns = self._get_visible_columns_in_order()

        # Remember currently selected item if any
        selected_items = self.feature_tree.selection()
        selected_id = selected_items[0] if selected_items else None

        # Clear the treeview
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        # Re-configure columns
        self.feature_tree["columns"] = visible_columns

        # Configure column properties
        for col in visible_columns:
            # Set appropriate column width and properties
            anchor = tk.W
            width = 80
            stretch = tk.NO

            if col == "#":
                width = 40
            elif col == "Label":
                width = 100
                stretch = tk.YES
            elif col in ["Start (s)", "End (s)", "Dur (s)"]:
                width = 70
            elif "rms" in col.lower() or "energy" in col.lower():
                width = 110
                stretch = tk.YES
            elif "centroid" in col.lower() or "position" in col.lower():
                width = 130
                stretch = tk.YES
            elif len(col) > 15:
                width = 120
                stretch = tk.YES

            self.feature_tree.heading(col, text=col, anchor=anchor)
            self.feature_tree.column(col, width=width, anchor=anchor, stretch=stretch)

        # Repopulate the treeview with current data
        if self.current_data:
            for item_id, values_dict in self.current_data:
                # Create a new row with only the visible columns in correct order
                row_values = []
                for col in visible_columns:
                    row_values.append(values_dict.get(col, ""))

                # Insert into treeview
                self.feature_tree.insert(
                    "", tk.END, iid=item_id, values=tuple(row_values)
                )

            # Restore selection if possible
            if selected_id:
                try:
                    self.feature_tree.selection_set(selected_id)
                    self.feature_tree.see(selected_id)
                except:
                    pass

    # --- Column Drag and Drop Functions ---

    def _start_column_drag(self, event):
        """Start dragging a column header."""
        if self.dragging:
            return

        # Identify the region and column being clicked
        region = self.feature_tree.identify_region(event.x, event.y)
        if region != "heading":
            return

        # Get the column identifier
        column_id = self.feature_tree.identify_column(event.x)

        # Convert column id (#1, #2, etc.) to column name
        if column_id.startswith("#"):
            col_idx = int(column_id[1:]) - 1
            visible_columns = self._get_visible_columns_in_order()

            if 0 <= col_idx < len(visible_columns):
                # Found a valid column to drag
                self.drag_column = visible_columns[col_idx]
                self.dragging = True

                # Change cursor to indicate dragging
                self.feature_tree.configure(cursor="exchange")

                # Save the x position for calculating drag direction
                self.drag_start_x = event.x

    def _column_drag_motion(self, event):
        """Handle column dragging motion."""
        # Only process if we're in a drag operation
        if not self.dragging or not self.drag_column:
            return

        # Provide visual feedback here if needed
        pass

    def _end_column_drag(self, event):
        """End column dragging and reposition the column."""
        # Only process if we're in a drag operation
        if not self.dragging or not self.drag_column:
            return

        try:
            # Reset cursor
            self.feature_tree.configure(cursor="")

            # Identify the target column
            target_id = self.feature_tree.identify_column(event.x)

            if target_id.startswith("#"):
                target_idx = int(target_id[1:]) - 1
                visible_columns = self._get_visible_columns_in_order()

                if 0 <= target_idx < len(visible_columns):
                    target_column = visible_columns[target_idx]

                    # Only reorder if dropping on a different column
                    if target_column != self.drag_column:
                        self._reorder_column(self.drag_column, target_column)
        finally:
            # Reset drag state
            self.dragging = False
            self.drag_column = None

    def _reorder_column(self, source_column, target_column):
        """Reorder columns by moving source_column before target_column."""
        if not self.column_order:
            # Initialize column order if needed
            self.column_order = self.all_columns.copy()

        # Create a working copy of the current order
        current_order = self.column_order.copy()

        # Find positions in the current order
        if source_column in current_order and target_column in current_order:
            # Remove source column from its current position
            current_order.remove(source_column)

            # Find the target position
            target_pos = current_order.index(target_column)

            # Insert source column at the target position
            current_order.insert(target_pos, source_column)

            # Update column order
            self.column_order = current_order

            # Refresh the treeview
            self._refresh_treeview()

            self._update_status(
                f"Moved column '{source_column}' before '{target_column}'"
            )

    # --- Feature Set Filtering Logic ---

    def on_feature_set_select(self, event=None):
        """Handles selection change event in the feature set listbox."""
        if self.loading_in_progress:
            print("Loading in progress, ignoring feature set selection")
            return

        selected_indices = self.feature_set_listbox.curselection()

        # Handle toggle behavior - if clicking already selected item, deselect it
        if self.selected_feature_set is not None and selected_indices:
            selected_index = selected_indices[0]
            if (
                selected_index < len(self.unique_feature_sets)
                and self.unique_feature_sets[selected_index]
                == self.selected_feature_set
            ):
                # User clicked the already selected item - deselect it
                print("Deselecting current feature set")
                self.feature_set_listbox.selection_clear(0, tk.END)
                self.selected_feature_set = None
                self._update_status("Feature set filter cleared")
                self._apply_filters()
                return

        # Handle normal selection or deselection
        if not selected_indices:
            # Nothing selected, clear filter
            print("No feature set selected, clearing filter")
            self.selected_feature_set = None
            self._apply_filters()
            return

        selected_index = selected_indices[0]
        if selected_index < 0 or selected_index >= len(self.unique_feature_sets):
            print(f"Error: Feature set index {selected_index} out of bounds")
            return

        # Store selected feature set for filtering
        self.selected_feature_set = self.unique_feature_sets[selected_index]
        print(f"Selected feature set: {self.selected_feature_set}")

        # Update status bar
        feature_set_str = ", ".join(self.selected_feature_set)
        self._update_status(f"Filtering by feature set: {feature_set_str}")

        # Apply filter
        self._apply_filters()

    def _check_file_matches_feature_set(self, file_path, feature_set):
        """Checks if a file contains sections with the specified feature set."""
        # Use the pre-calculated mapping for efficiency
        if file_path in self.file_to_feature_sets:
            feature_sets_in_file = self.file_to_feature_sets[file_path]
            return feature_set in feature_sets_in_file

        # Fall back to direct check if pre-calculated mapping not available
        if file_path not in self.data_cache:
            return False

        track_data = self.data_cache[file_path]
        if not isinstance(track_data, dict):
            return False

        features_list = track_data.get("section_features", [])
        if not isinstance(features_list, list):
            return False

        # Check each section in the file
        for section_dict in features_list:
            if not isinstance(section_dict, dict):
                continue

            # Get the feature keys for this section
            section_keys = tuple(
                sorted(
                    [
                        k
                        for k in section_dict.keys()
                        if k
                        not in [
                            "index",
                            "start_time",
                            "end_time",
                            "duration_sec",
                            "duration_bars",
                            "original_label",
                            "cluster_id",
                        ]
                    ]
                )
            )

            # Check if this section has the feature set we're looking for
            if section_keys == feature_set:
                return True

        return False

    def _reset_filters(self):
        """Resets all filters to their default states."""
        self.show_nan_only_var.set(False)  # Reset NaN filter
        self.selected_feature_set = None  # Clear feature set filter

        # Clear selection in feature set listbox
        self.feature_set_listbox.selection_clear(0, tk.END)

        # Update status
        self._update_status("Filters reset")

        # Apply filters (which will now show all files)
        self._apply_filters()

    # --- Caching and Loading Logic ---

    def _load_cache_or_start_load(self):
        """Tries to load persistent cache, otherwise starts background load."""
        try:
            # First, check if cache file exists and is readable
            if os.path.exists(CACHE_FILE_PATH) and os.path.getsize(CACHE_FILE_PATH) > 0:
                try:
                    # Try to load the cache
                    print(f"Loading persistent cache from: {CACHE_FILE_PATH}")
                    start_time = time.time()
                    cached_content = joblib.load(CACHE_FILE_PATH)
                    load_time = time.time() - start_time

                    # Validate cache structure
                    if (
                        isinstance(cached_content, dict)
                        and "data" in cached_content
                        and "timestamps" in cached_content
                    ):

                        self.data_cache = cached_content["data"]
                        self.cache_timestamps = cached_content["timestamps"]
                        self.full_file_paths = sorted(
                            list(self.cache_timestamps.keys())
                        )

                        print(
                            f"Cache loaded successfully in {load_time:.2f}s. {len(self.data_cache)} files."
                        )
                        self._update_status("Loaded data from cache. Analyzing...")

                        # Perform NaN check and feature set analysis on cached data
                        self._analyze_cached_data()

                        # Populate listbox based on cached paths and current filters
                        self._update_listbox()  # Force update regardless of filters initially

                        self._update_status("Loaded data from cache. Ready.")
                        return True
                    else:
                        print("Error: Invalid cache file format.")
                        # Continue to initial load
                except Exception as e:
                    print(f"Error loading persistent cache file: {e}")
                    traceback.print_exc()
                    # Try to remove corrupted cache
                    try:
                        os.remove(CACHE_FILE_PATH)
                        print(f"Removed corrupted cache file: {CACHE_FILE_PATH}")
                    except OSError as e:
                        print(f"Could not remove cache file: {e}")
            else:
                print("Persistent cache file not found or empty.")
        except Exception as e:
            print(f"Unexpected error during cache check: {e}")
            traceback.print_exc()

        # If we get here, we need to do an initial load
        print("Starting initial data load...")
        self._update_status("Cache not found or invalid. Loading all files...")
        self._initial_load()
        return False

    def _initial_load(self):
        """Starts the initial loading of all data in a background thread."""
        if self.loading_in_progress:
            return
        self.loading_in_progress = True
        load_thread = threading.Thread(
            target=self._load_all_data, args=(True,), daemon=True
        )
        load_thread.start()

    def _refresh_data(self):
        """Starts the data refresh process in a background thread."""
        if self.loading_in_progress:
            print("Load/Refresh already in progress.")
            return

        # Clear any existing cache file AND in-memory cache to force a complete reload
        try:
            # Clear file cache
            if os.path.exists(CACHE_FILE_PATH):
                os.remove(CACHE_FILE_PATH)
                print(f"Removed existing cache file for refresh: {CACHE_FILE_PATH}")

            # Clear in-memory cache
            self.data_cache = {}
            self.cache_timestamps = {}
            self.file_has_nan = {}
            self.file_to_feature_sets = {}
            print("Cleared all in-memory cache data to force complete reload")
        except OSError as e:
            print(f"Warning: Could not completely clear cache: {e}")

        self.loading_in_progress = True
        refresh_thread = threading.Thread(
            target=self._load_all_data, args=(False,), daemon=True
        )
        refresh_thread.start()

    def _update_status(self, message):
        """Helper to update status bar safely from any thread."""
        self.master.after(0, lambda: self.status_var.set(message))

    def _load_all_data(self, is_initial_load=False):
        """Scans folder, loads/reloads files into cache, analyzes, saves cache."""
        if not self.loading_in_progress:
            self.loading_in_progress = True

        print("\n=== STARTING DATA LOAD PROCESS ===")
        self.master.after(0, lambda: self.refresh_button.config(state=tk.DISABLED))
        self._update_status("Scanning for files...")

        # *** DEBUG: Check if PERFECT_FOLDER_PATH exists ***
        print(
            f"DEBUG: Checking existence of PERFECT_FOLDER_PATH: {PERFECT_FOLDER_PATH}"
        )
        if not os.path.isdir(PERFECT_FOLDER_PATH):
            print(
                f"ERROR: Directory not found - {PERFECT_FOLDER_PATH}"
            )  # Print error to console
            self.master.after(
                0,
                lambda: messagebox.showerror(
                    "Error", f"Directory not found:\n{PERFECT_FOLDER_PATH}"
                ),
            )
            self._update_status("Error: Directory not found.")
            self.loading_in_progress = False
            self.master.after(0, lambda: self.refresh_button.config(state=tk.NORMAL))
            return
        else:
            print(f"DEBUG: Directory found: {PERFECT_FOLDER_PATH}")

        try:
            # *** DEBUG: List directory contents ***
            try:
                all_dir_contents = os.listdir(PERFECT_FOLDER_PATH)
                print(
                    f"DEBUG: Contents of PERFECT_FOLDER_PATH ({len(all_dir_contents)} items)"
                )
                if (
                    len(all_dir_contents) < 10
                ):  # Only print all contents if not too many
                    print(all_dir_contents)
            except Exception as list_err:
                print(f"ERROR: Could not list directory contents: {list_err}")
                all_dir_contents = []  # Continue with empty list on error

            current_files = sorted(
                [
                    f
                    for f in all_dir_contents
                    if f.lower().endswith(FILENAME_SUFFIX.lower())
                ]
            )
            # *** DEBUG: Print filtered files ***
            print(
                f"DEBUG: Found {len(current_files)} files ending with '{FILENAME_SUFFIX}'"
            )

            # Removed the limitation to first 5 files
            # Process ALL files instead

            # Always consider it needs an update on initial load
            needs_listbox_update = True

            new_full_paths = [
                os.path.join(PERFECT_FOLDER_PATH, f) for f in current_files
            ]

            # Set the full file paths list now, so the listbox can be updated as soon as possible
            self.full_file_paths = new_full_paths

            # Update the listbox immediately with file names, even before loading data
            # This gives visual feedback to the user that something is happening
            self.master.after(0, self._update_listbox_with_names_only)

            loaded_count, reloaded_count, error_count = 0, 0, 0
            total_files = len(new_full_paths)
            temp_file_has_nan = {}  # Store NaN results temporarily
            self.file_to_feature_sets = {}  # Clear feature sets mapping

            # --- Load/Reload Files ---
            for i, file_path in enumerate(new_full_paths):
                filename = os.path.basename(file_path)
                self._update_status(f"Processing {i+1}/{total_files}: {filename}...")
                try:
                    current_mtime = os.path.getmtime(file_path)
                    cached_mtime = self.cache_timestamps.get(file_path)
                    track_data = None

                    # Load if not cached, or if modified
                    if (
                        file_path not in self.data_cache
                        or cached_mtime != current_mtime
                    ):
                        if file_path in self.data_cache:
                            reloaded_count += 1
                        else:
                            loaded_count += 1
                        print(f"Loading/Reloading: {filename}")

                        try:
                            track_data = joblib.load(file_path)
                            self.data_cache[file_path] = track_data
                            self.cache_timestamps[file_path] = current_mtime

                            # Force NaN check immediately after loading
                            has_nan = self._check_file_for_nan(track_data)
                            temp_file_has_nan[file_path] = has_nan
                            print(f"** NaN check for {filename}: {has_nan}")
                        except Exception as load_err:
                            print(f"Error loading file {filename}: {load_err}")
                            track_data = None
                            temp_file_has_nan[file_path] = False
                    else:
                        # Use existing cached data
                        track_data = self.data_cache.get(file_path)

                        # Always recheck NaN status
                        has_nan = self._check_file_for_nan(track_data)
                        temp_file_has_nan[file_path] = has_nan
                        print(
                            f"** Using cached data for {filename}, NaN check: {has_nan}"
                        )

                    # Build feature sets mapping for this file
                    if track_data:
                        self._build_feature_sets_for_file(file_path, track_data)
                    else:
                        self.file_to_feature_sets[file_path] = set()  # Empty set

                except Exception as e:
                    error_count += 1
                    print(f"Error processing file {filename}: {e}")
                    traceback.print_exc()
                    if file_path in self.data_cache:
                        del self.data_cache[file_path]
                    if file_path in self.cache_timestamps:
                        del self.cache_timestamps[file_path]
                    temp_file_has_nan[file_path] = False  # Assume no NaN on error
                    self.file_to_feature_sets[file_path] = set()  # Empty set

            # --- Clean up cache for deleted files ---
            current_file_set = set(new_full_paths)
            deleted_files = set(self.cache_timestamps.keys()) - current_file_set
            for deleted_path in deleted_files:
                print(
                    f"Removing deleted file from cache: {os.path.basename(deleted_path)}"
                )
                if deleted_path in self.data_cache:
                    del self.data_cache[deleted_path]
                if deleted_path in self.cache_timestamps:
                    del self.cache_timestamps[deleted_path]
                if deleted_path in self.file_has_nan:
                    del self.file_has_nan[deleted_path]  # Also remove from NaN cache
                if deleted_path in self.file_to_feature_sets:
                    del self.file_to_feature_sets[
                        deleted_path
                    ]  # Remove from feature sets mapping

            # Update the main NaN cache
            print("File NaN status summary:")
            for file_path, has_nan in temp_file_has_nan.items():
                print(f"  {os.path.basename(file_path)}: {has_nan}")

            self.file_has_nan = temp_file_has_nan

            # --- Analyze Feature Sets ---
            self._analyze_and_display_feature_sets()  # Analyze full cache

            # --- Save Updated Cache to Disk ---
            self._save_persistent_cache()

            # --- Update Listbox if needed (from GUI thread) ---
            print(f"DEBUG: Scheduling final listbox update.")
            self.master.after(0, self._update_listbox)  # Force a final update

            # --- Final Status Update ---
            status_msg = (
                f"Load/Refresh complete. {loaded_count} new, {reloaded_count} reloaded."
            )
            if deleted_files:
                status_msg += f" {len(deleted_files)} removed."
            if error_count > 0:
                status_msg += f" {error_count} errors."
            self._update_status(status_msg)

        except Exception as e:
            self.master.after(
                0,
                lambda: messagebox.showerror(
                    "Error During Load/Refresh", f"An unexpected error occurred:\n{e}"
                ),
            )
            self._update_status("Error during load/refresh.")
            print(traceback.format_exc())
        finally:
            self.loading_in_progress = False
            self.master.after(0, lambda: self.refresh_button.config(state=tk.NORMAL))

    def _update_listbox_with_names_only(self):
        """Updates the listbox with just filenames, before data is loaded."""
        self.file_listbox.delete(0, tk.END)
        if not self.full_file_paths:
            self.file_listbox.insert(tk.END, "(No files found)")
            return

        for file_path in self.full_file_paths:
            filename = os.path.basename(file_path)
            display_name = filename
            if display_name.lower().endswith(FILENAME_SUFFIX.lower()):
                display_name = display_name[: -len(FILENAME_SUFFIX)]
            self.file_listbox.insert(tk.END, display_name)

        self._update_status(f"Found {len(self.full_file_paths)} files. Loading data...")

    def _save_persistent_cache(self):
        """Saves the current in-memory cache to the persistent file."""
        try:
            print(
                f"Saving persistent cache to: {CACHE_FILE_PATH} ({len(self.data_cache)} items)"
            )
            start_time = time.time()
            cache_to_save = {
                "data": self.data_cache,
                "timestamps": self.cache_timestamps,
            }
            joblib.dump(cache_to_save, CACHE_FILE_PATH, compress=3)
            save_time = time.time() - start_time
            print(f"Cache saved successfully in {save_time:.2f}s.")
            return True
        except Exception as e:
            print(f"Error saving persistent cache: {e}")
            traceback.print_exc()
            return False

    # --- Analysis Helpers ---

    def _analyze_cached_data(self):
        """Performs NaN check and feature set analysis on currently cached data."""
        print("Analyzing cached data...")

        # Reset tracking variables
        temp_file_has_nan = {}
        self.file_to_feature_sets = {}  # Clear and rebuild feature sets mapping

        # Analyze each file
        for file_path, track_data in self.data_cache.items():
            filename = os.path.basename(file_path)

            # Check for NaN values
            has_nan = self._check_file_for_nan(track_data)
            temp_file_has_nan[file_path] = has_nan
            print(f"NaN check for {filename}: {has_nan}")

            # Build feature sets mapping for this file
            self._build_feature_sets_for_file(file_path, track_data)

        # Update the NaN tracking dictionary
        self.file_has_nan = temp_file_has_nan

        # Analyze and display feature sets
        self._analyze_and_display_feature_sets()

    def _build_feature_sets_for_file(self, file_path, track_data):
        """Identifies all unique feature sets in a file and stores them in mapping."""
        if not isinstance(track_data, dict):
            self.file_to_feature_sets[file_path] = set()
            return

        features_list = track_data.get("section_features", [])
        if not isinstance(features_list, list):
            self.file_to_feature_sets[file_path] = set()
            return

        feature_sets = set()
        for section_dict in features_list:
            if isinstance(section_dict, dict):
                # Extract feature keys (same as in _analyze_and_display_feature_sets)
                keys = tuple(
                    sorted(
                        [
                            k
                            for k in section_dict.keys()
                            if k
                            not in [
                                "index",
                                "start_time",
                                "end_time",
                                "duration_sec",
                                "duration_bars",
                                "original_label",
                                "cluster_id",
                            ]
                        ]
                    )
                )
                if keys:  # Only add non-empty sets
                    feature_sets.add(keys)

        self.file_to_feature_sets[file_path] = feature_sets

    def _check_file_for_nan(self, track_data):
        """Checks if any section feature in the track_data contains NaN."""
        print("Checking for NaN values in data...")

        # Force a check of the actual data displayed in the UI
        # This is a direct check rather than going through section_features
        # Look at what the user sees, not just the data structure
        try:
            if isinstance(track_data, dict) and "section_features" in track_data:
                for idx, section in enumerate(track_data.get("section_features", [])):
                    if not isinstance(section, dict):
                        continue

                    # Check each key-value pair in the section
                    for key, value in section.items():
                        # Convert to string for visual inspection
                        str_value = str(value)

                        # Explicit check for "NaN" string representation
                        # This is what appears in the UI and is visible to the user
                        if "NaN" in str_value or "nan" in str_value:
                            print(
                                f"FOUND NaN in section {idx}, key '{key}': {str_value}"
                            )
                            return True

                # Additional safety check - convert entire data to string and search
                data_str = str(track_data)
                if "NaN" in data_str or "nan" in data_str:
                    print(f"FOUND NaN string in full data structure")
                    return True

            return False  # No NaN found

        except Exception as e:
            print(f"Error checking for NaN: {e}")
            traceback.print_exc()
            return False  # Default to False on error

    def _analyze_and_display_feature_sets(self):
        """Analyzes cached data for unique feature sets and updates display."""
        print("Analyzing unique feature sets...")
        unique_sets = set()
        feature_counts = defaultdict(int)  # Count occurrences of each set

        for file_path, track_data in self.data_cache.items():
            if not isinstance(track_data, dict):
                continue
            features_list = track_data.get("section_features", [])
            if not isinstance(features_list, list):
                continue

            file_sets = set()  # Sets found in this specific file
            for section_dict in features_list:
                if isinstance(section_dict, dict):
                    # Get keys excluding basic info, sort them into a tuple
                    keys = tuple(
                        sorted(
                            [
                                k
                                for k in section_dict.keys()
                                if k
                                not in [
                                    "index",
                                    "start_time",
                                    "end_time",
                                    "duration_sec",
                                    "duration_bars",
                                    "original_label",
                                    "cluster_id",
                                ]
                            ]
                        )
                    )
                    if keys:  # Only add non-empty sets
                        unique_sets.add(keys)
                        file_sets.add(keys)

            # Count each unique set found in this file once
            for f_set in file_sets:
                feature_counts[f_set] += 1

        self.unique_feature_sets = sorted(
            list(unique_sets), key=len
        )  # Sort by number of features

        # --- Update Display (from GUI thread) ---
        self.master.after(0, self._update_feature_set_display, feature_counts)

    def _update_feature_set_display(self, feature_counts):
        """Updates the feature set listbox."""
        self.feature_set_listbox.delete(0, tk.END)
        if not self.unique_feature_sets:
            self.feature_set_listbox.insert(tk.END, "(No feature sets found)")
            return

        print(f"Found {len(self.unique_feature_sets)} unique feature sets.")
        for feature_set in self.unique_feature_sets:
            count = feature_counts.get(feature_set, 0)
            # Format the display string
            display_str = (
                f"({count} files) {len(feature_set)} features: {', '.join(feature_set)}"
            )
            self.feature_set_listbox.insert(tk.END, display_str)

    # --- Filtering Logic ---

    def _apply_filters(self):
        """Triggers the listbox update based on current filter settings."""
        print("Applying filters...")
        if self.loading_in_progress:
            print(" -> Load/Refresh in progress, deferring filter application.")
            return
        self._update_listbox()  # Update listbox using current filters

    def _update_listbox(self):
        """Updates the listbox content, applying current filters."""
        print("Updating listbox display with filters...")
        # Remember selection before clearing
        selected_indices = self.file_listbox.curselection()
        current_selection_idx = selected_indices[0] if selected_indices else -1
        # Get the path corresponding to the index *before* filtering
        # Use the current self.full_file_paths which reflects the *currently displayed* items
        selected_path = (
            self.full_file_paths[current_selection_idx]
            if current_selection_idx != -1
            and current_selection_idx < len(self.full_file_paths)
            else None
        )

        # Start with the full list of paths from the cache
        all_cached_paths = sorted(list(self.cache_timestamps.keys()))
        filtered_paths = []

        # --- Apply Filters ---
        show_nan = self.show_nan_only_var.get()
        print(f"NaN filter active: {show_nan}")

        for file_path in all_cached_paths:  # Iterate through ALL cached paths
            passes_filters = True

            # Debug the file
            filename = os.path.basename(file_path)
            has_nan = self.file_has_nan.get(file_path, False)
            print(f"Checking file: {filename}, has_nan={has_nan}")

            # Apply NaN Filter - show ONLY files WITH NaN values when checked
            if show_nan and not has_nan:
                print(f"  Filtering out {filename} because it has no NaN values")
                passes_filters = False

            # Apply Feature Set Filter
            if self.selected_feature_set and not self._check_file_matches_feature_set(
                file_path, self.selected_feature_set
            ):
                print(
                    f"  Filtering out {filename} because it doesn't match selected feature set"
                )
                passes_filters = False

            if passes_filters:
                print(f"  Including {filename} in filtered list")
                filtered_paths.append(file_path)
            else:
                print(f"  Excluding {filename} from filtered list")

        # --- Populate Listbox with Filtered Items ---
        self.file_listbox.delete(0, tk.END)  # Clear display
        self.full_file_paths = filtered_paths  # Update internal list to filtered list
        new_selection_index = -1

        if not filtered_paths:
            self.file_listbox.insert(tk.END, "(No files match filters)")
            self.clear_feature_display()
        else:
            for idx, file_path in enumerate(filtered_paths):
                filename = os.path.basename(file_path)
                display_name = filename
                if display_name.lower().endswith(FILENAME_SUFFIX.lower()):
                    display_name = display_name[: -len(FILENAME_SUFFIX)]
                self.file_listbox.insert(tk.END, display_name)
                # Check if this path matches the previously selected one
                if file_path == selected_path:
                    new_selection_index = idx

            # Try to restore selection if it's still in the filtered list
            if new_selection_index != -1:
                self.file_listbox.selection_set(new_selection_index)
                self.file_listbox.activate(new_selection_index)
                self.file_listbox.see(new_selection_index)
                # Ensure the selection event triggers
                self.master.after(100, lambda: self.on_file_select())
            elif self.file_listbox.size() > 0:
                self.clear_feature_display()  # Clear display if selection lost
            else:
                self.clear_feature_display()

        self._update_status(f"Displaying {len(filtered_paths)} files matching filters.")

    # --- Display Logic ---

    def on_file_select(self, event=None):
        """Handles the selection change event in the file listbox."""
        if self.loading_in_progress:
            print("Loading in progress, ignoring selection change")
            return

        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            print("No selection in file listbox")
            return

        selected_index = selected_indices[0]
        # Use the filtered self.full_file_paths list
        if selected_index < 0 or selected_index >= len(self.full_file_paths):
            print(
                f"Error: Selected index {selected_index} out of bounds for filtered file paths list ({len(self.full_file_paths)} items)."
            )
            return

        file_path = self.full_file_paths[selected_index]
        display_name = self.file_listbox.get(selected_index)

        self._update_status(f"Displaying data for {display_name}...")
        self.master.update_idletasks()

        # Detailed debug of cache access
        print(f"Checking cache for {file_path}")
        if file_path in self.data_cache:
            track_data = self.data_cache[file_path]
            print(f"Found in cache, data type: {type(track_data).__name__}")
            print(
                f"Track data keys: {list(track_data.keys()) if isinstance(track_data, dict) else 'Not a dictionary'}"
            )

            # Verify that section_features exist and are valid
            if isinstance(track_data, dict) and "section_features" in track_data:
                features = track_data["section_features"]
                print(
                    f"Section features found: {len(features) if isinstance(features, list) else 'Not a list'}"
                )
                # Print the first section if available for debugging
                if isinstance(features, list) and len(features) > 0:
                    print("First section sample:")
                    first_section = features[0]
                    if isinstance(first_section, dict):
                        for k, v in first_section.items():
                            print(f"  {k}: {v}")
                    else:
                        print(
                            f"  First section is not a dict but: {type(first_section).__name__}"
                        )
            else:
                print("Section features missing or invalid")

            self.display_track_features(track_data)
            self._update_status(f"Displayed features for {display_name}")
        else:
            print(f"Cache miss for {file_path}")
            self.master.after(
                0,
                lambda: messagebox.showwarning(
                    "Cache Miss",
                    f"Data for {display_name} not found in cache. Please Refresh Data.",
                ),
            )
            self.clear_feature_display()
            self._update_status(f"Data not cached for {display_name}")

    def clear_feature_display(self):
        """Clears the feature display Treeview."""
        # Clear the treeview
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        # Reset column configuration
        self.feature_tree["columns"] = ()

        # Clear data storage
        self.current_data = []

        # Reset column tracking
        self.all_columns = []
        self.column_order = []

        # Clear column control checkboxes
        for widget in self.column_checkbox_frame.winfo_children():
            widget.destroy()

        print("Feature display cleared")

    def display_track_features(self, track_data):
        """Populates the Treeview with features from the loaded track_data."""
        self.clear_feature_display()

        # Find the file path for this track data - use id() for reliable identity comparison
        file_path = None
        track_data_id = id(track_data)
        for path, data in self.data_cache.items():
            if id(data) == track_data_id:
                file_path = path
                break

        filename = os.path.basename(file_path) if file_path else "Unknown"
        print(f"Displaying track features for: {filename}")

        # Enhanced error checking and logging
        if not isinstance(track_data, dict):
            print(
                f"ERROR: Loaded data is not a dictionary but {type(track_data).__name__}"
            )
            self._update_status("Error: Invalid track data format")
            return

        # Check for section_features
        if "section_features" not in track_data:
            print("ERROR: 'section_features' key missing from track data")
            self._update_status("Error: No section features found in track data")
            keys_found = ", ".join(track_data.keys())
            print(f"Available keys: {keys_found}")
            return

        section_features_list = track_data.get("section_features")
        semantic_labels = track_data.get("semantic_labels", [])

        if not isinstance(section_features_list, list):
            print(
                f"ERROR: 'section_features' is not a list but {type(section_features_list).__name__}"
            )
            self._update_status("Error: Invalid section features format")
            return

        if not section_features_list:
            print("INFO: 'section_features' list is empty")
            self._update_status("No section features found in track")
            return

        num_sections = len(section_features_list)
        print(f"Displaying features for {num_sections} sections")

        # Collect all feature keys
        all_feature_keys = set()
        for section_dict in section_features_list:
            if isinstance(section_dict, dict):
                all_feature_keys.update(section_dict.keys())
            else:
                print(
                    f"WARNING: Section item is not a dict but {type(section_dict).__name__}"
                )

        # Set up columns
        standard_cols = ["#", "Label", "Start (s)", "End (s)", "Dur (s)"]
        dynamic_feature_keys = sorted(
            [
                k
                for k in all_feature_keys
                if k
                not in [
                    "index",
                    "start_time",
                    "end_time",
                    "duration_sec",
                    "duration_bars",
                    "original_label",
                    "cluster_id",
                ]
            ]
        )

        print(
            f"Dynamic feature keys ({len(dynamic_feature_keys)}): {dynamic_feature_keys}"
        )

        # Store all possible columns in their default order
        self.all_columns = standard_cols + dynamic_feature_keys

        # Initialize column order if empty
        if not self.column_order:
            self.column_order = self.all_columns.copy()

        # Set up column visibility controls
        self._setup_column_checkboxes(self.all_columns)

        # Process and display the data
        row_count = 0
        has_nan = False
        self.current_data = []  # Clear current data

        for i, section_dict in enumerate(section_features_list):
            if not isinstance(section_dict, dict):
                print(f"Skipping non-dict section at index {i}")
                continue

            try:
                # Create a full set of values for all possible columns
                values_dict = {
                    "#": i + 1,
                    "Label": semantic_labels[i] if i < len(semantic_labels) else "N/A",
                    "Start (s)": f"{section_dict.get('start_time', np.nan):.3f}",
                    "End (s)": f"{section_dict.get('end_time', np.nan):.3f}",
                    "Dur (s)": f"{section_dict.get('duration_sec', np.nan):.3f}",
                }

                # Add dynamic feature values
                section_has_nan = False
                for key in dynamic_feature_keys:
                    val = section_dict.get(key, np.nan)
                    if isinstance(val, (int, float, np.number)):
                        formatted_val = f"{val:.4f}" if np.isfinite(val) else "NaN"
                        if formatted_val == "NaN":
                            section_has_nan = True
                            has_nan = True
                            print(f"Found NaN value for key '{key}' in section {i}")
                    else:
                        formatted_val = str(val) if val is not None else "None"
                        if "NaN" in formatted_val or "nan" in formatted_val:
                            section_has_nan = True
                            has_nan = True
                            print(
                                f"Found NaN string for key '{key}' in section {i}: {formatted_val}"
                            )

                    values_dict[key] = formatted_val

                # Store the data for this row
                self.current_data.append((i, values_dict))

                # If we have visible columns already configured, display the row
                if self._get_visible_columns_in_order():
                    row_count += 1
            except Exception as e:
                print(f"Error processing section {i}: {e}")
                traceback.print_exc()

        # Display the data in the treeview
        self._refresh_treeview()

        # Update NaN status for the file
        if file_path:
            if has_nan:
                print(f"UPDATING NaN STATUS: File {filename} HAS NaN values!")
                self.file_has_nan[file_path] = True
            else:
                print(f"UPDATING NaN STATUS: File {filename} has NO NaN values")
                self.file_has_nan[file_path] = False

        self._update_status(
            f"Displayed {len(self.current_data)} sections with {len(self._get_visible_columns_in_order())} visible columns"
        )


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureInspectorApp(root)
    root.mainloop()
