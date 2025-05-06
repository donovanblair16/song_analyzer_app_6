# =============================================================================
# FILE: inspect_data.py
# PURPOSE: Provides a GUI to load and inspect section features from
#          analysis files stored in the 'completed_analyses/Perfect' folder.
#          Includes file list, feature display, caching, filtering, search,
#          multi-select, CSV export, and column sorting.
# USAGE: Run this script from the main project directory
#        (e.g., song_analyzer_app_6) or its subdirectory.
# MODIFIED:
# - Added Section # column to Feature Search results.
# - Implemented LEAN caching.
# - Renamed cache file.
# - Added Default column visibility and order configuration.
# - FIXED: Ensure column visibility checkboxes are created immediately.
# - ADDED: Search functionality (feature value & label transition).
# - ADDED: Calculation of global feature statistics.
# - ADDED: Dedicated Treeview for feature search results.
# - ADDED: Dedicated ScrolledText for transition search results.
# - ADDED: Label filter for feature search.
# - ADDED: GUI elements for transition search.
# - ADDED: Multi-select for file listbox.
# - ADDED: "Display Selected Files" button.
# - ADDED: "Export Displayed Data" button and functionality.
# - ADDED: "Song Name" and "Song ID" columns to main feature view.
# - ADDED: Right-click column header sorting for main feature view.
# - (Previous fixes and features retained)
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd # Keep for potential future use, but using csv for export now
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font as tkFont, filedialog # Added filedialog
import traceback
import time
import threading
from collections import defaultdict
import math # For isnan/isinf
import csv # Added for CSV export

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_BASE_FOLDER = os.path.dirname(SCRIPT_DIR)
ANALYSIS_BASE_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "completed_analyses")
PERFECT_SUBFOLDER = "Perfect"
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, PERFECT_SUBFOLDER)
FILENAME_SUFFIX = ".analysis.joblib"

# --- Lean Cache File ---
CACHE_FILENAME = "feature_inspector_lean_cache.joblib"
CACHE_FILE_PATH = os.path.join(SCRIPT_DIR, CACHE_FILENAME)

# --- Essential Keys for Lean Cache ---
ESSENTIAL_KEYS_FOR_CACHE = ["section_features", "semantic_labels"]

# --- Default Column Visibility and Order ---
# <<< MODIFIED: Added new default columns >>>
DEFAULT_VISIBLE_COLUMNS = [
    "Song Name", "Song ID", "#", "Label", "Start (s)", "End (s)", "Dur (s)",
    "relative_rms", "delta_rms", "label_proportion", "low_energy_norm",
    "position_context", "relative_position",
]

# --- Search Query Types ---
QUERY_TYPES = [
    "Greater Than (>)",
    "Less Than (<)",
    "Between (A <= val <= B)",
    "Outside Range (val < A or val > B)",
    "Std Dev Above Mean (> mean + N*std)",
    "Std Dev Below Mean (< mean - N*std)",
]

# --- DEBUG: Print calculated paths ---
print(f"--- DEBUG PATHS ---")
print(f"Script Dir: {SCRIPT_DIR}")
print(f"Project Base: {PROJECT_BASE_FOLDER}")
print(f"Analysis Base: {ANALYSIS_BASE_FOLDER}")
print(f"Perfect Folder: {PERFECT_FOLDER_PATH}")
print(f"Cache Path: {CACHE_FILE_PATH}")
print(f"--- END DEBUG PATHS ---")


# --- Main Application Class ---
class FeatureInspectorApp:
    """
    Tkinter GUI application for inspecting section features in analysis files
    with persistent LEAN caching, filtering, and search functionality.
    """
    def __init__(self, master):
        """Initialize the application."""
        self.master = master
        master.title("Perfect Analysis Feature Inspector (Lean Cached + Search + Export + Sort)")
        master.geometry("1500x800") # Keep size, might need adjustment

        # Data Caching & State
        self.data_cache = {}
        self.cache_timestamps = {}
        self.full_file_paths = [] # List of full paths currently loaded/filtered
        self.file_has_nan = {}
        self.unique_feature_sets = []
        self.loading_in_progress = False
        self.selected_feature_set = None
        self.file_to_feature_sets = {}
        self.feature_stats = {}
        self.unique_labels = []
        self.song_name_to_id = {} # For Song ID column
        self.sorted_file_list_for_id = [] # To maintain consistent ID mapping

        # Column visibility and order tracking
        self.all_columns = []
        self.column_visibility = {}
        self.column_order = DEFAULT_VISIBLE_COLUMNS[:] # Use copy
        self.current_data = [] # Holds data currently displayed in feature_tree (tuples)
        self.dragging = False
        self.drag_column = None

        # Sorting state
        self.sort_column = None
        self.sort_reverse = False

        # --- Tkinter Variables ---
        self.show_nan_only_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Initializing...")
        # Feature Search Vars
        self.search_label_filter_var = tk.StringVar(value="All Sections")
        self.search_feature_var = tk.StringVar()
        self.search_query_type_var = tk.StringVar(value=QUERY_TYPES[0])
        self.search_value_a_var = tk.StringVar()
        self.search_value_b_var = tk.StringVar()
        # Transition Search Vars (NEW)
        self.search_from_label_var = tk.StringVar()
        self.search_to_label_var = tk.StringVar()


        # --- Create Custom Font ---
        default_font = tkFont.nametofont("TkDefaultFont")
        listbox_font_size = default_font.actual()["size"] # Keep listbox standard size now
        self.listbox_font = tkFont.Font(size=listbox_font_size)
        self.small_font = tkFont.Font(size=default_font.actual()["size"] - 1)

        # --- Create Main UI Structure ---
        self.main_paned_window = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Pane: File List, Filters, Search Controls ---
        self.left_pane_outer = ttk.Frame(self.main_paned_window, width=350) # Slightly wider
        self.main_paned_window.add(self.left_pane_outer, weight=1)

        # File List Area
        file_list_frame = ttk.LabelFrame(self.left_pane_outer, text="Analysis Files")
        file_list_frame.pack(pady=(0, 5), padx=5, fill=tk.BOTH, expand=True)
        list_frame = ttk.Frame(file_list_frame); list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.file_list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        # <<< MODIFIED: Enable extended selection >>>
        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=self.file_list_scrollbar.set,
            exportselection=False,
            font=self.listbox_font,
            selectmode=tk.EXTENDED # Allow multi-select
        )
        self.file_list_scrollbar.config(command=self.file_listbox.yview); self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True);
        # <<< MODIFIED: Bind selection change to a different handler >>>
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_selection_change)

        # <<< ADDED: Buttons for file list actions >>>
        file_button_frame = ttk.Frame(file_list_frame)
        file_button_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        self.display_selected_button = ttk.Button(file_button_frame, text="Display Selected Files", command=self.display_selected_files)
        self.display_selected_button.pack(side=tk.LEFT, padx=(0,5), expand=True, fill=tk.X)
        self.display_all_button = ttk.Button(file_button_frame, text="Display All Files", command=self.display_all_files)
        self.display_all_button.pack(side=tk.LEFT, padx=(0,0), expand=True, fill=tk.X)


        # Filter & Control Area
        control_frame = ttk.Frame(self.left_pane_outer)
        control_frame.pack(fill=tk.X, pady=5, padx=5, expand=False)
        self.nan_filter_check = ttk.Checkbutton(control_frame, text="Show Files with NaN Features Only", variable=self.show_nan_only_var, command=self._apply_filters)
        self.nan_filter_check.pack(anchor=tk.W)
        self.refresh_button = ttk.Button(control_frame, text="Refresh Data", command=self._refresh_data)
        self.refresh_button.pack(pady=5, fill=tk.X)
        self.reset_button = ttk.Button(control_frame, text="Reset Filters", command=self._reset_filters)
        self.reset_button.pack(pady=5, fill=tk.X)

        # Feature Set Display Area
        feature_set_frame = ttk.LabelFrame(self.left_pane_outer, text="Unique Feature Sets (Click to Filter)")
        feature_set_frame.pack(pady=(5, 5), padx=5, fill=tk.X, expand=False)
        self.feature_set_listbox = tk.Listbox(feature_set_frame, height=5, font=self.small_font)
        feature_set_scrollbar = ttk.Scrollbar(feature_set_frame, orient=tk.VERTICAL, command=self.feature_set_listbox.yview)
        self.feature_set_listbox.config(yscrollcommand=feature_set_scrollbar.set); feature_set_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_set_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        self.feature_set_listbox.bind("<<ListboxSelect>>", self.on_feature_set_select)

        # --- Feature Search Control Area ---
        self.search_frame = ttk.LabelFrame(self.left_pane_outer, text="Search Feature Values")
        self.search_frame.pack(pady=(5, 5), padx=5, fill=tk.X, expand=False)
        # Label Filter Selection
        ttk.Label(self.search_frame, text="Label Filter:").grid(row=0, column=0, padx=(5,2), pady=5, sticky=tk.W)
        self.search_label_combo = ttk.Combobox(self.search_frame, textvariable=self.search_label_filter_var, state='disabled', width=25)
        self.search_label_combo['values'] = ["All Sections"]; self.search_label_combo.current(0)
        self.search_label_combo.grid(row=0, column=1, columnspan=3, padx=(0,5), pady=5, sticky=tk.EW)
        # Feature Selection
        ttk.Label(self.search_frame, text="Feature:").grid(row=1, column=0, padx=(5,2), pady=5, sticky=tk.W)
        self.search_feature_combo = ttk.Combobox(self.search_frame, textvariable=self.search_feature_var, state='disabled', width=25)
        self.search_feature_combo.grid(row=1, column=1, columnspan=3, padx=(0,5), pady=5, sticky=tk.EW)
        # Query Type Selection
        ttk.Label(self.search_frame, text="Condition:").grid(row=2, column=0, padx=(5,2), pady=5, sticky=tk.W)
        self.search_query_type_combo = ttk.Combobox(self.search_frame, textvariable=self.search_query_type_var, values=QUERY_TYPES, state='readonly', width=25)
        self.search_query_type_combo.grid(row=2, column=1, columnspan=3, padx=(0,5), pady=5, sticky=tk.EW)
        self.search_query_type_combo.bind("<<ComboboxSelected>>", self._update_search_value_fields)
        # Value A Entry
        self.search_value_a_label = ttk.Label(self.search_frame, text="Value A:")
        self.search_value_a_label.grid(row=3, column=0, padx=(5,2), pady=5, sticky=tk.W)
        self.search_value_a_entry = ttk.Entry(self.search_frame, textvariable=self.search_value_a_var, width=10)
        self.search_value_a_entry.grid(row=3, column=1, padx=(0,5), pady=5, sticky=tk.W)
        # Value B Entry
        self.search_value_b_label = ttk.Label(self.search_frame, text="Value B:")
        self.search_value_b_label.grid(row=3, column=2, padx=(5,2), pady=5, sticky=tk.W)
        self.search_value_b_entry = ttk.Entry(self.search_frame, textvariable=self.search_value_b_var, width=10)
        self.search_value_b_entry.grid(row=3, column=3, padx=(0,5), pady=5, sticky=tk.W)
        # Search Button
        self.search_button = ttk.Button(self.search_frame, text="Search Values", command=self._perform_search)
        self.search_button.grid(row=4, column=0, columnspan=4, pady=(10, 5))
        self._update_search_value_fields() # Initial setup

        # --- Transition Search Control Area (NEW) ---
        self.transition_search_frame = ttk.LabelFrame(self.left_pane_outer, text="Search Label Transitions")
        self.transition_search_frame.pack(pady=(5, 0), padx=5, fill=tk.X, expand=False)
        # From Label
        ttk.Label(self.transition_search_frame, text="From:").grid(row=0, column=0, padx=(5,2), pady=5, sticky=tk.W)
        self.search_from_label_combo = ttk.Combobox(self.transition_search_frame, textvariable=self.search_from_label_var, state='disabled', width=12)
        self.search_from_label_combo.grid(row=0, column=1, padx=(0,5), pady=5, sticky=tk.W)
        # To Label
        ttk.Label(self.transition_search_frame, text="To:").grid(row=0, column=2, padx=(5,2), pady=5, sticky=tk.W)
        self.search_to_label_combo = ttk.Combobox(self.transition_search_frame, textvariable=self.search_to_label_var, state='disabled', width=12)
        self.search_to_label_combo.grid(row=0, column=3, padx=(0,5), pady=5, sticky=tk.W)
        # Search Transition Button
        self.search_transition_button = ttk.Button(self.transition_search_frame, text="Search Transition", command=self._perform_transition_search)
        self.search_transition_button.grid(row=1, column=0, columnspan=4, pady=(5, 5))


        # --- Middle Pane: Column Visibility Controls ---
        self.column_control_frame = ttk.Frame(self.main_paned_window, width=200)
        self.main_paned_window.add(self.column_control_frame, weight=1)
        # (Column control setup remains the same as before)
        column_control_label_frame = ttk.LabelFrame(self.column_control_frame, text="Column Visibility")
        column_control_label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        column_canvas_frame = ttk.Frame(column_control_label_frame); column_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.column_scroll = ttk.Scrollbar(column_canvas_frame, orient=tk.VERTICAL)
        self.column_canvas = tk.Canvas(column_canvas_frame, yscrollcommand=self.column_scroll.set, highlightthickness=0)
        self.column_scroll.config(command=self.column_canvas.yview); self.column_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.column_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.column_checkbox_frame = ttk.Frame(self.column_canvas) # Holds the checkboxes
        self.column_canvas_window = self.column_canvas.create_window((0, 0), window=self.column_checkbox_frame, anchor=tk.NW, tags="self.column_checkbox_frame")
        column_button_frame = ttk.Frame(column_control_label_frame); column_button_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(column_button_frame, text="Select All", command=self._select_all_columns).pack(side=tk.LEFT, padx=2)
        ttk.Button(column_button_frame, text="Deselect All", command=self._deselect_all_columns).pack(side=tk.LEFT, padx=2)
        ttk.Button(column_button_frame, text="Reset Order", command=self._reset_column_order).pack(side=tk.LEFT, padx=2)
        ttk.Label(column_control_label_frame, text="Drag column headers to reorder", font=self.small_font, wraplength=180).pack(pady=(0, 5))
        self.column_checkbox_frame.bind("<Configure>", self._on_checkbox_frame_configure)
        self.column_canvas.bind("<Configure>", self._on_column_canvas_configure)


        # --- Right Pane: Feature Display Treeview ---
        self.right_frame = ttk.Frame(self.main_paned_window, width=550)
        self.main_paned_window.add(self.right_frame, weight=3)
        # <<< ADDED: Export Button >>>
        export_button_frame = ttk.Frame(self.right_frame)
        export_button_frame.pack(fill=tk.X, pady=(0,5))
        ttk.Label(export_button_frame, text="Section Features:").pack(side=tk.LEFT, anchor=tk.W)
        self.export_button = ttk.Button(export_button_frame, text="Export Displayed Data", command=self._export_to_csv)
        self.export_button.pack(side=tk.RIGHT, padx=(0,5))
        # <<< END ADDED >>>
        tree_frame = ttk.Frame(self.right_frame); tree_frame.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical"); hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.feature_tree = ttk.Treeview(tree_frame, columns=(), show="headings", yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=self.feature_tree.yview); hsb.config(command=self.feature_tree.xview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y); hsb.pack(side=tk.BOTTOM, fill=tk.X); self.feature_tree.pack(fill=tk.BOTH, expand=True)
        self.feature_tree.bind("<ButtonPress-1>", self._start_column_drag); self.feature_tree.bind("<B1-Motion>", self._column_drag_motion); self.feature_tree.bind("<ButtonRelease-1>", self._end_column_drag)
        # <<< ADDED: Binding for column sorting >>>
        self.feature_tree.bind("<Button-3>", self._on_column_header_right_click) # Mac: Ctrl-Click or Right-Click


        # --- Far Right Pane: Search Results ---
        self.search_results_outer_frame = ttk.Frame(self.main_paned_window, width=400) # Added new pane
        self.main_paned_window.add(self.search_results_outer_frame, weight=2)

        # Feature Search Results Treeview
        feature_results_frame = ttk.LabelFrame(self.search_results_outer_frame, text="Feature Search Results")
        feature_results_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5)) # Expand vertically
        results_tree_frame = ttk.Frame(feature_results_frame); results_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_vsb = ttk.Scrollbar(results_tree_frame, orient="vertical"); results_hsb = ttk.Scrollbar(results_tree_frame, orient="horizontal")
        search_cols = ("File", "Section #", "Section Label", "Feature", "Value")
        self.search_results_tree = ttk.Treeview(results_tree_frame, columns=search_cols, show="headings", yscrollcommand=results_vsb.set, xscrollcommand=results_hsb.set)
        results_vsb.config(command=self.search_results_tree.yview); results_hsb.config(command=self.search_results_tree.xview)
        self.search_results_tree.heading("File", text="File"); self.search_results_tree.column("File", width=150, stretch=tk.YES)
        self.search_results_tree.heading("Section #", text="Sec #"); self.search_results_tree.column("Section #", width=50, stretch=tk.NO, anchor=tk.CENTER)
        self.search_results_tree.heading("Section Label", text="Label"); self.search_results_tree.column("Section Label", width=80, stretch=tk.NO)
        self.search_results_tree.heading("Feature", text="Feature"); self.search_results_tree.column("Feature", width=100, stretch=tk.YES)
        self.search_results_tree.heading("Value", text="Value"); self.search_results_tree.column("Value", width=80, stretch=tk.NO, anchor=tk.E)
        results_vsb.pack(side=tk.RIGHT, fill=tk.Y); results_hsb.pack(side=tk.BOTTOM, fill=tk.X); self.search_results_tree.pack(fill=tk.BOTH, expand=True)

        # Transition Search Results Text Area (NEW)
        transition_results_frame = ttk.LabelFrame(self.search_results_outer_frame, text="Transition Search Results")
        transition_results_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0)) # Expand vertically
        self.transition_results_text = scrolledtext.ScrolledText(transition_results_frame, wrap=tk.WORD, height=8, state=tk.DISABLED) # Start disabled
        self.transition_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        # --- Status Bar ---
        status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Load Cache or Start Initial Load ---
        self.master.after(100, self._load_cache_or_start_load)

    # --- Canvas Config Handlers (Unchanged) ---
    def _on_checkbox_frame_configure(self, event): self.column_canvas.configure(scrollregion=self.column_canvas.bbox("all"))
    def _on_column_canvas_configure(self, event): self.column_canvas.itemconfig(self.column_canvas_window, width=event.width)

    # --- Column Control Functions (Unchanged) ---
    def _setup_column_checkboxes(self, columns):
        for widget in self.column_checkbox_frame.winfo_children(): widget.destroy()
        self.column_visibility = {}
        # print(f"DEBUG: Setting up checkboxes for columns: {columns}") # Less verbose
        # print(f"DEBUG: Default visible columns: {DEFAULT_VISIBLE_COLUMNS}") # Less verbose
        for i, col in enumerate(columns):
            # <<< MODIFIED: Use self.column_order for default visibility >>>
            is_visible_by_default = col in self.column_order
            var = tk.BooleanVar(value=is_visible_by_default)
            checkbox = ttk.Checkbutton(self.column_checkbox_frame, text=col, variable=var, command=lambda c=col: self._toggle_column_visibility(c))
            checkbox.pack(anchor=tk.W, pady=1, padx=5, fill=tk.X)
            self.column_visibility[col] = var

    def _toggle_column_visibility(self, column):
        if column in self.column_visibility:
            # print(f"Toggling column '{column}' to state: {self.column_visibility[column].get()}") # Less verbose
            self._refresh_treeview()
        else: print(f"Warning: Tried to toggle unknown column '{column}'")

    def _select_all_columns(self):
        for col, var in self.column_visibility.items(): var.set(True)
        self._refresh_treeview(); self._update_status("All columns selected")

    def _deselect_all_columns(self):
        # <<< MODIFIED: Keep essential columns visible >>>
        essential_cols = ["Song Name", "Song ID", "#", "Label"]
        for col, var in self.column_visibility.items():
            var.set(col in essential_cols or col in DEFAULT_VISIBLE_COLUMNS)
        self._refresh_treeview(); self._update_status("Deselected non-default columns")

    def _reset_column_order(self):
        self.column_order = DEFAULT_VISIBLE_COLUMNS[:]
        for col in self.all_columns:
            if col not in self.column_order: self.column_order.append(col)
        # Reset visibility to default as well
        for col, var in self.column_visibility.items():
            var.set(col in DEFAULT_VISIBLE_COLUMNS)
        self._refresh_treeview(); self._update_status("Column order and visibility reset to default")

    def _get_visible_columns_in_order(self):
        if not self.column_order: self.column_order = self.all_columns[:]
        visible_columns = []
        # First add columns from column_order that are visible
        for col in self.column_order:
            var = self.column_visibility.get(col)
            if var is not None and var.get():
                visible_columns.append(col)
        # Add any other visible columns that might not be in column_order yet
        for col, var in self.column_visibility.items():
             if var.get() and col not in visible_columns:
                 print(f"Warning: Visible column '{col}' not found in current order. Appending.")
                 visible_columns.append(col)
        return visible_columns

    def _refresh_treeview(self):
        """Refreshes the main feature treeview based on current data, visibility, and order."""
        visible_columns = self._get_visible_columns_in_order()
        print(f"DEBUG: Refreshing treeview with {len(self.current_data)} items and visible columns: {visible_columns}")
        selected_items = self.feature_tree.selection();
        # Store the actual item IDs (which are tuples: (file_path, section_index))
        selected_ids = [self.feature_tree.item(item_id)['tags'] for item_id in selected_items if self.feature_tree.item(item_id)['tags']]
        selected_id_to_restore = selected_ids[0] if selected_ids else None

        # Clear existing items
        for item in self.feature_tree.get_children(): self.feature_tree.delete(item)

        if visible_columns:
            self.feature_tree["columns"] = visible_columns
            for col in visible_columns:
                # Define column properties (adjust widths as needed)
                anchor = tk.W; width = 80; stretch = tk.YES
                if col == "#": width = 40; stretch = tk.NO; anchor = tk.CENTER
                elif col == "Song ID": width = 60; stretch = tk.NO; anchor = tk.CENTER
                elif col == "Song Name": width = 150; stretch = tk.YES
                elif col == "Label": width = 100; stretch = tk.NO
                elif col in ["Start (s)", "End (s)", "Dur (s)"]: width = 70; stretch = tk.NO; anchor = tk.E
                elif "rms" in col.lower() or "energy" in col.lower(): width = 110; stretch = tk.YES; anchor = tk.E
                elif "centroid" in col.lower() or "position" in col.lower(): width = 130; stretch = tk.YES; anchor = tk.E
                elif len(col) > 15: width = 120; stretch = tk.YES

                # <<< MODIFIED: Add command binding for sorting >>>
                self.feature_tree.heading(col, text=col, anchor=anchor,
                                         command=lambda c=col: self._sort_treeview(c))
                self.feature_tree.column(col, width=width, anchor=anchor, stretch=stretch)
        else:
            self.feature_tree["columns"] = ()

        # Repopulate with current data
        if self.current_data and visible_columns:
            for item_data in self.current_data:
                # Ensure item_data has the expected structure (file_path, section_index, values_dict)
                if len(item_data) == 3:
                    file_path, section_index, values_dict = item_data
                    row_values = [values_dict.get(col, "") for col in visible_columns]
                    # Use a unique tuple as the item ID and also store it as a tag for easy retrieval
                    item_id_tuple = (file_path, section_index)
                    self.feature_tree.insert("", tk.END, iid=f"{file_path}_{section_index}", values=tuple(row_values), tags=item_id_tuple)
                else:
                    print(f"Warning: Skipping malformed item in self.current_data: {item_data}")

            # Restore selection if possible
            if selected_id_to_restore:
                try:
                    # Find the item ID string corresponding to the tuple tag
                    item_id_str_to_select = None
                    for item_id_str in self.feature_tree.get_children():
                        if self.feature_tree.item(item_id_str)['tags'] == selected_id_to_restore:
                            item_id_str_to_select = item_id_str
                            break
                    if item_id_str_to_select:
                        self.feature_tree.selection_set(item_id_str_to_select)
                        self.feature_tree.see(item_id_str_to_select)
                except Exception as e:
                    print(f"Warning: Could not restore selection after refresh/sort: {e}")

        elif not visible_columns:
            print("DEBUG: No columns are visible, treeview not populated.")


    # --- Column Drag and Drop Functions (Unchanged) ---
    def _start_column_drag(self, event):
        if self.dragging: return
        region = self.feature_tree.identify_region(event.x, event.y)
        if region != "heading": return
        column_id = self.feature_tree.identify_column(event.x)
        if column_id.startswith("#"):
            col_idx = int(column_id[1:]) - 1; visible_columns = self._get_visible_columns_in_order()
            if 0 <= col_idx < len(visible_columns):
                self.drag_column = visible_columns[col_idx]; self.dragging = True
                self.feature_tree.configure(cursor="exchange"); self.drag_start_x = event.x
    def _column_drag_motion(self, event): pass # Motion logic can be added if visual feedback is needed
    def _end_column_drag(self, event):
        if not self.dragging or not self.drag_column: return
        try:
            self.feature_tree.configure(cursor="")
            target_id = self.feature_tree.identify_column(event.x)
            if target_id.startswith("#"):
                target_idx = int(target_id[1:]) - 1; visible_columns = self._get_visible_columns_in_order()
                if 0 <= target_idx < len(visible_columns):
                    target_column = visible_columns[target_idx]
                    if target_column != self.drag_column: self._reorder_column(self.drag_column, target_column)
        finally: self.dragging = False; self.drag_column = None
    def _reorder_column(self, source_column, target_column):
        if not self.column_order: self.column_order = self.all_columns[:]
        current_order = self.column_order[:]
        if source_column in current_order and target_column in current_order:
            current_order.remove(source_column)
            try:
                target_pos = current_order.index(target_column); current_order.insert(target_pos, source_column)
                self.column_order = current_order; self._refresh_treeview()
                self._update_status(f"Moved column '{source_column}' before '{target_column}'")
            except ValueError: print(f"Error: Target column '{target_column}' not found."); self.column_order.append(source_column); self._refresh_treeview()
        else: print(f"Warning: Source '{source_column}' or Target '{target_column}' not in current column order.")

    # --- Feature Set Filtering Logic (Unchanged) ---
    def on_feature_set_select(self, event=None):
        if self.loading_in_progress: print("Loading in progress, ignoring feature set selection"); return
        selected_indices = self.feature_set_listbox.curselection()
        if self.selected_feature_set is not None and selected_indices:
            selected_index = selected_indices[0]
            if selected_index < len(self.unique_feature_sets) and self.unique_feature_sets[selected_index] == self.selected_feature_set:
                print("Deselecting current feature set"); self.feature_set_listbox.selection_clear(0, tk.END)
                self.selected_feature_set = None; self._update_status("Feature set filter cleared"); self._apply_filters(); return
        if not selected_indices: print("No feature set selected, clearing filter"); self.selected_feature_set = None; self._apply_filters(); return
        selected_index = selected_indices[0]
        if selected_index < 0 or selected_index >= len(self.unique_feature_sets): print(f"Error: Feature set index {selected_index} out of bounds"); return
        self.selected_feature_set = self.unique_feature_sets[selected_index]; print(f"Selected feature set: {self.selected_feature_set}")
        feature_set_str = ", ".join(self.selected_feature_set); self._update_status(f"Filtering by feature set: {feature_set_str}"); self._apply_filters(); return

    def _check_file_matches_feature_set(self, file_path, feature_set):
        if file_path in self.file_to_feature_sets: return feature_set in self.file_to_feature_sets[file_path]
        return False # Should rely on precomputed mapping

    def _reset_filters(self):
        self.show_nan_only_var.set(False); self.selected_feature_set = None
        self.feature_set_listbox.selection_clear(0, tk.END); self._update_status("Filters reset"); self._apply_filters()

    # --- Caching and Loading Logic ---
    def _load_cache_or_start_load(self):
        """Tries to load persistent LEAN cache, otherwise starts background load."""
        try:
            if os.path.exists(CACHE_FILE_PATH) and os.path.getsize(CACHE_FILE_PATH) > 0:
                try:
                    print(f"Loading persistent lean cache from: {CACHE_FILE_PATH}")
                    start_time = time.time(); cached_content = joblib.load(CACHE_FILE_PATH); load_time = time.time() - start_time
                    if isinstance(cached_content, dict) and "data" in cached_content and "timestamps" in cached_content:
                        first_item_key = next(iter(cached_content["data"]), None)
                        if first_item_key:
                            first_item_data = cached_content["data"][first_item_key]
                            if isinstance(first_item_data, dict) and all(key in first_item_data for key in ESSENTIAL_KEYS_FOR_CACHE):
                                print("Lean cache structure validated.")
                                self.data_cache = cached_content["data"]; self.cache_timestamps = cached_content["timestamps"]
                                self.feature_stats = cached_content.get("stats", {}) # Load stats if present
                                print(f"Loaded feature stats from cache: {bool(self.feature_stats)}")
                                # <<< MODIFIED: Store sorted list for consistent IDs >>>
                                self.sorted_file_list_for_id = sorted(list(self.cache_timestamps.keys()))
                                self.full_file_paths = self.sorted_file_list_for_id[:] # Initialize full list
                                # Create Song ID mapping
                                self._create_song_id_map()
                                # <<< END MODIFICATION >>>
                                print(f"Lean cache loaded successfully in {load_time:.2f}s. {len(self.data_cache)} files.")
                                self._update_status("Loaded lean data from cache. Analyzing...")
                                self._analyze_cached_data(analyze_stats=not bool(self.feature_stats)) # Analyze stats only if not loaded
                                self._update_listbox(); self._update_status("Loaded lean data from cache. Ready."); return True
                            else: print("Error: Cache file does not contain expected lean data structure. Rebuilding.")
                        else: print("Warning: Loaded cache is empty. Rebuilding.")
                    else: print("Error: Invalid cache file format. Rebuilding.")
                    try: os.remove(CACHE_FILE_PATH); print(f"Removed invalid/old cache file: {CACHE_FILE_PATH}")
                    except OSError as e: print(f"Could not remove cache file: {e}")
                except Exception as e:
                    print(f"Error loading persistent cache file: {e}"); traceback.print_exc()
                    try: os.remove(CACHE_FILE_PATH); print(f"Removed corrupted cache file: {CACHE_FILE_PATH}")
                    except OSError as e: print(f"Could not remove cache file: {e}")
            else: print("Persistent lean cache file not found or empty.")
        except Exception as e: print(f"Unexpected error during cache check: {e}"); traceback.print_exc()
        print("Starting initial data load (will create lean cache)...")
        self._update_status("Cache not found or invalid. Loading all files to create lean cache...")
        self._initial_load()
        return False

    def _initial_load(self):
        if self.loading_in_progress: return
        self.loading_in_progress = True
        load_thread = threading.Thread(target=self._load_all_data, args=(True,), daemon=True); load_thread.start()

    def _refresh_data(self):
        if self.loading_in_progress: print("Load/Refresh already in progress."); return
        try:
            if os.path.exists(CACHE_FILE_PATH): os.remove(CACHE_FILE_PATH); print(f"Removed existing lean cache file for refresh: {CACHE_FILE_PATH}")
            self.data_cache = {}; self.cache_timestamps = {}; self.file_has_nan = {}; self.file_to_feature_sets = {}; self.feature_stats = {} # Clear stats too
            self.song_name_to_id = {}; self.sorted_file_list_for_id = [] # Clear ID mapping
            print("Cleared all in-memory cache data to force complete reload")
        except OSError as e: print(f"Warning: Could not completely clear cache: {e}")
        self.loading_in_progress = True
        refresh_thread = threading.Thread(target=self._load_all_data, args=(False,), daemon=True); refresh_thread.start()

    def _update_status(self, message): self.master.after(0, lambda: self.status_var.set(message))

    def _load_all_data(self, is_initial_load=False):
        if not self.loading_in_progress: self.loading_in_progress = True
        print("\n=== STARTING DATA LOAD PROCESS (Lean Cache Mode) ===")
        self.master.after(0, lambda: self.refresh_button.config(state=tk.DISABLED))
        self._update_status("Scanning for files...")
        if not os.path.isdir(PERFECT_FOLDER_PATH):
            print(f"ERROR: Directory not found - {PERFECT_FOLDER_PATH}")
            self.master.after(0, lambda: messagebox.showerror("Error", f"Directory not found:\n{PERFECT_FOLDER_PATH}"))
            self._update_status("Error: Directory not found."); self.loading_in_progress = False
            self.master.after(0, lambda: self.refresh_button.config(state=tk.NORMAL)); return
        else: print(f"DEBUG: Directory found: {PERFECT_FOLDER_PATH}")
        try:
            try: all_dir_contents = os.listdir(PERFECT_FOLDER_PATH)
            except Exception as list_err: print(f"ERROR: Could not list directory contents: {list_err}"); all_dir_contents = []
            current_files = sorted([f for f in all_dir_contents if f.lower().endswith(FILENAME_SUFFIX.lower())])
            print(f"DEBUG: Found {len(current_files)} files ending with '{FILENAME_SUFFIX}'")
            new_full_paths = [os.path.join(PERFECT_FOLDER_PATH, f) for f in current_files]
            # <<< MODIFIED: Update sorted list for IDs and full list >>>
            self.sorted_file_list_for_id = new_full_paths[:] # Store sorted list
            self.full_file_paths = new_full_paths[:] # Initialize full list
            self._create_song_id_map() # Create IDs based on sorted list
            # <<< END MODIFICATION >>>
            self.master.after(0, self._update_listbox_with_names_only) # Update display
            loaded_count, reloaded_count, error_count = 0, 0, 0; total_files = len(new_full_paths)
            temp_data_cache = {}; temp_timestamps = {}; temp_file_has_nan = {}; self.file_to_feature_sets = {}
            for i, file_path in enumerate(new_full_paths):
                filename = os.path.basename(file_path); self._update_status(f"Processing {i+1}/{total_files}: {filename}...")
                try:
                    current_mtime = os.path.getmtime(file_path); cached_mtime = self.cache_timestamps.get(file_path); lean_data = None
                    # --- Load/Reload Logic (Unchanged) ---
                    if cached_mtime is None or cached_mtime != current_mtime:
                        if cached_mtime is not None: reloaded_count += 1
                        else: loaded_count += 1
                        try:
                            full_track_data = joblib.load(file_path)
                            if not isinstance(full_track_data, dict): print(f" -> Warning: Loaded file {filename} is not a dictionary. Skipping."); continue
                            lean_data = {}; missing_essential = []
                            for key in ESSENTIAL_KEYS_FOR_CACHE:
                                if key in full_track_data: lean_data[key] = full_track_data[key]
                                else: missing_essential.append(key)
                            if missing_essential: print(f" -> Warning: File {filename} missing essential keys: {missing_essential}. Skipping."); continue
                            temp_data_cache[file_path] = lean_data; temp_timestamps[file_path] = current_mtime;
                        except Exception as load_err: print(f"Error loading file {filename}: {load_err}"); lean_data = None; error_count += 1; continue
                    else:
                        lean_data = self.data_cache.get(file_path)
                        if lean_data is not None: temp_data_cache[file_path] = lean_data; temp_timestamps[file_path] = cached_mtime
                        else: print(f" -> Warning: Timestamp match but no data in memory for {filename}. Will attempt reload on next refresh."); continue
                    # --- End Load/Reload Logic ---
                    if lean_data is not None:
                        has_nan = self._check_file_for_nan(lean_data); temp_file_has_nan[file_path] = has_nan
                        self._build_feature_sets_for_file(file_path, lean_data)
                    else: temp_file_has_nan[file_path] = False; self.file_to_feature_sets[file_path] = set()
                except Exception as e: error_count += 1; print(f"Error processing file {filename}: {e}"); traceback.print_exc()
            self.data_cache = temp_data_cache; self.cache_timestamps = temp_timestamps; self.file_has_nan = temp_file_has_nan
            self._analyze_cached_data(analyze_stats=True); # Always analyze stats on full load/refresh
            self._save_persistent_cache()
            print(f"DEBUG: Scheduling final listbox update."); self.master.after(0, self._update_listbox)
            status_msg = f"Load/Refresh complete. {loaded_count} new, {reloaded_count} reloaded."
            if error_count > 0: status_msg += f" {error_count} errors."
            self._update_status(status_msg)
        except Exception as e: self.master.after(0, lambda: messagebox.showerror("Error During Load/Refresh", f"An unexpected error occurred:\n{e}")); self._update_status("Error during load/refresh."); print(traceback.format_exc())
        finally: self.loading_in_progress = False; self.master.after(0, lambda: self.refresh_button.config(state=tk.NORMAL))

    def _update_listbox_with_names_only(self):
        self.file_listbox.delete(0, tk.END)
        if not self.full_file_paths: self.file_listbox.insert(tk.END, "(No files found)"); return
        for file_path in self.full_file_paths: # Use the potentially filtered list
            filename = os.path.basename(file_path); display_name = filename
            if display_name.lower().endswith(FILENAME_SUFFIX.lower()): display_name = display_name[:-len(FILENAME_SUFFIX)]
            self.file_listbox.insert(tk.END, display_name)
        self._update_status(f"Found {len(self.full_file_paths)} files. Loading data...")

    def _save_persistent_cache(self):
        try:
            print(f"Saving persistent lean cache to: {CACHE_FILE_PATH} ({len(self.data_cache)} items)")
            start_time = time.time();
            # Include feature stats in the cache
            cache_to_save = {"data": self.data_cache, "timestamps": self.cache_timestamps, "stats": self.feature_stats}
            joblib.dump(cache_to_save, CACHE_FILE_PATH, compress=3); save_time = time.time() - start_time
            print(f"Lean cache saved successfully in {save_time:.2f}s."); return True
        except Exception as e: print(f"Error saving persistent lean cache: {e}"); traceback.print_exc(); return False

    # --- Analysis Helpers (Operate on lean data) ---
    def _analyze_cached_data(self, analyze_stats=True):
        """Performs NaN check, feature set analysis, and optionally STATS calculation."""
        print("Analyzing cached lean data...")
        temp_file_has_nan = {}
        self.file_to_feature_sets = {}
        all_numeric_features_global = defaultdict(list)
        all_labels_found = set() # Set to collect unique labels

        # --- Pass 1: Collect data for stats, check NaN, build feature sets, find labels ---
        for file_path, lean_data in self.data_cache.items():
            filename = os.path.basename(file_path)
            has_nan = self._check_file_for_nan(lean_data)
            temp_file_has_nan[file_path] = has_nan
            self._build_feature_sets_for_file(file_path, lean_data)

            # Collect labels
            if isinstance(lean_data, dict) and "semantic_labels" in lean_data:
                 labels = lean_data.get("semantic_labels", [])
                 if isinstance(labels, list):
                      all_labels_found.update(labels) # Add labels from this file

            # Collect numeric features only if stats need recalculating
            if analyze_stats and isinstance(lean_data, dict) and "section_features" in lean_data:
                for section in lean_data.get("section_features", []):
                    if isinstance(section, dict):
                        for key, value in section.items():
                            try:
                                float_val = float(value)
                                if np.isfinite(float_val):
                                    all_numeric_features_global[key].append(float_val)
                            except (ValueError, TypeError): continue

        # --- Calculate Global Stats (only if requested) ---
        if analyze_stats:
            self.feature_stats = {}
            numeric_feature_keys_for_search = []
            print("Calculating global feature statistics...")
            for key, values in all_numeric_features_global.items():
                if len(values) >= 2:
                    mean = np.mean(values); std = np.std(values)
                    self.feature_stats[key] = {'mean': mean, 'std': std if std > 1e-9 else 0.0}
                    numeric_feature_keys_for_search.append(key)
                else: self.feature_stats[key] = {'mean': np.nan, 'std': np.nan}
            # Update Search Feature Combobox
            sorted_numeric_keys = sorted(numeric_feature_keys_for_search)
            self.master.after(0, lambda: self.search_feature_combo.config(values=sorted_numeric_keys, state='readonly' if sorted_numeric_keys else 'disabled')) # Enable if keys found
            if sorted_numeric_keys: self.master.after(0, lambda: self.search_feature_var.set(sorted_numeric_keys[0]))
            else: self.master.after(0, lambda: self.search_feature_var.set(""))
        else:
             print("Skipping stats calculation (loaded from cache or not requested).")
             # Still update combobox based on loaded stats keys
             loaded_stat_keys = sorted([k for k, v in self.feature_stats.items() if not math.isnan(v['mean'])])
             self.master.after(0, lambda: self.search_feature_combo.config(values=loaded_stat_keys, state='readonly' if loaded_stat_keys else 'disabled'))
             if loaded_stat_keys: self.master.after(0, lambda: self.search_feature_var.set(loaded_stat_keys[0]))
             else: self.master.after(0, lambda: self.search_feature_var.set(""))


        # --- Update Label Filter Comboboxes (Feature Search and Transition Search) ---
        self.unique_labels = sorted(list(all_labels_found))
        label_combo_values = ["All Sections"] + self.unique_labels
        # Update feature search label filter
        self.master.after(0, lambda: self.search_label_combo.config(values=label_combo_values, state='readonly'))
        self.master.after(0, lambda: self.search_label_filter_var.set("All Sections"))
        # Update transition search label filters
        self.master.after(0, lambda: self.search_from_label_combo.config(values=self.unique_labels, state='readonly' if self.unique_labels else 'disabled'))
        self.master.after(0, lambda: self.search_to_label_combo.config(values=self.unique_labels, state='readonly' if self.unique_labels else 'disabled'))
        # Set default selection for transition combos if possible
        if self.unique_labels:
            self.master.after(0, lambda: self.search_from_label_var.set(self.unique_labels[0]))
            self.master.after(0, lambda: self.search_to_label_var.set(self.unique_labels[0]))
        else:
            self.master.after(0, lambda: self.search_from_label_var.set(""))
            self.master.after(0, lambda: self.search_to_label_var.set(""))


        # --- Finalize other analysis ---
        self.file_has_nan = temp_file_has_nan
        self._analyze_and_display_feature_sets()

        # --- Setup Column Checkboxes ---
        all_keys_found = set()
        for data_dict in self.data_cache.values():
             if isinstance(data_dict, dict) and "section_features" in data_dict:
                  for section in data_dict.get("section_features", []):
                       if isinstance(section, dict): all_keys_found.update(section.keys())
        # <<< MODIFIED: Add new standard columns >>>
        standard_cols = ["Song Name", "Song ID", "#", "Label", "Start (s)", "End (s)", "Dur (s)"]
        dynamic_keys = sorted([k for k in all_keys_found if k not in ["index", "start_time", "end_time", "duration_sec", "duration_bars", "original_label", "cluster_id"]])
        self.all_columns = standard_cols + dynamic_keys
        # Update default order if necessary
        if "Song Name" not in self.column_order: self.column_order.insert(0, "Song Name")
        if "Song ID" not in self.column_order: self.column_order.insert(1, "Song ID")
        # Ensure all columns are accounted for
        for col in self.all_columns:
            if col not in self.column_order: self.column_order.append(col)
        self.master.after(0, lambda: self._setup_column_checkboxes(self.all_columns))

    # <<< ADDED: Method to create Song ID mapping >>>
    def _create_song_id_map(self):
        """Creates a mapping from sorted file path to a unique ID."""
        self.song_name_to_id = {file_path: i + 1 for i, file_path in enumerate(self.sorted_file_list_for_id)}
        print(f"DEBUG: Created Song ID map for {len(self.song_name_to_id)} files.")

    def _build_feature_sets_for_file(self, file_path, lean_data):
        # (Code remains the same)
        if not isinstance(lean_data, dict): self.file_to_feature_sets[file_path] = set(); return
        features_list = lean_data.get("section_features", [])
        if not isinstance(features_list, list): self.file_to_feature_sets[file_path] = set(); return
        feature_sets = set()
        for section_dict in features_list:
            if isinstance(section_dict, dict):
                keys = tuple(sorted([k for k in section_dict.keys() if k not in ["index", "start_time", "end_time", "duration_sec", "duration_bars", "original_label", "cluster_id"]]))
                if keys: feature_sets.add(keys)
        self.file_to_feature_sets[file_path] = feature_sets

    def _check_file_for_nan(self, lean_data):
        # (Code remains the same)
        try:
            if isinstance(lean_data, dict) and "section_features" in lean_data:
                for idx, section in enumerate(lean_data.get("section_features", [])):
                    if not isinstance(section, dict): continue
                    for key, value in section.items():
                        try:
                            if isinstance(value, (float, np.floating)) and not np.isfinite(value): return True
                        except: pass
                        str_value = str(value)
                        if "NaN" in str_value or "nan" in str_value: return True
            return False
        except Exception as e: print(f"Error checking for NaN: {e}"); traceback.print_exc(); return False

    def _analyze_and_display_feature_sets(self):
        # (Code remains the same)
        print("Analyzing unique feature sets from lean cache..."); unique_sets = set(); feature_counts = defaultdict(int)
        for file_path, lean_data in self.data_cache.items():
            if not isinstance(lean_data, dict): continue
            features_list = lean_data.get("section_features", [])
            if not isinstance(features_list, list): continue
            file_sets = set()
            for section_dict in features_list:
                if isinstance(section_dict, dict):
                    keys = tuple(sorted([k for k in section_dict.keys() if k not in ["index", "start_time", "end_time", "duration_sec", "duration_bars", "original_label", "cluster_id"]]))
                    if keys: unique_sets.add(keys); file_sets.add(keys)
            for f_set in file_sets: feature_counts[f_set] += 1
        self.unique_feature_sets = sorted(list(unique_sets), key=len)
        self.master.after(0, self._update_feature_set_display, feature_counts)

    def _update_feature_set_display(self, feature_counts):
        # (Code remains the same)
        self.feature_set_listbox.delete(0, tk.END)
        if not self.unique_feature_sets: self.feature_set_listbox.insert(tk.END, "(No feature sets found)"); return
        print(f"Found {len(self.unique_feature_sets)} unique feature sets.")
        for feature_set in self.unique_feature_sets:
            count = feature_counts.get(feature_set, 0)
            display_str = f"({count} files) {len(feature_set)} features: {', '.join(feature_set)}"
            self.feature_set_listbox.insert(tk.END, display_str)

    # --- Filtering Logic (Unchanged) ---
    def _apply_filters(self):
        print("Applying filters...");
        if self.loading_in_progress: print(" -> Load/Refresh in progress, deferring filter application."); return
        self._update_listbox()

    def _update_listbox(self):
        """Updates the file listbox based on current filters."""
        print("Updating listbox display with filters...")
        selected_indices = self.file_listbox.curselection() # Get current selections
        selected_paths = [self.full_file_paths[i] for i in selected_indices if 0 <= i < len(self.full_file_paths)]

        all_cached_paths = sorted(list(self.cache_timestamps.keys())) # Use the full list from cache
        filtered_paths = []
        show_nan = self.show_nan_only_var.get()
        for file_path in all_cached_paths:
            passes_filters = True
            has_nan = self.file_has_nan.get(file_path, False)
            if show_nan and not has_nan: passes_filters = False
            if self.selected_feature_set and not self._check_file_matches_feature_set(file_path, self.selected_feature_set): passes_filters = False
            if passes_filters: filtered_paths.append(file_path)

        self.file_listbox.delete(0, tk.END)
        self.full_file_paths = filtered_paths # Update the list used by the listbox

        if not filtered_paths:
            self.file_listbox.insert(tk.END, "(No files match filters)")
            self.clear_feature_display()
        else:
            new_selection_indices = []
            for idx, file_path in enumerate(filtered_paths):
                filename = os.path.basename(file_path)
                display_name = filename
                if display_name.lower().endswith(FILENAME_SUFFIX.lower()):
                    display_name = display_name[:-len(FILENAME_SUFFIX)]
                self.file_listbox.insert(tk.END, display_name)
                if file_path in selected_paths: # Check if this path was previously selected
                    new_selection_indices.append(idx)

            # Restore selection
            if new_selection_indices:
                for idx in new_selection_indices:
                    self.file_listbox.selection_set(idx)
                self.file_listbox.activate(new_selection_indices[0]) # Activate first selected
                self.file_listbox.see(new_selection_indices[0]) # Ensure first selected is visible
            elif self.file_listbox.size() > 0:
                # If previous selection is gone, maybe select first item? Or clear display?
                self.clear_feature_display() # Clear display if selection changed drastically

        self._update_status(f"Displaying {len(filtered_paths)} files matching filters.")


    # --- Display Logic (Uses lean data) ---
    # <<< MODIFIED: Handler for selection change (doesn't auto-display) >>>
    def on_file_selection_change(self, event=None):
        """Updates status based on selection, does not automatically display."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            self._update_status("No files selected.")
            self.clear_feature_display()
        elif len(selected_indices) == 1:
             idx = selected_indices[0]
             if 0 <= idx < len(self.full_file_paths):
                 display_name = self.file_listbox.get(idx)
                 self._update_status(f"Selected: {display_name}")
             else:
                  self._update_status("Selection index out of bounds.")
        else:
            self._update_status(f"Selected {len(selected_indices)} files.")
        # Do NOT call display method here automatically

    # <<< ADDED: Method to display selected files >>>
    def display_selected_files(self):
        """Gathers data for selected files and displays it."""
        if self.loading_in_progress: print("Loading in progress, cannot display."); return
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Display Info", "No files selected in the list.")
            return

        paths_to_display = [self.full_file_paths[i] for i in selected_indices if 0 <= i < len(self.full_file_paths)]
        if not paths_to_display:
            messagebox.showerror("Display Error", "Selected indices are out of bounds.")
            return

        self._display_aggregated_data(paths_to_display)

    # <<< ADDED: Method to display all files >>>
    def display_all_files(self):
        """Gathers data for ALL currently filtered files and displays it."""
        if self.loading_in_progress: print("Loading in progress, cannot display."); return
        if not self.full_file_paths:
             messagebox.showinfo("Display Info", "No files available to display (check filters).")
             return

        # Use self.full_file_paths which respects current filters
        self._display_aggregated_data(self.full_file_paths)
        # Select all in listbox visually
        self.file_listbox.selection_set(0, tk.END)

    # <<< ADDED: Central method to prepare and display aggregated data >>>
    def _display_aggregated_data(self, file_paths_to_display):
        """Prepares and displays data from a list of file paths."""
        print(f"Preparing to display data for {len(file_paths_to_display)} files...")
        self._update_status(f"Loading data for {len(file_paths_to_display)} files...")
        self.master.update_idletasks()

        aggregated_data = []
        files_processed = 0
        total_sections = 0

        # --- Ensure Song ID map is up-to-date ---
        if not self.song_name_to_id or set(self.sorted_file_list_for_id) != set(self.cache_timestamps.keys()):
            print("Warning: Song ID map might be outdated or missing. Recreating...")
            self.sorted_file_list_for_id = sorted(list(self.cache_timestamps.keys()))
            self._create_song_id_map()

        # --- Prepare data for Treeview ---
        for file_path in file_paths_to_display:
            if file_path in self.data_cache:
                lean_track_data = self.data_cache[file_path]
                filename_short = os.path.basename(file_path)
                if filename_short.lower().endswith(FILENAME_SUFFIX.lower()):
                    filename_short = filename_short[:-len(FILENAME_SUFFIX)]
                song_id = self.song_name_to_id.get(file_path, "N/A") # Get song ID

                if isinstance(lean_track_data, dict) and "section_features" in lean_track_data:
                    section_features_list = lean_track_data.get("section_features", [])
                    semantic_labels = lean_track_data.get("semantic_labels", [])
                    if isinstance(section_features_list, list) and len(section_features_list) == len(semantic_labels):
                        for i, section_dict in enumerate(section_features_list):
                            if not isinstance(section_dict, dict): continue
                            try:
                                # <<< MODIFIED: Add Song Name and Song ID >>>
                                values_dict = {
                                    "Song Name": filename_short,
                                    "Song ID": song_id,
                                    "#": i + 1,
                                    "Label": semantic_labels[i] if i < len(semantic_labels) else "N/A",
                                    "Start (s)": f"{section_dict.get('start_time', np.nan):.3f}",
                                    "End (s)": f"{section_dict.get('end_time', np.nan):.3f}",
                                    "Dur (s)": f"{section_dict.get('duration_sec', np.nan):.3f}"
                                }
                                # <<< MODIFIED: Update standard columns list >>>
                                standard_cols = ["Song Name", "Song ID", "#", "Label", "Start (s)", "End (s)", "Dur (s)"]
                                dynamic_feature_keys = [col for col in self.all_columns if col not in standard_cols]
                                for key in dynamic_feature_keys:
                                    val = section_dict.get(key, np.nan)
                                    formatted_val = ""
                                    if isinstance(val, (int, float, np.number)):
                                        formatted_val = f"{val:.4f}" if np.isfinite(val) else "NaN"
                                    else:
                                        formatted_val = str(val) if val is not None else "None"
                                    values_dict[key] = formatted_val
                                # <<< MODIFIED: Store tuple for unique identification >>>
                                aggregated_data.append((file_path, i, values_dict))
                                total_sections += 1
                            except Exception as e:
                                print(f"Error processing section {i} of {filename_short}: {e}")
                                traceback.print_exc()
                        files_processed += 1
                    else:
                        print(f"Warning: Mismatched features/labels or not list for {filename_short}")
                else:
                    print(f"Warning: Invalid or missing section_features for {filename_short}")
            else:
                print(f"Warning: Data not found in cache for {file_path}")

        # --- Update Treeview ---
        self.current_data = aggregated_data # Store the aggregated data
        self.sort_column = None # Reset sort when displaying new data
        self.sort_reverse = False
        self._refresh_treeview() # Populate the treeview

        self._update_status(f"Displayed {total_sections} sections from {files_processed} files.")


    def clear_feature_display(self):
        """Clears the main feature treeview."""
        for item in self.feature_tree.get_children(): self.feature_tree.delete(item)
        # Keep columns defined, just clear rows
        # self.feature_tree["columns"] = ()
        self.current_data = [];
        print("Feature display cleared")

    def display_track_features(self, lean_track_data):
        """(Deprecated) Use _display_aggregated_data instead."""
        print("Warning: display_track_features is deprecated. Use _display_aggregated_data.")
        # This function is no longer called directly by on_file_select
        # If needed for single file display elsewhere, adapt _display_aggregated_data
        pass


    # --- Search Functionality ---
    def _update_search_value_fields(self, event=None):
        """Enable/disable Value A/B entry fields based on query type."""
        query_type = self.search_query_type_var.get()
        if "Std Dev" in query_type:
            self.search_value_a_label.config(text="N (Std Devs):")
            self.search_value_b_label.grid_remove(); self.search_value_b_entry.grid_remove()
            self.search_value_a_entry.config(state=tk.NORMAL)
        elif "Between" in query_type or "Outside" in query_type:
            self.search_value_a_label.config(text="Value A:")
            self.search_value_b_label.grid(); self.search_value_b_entry.grid()
            self.search_value_a_entry.config(state=tk.NORMAL); self.search_value_b_entry.config(state=tk.NORMAL)
        elif "Greater" in query_type or "Less" in query_type:
            self.search_value_a_label.config(text="Value:")
            self.search_value_b_label.grid_remove(); self.search_value_b_entry.grid_remove()
            self.search_value_a_entry.config(state=tk.NORMAL)
        else:
            self.search_value_a_label.config(text="Value A:"); self.search_value_b_label.grid_remove()
            self.search_value_b_entry.grid_remove(); self.search_value_a_entry.config(state=tk.DISABLED)

    def _perform_search(self):
        """Executes the feature value search based on selected criteria."""
        print("--- Performing Feature Value Search ---")
        self._update_status("Searching feature values...")
        self.master.update_idletasks()

        for item in self.search_results_tree.get_children(): self.search_results_tree.delete(item)
        # Clear the transition results text area as well
        self.transition_results_text.config(state=tk.NORMAL); self.transition_results_text.delete('1.0', tk.END); self.transition_results_text.config(state=tk.DISABLED)


        label_filter = self.search_label_filter_var.get()
        feature = self.search_feature_var.get()
        query_type = self.search_query_type_var.get()
        val_a_str = self.search_value_a_var.get()
        val_b_str = self.search_value_b_var.get()

        print(f"Search Params: Label='{label_filter}', Feature='{feature}', Query='{query_type}', ValA='{val_a_str}', ValB='{val_b_str}'")

        if not feature: messagebox.showerror("Search Error", "Please select a feature to search."); self._update_status("Search Error: No feature selected."); return

        try: # Input validation
            val_a = float(val_a_str) if val_a_str else None
            val_b = float(val_b_str) if val_b_str else None
            if query_type in ["Greater Than (>)", "Less Than (<)", "Std Dev Above Mean (> mean + N*std)", "Std Dev Below Mean (< mean - N*std)"]:
                if val_a is None: raise ValueError("Value A / N cannot be empty.")
                if "Std Dev" in query_type and val_a <= 0: raise ValueError("Std Dev multiplier (N) must be positive.")
            elif query_type in ["Between (A <= val <= B)", "Outside Range (val < A or val > B)"]:
                if val_a is None or val_b is None: raise ValueError("Both Value A and Value B must be provided.")
                if val_a >= val_b: raise ValueError("Value A must be less than Value B for range queries.")
        except ValueError as e: messagebox.showerror("Search Error", f"Invalid input value(s).\n{e}"); self._update_status(f"Search Error: Invalid input."); return

        mean = None; std = None
        if "Std Dev" in query_type:
            if feature not in self.feature_stats or math.isnan(self.feature_stats[feature]['mean']):
                 messagebox.showerror("Search Error", f"Statistics not available for feature '{feature}'."); self._update_status(f"Search Error: Stats missing for {feature}."); return
            stats = self.feature_stats[feature]; mean = stats['mean']; std = stats['std']
            if std == 0.0: messagebox.showwarning("Search Warning", f"Feature '{feature}' has zero standard deviation.")

        results = []; files_searched = 0; sections_searched = 0; matches_found = 0
        for file_path, lean_data in self.data_cache.items():
            files_searched += 1; filename = os.path.basename(file_path)
            if filename.lower().endswith(FILENAME_SUFFIX.lower()): filename = filename[:-len(FILENAME_SUFFIX)]
            if not isinstance(lean_data, dict): continue
            section_features = lean_data.get("section_features", []); semantic_labels = lean_data.get("semantic_labels", [])
            if not isinstance(section_features, list) or len(section_features) != len(semantic_labels): continue

            for i, section in enumerate(section_features):
                sections_searched += 1; label = semantic_labels[i]
                if not isinstance(section, dict): continue
                if label_filter != "All Sections" and label != label_filter: continue # Apply label filter
                value_raw = section.get(feature)
                try: value = float(value_raw);
                except (ValueError, TypeError): continue
                if not math.isfinite(value): continue # Skip NaN/Inf

                match = False
                try:
                    if query_type == "Greater Than (>)" and value > val_a: match = True
                    elif query_type == "Less Than (<)" and value < val_a: match = True
                    elif query_type == "Between (A <= val <= B)" and val_a <= value <= val_b: match = True
                    elif query_type == "Outside Range (val < A or val > B)" and (value < val_a or value > val_b): match = True
                    elif query_type == "Std Dev Above Mean (> mean + N*std)" and std is not None:
                        threshold = mean + val_a * std;
                        if value > threshold: match = True
                    elif query_type == "Std Dev Below Mean (< mean - N*std)" and std is not None:
                        threshold = mean - val_a * std;
                        if value < threshold: match = True
                except Exception as comp_e: print(f"Error during comparison: {comp_e}"); continue

                if match:
                    matches_found += 1
                    # Add section number (i+1) to results tuple
                    results.append((filename, i + 1, label, feature, f"{value:.4f}"))

        print(f"Search complete. Found {matches_found} matches in {sections_searched} sections across {files_searched} files.")
        if results:
            for item in results:
                # Insert item with 5 values
                self.search_results_tree.insert("", tk.END, values=item)
            self._update_status(f"Search complete. Found {matches_found} results.")
        else: self._update_status("Search complete. No matching sections found.")

    def _perform_transition_search(self):
        """Executes the search for specific label transitions."""
        print("--- Performing Transition Search ---")
        self._update_status("Searching transitions...")
        self.master.update_idletasks()

        # Clear previous results (both feature and transition)
        for item in self.search_results_tree.get_children(): self.search_results_tree.delete(item)
        self.transition_results_text.config(state=tk.NORMAL); self.transition_results_text.delete('1.0', tk.END)

        # Get selected labels
        from_label = self.search_from_label_var.get()
        to_label = self.search_to_label_var.get()

        # Validate selections
        if not from_label or not to_label:
            messagebox.showerror("Transition Search Error", "Please select both a 'From' and 'To' label.")
            self._update_status("Transition Search Error: Labels not selected.")
            self.transition_results_text.config(state=tk.DISABLED); return
        if from_label == to_label:
             messagebox.showwarning("Transition Search Info", "Searching for consecutive identical labels.")
             # Allow searching for e.g., Drop -> Drop if desired

        print(f"Searching for transition: {from_label} -> {to_label}")

        # --- Iterate through cached data ---
        matching_files = set()
        files_searched = 0
        transitions_found_total = 0

        for file_path, lean_data in self.data_cache.items():
            files_searched += 1
            filename = os.path.basename(file_path)
            if filename.lower().endswith(FILENAME_SUFFIX.lower()):
                filename = filename[:-len(FILENAME_SUFFIX)] # Clean filename

            if not isinstance(lean_data, dict): continue
            semantic_labels = lean_data.get("semantic_labels", [])
            if not isinstance(semantic_labels, list) or len(semantic_labels) < 2:
                continue # Need at least two labels for a transition

            # Check for the transition pair
            found_in_file = False
            for i in range(len(semantic_labels) - 1):
                if semantic_labels[i] == from_label and semantic_labels[i+1] == to_label:
                    matching_files.add(filename)
                    found_in_file = True
                    transitions_found_total += 1
                    # break # Optional: Stop searching this file once found? Or count all occurrences?
                    # Let's count all occurrences for now, but only list the file once.

        # --- Display Results ---
        print(f"Transition search complete. Found transition in {len(matching_files)} files.")
        if matching_files:
            result_text = f"Found '{from_label} -> {to_label}' transition in the following files:\n\n"
            result_text += "\n".join(sorted(list(matching_files)))
            self.transition_results_text.insert('1.0', result_text)
            self._update_status(f"Transition search complete. Found in {len(matching_files)} files.")
        else:
            self.transition_results_text.insert('1.0', f"No files found with the transition '{from_label} -> {to_label}'.")
            self._update_status("Transition search complete. No matches found.")

        self.transition_results_text.config(state=tk.DISABLED) # Make read-only

    # <<< ADDED: Export to CSV Function >>>
    def _export_to_csv(self):
        """Exports the data currently displayed in the main feature treeview to a CSV file."""
        print("Exporting displayed data to CSV...")
        self._update_status("Exporting data...")
        self.master.update_idletasks()

        # Get visible columns in their current display order
        visible_columns = self._get_visible_columns_in_order()
        if not visible_columns:
            messagebox.showwarning("Export Error", "No columns are visible to export.")
            self._update_status("Export cancelled: No visible columns.")
            return

        # Get data currently displayed in the treeview
        # This uses the order currently shown, respecting any sorting
        tree_items = self.feature_tree.get_children('')
        if not tree_items:
            messagebox.showinfo("Export Info", "No data currently displayed in the table to export.")
            self._update_status("Export cancelled: No data displayed.")
            return

        # Ask user for save location
        default_filename = f"feature_inspector_export_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            title="Save Displayed Data as CSV",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if not file_path:
            self._update_status("Export cancelled by user.")
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header row
                writer.writerow(visible_columns)
                # Write data rows
                for item_id in tree_items:
                    row_values = self.feature_tree.item(item_id)['values']
                    writer.writerow(row_values)
            self._update_status(f"Data successfully exported to {os.path.basename(file_path)}")
            messagebox.showinfo("Export Successful", f"Data exported to:\n{file_path}")
        except Exception as e:
            self._update_status("Export failed!")
            messagebox.showerror("Export Error", f"Failed to write CSV file:\n{e}")
            traceback.print_exc()

    # <<< ADDED: Sorting Functionality >>>
    def _on_column_header_right_click(self, event):
        """Handles right-click on Treeview column header for sorting."""
        region = self.feature_tree.identify_region(event.x, event.y)
        if region != "heading":
            return # Click wasn't on a heading

        column_id_str = self.feature_tree.identify_column(event.x) # e.g., '#1', '#2'
        try:
            # Convert Treeview column ID (like '#3') to actual column name
            col_index = int(column_id_str.replace('#', '')) - 1
            visible_cols = self._get_visible_columns_in_order()
            if 0 <= col_index < len(visible_cols):
                column_name = visible_cols[col_index]
            else:
                print(f"Warning: Could not map column ID {column_id_str} to visible column.")
                return
        except (ValueError, IndexError):
            print(f"Warning: Could not parse column ID {column_id_str} for sorting.")
            return

        # Create popup menu
        sort_menu = tk.Menu(self.master, tearoff=0)
        sort_menu.add_command(label=f"Sort Ascending by '{column_name}'",
                              command=lambda: self._sort_treeview(column_name, reverse=False))
        sort_menu.add_command(label=f"Sort Descending by '{column_name}'",
                              command=lambda: self._sort_treeview(column_name, reverse=True))

        # Display the menu at the cursor position
        sort_menu.tk_popup(event.x_root, event.y_root)


    def _sort_treeview(self, column, reverse=False):
        """Sorts the data in self.current_data and refreshes the treeview."""
        print(f"Sorting by column '{column}', reverse={reverse}")
        self._update_status(f"Sorting by '{column}'...")
        self.master.update_idletasks()

        if not self.current_data:
            print("No data to sort.")
            self._update_status("No data to sort.")
            return

        # --- Sorting Logic ---
        # Define a key function to handle potential type errors during sort
        def sort_key(item_tuple):
            # item_tuple is expected to be (file_path, section_index, values_dict)
            if len(item_tuple) != 3 or not isinstance(item_tuple[2], dict):
                return None # Should not happen, but handles malformed data

            values_dict = item_tuple[2]
            val = values_dict.get(column)

            # Attempt numerical conversion for sorting
            try:
                # Handle potential 'NaN' strings or actual NaN/inf
                if isinstance(val, str) and val.lower() == 'nan':
                    return float('inf') # Treat string 'NaN' as largest value
                float_val = float(val)
                if not math.isfinite(float_val):
                     # Treat actual NaN/inf as largest value
                     # Use negative infinity if sorting descending to put them last
                    return float('-inf') if reverse else float('inf')
                return float_val
            except (ValueError, TypeError):
                 # Fallback to string sorting (case-insensitive) if conversion fails
                 return str(val).lower() if val is not None else ""


        try:
            # Sort the underlying data list (self.current_data)
            self.current_data.sort(key=sort_key, reverse=reverse)
            self.sort_column = column # Store last sort criteria
            self.sort_reverse = reverse
        except Exception as e:
             print(f"Error during sorting: {e}")
             traceback.print_exc()
             messagebox.showerror("Sort Error", f"Could not sort by column '{column}'.\nError: {e}")
             self._update_status("Sort failed.")
             return

        # Refresh the treeview display with the sorted data
        self._refresh_treeview()
        sort_dir = "Descending" if reverse else "Ascending"
        self._update_status(f"Sorted by '{column}' ({sort_dir})")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureInspectorApp(root)
    root.mainloop()
