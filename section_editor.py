# =============================================================================
# FILE: section_editor.py
# Contains the SectionEditor class (a ttk.Frame) for displaying and
# editing section labels and colors in a Treeview using a pop-up dialog.
# Includes a color indicator column; row background color removed.
# Uses "Body" instead of "Verse". Adds "Fade Out" label.
# Right-click now calls main app's pop-up method.
# =============================================================================

import tkinter as tk
from tkinter import ttk
import traceback
import numpy as np # Needed for calculating bar number
from debug_utils import debug_print

# --- Constants ---
# These are now primarily defined here and imported by main_app if needed elsewhere
COLOR_NAME_MAP = {
    "Dark Red":   '#8B0000',
    "Red":        '#FF0000',
    "Orange":     '#FFA500',
    "Dark Green": '#014421',
    "Light Green":'#7CCD7C',
    "Light Blue": '#ADD8E6',
    "Fade Out":   '#8A2BE2' # BlueViolet for Fade Out
}
HEX_TO_COLOR_NAME = {v: k for k, v in COLOR_NAME_MAP.items()}
ALLOWED_LABELS = ["Intro", "Body", "Breakdown", "Build", "Drop", "Outro", "Fill", "Fade Out"]


class SectionEditor(ttk.Frame):
    """
    A ttk.Frame containing a Treeview for editing section labels and colors
    via a pop-up dialog, and an 'Update Sections' button. Includes color indicator column.
    """
    def __init__(self, master, apply_callback=None, app_ref=None, **kwargs): # Added app_ref
        """
        Initialize the SectionEditor frame.

        Args:
            master: The parent widget (likely the editor_outer_frame).
            apply_callback (callable): A function to call when the 'Update'
                                       button is pressed.
            app_ref (AudioAnalyzerApp): Reference to the main application instance.
            **kwargs: Additional keyword arguments for the ttk.Frame.
        """
        super().__init__(master, padding="5", **kwargs)
        self.apply_callback = apply_callback
        self.app = app_ref # Store reference to main app

        # --- Create Widgets ---
        self._setup_widgets()

    def _setup_widgets(self):
        """Creates and packs the Treeview, Scrollbar, and Button."""
        debug_print("INITIALIZATION", "Setting up SectionEditor widgets")

        # Treeview for editing
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ('Clr', '#', 'Start (Bar)', 'Dur (Bars)', 'Label', 'Color')
        self.tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=10)

        # Define headings and column widths
        self.tree.heading('Clr', text='', anchor=tk.CENTER); self.tree.column('Clr', width=30, stretch=tk.NO, anchor=tk.CENTER)
        self.tree.heading('#', text='#', anchor=tk.W); self.tree.column('#', width=40, stretch=tk.NO, anchor=tk.W)
        self.tree.heading('Start (Bar)', text='Start (Bar)', anchor=tk.W); self.tree.column('Start (Bar)', width=70, stretch=tk.NO, anchor=tk.W)
        self.tree.heading('Dur (Bars)', text='Dur (Bars)', anchor=tk.W); self.tree.column('Dur (Bars)', width=70, stretch=tk.NO, anchor=tk.W)
        self.tree.heading('Label', text='Label', anchor=tk.W); self.tree.column('Label', width=120, stretch=tk.YES)
        self.tree.heading('Color', text='Color', anchor=tk.W); self.tree.column('Color', width=100, stretch=tk.YES)

        # Scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview); self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y); self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        debug_print("INITIALIZATION", f"Setting up color tags for {len(COLOR_NAME_MAP)} colors")

        # Define tags for FOREGROUND colors only
        self.color_tags = {}
        for color_name, color_hex in COLOR_NAME_MAP.items():
            tag_name = f"color_{color_name.lower().replace(' ', '_')}"
            self.color_tags[color_hex] = tag_name
            try:
                self.tree.tag_configure(tag_name, foreground=color_hex)
                debug_print(
                    "INITIALIZATION", f"Configured tag '{tag_name}' with fg={color_hex}"
                )
            except tk.TclError as e: 
                debug_print(
                    "INITIALIZATION",
                    f"Error: Could not configure tag '{tag_name}' - {e}",
                )

        self.tree.tag_configure('custom_color', foreground='grey')
        self.color_tags['default'] = 'custom_color'
        debug_print("INITIALIZATION", "Added default 'custom_color' tag")

        # Bind right-click event for editing
        self.tree.bind("<Button-3>", self._on_cell_edit_start)
        self.tree.bind("<Button-2>", self._on_cell_edit_start)
        debug_print("INITIALIZATION", "Bound edit events to mouse buttons 2 and 3")

        # Update button
        self.update_button = ttk.Button(self, text="Update Sections", command=self._trigger_apply_edits, state=tk.DISABLED)
        self.update_button.pack(pady=5)
        debug_print("INITIALIZATION", "SectionEditor widgets setup complete")

    def populate(self, track_data):
        """Fills the Treeview editor with current section data and applies row color tags."""
        print("DEBUG SectionEditor: Populating...")
        self.clear()
        required_keys = ['section_features', 'semantic_labels', 'label_colors', 'trim_offset_sec', 'seconds_per_bar']
        if not track_data or not all(k in track_data and track_data[k] is not None for k in required_keys):
            print(f"DEBUG SectionEditor: Missing data for population."); self.update_button.config(state=tk.DISABLED); return

        features = track_data['section_features']; labels = track_data['semantic_labels']; colors = track_data['label_colors']
        trim_offset = track_data['trim_offset_sec']; sec_per_bar = track_data['seconds_per_bar']; num_sections = len(features)

        if not (len(labels) == num_sections and len(colors) == num_sections):
            print("DEBUG SectionEditor: Data list length mismatch."); self.update_button.config(state=tk.DISABLED); return

        for i in range(num_sections):
            f = features[i]; start_time_abs = f.get('start_time', 0.0); duration_bars = f.get('duration_bars', 0)
            current_label = labels[i] if i < len(labels) else 'N/A'
            color_hex = colors[i] if i < len(colors) else '#808080'; color_name = HEX_TO_COLOR_NAME.get(color_hex, "Custom")
            start_bar_str = 'N/A'
            if sec_per_bar > 1e-6: start_bar_num = round((start_time_abs - trim_offset) / sec_per_bar) + 1; start_bar_str = str(start_bar_num)
            row_tag = self.color_tags.get(color_hex, self.color_tags['default'])
            color_square = '■'
            try:
                self.tree.insert('', tk.END, iid=i, values=(color_square, i + 1, start_bar_str, duration_bars, current_label, color_name), tags=(row_tag,))
            except Exception as e:
                print(f"Error inserting row {i} into treeview: {e}")
        self.update_button.config(state=tk.NORMAL); print(f"DEBUG SectionEditor: Populated with {num_sections} sections.")

    def clear(self):
        """Clears the Treeview and disables the update button."""
        print("DEBUG SectionEditor: Clearing.")
        if self.tree and self.tree.winfo_exists(): self.tree.delete(*self.tree.get_children())
        if self.update_button and self.update_button.winfo_exists(): self.update_button.config(state=tk.DISABLED)

    def _on_cell_edit_start(self, event):
        """Handles right-click on the Treeview to trigger the main app's edit pop-up."""
        print(f"--- DEBUG SectionEditor: _on_cell_edit_start triggered by event: {event} ---")
        if not self.app: print("ERROR SectionEditor: Main app reference not set."); return

        item_id_str = self.tree.identify_row(event.y) # Internal ID (0, 1, 2...)
        if not item_id_str: print("DEBUG SectionEditor: Click was not on a valid row."); return

        try:
            section_index = int(item_id_str) # The item ID is the section index
            print(f"DEBUG SectionEditor: Requesting edit pop-up for section index {section_index}")
            # Call the main app's method to show the pop-up
            self.app._show_section_edit_popup(section_index, event)
        except ValueError:
            print(f"Error: Could not convert item ID '{item_id_str}' to section index.")
        except Exception as e:
            print(f"Error calling main app's edit popup: {e}")
            traceback.print_exc()

    def _commit_popup_edit(self, popup, item_id, label_combo, color_combo):
        """Reads values from pop-up, updates Treeview, destroys pop-up. (Called by main app's popup OK button)"""
        new_label = label_combo.get(); new_color_name = color_combo.get()
        print(f"DEBUG SectionEditor: Committing pop-up edit for row {item_id}. Label='{new_label}', Color='{new_color_name}'")
        try:
            # Update the Treeview display immediately
            self.tree.set(item_id, column='Label', value=new_label)
            self.tree.set(item_id, column='Color', value=new_color_name)
            # Update the row tag based on the new color
            new_color_hex = COLOR_NAME_MAP.get(new_color_name, '#808080')
            new_tag = self.color_tags.get(new_color_hex, self.color_tags['default'])
            self.tree.item(item_id, tags=(new_tag,)) # Apply new tag (colors all text)
            print(f"DEBUG SectionEditor: Treeview display updated for row {item_id} with tag '{new_tag}'.")
        except Exception as e:
            print(f"Error updating treeview from pop-up: {e}")
        popup.destroy() # Destroy the pop-up window passed from main app

    def _trigger_apply_edits(self):
        """Reads data from treeview and calls the apply_callback."""
        print("DEBUG SectionEditor: Update button clicked.")
        if not self.apply_callback: print("ERROR SectionEditor: No apply_callback defined."); return
        try:
            all_items = self.tree.get_children()
            if not all_items: print("DEBUG SectionEditor: No items in tree to apply."); return
            new_labels = []; new_colors_hex = []
            for item_id in all_items:
                values = self.tree.item(item_id, 'values')
                if len(values) < 6: continue
                edited_label = values[4]; edited_color_name = values[5]
                if edited_label not in ALLOWED_LABELS: print(f"Warning: Invalid label '{edited_label}' found for item {item_id}. Using fallback 'Body'."); edited_label = "Body"
                color_hex = COLOR_NAME_MAP.get(edited_color_name, '#808080')
                new_labels.append(edited_label); new_colors_hex.append(color_hex)
            print(f"DEBUG SectionEditor: Applying {len(new_labels)} edits via callback.")
            self.apply_callback(new_labels, new_colors_hex)
        except Exception as e:
            print(f"ERROR SectionEditor: Failed during apply edits trigger: {e}")
            traceback.print_exc()
