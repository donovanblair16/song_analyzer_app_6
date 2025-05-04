import tkinter as tk
from tkinter import filedialog, scrolledtext
import joblib
import os
import numpy as np
import pandas as pd
from tkinter import ttk


class FeatureInspectorTool:
    def __init__(self, master):
        self.master = master
        master.title("Section Features Inspector")
        master.geometry("1000x800")

        # Create main frames
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)

        self.content_frame = tk.Frame(master)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Add file selection components
        self.file_button = tk.Button(
            self.top_frame, text="Select Joblib File", command=self.select_file
        )
        self.file_button.pack(side=tk.LEFT, padx=5)

        self.file_path_var = tk.StringVar()
        self.file_path_entry = tk.Entry(
            self.top_frame, textvariable=self.file_path_var, width=80
        )
        self.file_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.inspect_button = tk.Button(
            self.top_frame, text="Inspect Features", command=self.inspect_features
        )
        self.inspect_button.pack(side=tk.LEFT, padx=5)

        # Create notebook for different views
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.summary_tab = tk.Frame(self.notebook)
        self.features_tab = tk.Frame(self.notebook)
        self.details_tab = tk.Frame(self.notebook)

        self.notebook.add(self.summary_tab, text="Summary")
        self.notebook.add(self.features_tab, text="Features Table")
        self.notebook.add(self.details_tab, text="Feature Details")

        # Set up summary tab
        self.summary_text = scrolledtext.ScrolledText(self.summary_tab, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Set up features table tab
        self.features_frame = tk.Frame(self.features_tab)
        self.features_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create Treeview for features
        self.tree_frame = tk.Frame(self.features_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree_scroll_y = tk.Scrollbar(self.tree_frame, orient="vertical")
        self.tree_scroll_x = tk.Scrollbar(self.tree_frame, orient="horizontal")
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.features_tree = ttk.Treeview(
            self.tree_frame,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set,
        )
        self.features_tree.pack(fill=tk.BOTH, expand=True)

        self.tree_scroll_y.config(command=self.features_tree.yview)
        self.tree_scroll_x.config(command=self.features_tree.xview)

        # Set up feature details tab
        self.details_text = scrolledtext.ScrolledText(self.details_tab, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(
            master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Data storage
        self.track_data = None

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Joblib File",
            filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")],
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"File selected: {os.path.basename(file_path)}")

    def inspect_features(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            self.status_var.set("Error: Please select a valid file.")
            return

        try:
            self.status_var.set(f"Loading file: {os.path.basename(file_path)}...")
            self.master.update_idletasks()

            # Load the joblib file
            self.track_data = joblib.load(file_path)

            # Update the UI with data
            self.update_summary_tab()
            self.update_features_tab()
            self.update_details_tab()

            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()

    def update_summary_tab(self):
        """Update the summary tab with overview information."""
        if not self.track_data:
            return

        self.summary_text.delete(1.0, tk.END)

        # Check if it's a dictionary
        if not isinstance(self.track_data, dict):
            self.summary_text.insert(
                tk.END,
                f"Data is not a dictionary but {type(self.track_data).__name__}\n",
            )
            return

        # Get basic info
        self.summary_text.insert(tk.END, "=== TRACK DATA SUMMARY ===\n\n")

        # List all top-level keys
        self.summary_text.insert(
            tk.END, f"Top-level keys ({len(self.track_data.keys())}):\n"
        )
        for key in sorted(self.track_data.keys()):
            value = self.track_data[key]
            if isinstance(value, (list, np.ndarray)):
                self.summary_text.insert(
                    tk.END, f"• {key}: {type(value).__name__} with {len(value)} items\n"
                )
            else:
                self.summary_text.insert(tk.END, f"• {key}: {type(value).__name__}\n")

        # Check for section_features specifically
        self.summary_text.insert(tk.END, "\n=== SECTION FEATURES ===\n")
        if "section_features" not in self.track_data:
            self.summary_text.insert(tk.END, "No 'section_features' found in data.\n")
            return

        section_features = self.track_data["section_features"]
        if not isinstance(section_features, list):
            self.summary_text.insert(
                tk.END,
                f"'section_features' is not a list but {type(section_features).__name__}\n",
            )
            return

        self.summary_text.insert(tk.END, f"Found {len(section_features)} sections\n\n")

        # Find all unique keys across all sections
        all_feature_keys = set()
        for section in section_features:
            if isinstance(section, dict):
                all_feature_keys.update(section.keys())

        # Display all feature keys
        self.summary_text.insert(
            tk.END, f"All feature keys across sections ({len(all_feature_keys)}):\n"
        )
        for key in sorted(all_feature_keys):
            self.summary_text.insert(tk.END, f"• {key}\n")

        # Specifically check for relative_rms
        self.summary_text.insert(tk.END, "\n=== RELATIVE RMS CHECK ===\n")
        if "relative_rms" in all_feature_keys:
            self.summary_text.insert(
                tk.END, "✓ 'relative_rms' key IS PRESENT in section features\n"
            )

            # Check if any sections have relative_rms defined
            sections_with_rms = sum(
                1
                for s in section_features
                if isinstance(s, dict) and "relative_rms" in s
            )
            self.summary_text.insert(
                tk.END,
                f"✓ {sections_with_rms}/{len(section_features)} sections have 'relative_rms' defined\n",
            )

            # Sample some values
            self.summary_text.insert(tk.END, "\nSample relative_rms values:\n")
            for i, section in enumerate(section_features[:5]):
                if isinstance(section, dict) and "relative_rms" in section:
                    val = section["relative_rms"]
                    self.summary_text.insert(tk.END, f"Section {i}: {val}\n")
        else:
            self.summary_text.insert(
                tk.END, "✗ 'relative_rms' key is NOT present in any section features\n"
            )

    def update_features_tab(self):
        """Create a table showing all sections and their features."""
        if not self.track_data:
            return

        # Clear existing table
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)

        # Check for section features
        if "section_features" not in self.track_data or not isinstance(
            self.track_data["section_features"], list
        ):
            return

        section_features = self.track_data["section_features"]
        if not section_features:
            return

        # Get labels if available
        semantic_labels = self.track_data.get("semantic_labels", [])

        # Get all unique keys across sections
        all_keys = set()
        for section in section_features:
            if isinstance(section, dict):
                all_keys.update(section.keys())

        # Create columns
        columns = ["#", "Label", "Start", "End"]
        feature_columns = sorted(
            [
                k
                for k in all_keys
                if k
                not in [
                    "index",
                    "start_time",
                    "end_time",
                    "duration_sec",
                    "duration_bars",
                ]
            ]
        )
        all_columns = columns + feature_columns

        # Configure tree
        self.features_tree["columns"] = all_columns
        self.features_tree.column("#0", width=0, stretch=tk.NO)

        for col in all_columns:
            width = 100
            if col == "#":
                width = 40
            self.features_tree.column(col, width=width, anchor=tk.W)
            self.features_tree.heading(col, text=col)

        # Add rows
        for i, section in enumerate(section_features):
            if not isinstance(section, dict):
                continue

            values = [
                i + 1,
                semantic_labels[i] if i < len(semantic_labels) else "Unknown",
                section.get("start_time", "N/A"),
                section.get("end_time", "N/A"),
            ]

            # Add feature columns
            for key in feature_columns:
                val = section.get(key, "N/A")
                if isinstance(val, (float, np.float32, np.float64)):
                    if np.isnan(val):
                        values.append("NaN")
                    else:
                        values.append(f"{val:.4f}")
                else:
                    values.append(str(val))

            self.features_tree.insert("", tk.END, values=values)

    def update_details_tab(self):
        """Show detailed examination of key features."""
        if not self.track_data:
            return

        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "=== FEATURE DETAILS ===\n\n")

        # Check for section features
        if "section_features" not in self.track_data or not isinstance(
            self.track_data["section_features"], list
        ):
            self.details_text.insert(tk.END, "No valid section features found.\n")
            return

        section_features = self.track_data["section_features"]
        if not section_features:
            self.details_text.insert(tk.END, "Section features list is empty.\n")
            return

        # Focus on specific keys including relative_rms
        key_check = ["relative_rms", "avg_rms", "peak_rms", "relative_position"]

        self.details_text.insert(
            tk.END, "Checking for important keys in each section:\n\n"
        )

        # Print detailed info for first 5 sections
        semantic_labels = self.track_data.get("semantic_labels", [])

        for i, section in enumerate(section_features[:5]):
            if not isinstance(section, dict):
                self.details_text.insert(tk.END, f"Section {i} is not a dictionary\n")
                continue

            label = semantic_labels[i] if i < len(semantic_labels) else "Unknown"
            self.details_text.insert(tk.END, f"Section {i} (Label: {label}):\n")

            # Check each important key
            for key in key_check:
                if key in section:
                    val = section[key]
                    self.details_text.insert(tk.END, f"  • {key}: {val}\n")
                else:
                    self.details_text.insert(tk.END, f"  • {key}: NOT FOUND\n")

            # Print out all keys in this section
            self.details_text.insert(tk.END, f"\n  All keys ({len(section.keys())}):\n")
            for k in sorted(section.keys()):
                val = section[k]
                if isinstance(val, (float, np.float32, np.float64)):
                    if np.isnan(val):
                        val_str = "NaN"
                    else:
                        val_str = f"{val:.4f}"
                else:
                    val_str = str(val)

                # Truncate very long values
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."

                self.details_text.insert(tk.END, f"    - {k}: {val_str}\n")

            self.details_text.insert(tk.END, "\n" + "-" * 50 + "\n\n")

        # Analyze all sections for relative_rms problems
        self.details_text.insert(
            tk.END, "=== RELATIVE RMS ANALYSIS (ALL SECTIONS) ===\n\n"
        )

        # Check if all sections have relative_rms
        sections_with_rms = sum(
            1 for s in section_features if isinstance(s, dict) and "relative_rms" in s
        )
        self.details_text.insert(
            tk.END,
            f"{sections_with_rms}/{len(section_features)} sections have relative_rms defined\n",
        )

        # Check for sections missing relative_rms
        if sections_with_rms < len(section_features):
            self.details_text.insert(tk.END, "\nSections missing relative_rms:\n")
            for i, section in enumerate(section_features):
                if isinstance(section, dict) and "relative_rms" not in section:
                    label = (
                        semantic_labels[i] if i < len(semantic_labels) else "Unknown"
                    )
                    self.details_text.insert(
                        tk.END, f"• Section {i} (Label: {label})\n"
                    )

        # Statistics on relative_rms values
        rms_values = [
            s.get("relative_rms")
            for s in section_features
            if isinstance(s, dict) and "relative_rms" in s
        ]

        finite_values = [
            v
            for v in rms_values
            if isinstance(v, (float, np.float32, np.float64)) and np.isfinite(v)
        ]
        nan_values = [
            v
            for v in rms_values
            if isinstance(v, (float, np.float32, np.float64)) and np.isnan(v)
        ]

        self.details_text.insert(tk.END, f"\nRelative RMS statistics:\n")
        self.details_text.insert(tk.END, f"• Total values: {len(rms_values)}\n")
        self.details_text.insert(tk.END, f"• Finite values: {len(finite_values)}\n")
        self.details_text.insert(tk.END, f"• NaN values: {len(nan_values)}\n")

        if finite_values:
            self.details_text.insert(tk.END, f"• Min value: {min(finite_values):.4f}\n")
            self.details_text.insert(tk.END, f"• Max value: {max(finite_values):.4f}\n")
            self.details_text.insert(
                tk.END, f"• Mean value: {np.mean(finite_values):.4f}\n"
            )


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureInspectorTool(root)
    root.mainloop()
