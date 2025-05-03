import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import joblib
import numpy as np
import traceback
from collections import defaultdict


class TransitionFinderApp:
    """
    A GUI application to scan analysis files for specific label transitions
    (e.g., Outro -> Breakdown).
    """

    def __init__(self, master):
        self.master = master
        master.title("Analysis Transition Finder")
        master.geometry("700x550")  # Adjusted size

        # --- State Variables ---
        self.folder_path = tk.StringVar()
        self.all_labels = []  # List of unique labels found in data
        self.from_label_var = tk.StringVar()
        self.to_label_var = tk.StringVar()

        # --- Build GUI ---
        self.create_widgets()

    def create_widgets(self):
        """Creates and lays out the GUI widgets."""
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Controls Frame ---
        controls_frame = ttk.LabelFrame(
            main_frame, text="Scan Configuration", padding="10"
        )
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)  # Allow folder path label to expand
        controls_frame.columnconfigure(
            3, weight=1
        )  # Allow To label combo to expand slightly

        # Folder Selection
        ttk.Button(
            controls_frame, text="Select Folder...", command=self.select_folder
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(
            controls_frame, textvariable=self.folder_path, state="readonly", width=60
        ).grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Transition Selection
        ttk.Label(controls_frame, text="Find Transition:").grid(
            row=1, column=0, padx=(5, 0), pady=10, sticky="e"
        )
        self.from_label_combo = ttk.Combobox(
            controls_frame, textvariable=self.from_label_var, state="readonly", width=20
        )
        self.from_label_combo.grid(row=1, column=1, padx=(5, 0), pady=10, sticky="w")

        ttk.Label(controls_frame, text="->").grid(
            row=1, column=2, padx=5, pady=10
        )  # Arrow label

        self.to_label_combo = ttk.Combobox(
            controls_frame, textvariable=self.to_label_var, state="readonly", width=20
        )
        self.to_label_combo.grid(row=1, column=3, padx=(0, 5), pady=10, sticky="w")

        # Scan Button
        self.scan_button = ttk.Button(
            controls_frame,
            text="Scan for Transition",
            command=self.run_scan,
            state=tk.DISABLED,
        )
        self.scan_button.grid(row=2, column=0, columnspan=4, pady=10)

        # --- Results Frame ---
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Use Treeview for structured results
        columns = ("filename", "index", "from_label", "to_label")
        self.results_tree = ttk.Treeview(
            results_frame, columns=columns, show="headings", height=15
        )

        # Define headings
        self.results_tree.heading("filename", text="Filename")
        self.results_tree.heading("index", text="Index (of From Label)")
        self.results_tree.heading("from_label", text="From Label")
        self.results_tree.heading("to_label", text="To Label")

        # Define column widths and alignment
        self.results_tree.column("filename", width=350, anchor=tk.W)
        self.results_tree.column("index", width=100, anchor=tk.CENTER)
        self.results_tree.column("from_label", width=100, anchor=tk.W)
        self.results_tree.column("to_label", width=100, anchor=tk.W)

        # Add scrollbars
        vsb = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.results_tree.yview
        )
        hsb = ttk.Scrollbar(
            results_frame, orient="horizontal", command=self.results_tree.xview
        )
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout for treeview and scrollbars
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Configure resizing behavior
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Select a folder to begin.")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_folder(self):
        """Opens a dialog to select the folder containing analysis files."""
        # Suggest starting in the parent directory of the previous default
        initial_dir = os.path.dirname(
            os.path.dirname(
                "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"
            )
        )
        selected_path = filedialog.askdirectory(
            title="Select Folder Containing Analysis Files (e.g., 'Perfect')",
            initialdir=initial_dir,
        )
        if selected_path:
            self.folder_path.set(selected_path)
            self.status_var.set(
                f"Folder selected: {os.path.basename(selected_path)}. Extracting labels..."
            )
            self.master.update_idletasks()  # Update GUI immediately
            self.extract_labels()
            self.scan_button.config(state=tk.NORMAL)  # Enable scan button
            self.status_var.set(
                f"Ready to scan folder: {os.path.basename(selected_path)}"
            )
        else:
            self.status_var.set("Folder selection cancelled.")

    def extract_labels(self):
        """Scans the first valid joblib file to get available labels."""
        folder = self.folder_path.get()
        if not folder or not os.path.isdir(folder):
            self.status_var.set("Error: Invalid folder path.")
            return

        self.all_labels = set()  # Use set for unique labels

        try:
            found_file = False
            for filename in os.listdir(folder):
                if filename.endswith(".joblib"):
                    file_path = os.path.join(folder, filename)
                    try:
                        track_data = joblib.load(file_path)
                        if isinstance(track_data, dict):
                            # Get unique labels
                            semantic_labels = track_data.get("semantic_labels")
                            if semantic_labels and isinstance(
                                semantic_labels, (list, np.ndarray)
                            ):
                                self.all_labels.update(semantic_labels)
                                found_file = True
                                break  # Stop after processing the first valid file with labels
                    except Exception as e:
                        print(f"Warning: Could not load or process {filename}: {e}")
                        continue  # Try next file

            if not found_file:
                raise FileNotFoundError(
                    "No valid .joblib analysis files with 'semantic_labels' found or processed."
                )

            # Update comboboxes
            self.all_labels = sorted(
                [str(lbl) for lbl in self.all_labels]
            )  # Ensure strings and sort
            self.from_label_combo["values"] = self.all_labels
            self.to_label_combo["values"] = self.all_labels

            # Set default selections if lists are not empty
            if self.all_labels:
                self.from_label_var.set(self.all_labels[0])
                self.to_label_var.set(self.all_labels[0])

            self.status_var.set("Labels extracted. Ready to scan.")

        except Exception as e:
            self.status_var.set(f"Error extracting labels: {e}")
            messagebox.showerror("Error", f"Could not extract labels from folder:\n{e}")
            self.from_label_combo["values"] = []
            self.to_label_combo["values"] = []

    def run_scan(self):
        """Performs the scan based on user selections."""
        folder = self.folder_path.get()
        from_label = self.from_label_var.get()
        to_label = self.to_label_var.get()

        # --- Validate Inputs ---
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder first.")
            return
        if not from_label:
            messagebox.showerror("Error", "Please select a 'From Label'.")
            return
        if not to_label:
            messagebox.showerror("Error", "Please select a 'To Label'.")
            return

        # --- Clear previous results ---
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.status_var.set(
            f"Scanning for '{from_label}' -> '{to_label}' transitions..."
        )
        self.master.update_idletasks()

        # --- Scan Files ---
        found_results = []
        files_scanned = 0
        errors = 0

        try:
            for filename in os.listdir(folder):
                if filename.endswith(".joblib"):
                    files_scanned += 1
                    file_path = os.path.join(folder, filename)
                    try:
                        track_data = joblib.load(file_path)
                        if not isinstance(track_data, dict):
                            continue

                        semantic_labels = track_data.get("semantic_labels")

                        if (
                            semantic_labels is None
                            or not isinstance(semantic_labels, (list, np.ndarray))
                            or len(semantic_labels) < 2
                        ):  # Need at least 2 labels for a transition
                            continue  # Skip files with missing/invalid/short labels

                        # Iterate through labels looking for the transition
                        for i in range(len(semantic_labels) - 1):
                            current_label = str(
                                semantic_labels[i]
                            )  # Ensure string comparison
                            next_label = str(semantic_labels[i + 1])

                            if current_label == from_label and next_label == to_label:
                                found_results.append(
                                    (filename, i, from_label, to_label)
                                )

                    except Exception as load_err:
                        print(f"Error loading/processing {filename}: {load_err}")
                        errors += 1

            # --- Display Results ---
            if found_results:
                for result_item in found_results:
                    self.results_tree.insert("", tk.END, values=result_item)
                self.status_var.set(
                    f"Scan complete. Found {len(found_results)} matching transitions in {files_scanned} files."
                )
            else:
                self.status_var.set(
                    f"Scan complete. No '{from_label}' -> '{to_label}' transitions found in {files_scanned} files."
                )

            if errors > 0:
                messagebox.showwarning(
                    "Scan Warning",
                    f"Encountered errors while processing {errors} files. Check console for details.",
                )

        except Exception as scan_err:
            self.status_var.set(f"Error during scan: {scan_err}")
            messagebox.showerror(
                "Scan Error",
                f"An unexpected error occurred during the scan:\n{scan_err}",
            )
            traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TransitionFinderApp(root)
    root.mainloop()
