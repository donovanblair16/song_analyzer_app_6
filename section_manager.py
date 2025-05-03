"""
Section management module for the Audio Analyzer Tool.

This module defines the SectionManager class which handles operations on audio sections
including merging, splitting, shifting, and recalculating features. It works closely
with the main application to modify and update section data.

The class leverages the extract_section_features function from audio_analysis_wrapper
to ensure consistency with the initial analysis and to maintain specialized handling
like the 75% duration calculation for Drop sections.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import traceback
import copy
import math
import scipy.stats
import importlib.util

# Try to import extract_section_features from the wrapper module
try:
    from audio_analysis_wrapper import extract_section_features

    FEATURE_EXTRACTION_SOURCE = "audio_analysis_wrapper"
    print(
        "DEBUG: Successfully imported extract_section_features from audio_analysis_wrapper"
    )
except ImportError as e:
    print(
        f"WARNING: Could not import extract_section_features from audio_analysis_wrapper: {e}"
    )
    extract_section_features = None
    FEATURE_EXTRACTION_SOURCE = "import error"

# Required constants for section operations
MIN_SPLIT_SECTION_DURATION_SEC = (
    1.0  # Minimum section duration allowed after a split (in seconds)
)

# Required constants from main_app.py
MIN_SPLIT_SECTION_DURATION_SEC = (
    1.0  # Minimum section duration allowed after a split (in seconds)
)


class SectionManager:
    """
    Manages operations on audio sections including merging, splitting, shifting, and feature recalculation.

    This class extracts section-related functionality from the main AudioAnalyzerApp class
    to improve modularity and maintainability.

    Attributes:
        app (AudioAnalyzerApp): The parent application instance, used to access UI elements,
                               track data, and other application components.
    """

    def __init__(self, app_instance):
        """Initialize the SectionManager with a reference to the parent application.

        Args:
            app_instance (AudioAnalyzerApp): The parent application instance.
        """
        self.app = app_instance

    def merge_section(self, remove_boundary_index):
        """Merges two adjacent sections by removing the specified boundary.

        Removes the start time at `remove_boundary_index` and the corresponding
        label, color, and feature dictionary at that index. Updates the end time
        and duration of the preceding section. Recalculates features for the
        newly merged section. Clears any existing HMM results. Updates plots
        and UI state.

        Args:
            remove_boundary_index (int): The index of the section start time
                                         (boundary) to remove. This corresponds
                                         to the index of the *second* section
                                         in the pair being merged.

        Note:
            Operates directly on the `self.app.track_data[1]` dictionary.
        """
        print(
            f"DEBUG: Attempting to merge by removing boundary at index {remove_boundary_index}"
        )
        if not self.app.track_data.get(1):
            messagebox.showerror("Merge Error", "No track data loaded.")
            return

        t_data = self.app.track_data[1]
        # Get references to the lists within track_data
        section_starts = t_data.get("section_starts")
        semantic_labels = t_data.get("semantic_labels")
        label_colors = t_data.get("label_colors")
        section_features = t_data.get("section_features")  # List of dictionaries
        # Optional lists
        cluster_labels = t_data.get("cluster_labels")
        labels_before_cleanup = t_data.get("labels_before_cleanup")

        # Validate data structure
        if not all(
            [
                isinstance(l, list)
                for l in [
                    section_starts,
                    semantic_labels,
                    label_colors,
                    section_features,
                ]
            ]
        ):
            messagebox.showerror(
                "Merge Error", "Core section data lists are missing or invalid."
            )
            print("ERROR: Core section data lists missing for merge.")
            return

        num_sections_before_merge = len(section_starts)
        # Validate boundary index
        if (
            remove_boundary_index <= 0
            or remove_boundary_index >= num_sections_before_merge
        ):
            messagebox.showerror(
                "Merge Error",
                f"Invalid boundary index {remove_boundary_index} for merging.",
            )
            print(
                f"ERROR: Invalid boundary index {remove_boundary_index} for merging {num_sections_before_merge} sections."
            )
            return

        # Indices for the sections involved
        keep_section_idx = remove_boundary_index - 1  # The section that will grow
        remove_section_idx = remove_boundary_index  # The section being absorbed

        try:
            # Get times for duration calculation
            kept_start_time = section_features[keep_section_idx]["start_time"]
            removed_end_time = section_features[remove_section_idx]["end_time"]
            new_duration_sec = removed_end_time - kept_start_time
            new_duration_bars = (
                round(new_duration_sec / t_data["seconds_per_bar"])
                if t_data.get("seconds_per_bar", 0) > 0
                else 0
            )

            # --- Remove the boundary and corresponding data ---
            print(
                f"DEBUG: Removing data for section index {remove_section_idx} (boundary index {remove_boundary_index})"
            )
            # Remove the start time that defines the boundary
            del section_starts[remove_boundary_index]
            # Remove the data associated with the second section
            del semantic_labels[remove_section_idx]
            del label_colors[remove_section_idx]
            del section_features[remove_section_idx]
            # Remove from optional lists if they exist and have the correct length
            if cluster_labels and len(cluster_labels) == num_sections_before_merge:
                del cluster_labels[remove_section_idx]
            if (
                labels_before_cleanup
                and len(labels_before_cleanup) == num_sections_before_merge
            ):
                del labels_before_cleanup[remove_section_idx]

            # --- Update the kept section's data ---
            print(
                f"DEBUG: Updating kept section index {keep_section_idx} end time and duration."
            )
            section_features[keep_section_idx]["end_time"] = removed_end_time
            section_features[keep_section_idx]["duration_sec"] = new_duration_sec
            section_features[keep_section_idx]["duration_bars"] = new_duration_bars

            # --- Recalculate features for the merged section ---
            print(
                f"DEBUG: Recalculating features for merged section {keep_section_idx} ({kept_start_time:.2f} - {removed_end_time:.2f})"
            )
            merged_features = self.recalculate_section_features(
                t_data, keep_section_idx
            )
            if merged_features:
                # Update all recalculated keys in the existing feature dict
                feature_keys_to_update = list(merged_features.keys())
                for key in feature_keys_to_update:
                    section_features[keep_section_idx][key] = merged_features[key]
                print(
                    "DEBUG: Features recalculated and updated in section_features list."
                )
            else:
                print(
                    "WARNING: Feature recalculation failed for merged section. Using original features."
                )
                messagebox.showwarning(
                    "Merge Warning",
                    "Feature recalculation failed for the merged section. "
                    + "Some audio features may be inaccurate until the next full analysis.",
                )

            # --- Update indices in subsequent feature dictionaries ---
            # Indices need to be decremented from the removed section onwards
            for i in range(
                keep_section_idx + 1, len(section_features)
            ):  # Start from the one after the kept section
                if "index" in section_features[i]:
                    section_features[i]["index"] -= 1  # Decrement index
                else:
                    # This indicates a potential issue with data consistency
                    print(
                        f"Warning: 'index' key missing in section_features at list index {i} during merge update."
                    )

            # --- Update the main track_data dictionary (redundant but safe) ---
            t_data["section_starts"] = section_starts
            t_data["semantic_labels"] = semantic_labels
            t_data["label_colors"] = label_colors
            t_data["section_features"] = section_features
            if cluster_labels:
                t_data["cluster_labels"] = cluster_labels
            if labels_before_cleanup:
                t_data["labels_before_cleanup"] = labels_before_cleanup

            # --- Clear HMM results as they are now invalid ---
            if "hmm_semantic_labels" in t_data:
                del t_data["hmm_semantic_labels"]
            if "hmm_label_colors" in t_data:
                del t_data["hmm_label_colors"]
            if "hmm_section_starts" in t_data:
                del t_data["hmm_section_starts"]
            self.app.show_hmm_var.set(False)  # Switch back to original view
            self.app.ui_manager.update_hmm_button_state()  # Disable HMM buttons

            print(
                f"DEBUG: Merge successful. Removed section at original index {remove_section_idx}. Updated section at index {keep_section_idx}."
            )
            self.app.status_label.config(text="Sections Merged", foreground="blue")

            # --- Refresh plots and editor ---
            self.app.plot_manager.display_analysis_results()  # Update plots
            self.app.ui_manager.update_save_button_state  # Enable save button

        except IndexError as e:
            messagebox.showerror(
                "Merge Error",
                f"Index error during merge: {e}. Lists might be inconsistent.",
            )
            print(f"ERROR: Index error during merge: {e}")
            traceback.print_exc()
        except Exception as e:
            messagebox.showerror(
                "Merge Error", f"An unexpected error occurred during merge:\n{e}"
            )
            print(f"ERROR: Unexpected error during merge: {e}")
            traceback.print_exc()

    def split_section(self, section_index, split_time):
        """Splits a section at the given absolute time point.

        Modifies the `app.track_data[1]` dictionary by:
        1. Inserting the `split_time` into the `section_starts` list.
        2. Duplicating the label, color, and feature dictionary for the split section.
        3. Updating the end time/duration of the first part of the split section.
        4. Updating the start time/duration/index of the second part (newly inserted section).
        5. Updating the indices of all subsequent sections.
        6. Recalculating features for both newly formed sections.
        7. Clearing any existing HMM results.
        8. Refreshing plots and UI state.

        Args:
            section_index (int): The zero-based index of the section to split.
            split_time (float): The absolute time (in seconds) at which to split.

        Note:
            Operates directly on the `self.app.track_data[1]` dictionary.
        """
        print(
            f"DEBUG: Executing split for section {section_index} at time {split_time:.3f}"
        )
        if not self.app.track_data.get(1):
            return  # Should already be checked
        t_data = self.app.track_data[1]

        # --- Get original data lists ---
        section_starts = t_data.get("section_starts")
        semantic_labels = t_data.get("semantic_labels")
        label_colors = t_data.get("label_colors")
        section_features = t_data.get("section_features")  # List of dictionaries
        # Optional lists that also need modification
        cluster_labels = t_data.get("cluster_labels")
        labels_before_cleanup = t_data.get("labels_before_cleanup")

        # Basic validation
        if not all(
            [
                isinstance(l, list)
                for l in [
                    section_starts,
                    semantic_labels,
                    label_colors,
                    section_features,
                ]
            ]
        ):
            messagebox.showerror(
                "Split Error", "Core section data lists missing or invalid."
            )
            return
        num_sections_before = len(section_starts)
        if not (0 <= section_index < num_sections_before):
            messagebox.showerror(
                "Split Error", f"Invalid section index {section_index} for split."
            )
            return

        try:
            insert_index = (
                section_index + 1
            )  # New boundary/section goes after the current one
            original_start_time = section_starts[section_index]
            original_end_time = section_features[section_index][
                "end_time"
            ]  # Get original end time before modification

            # --- Insert new boundary and duplicate labels/colors ---
            print(f" -> Inserting boundary at {split_time:.3f}s (index {insert_index})")
            section_starts.insert(insert_index, split_time)
            original_label = semantic_labels[
                section_index
            ]  # Label of the section being split
            semantic_labels.insert(insert_index, original_label)  # Duplicate label
            original_color = label_colors[section_index]
            label_colors.insert(insert_index, original_color)  # Duplicate color

            # Duplicate optional list items if they exist and have the correct length
            if cluster_labels and len(cluster_labels) == num_sections_before:
                cluster_labels.insert(insert_index, cluster_labels[section_index])
            if (
                labels_before_cleanup
                and len(labels_before_cleanup) == num_sections_before
            ):
                labels_before_cleanup.insert(
                    insert_index, labels_before_cleanup[section_index]
                )

            # --- Handle section_features list ---
            original_feature_dict = section_features[section_index]
            # Create a deep copy for the second part to avoid modifying shared references later
            new_section_feature_dict = copy.deepcopy(original_feature_dict)

            # 1. Update first part (section_index) - times and durations
            print(f" -> Updating section {section_index} end time to {split_time:.3f}s")
            section_features[section_index]["end_time"] = split_time
            section_features[section_index]["duration_sec"] = (
                split_time - original_start_time
            )
            section_features[section_index]["duration_bars"] = (
                round(
                    section_features[section_index]["duration_sec"]
                    / t_data["seconds_per_bar"]
                )
                if t_data.get("seconds_per_bar", 0) > 0
                else 0
            )
            # Keep original start time and index ('index' key should already be correct)

            # 2. Update second part (newly inserted dict) - times, durations, and index
            print(
                f" -> Creating new section {insert_index} from {split_time:.3f}s to {original_end_time:.3f}s"
            )
            new_section_feature_dict["start_time"] = split_time
            new_section_feature_dict["end_time"] = (
                original_end_time  # End time is the original end time
            )
            new_section_feature_dict["duration_sec"] = original_end_time - split_time
            new_section_feature_dict["duration_bars"] = (
                round(
                    new_section_feature_dict["duration_sec"] / t_data["seconds_per_bar"]
                )
                if t_data.get("seconds_per_bar", 0) > 0
                else 0
            )
            new_section_feature_dict["index"] = (
                insert_index  # Set correct index for the new section
            )
            new_section_feature_dict["original_label"] = (
                original_label  # Ensure label consistency
            )

            # Insert the new feature dictionary into the list at the correct position
            section_features.insert(insert_index, new_section_feature_dict)

            # 3. Update indices for all subsequent feature dictionaries
            print(f" -> Updating indices for sections {insert_index + 1} onwards...")
            for i in range(insert_index + 1, len(section_features)):
                if "index" in section_features[i]:
                    section_features[i]["index"] += 1  # Increment index
                else:
                    # This indicates a potential issue with data consistency
                    print(
                        f"Warning: 'index' key missing in section_features at list index {i} during split index update."
                    )

            # --- Recalculate features for the two new sections ---
            # Track success/failure for warning message
            recalc_success = True

            # Recalculate for the first part (index section_index)
            print(
                f"DEBUG: Recalculating features for split section part 1 (index {section_index})"
            )
            recalculated_part1 = self.recalculate_section_features(
                t_data, section_index
            )
            if recalculated_part1:
                # Update all recalculated keys in the existing feature dict
                feature_keys_to_update = list(recalculated_part1.keys())
                for key in feature_keys_to_update:
                    section_features[section_index][key] = recalculated_part1[key]
                print(" -> Part 1 features updated.")
            else:
                print(" -> WARNING: Feature recalculation failed for part 1.")
                recalc_success = False

            # Recalculate for the second part (index insert_index)
            print(
                f"DEBUG: Recalculating features for split section part 2 (index {insert_index})"
            )
            recalculated_part2 = self.recalculate_section_features(t_data, insert_index)
            if recalculated_part2:
                # Update all recalculated keys in the newly inserted feature dict
                feature_keys_to_update = list(recalculated_part2.keys())
                for key in feature_keys_to_update:
                    section_features[insert_index][key] = recalculated_part2[key]
                print(" -> Part 2 features updated.")
            else:
                print(" -> WARNING: Feature recalculation failed for part 2.")
                recalc_success = False

            # Show warning if either recalculation failed
            if not recalc_success:
                messagebox.showwarning(
                    "Split Warning",
                    "Feature recalculation failed for one or both split sections. "
                    + "Some audio features may be inaccurate until the next full analysis.",
                )

            # --- Update the main track_data dictionary (redundant but safe) ---
            t_data["section_starts"] = section_starts
            t_data["semantic_labels"] = semantic_labels
            t_data["label_colors"] = label_colors
            t_data["section_features"] = section_features
            if cluster_labels:
                t_data["cluster_labels"] = cluster_labels
            if labels_before_cleanup:
                t_data["labels_before_cleanup"] = labels_before_cleanup

            # --- Clear any existing HMM results as they are now invalid ---
            if "hmm_semantic_labels" in t_data:
                del t_data["hmm_semantic_labels"]
            if "hmm_label_colors" in t_data:
                del t_data["hmm_label_colors"]
            if "hmm_section_starts" in t_data:
                del t_data["hmm_section_starts"]
            self.app.show_hmm_var.set(False)  # Switch back to original view
            self.app.ui_manager.update_hmm_button_state()  # Disable HMM buttons

            print(
                f"DEBUG: Split successful. Section {section_index} split into {section_index} and {insert_index}."
            )
            self.app.status_label.config(text="Section Split", foreground="blue")

            # --- Refresh plots and editor ---
            self.app.plot_manager.display_analysis_results()  # Update plots
            self.app.ui_manager.update_save_button_state  # Enable save button

        except Exception as e:
            messagebox.showerror(
                "Split Error", f"An unexpected error occurred during split:\n{e}"
            )
            print(f"ERROR: Unexpected error during split: {e}")
            traceback.print_exc()

    def shift_sections(self, start_bar, shift_bars):
        """
        Applies a shift to section boundaries starting from a specified bar.

        Modifies `section_starts` and updates `start_time`/`end_time` in
        `section_features`. Recalculates features for all affected sections.

        Args:
            start_bar (int): The 1-based bar number from which to start shifting.
            shift_bars (int): The number of bars to shift (positive or negative).
        """
        print(
            f"DEBUG MainApp: Applying shift - Start Bar: {start_bar}, Shift Amount: {shift_bars} bars"
        )
        track_num = 1  # Assuming single track mode
        td = self.app.track_data.get(track_num)
        if not td:
            return  # Should not happen

        # --- Get necessary data ---
        section_starts = td.get("section_starts")
        section_features = td.get("section_features")
        seconds_per_bar = td.get("seconds_per_bar")
        trim_offset = td.get("trim_offset_sec", 0)
        duration_processed = td.get("duration_processed")

        if (
            not all([isinstance(l, list) for l in [section_starts, section_features]])
            or seconds_per_bar is None
            or seconds_per_bar <= 0
        ):
            messagebox.showerror(
                "Shift Error",
                "Cannot apply shift due to missing or invalid track data (starts, features, seconds_per_bar).",
            )
            return

        num_sections = len(section_starts)
        if num_sections == 0:
            messagebox.showinfo("Shift Sections", "No sections found to shift.")
            return

        # --- Calculations ---
        shift_seconds = shift_bars * seconds_per_bar
        # Convert start_bar (1-based) to the absolute time corresponding to the START of that bar
        start_shift_time_abs = (start_bar - 1) * seconds_per_bar + trim_offset

        # Find the index of the first section boundary to shift
        # We shift boundaries that are AT or AFTER the start_shift_time_abs
        start_index = -1
        for i in range(num_sections):
            if (
                section_starts[i] >= start_shift_time_abs - 1e-6
            ):  # Use epsilon for float comparison
                start_index = i
                break

        if start_index == -1:
            messagebox.showinfo(
                "Shift Sections",
                f"No section boundaries found at or after Bar {start_bar} to shift.",
            )
            return

        print(
            f" -> Shifting boundaries from index {start_index} onwards by {shift_seconds:.3f} seconds."
        )

        # --- Apply Shift and Validate ---
        original_section_starts = list(section_starts)  # Keep a copy for validation
        shifted_indices = []

        for i in range(start_index, num_sections):
            new_start_time = section_starts[i] + shift_seconds

            # **Validation 1: Prevent negative start times**
            if new_start_time < 0.0:
                messagebox.showerror(
                    "Shift Error",
                    f"Shifting boundary {i+1} by {shift_bars} bars would result in a negative start time ({new_start_time:.2f}s). Shift cancelled.",
                )
                return  # Abort shift

            # **Validation 2: Prevent overlaps (simple check: shifted time must be < next original time)**
            # This check is tricky because the next boundary also shifts. A better check might be
            # ensuring the order remains the same, which adding a constant value guarantees,
            # unless the shift is so large it goes past the track duration.
            # We will check against track duration later.

            section_starts[i] = new_start_time
            shifted_indices.append(i)

        # --- Update section_features times ---
        print(
            f" -> Updating start/end times in section_features for indices {shifted_indices}"
        )
        track_end_time_abs = (
            duration_processed + trim_offset if duration_processed else float("inf")
        )

        for i in range(num_sections):
            # Update start time based on the (potentially shifted) value in section_starts
            section_features[i]["start_time"] = section_starts[i]

            # Update end time based on the start time of the *next* section, or track end
            if i + 1 < num_sections:
                section_features[i]["end_time"] = section_starts[i + 1]
            else:
                # Last section's end time is the track duration (absolute)
                section_features[i]["end_time"] = track_end_time_abs

            # **Validation 3: Check end time doesn't exceed track duration (allow slight tolerance)**
            if section_features[i]["end_time"] > track_end_time_abs + 1e-6:
                messagebox.showwarning(
                    "Shift Warning",
                    f"Shift resulted in section {i+1} end time ({section_features[i]['end_time']:.2f}s) potentially exceeding track duration ({track_end_time_abs:.2f}s). Check results carefully.",
                )
                # Optionally clamp the end time?
                # section_features[i]['end_time'] = track_end_time_abs

            # Update duration (should remain constant, but recalculate for safety)
            section_features[i]["duration_sec"] = (
                section_features[i]["end_time"] - section_features[i]["start_time"]
            )
            section_features[i]["duration_bars"] = (
                round(section_features[i]["duration_sec"] / seconds_per_bar)
                if seconds_per_bar > 0
                else 0
            )

            # Ensure duration isn't negative due to potential clamping/float issues
            if section_features[i]["duration_sec"] < 0:
                print(
                    f"Warning: Negative duration calculated for section {i}. Clamping to 0."
                )
                section_features[i]["duration_sec"] = 0
                section_features[i]["duration_bars"] = 0

        # --- Recalculate Features for shifted sections ---
        print(
            f" -> Recalculating features for shifted section indices: {shifted_indices}"
        )
        recalc_failed = False
        recalc_success_count = 0
        recalc_fail_count = 0

        for idx in shifted_indices:
            recalculated_dict = self.recalculate_section_features(td, idx)
            if recalculated_dict:
                # Update features in the list
                feature_keys_to_update = list(recalculated_dict.keys())
                for key in feature_keys_to_update:
                    section_features[idx][key] = recalculated_dict[key]
                recalc_success_count += 1
            else:
                print(
                    f" -> WARNING: Feature recalculation failed for shifted section index {idx}."
                )
                recalc_failed = True
                recalc_fail_count += 1

        # Show appropriate warning if any recalculations failed
        if recalc_failed:
            total_sections = len(shifted_indices)
            messagebox.showwarning(
                "Shift Warning",
                f"Feature recalculation failed for {recalc_fail_count} of {total_sections} shifted sections. "
                + "Some audio features may be inaccurate until the next full analysis.",
            )
            print(
                f" -> Recalculation summary: {recalc_success_count} succeeded, {recalc_fail_count} failed"
            )

        # --- Update track_data (redundant but safe) ---
        td["section_starts"] = section_starts
        td["section_features"] = section_features

        # --- Clear HMM results ---
        if "hmm_semantic_labels" in td:
            del td["hmm_semantic_labels"]
        if "hmm_label_colors" in td:
            del td["hmm_label_colors"]
        if "hmm_section_starts" in td:
            del td["hmm_section_starts"]
        self.app.show_hmm_var.set(False)
        self.app.ui_manager.update_hmm_button_state()

        # --- Update UI ---
        print(" -> Refreshing plots and editor...")
        self.app.plot_manager.display_analysis_results()  # This also updates the editor
        self.app.status_label.config(
            text=f"Sections shifted by {shift_bars} bars from Bar {start_bar}.",
            foreground="blue",
        )
        self.app.ui_manager.update_save_button_state
        print("--- Section Shift Complete ---")

    def validate_split_time(self, section_index, split_time):
        """Validates if a given time is valid for splitting a section.

        Checks if the split time is within the section's boundaries and if the
        resulting sections would meet the minimum duration requirement.

        Args:
            section_index (int): The zero-based index of the section to check.
            split_time (float): The absolute time point (in seconds) for the potential split.

        Returns:
            tuple: (is_valid, error_message) where is_valid is a boolean and
                   error_message is a string (None if valid).
        """
        t_data = self.app.track_data.get(1)
        if not t_data:
            return False, "No track data loaded."

        section_starts = t_data.get("section_starts")
        num_sections = len(section_starts) if section_starts else 0

        if not (0 <= section_index < num_sections):
            return False, f"Invalid section index {section_index}."

        original_start_time = section_starts[section_index]
        # Calculate original end time carefully
        original_end_time = (
            section_starts[section_index + 1]
            if section_index + 1 < num_sections
            else t_data.get("duration_processed", float("inf"))
            + t_data.get("trim_offset_sec", 0)
        )

        # Use a small epsilon for time comparisons to avoid floating point issues near boundaries
        epsilon = 1e-6
        # Check if split time is strictly within the section
        if not (
            original_start_time + epsilon < split_time < original_end_time - epsilon
        ):
            return (
                False,
                f"Split time {split_time:.3f}s must be strictly within the section boundaries ({original_start_time:.3f}s - {original_end_time:.3f}s).",
            )

        # Check if resulting sections meet minimum duration
        if (split_time - original_start_time < MIN_SPLIT_SECTION_DURATION_SEC) or (
            original_end_time - split_time < MIN_SPLIT_SECTION_DURATION_SEC
        ):
            return (
                False,
                f"Resulting sections would be shorter than minimum duration ({MIN_SPLIT_SECTION_DURATION_SEC:.1f}s).",
            )

        return True, None

    def recalculate_section_features(self, track_data, section_idx):
        """Recalculates features for a specific section after a merge, split or shift operation.

        Uses the extract_section_features function from audio_analysis_wrapper module to ensure
        consistency with the initial analysis. This leverages special handling in the original
        function, such as the 75% duration calculation for Drop sections.

        Args:
            track_data (dict): The main track data dictionary containing all audio analysis data.
            section_idx (int): The zero-based index within the section_features list to recalculate.

        Returns:
            dict or None: A dictionary containing the recalculated feature values or None if
                         the recalculation fails.
        """
        print(
            f"DEBUG SectionManager: Recalculating features for section index {section_idx}"
        )

        try:
            # --- Validate input data ---
            section_features_list = track_data.get("section_features", [])
            semantic_labels_list = track_data.get("semantic_labels", [])

            if not (0 <= section_idx < len(section_features_list)):
                print(f"ERROR Recalc: Invalid section index {section_idx}")
                return None

            if not (0 <= section_idx < len(semantic_labels_list)):
                print(
                    f"ERROR Recalc: Semantic labels list doesn't contain index {section_idx}"
                )
                return None

            # --- Try to use the extract_section_features function ---
            if extract_section_features is not None:
                print(
                    f"DEBUG Recalc: Using extract_section_features from {FEATURE_EXTRACTION_SOURCE}"
                )

                # Create a temporary track_data with only the section we want to recalculate
                temp_track_data = track_data.copy()
                temp_track_data["section_features"] = [
                    track_data["section_features"][section_idx]
                ]
                temp_track_data["semantic_labels"] = [
                    track_data["semantic_labels"][section_idx]
                ]

                try:
                    # Run the extraction on our temporary data
                    updated_features_list, _ = extract_section_features(temp_track_data)

                    if updated_features_list and len(updated_features_list) > 0:
                        # Success - return the recalculated features
                        print(
                            f"DEBUG Recalc: Successfully recalculated features for section {section_idx}"
                        )
                        return updated_features_list[0]
                    else:
                        print(
                            f"WARNING Recalc: extract_section_features returned no data for section {section_idx}"
                        )
                except Exception as feature_error:
                    print(
                        f"ERROR Recalc: extract_section_features failed: {feature_error}"
                    )
                    traceback.print_exc()
            else:
                print("WARNING Recalc: extract_section_features function not available")

            # --- FALLBACK: If we can't use the feature extraction module, use basic recalculation ---
            print(
                "WARNING Recalc: Falling back to basic feature recalculation (less accurate)"
            )

            # Get required section info
            section_info = section_features_list[section_idx]
            start_time_abs = section_info.get("start_time")
            end_time_abs = section_info.get("end_time")

            # Base data from track_data
            trim_offset = track_data.get("trim_offset_sec", 0)
            duration_processed = track_data.get("duration_processed")

            # Calculate basic features
            basic_features = {}

            # 1. Relative position (basic feature that most sections need)
            rel_pos = np.nan
            if (
                start_time_abs is not None
                and duration_processed is not None
                and duration_processed > 0
            ):
                rel_pos = max(0.0, min(start_time_abs / duration_processed, 1.0))
            basic_features["relative_position"] = rel_pos

            # 2. Set defaults for essential features to prevent errors
            basic_features["avg_rms"] = 0.0
            basic_features["peak_rms"] = 0.0
            basic_features["rms_std_dev"] = 0.0
            basic_features["rms_trend"] = 0.0
            basic_features["spectral_centroid_avg"] = 0.0
            basic_features["delta_rms"] = 0.0
            basic_features["delta_centroid"] = 0.0

            print(
                "WARNING Recalc: Returning limited features. Full recalculation failed."
            )
            return basic_features

        except Exception as e:
            print(
                f"ERROR Recalc: Unhandled exception in recalculate_section_features: {e}"
            )
            traceback.print_exc()
            return None
