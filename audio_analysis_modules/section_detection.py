# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/section_detection.py

"""
Functions for detecting section boundaries in audio tracks.
"""

import numpy as np


def detect_sections(track_data):
    """
    Performs section detection based on RMS energy changes across bars.

    Args:
        track_data (dict): Dictionary containing 'bar_rms_data', 'bar_starts_absolute',
                           'trim_offset_sec'.

    Returns:
        tuple: (list of section start times (absolute), list of initial section labels ('Start', 'S1', 'S2'...))
               Returns ([], []) on failure or if input data is missing.
    """
    # Retrieve necessary data
    bar_rms_data = track_data.get("bar_rms_data")
    bar_starts = track_data.get("bar_starts_absolute")  # Absolute times
    trim_offset = track_data.get("trim_offset_sec", 0)

    # Validate inputs
    if bar_rms_data is None or bar_starts is None:
        print("Warning: Missing bar RMS or start times for section detection.")
        return [], []
    num_bars = len(bar_rms_data)
    if num_bars <= 0:
        print("Warning: No bars found for section detection.")
        return [], []
    if len(bar_starts) != num_bars:
        print(
            f"Warning: Mismatch between bar_starts ({len(bar_starts)}) and bar_rms ({num_bars})."
        )
        # Decide how to handle - return empty or try to proceed? Returning empty for safety.
        return [], []

    # Initialize lists for results
    section_starts = []
    section_labels = []
    section_index = 1  # Start with S1
    drop_rms_reference = []  # Store RMS of high-energy sections to adapt fill threshold

    i = 0  # Bar index
    while i < num_bars:
        # Check bounds
        if i >= len(bar_rms_data):
            break

        # Update reference RMS for drop sections
        current_rms = bar_rms_data[i]
        if (
            np.isfinite(current_rms) and current_rms > 0.35
        ):  # Threshold to consider it high energy
            drop_rms_reference.append(current_rms)
        # Use average of recent high-energy sections, or default if none found yet
        drop_avg = np.mean(drop_rms_reference) if drop_rms_reference else 0.35

        # --- Fill Detection Logic ---
        matched_fill = False
        # Look ahead for potential fills of different lengths (e.g., 4, 3, 2, 1 bars)
        for fill_length in [4, 3, 2, 1]:
            rebound_idx = (
                i + fill_length
            )  # Index where energy should rebound after fill
            if rebound_idx < num_bars:
                fill_indices = range(i, i + fill_length)
                # Check if all bars in the potential fill have low RMS relative to drop average
                is_fill = all(
                    idx < len(bar_rms_data)
                    and np.isfinite(bar_rms_data[idx])
                    and bar_rms_data[idx] < 0.65 * drop_avg
                    for idx in fill_indices
                )
                # Check if the bar immediately after the fill rebounds in energy
                rebounds = (
                    rebound_idx < len(bar_rms_data)
                    and np.isfinite(bar_rms_data[rebound_idx])
                    and bar_rms_data[rebound_idx] > 0.9 * drop_avg
                )
                if is_fill and rebounds:
                    # Found a fill section
                    section_starts.append(bar_starts[i])  # Start of the fill
                    section_labels.append("Fill")
                    # Start a new section after the fill (where energy rebounds)
                    section_starts.append(bar_starts[rebound_idx])
                    section_labels.append(f"S{section_index}")
                    section_index += 1
                    # Advance main loop index past the fill and a buffer (e.g., 8 bars)
                    i += fill_length + 8
                    matched_fill = True
                    break  # Stop checking shorter fill lengths once a match is found
        if matched_fill:
            continue  # Continue to next iteration of the main while loop

        # --- Standard Section Boundary Detection (if no fill found) ---
        # Look at energy change over an 8-bar window (4 bars vs next 4 bars)
        if i + 8 <= num_bars:
            slice1 = bar_rms_data[i : min(i + 4, num_bars)]
            slice2 = bar_rms_data[min(i + 4, num_bars) : min(i + 8, num_bars)]
            # Calculate mean RMS for each slice, ignoring NaNs/Infs
            slice1_rms = [val for val in slice1 if np.isfinite(val)]
            slice2_rms = [val for val in slice2 if np.isfinite(val)]
            e1 = np.mean(slice1_rms) if slice1_rms else 0.0
            e2 = np.mean(slice2_rms) if slice2_rms else 0.0
            # Calculate relative change, handle division by zero
            delta = abs(e2 - e1) / e1 if e1 > 1e-9 else (1.0 if abs(e2) > 1e-9 else 0.0)

            # Add start of this 8-bar block if it's a new section start
            if (
                not section_starts or abs(bar_starts[i] - section_starts[-1]) > 1e-3
            ):  # Avoid duplicate starts
                section_starts.append(bar_starts[i])
                section_labels.append(f"S{section_index}")
                section_index += 1

            # If significant energy change occurs after 4 bars, add another boundary
            if delta >= 0.5 and (i + 4 < num_bars):
                if (
                    not section_starts
                    or abs(bar_starts[i + 4] - section_starts[-1]) > 1e-3
                ):
                    section_starts.append(bar_starts[i + 4])
                    section_labels.append(f"S{section_index}")
                    section_index += 1
            # Advance by the window size
            i += 8
        else:
            # Handle remaining bars at the end
            if i < num_bars:
                # Add start if it's a new section
                if not section_starts or abs(bar_starts[i] - section_starts[-1]) > 1e-3:
                    section_starts.append(bar_starts[i])
                    section_labels.append(f"S{section_index}")
                    # No need to increment section_index here, it's the last one
            break  # Exit loop, no more full windows to check

    # --- Post-processing ---
    # Sort sections by start time and remove duplicates (just in case)
    if section_starts:
        section_starts_np = np.array(section_starts)
        section_labels_np = np.array(section_labels)
        sorted_indices = np.argsort(section_starts_np)
        section_starts_sorted = section_starts_np[sorted_indices]
        section_labels_sorted = section_labels_np[sorted_indices]
        # Find unique start times and keep the first label associated with them
        unique_starts, unique_indices = np.unique(
            section_starts_sorted, return_index=True
        )
        section_starts = list(unique_starts)
        section_labels = list(section_labels_sorted[unique_indices])

    # Ensure the very first section starts at the beginning (trim_offset) and label it 'Start'
    if (
        not section_starts or section_starts[0] > trim_offset + 1e-3
    ):  # If no sections or first starts too late
        section_starts.insert(0, trim_offset)
        section_labels.insert(0, "Start")
    elif (
        section_starts and section_labels
    ):  # If sections exist, ensure first is 'Start'
        if abs(section_starts[0] - trim_offset) < 1e-3:
            section_labels[0] = "Start"
        else:  # If first section doesn't start near trim_offset, insert 'Start'
            section_starts.insert(0, trim_offset)
            section_labels.insert(0, "Start")

    return section_starts, section_labels
