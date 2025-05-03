# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/feature_extraction.py

"""Functions for feature extraction from audio sections.

MODIFIED: For sections labeled "Drop", RMS-based features (avg_rms, relative_rms,
          rms_std_dev_section, rms_trend) are calculated using only the
          first 75% of the section's duration to mitigate the effect of quieter tails.
ADDED: position_context feature calculation.
"""

import numpy as np
import scipy.stats
import librosa
import traceback  # Added for more detailed error printing
from collections import Counter  # Added for label_proportion calculation


def extract_section_features(track_data):
    """Retrieves section features (as a list of dictionaries) and semantic labels
    from the track_data. Calculates and adds features like 'relative_position',
    'low_energy_norm', standard deviations, delta features, crest factor,
    spectral slope, 'relative_rms', 'rms_trend', 'position_context', AND
    'label_proportion'.

    MODIFICATION: RMS features for "Drop" sections use only the first 75% duration.

    Args:
        track_data (dict): Dictionary loaded from analysis file, expected
                           to contain base data like 'section_features',
                           'semantic_labels', 'rms', 'rms_times', etc.

    Returns:
        tuple: A tuple containing (section_features_list_of_dicts, semantic_labels_list).
               The first element is a list of dictionaries with calculated features.
               The second element is a list of corresponding semantic labels.
               Returns ([], []) if essential data is missing or mismatched.

    """

    # --- Retrieve necessary base data ---
    section_features_list_of_dicts = track_data.get("section_features", [])
    semantic_labels_list = track_data.get("semantic_labels", [])
    duration_processed = track_data.get("duration_processed")
    low_energy_norm = track_data.get("low_energy_norm")
    low_energy_times = track_data.get("low_energy_times")
    trim_offset_sec = track_data.get("trim_offset_sec", 0)
    rms_frames = track_data.get("rms")
    rms_times = track_data.get("rms_times")
    spectral_centroid_frames = track_data.get("spectral_centroid_frames")
    times_absolute = track_data.get(
        "times_absolute"
    )  # Absolute times for spectral frames

    # --- Initial Validation ---
    print(f"\n=== DEBUG: Inside extract_section_features ===")
    valid_input = True
    if (
        not isinstance(section_features_list_of_dicts, list)
        or not section_features_list_of_dicts
    ):
        print(" -> ERROR: 'section_features' missing, empty, or not a list.")
        valid_input = False
    if not isinstance(semantic_labels_list, list):
        print(" -> ERROR: 'semantic_labels' is not a list.")
        valid_input = False
    if valid_input and len(section_features_list_of_dicts) != len(semantic_labels_list):
        print(
            f" -> ERROR: Mismatched section data lengths. Features: {len(section_features_list_of_dicts)}, Labels: {len(semantic_labels_list)}."
        )
        valid_input = False
    if duration_processed is None or duration_processed <= 0:
        print(" -> WARNING: 'duration_processed' missing or invalid.")

    if rms_frames is None or rms_times is None:
        print(
            " -> FATAL ERROR: Frame-based 'rms' or 'rms_times' data is MISSING. Cannot calculate RMS features."
        )
        return [], []
    if (
        not isinstance(rms_frames, np.ndarray)
        or not isinstance(rms_times, np.ndarray)
        or rms_frames.shape != rms_times.shape
    ):
        print(
            f" -> FATAL ERROR: Invalid 'rms' ({rms_frames.shape if isinstance(rms_frames, np.ndarray) else type(rms_frames)}) or 'rms_times' ({rms_times.shape if isinstance(rms_times, np.ndarray) else type(rms_times)}) data."
        )
        return [], []
    print(
        f" -> DEBUG: Found rms data (shape: {rms_frames.shape}) and rms_times (shape: {rms_times.shape}). Trim offset: {trim_offset_sec}"
    )

    if low_energy_norm is None or low_energy_times is None:
        print(" -> WARNING: 'low_energy_norm' or 'low_energy_times' missing.")
    if spectral_centroid_frames is None or times_absolute is None:
        print(
            " -> WARNING: Frame-based 'spectral_centroid_frames' or 'times_absolute' missing."
        )

    if not valid_input:
        return [], []
    print(
        f" -> Found {len(section_features_list_of_dicts)} sections. Processing features..."
    )

    # Calculate relative times ONCE
    spec_times_rel = None
    if times_absolute is not None and isinstance(times_absolute, np.ndarray):
        try:
            spec_times_rel = times_absolute - float(
                trim_offset_sec
            )  # Ensure float conversion
        except Exception as e:
            print(f" -> WARNING: Could not calculate spec_times_rel: {e}")

    rms_times_rel = None
    if rms_times is not None:
        try:
            rms_times_rel = rms_times.astype(float) - float(
                trim_offset_sec
            )  # Ensure float conversion
        except Exception as e:
            print(f" -> WARNING: Could not calculate rms_times_rel: {e}")

    # --- First Pass: Calculate avg_rms (potentially modified for Drops) and find track maximum ---
    all_section_avg_rms = []
    max_track_rms = 0.0
    print(" -> DEBUG: Starting First Pass (Calculating avg_rms per section)...")

    for i, section_dict in enumerate(section_features_list_of_dicts):
        current_avg_rms = np.nan
        start_time_rel, end_time_rel = np.nan, np.nan
        current_label = (
            semantic_labels_list[i] if i < len(semantic_labels_list) else None
        )

        if isinstance(section_dict, dict):
            start_time_abs = section_dict.get("start_time")
            end_time_abs = section_dict.get("end_time")

            if start_time_abs is not None and end_time_abs is not None:
                try:
                    start_time_rel = float(start_time_abs) - float(trim_offset_sec)
                    end_time_rel = float(end_time_abs) - float(trim_offset_sec)
                except (ValueError, TypeError) as time_err:
                    print(
                        f" -> DEBUG: Section {i}: Error converting times: {time_err}."
                    )
                    start_time_rel, end_time_rel = np.nan, np.nan

            # *** MODIFICATION START: Determine end time for RMS calculation ***
            rms_calc_end_time_rel = end_time_rel  # Default to full section end time
            if (
                current_label == "Drop"
                and not np.isnan(start_time_rel)
                and not np.isnan(end_time_rel)
                and end_time_rel > start_time_rel
            ):
                section_duration_rel = end_time_rel - start_time_rel
                rms_calc_end_time_rel = start_time_rel + (section_duration_rel * 0.75)
                # Optional Debug Print:
                # print(f"   -> DEBUG Drop RMS Time: Section {i}, Full End={end_time_rel:.3f}, RMS End={rms_calc_end_time_rel:.3f}")
            # *** MODIFICATION END ***

            # Proceed only if times are valid
            if (
                not np.isnan(start_time_rel)
                and not np.isnan(rms_calc_end_time_rel)
                and rms_times_rel is not None
            ):
                try:
                    # Use rms_calc_end_time_rel for the mask
                    rms_mask = (rms_times_rel >= start_time_rel) & (
                        rms_times_rel < rms_calc_end_time_rel
                    )  # Use potentially shorter end time

                    if (
                        isinstance(rms_mask, np.ndarray)
                        and rms_mask.shape == rms_frames.shape
                    ):
                        section_rms_frames = rms_frames[rms_mask]
                        finite_section_rms = section_rms_frames[
                            np.isfinite(section_rms_frames)
                        ]
                        if finite_section_rms.size > 0:
                            current_avg_rms = np.mean(finite_section_rms)
                            if (
                                np.isfinite(current_avg_rms)
                                and current_avg_rms > max_track_rms
                            ):
                                max_track_rms = current_avg_rms
                    else:
                        print(f" -> WARNING: Section {i}: Invalid RMS mask generated.")
                except Exception as e:
                    print(f" -> ERROR calculating avg_rms for section {i}: {e}")
                    traceback.print_exc()
                    current_avg_rms = np.nan
        else:
            print(f" -> WARNING: Section {i} data is not a dictionary.")
            current_avg_rms = np.nan

        all_section_avg_rms.append(current_avg_rms)

    print(
        f" -> DEBUG: First Pass Complete. Max Avg RMS found for track: {max_track_rms:.4f}"
    )
    if max_track_rms < 1e-9:
        print(" -> WARNING: Maximum track RMS is near zero. Relative RMS will be NaN.")
        max_track_rms = np.nan

    # --- Second Pass: Calculate all features including relative_rms ---
    print(" -> DEBUG: Starting Second Pass (Calculating all features)...")
    prev_avg_rms = np.nan
    prev_avg_centroid = np.nan

    for i, section_dict in enumerate(section_features_list_of_dicts):
        if not isinstance(section_dict, dict):
            section_dict = {}
            section_features_list_of_dicts[i] = section_dict

        current_label = (
            semantic_labels_list[i] if i < len(semantic_labels_list) else None
        )
        current_avg_rms = (
            all_section_avg_rms[i] if i < len(all_section_avg_rms) else np.nan
        )
        section_dict["avg_rms"] = current_avg_rms  # Store potentially modified avg_rms

        # Calculate Relative RMS using the potentially modified avg_rms
        relative_rms = np.nan
        if not np.isnan(current_avg_rms) and not np.isnan(max_track_rms):
            relative_rms = current_avg_rms / max_track_rms
        section_dict["relative_rms"] = relative_rms

        # --- Calculate other features ---
        start_time_abs = section_dict.get("start_time")
        end_time_abs = section_dict.get("end_time")
        start_time_rel, end_time_rel = np.nan, np.nan
        if start_time_abs is not None and end_time_abs is not None:
            try:
                start_time_rel = float(start_time_abs) - float(trim_offset_sec)
                end_time_rel = float(end_time_abs) - float(trim_offset_sec)
            except (ValueError, TypeError):
                start_time_rel, end_time_rel = np.nan, np.nan

        # *** MODIFICATION START: Determine end time for RMS calculations (Std Dev, Trend) ***
        rms_calc_end_time_rel = end_time_rel  # Default
        if (
            current_label == "Drop"
            and not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
            and end_time_rel > start_time_rel
        ):
            section_duration_rel = end_time_rel - start_time_rel
            rms_calc_end_time_rel = start_time_rel + (section_duration_rel * 0.75)
        # *** MODIFICATION END ***

        # Use full section duration (end_time_rel) for non-RMS features unless specified otherwise
        section_end_time_rel = end_time_rel

        # Peak RMS is calculated over full section usually (less affected by tail)
        current_peak_rms = np.nan
        if (
            rms_frames is not None
            and rms_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(section_end_time_rel)
        ):
            full_rms_mask = (rms_times_rel >= start_time_rel) & (
                rms_times_rel < section_end_time_rel
            )
            if (
                isinstance(full_rms_mask, np.ndarray)
                and full_rms_mask.shape == rms_frames.shape
            ):
                full_section_rms = rms_frames[full_rms_mask][
                    np.isfinite(rms_frames[full_rms_mask])
                ]
                if full_section_rms.size > 0:
                    current_peak_rms = np.max(full_section_rms)
        section_dict["peak_rms"] = (
            current_peak_rms  # Store peak calculated over full section
        )

        # Spectral Centroid Avg is calculated over full section
        current_avg_centroid = np.nan
        if (
            spectral_centroid_frames is not None
            and spec_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(section_end_time_rel)
            and spec_times_rel.shape == spectral_centroid_frames.shape
        ):
            centroid_mask = (spec_times_rel >= start_time_rel) & (
                spec_times_rel < section_end_time_rel
            )
            if (
                isinstance(centroid_mask, np.ndarray)
                and centroid_mask.shape == spectral_centroid_frames.shape
            ):
                section_centroid_vals = spectral_centroid_frames[centroid_mask][
                    np.isfinite(spectral_centroid_frames[centroid_mask])
                ]
                if section_centroid_vals.size > 0:
                    current_avg_centroid = np.mean(section_centroid_vals)
        section_dict["spectral_centroid_avg"] = (
            current_avg_centroid  # Store centroid calculated over full section
        )

        # 1. Relative Position (uses full section start)
        relative_position = np.nan
        if (
            start_time_abs is not None
            and duration_processed is not None
            and duration_processed > 0
        ):
            try:
                relative_position = float(start_time_abs) / float(duration_processed)
                relative_position = max(0.0, min(relative_position, 1.0))
            except (ValueError, TypeError):
                relative_position = np.nan
        section_dict["relative_position"] = relative_position

        # 1b. Position Context
        position_context = 0.0
        if current_label in ["Intro", "Outro"]:
            position_context = 1.0
        section_dict["position_context"] = position_context

        # 2. Average Low-End Energy (uses full section duration)
        avg_low_energy = np.nan
        if (
            low_energy_norm is not None
            and low_energy_times is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(section_end_time_rel)
            and isinstance(low_energy_times, np.ndarray)
            and low_energy_times.shape == low_energy_norm.shape
        ):
            try:
                le_mask = (low_energy_times >= start_time_rel) & (
                    low_energy_times < section_end_time_rel
                )  # Use full duration
                if (
                    isinstance(le_mask, np.ndarray)
                    and le_mask.shape == low_energy_norm.shape
                ):
                    finite_vals = low_energy_norm[le_mask][
                        np.isfinite(low_energy_norm[le_mask])
                    ]
                    if finite_vals.size > 0:
                        avg_low_energy = np.mean(finite_vals)
                else:
                    print(f" -> WARNING: Section {i}: Invalid Low Energy mask.")
            except Exception as e:
                print(f" -> ERROR calculating low energy for section {i}: {e}")
        section_dict["low_energy_norm"] = avg_low_energy

        # 3. Std Dev of RMS (uses potentially shorter duration for Drops)
        std_dev_rms = np.nan
        if (
            rms_frames is not None
            and rms_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(rms_calc_end_time_rel)
        ):
            try:
                rms_mask = (rms_times_rel >= start_time_rel) & (
                    rms_times_rel < rms_calc_end_time_rel
                )  # Use potentially shorter end time
                if (
                    isinstance(rms_mask, np.ndarray)
                    and rms_mask.shape == rms_frames.shape
                ):
                    section_rms_frames = rms_frames[rms_mask]
                    finite_section_rms = section_rms_frames[
                        np.isfinite(section_rms_frames)
                    ]
                    if finite_section_rms.size >= 2:
                        std_dev_rms = np.std(finite_section_rms)
                    elif finite_section_rms.size == 1:
                        std_dev_rms = 0.0
                else:
                    print(f" -> WARNING: Section {i}: Invalid RMS mask for std dev.")
            except Exception as e:
                print(f" -> ERROR calculating RMS std dev for section {i}: {e}")
        section_dict["rms_std_dev_section"] = std_dev_rms

        # 4. Std Dev of Spectral Centroid (uses full section duration)
        std_dev_centroid = np.nan
        if (
            spectral_centroid_frames is not None
            and spec_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(section_end_time_rel)
            and spec_times_rel.shape == spectral_centroid_frames.shape
        ):
            try:
                centroid_mask = (spec_times_rel >= start_time_rel) & (
                    spec_times_rel < section_end_time_rel
                )  # Use full duration
                if (
                    isinstance(centroid_mask, np.ndarray)
                    and centroid_mask.shape == spectral_centroid_frames.shape
                ):
                    section_centroid_frames = spectral_centroid_frames[centroid_mask]
                    finite_section_centroid = section_centroid_frames[
                        np.isfinite(section_centroid_frames)
                    ]
                    if finite_section_centroid.size >= 2:
                        std_dev_centroid = np.std(finite_section_centroid)
                    elif finite_section_centroid.size == 1:
                        std_dev_centroid = 0.0
                else:
                    print(
                        f" -> WARNING: Section {i}: Invalid Centroid mask for std dev."
                    )
            except Exception as e:
                print(f" -> ERROR calculating Centroid std dev for section {i}: {e}")
        section_dict["centroid_std_dev_section"] = std_dev_centroid

        # 5. Delta RMS (uses the potentially modified avg_rms)
        delta_rms = 0.0
        if i > 0:
            if not np.isnan(current_avg_rms) and not np.isnan(prev_avg_rms):
                delta_rms = current_avg_rms - prev_avg_rms
            else:
                delta_rms = np.nan
        section_dict["delta_rms"] = delta_rms

        # 6. Delta Centroid (uses full section avg_centroid)
        delta_centroid = 0.0
        if i > 0:
            if not np.isnan(current_avg_centroid) and not np.isnan(prev_avg_centroid):
                delta_centroid = current_avg_centroid - prev_avg_centroid
            else:
                delta_centroid = np.nan
        section_dict["delta_centroid"] = delta_centroid

        # 7. Crest Factor (uses potentially modified avg_rms and full section peak_rms)
        crest_factor = np.nan
        if not np.isnan(current_peak_rms) and not np.isnan(current_avg_rms):
            if current_avg_rms > 1e-9:
                crest_factor = current_peak_rms / current_avg_rms
            else:
                crest_factor = (
                    1.0  # Avoid division by zero if avg is tiny but peak exists
                )
        section_dict["crest_factor"] = crest_factor

        # 8. Spectral Centroid Slope (uses full section duration)
        spectral_centroid_slope = np.nan
        if (
            spectral_centroid_frames is not None
            and spec_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(section_end_time_rel)
            and spec_times_rel.shape == spectral_centroid_frames.shape
        ):
            try:
                centroid_mask = (spec_times_rel >= start_time_rel) & (
                    spec_times_rel < section_end_time_rel
                )  # Use full duration
                if (
                    isinstance(centroid_mask, np.ndarray)
                    and centroid_mask.shape == spectral_centroid_frames.shape
                ):
                    centroid_vals = spectral_centroid_frames[centroid_mask]
                    time_vals = spec_times_rel[centroid_mask]
                    finite_mask = np.isfinite(centroid_vals) & np.isfinite(time_vals)
                    centroid_vals = centroid_vals[finite_mask]
                    time_vals = time_vals[finite_mask]
                    if centroid_vals.size > 1:
                        rel_times = time_vals - time_vals[0]
                        slope, _, _, _, _ = scipy.stats.linregress(
                            rel_times, centroid_vals
                        )
                        spectral_centroid_slope = slope if np.isfinite(slope) else 0.0
                    elif centroid_vals.size <= 1:
                        spectral_centroid_slope = 0.0
                else:
                    print(f" -> WARNING: Section {i}: Invalid Centroid mask for slope.")
            except Exception as e:
                print(
                    f" -> WARNING: Failed to calculate spectral centroid slope for section {i}: {e}"
                )
                spectral_centroid_slope = 0.0
        section_dict["spectral_centroid_slope"] = spectral_centroid_slope

        # 9. RMS Trend (Slope) (uses potentially shorter duration for Drops)
        rms_trend = np.nan
        if (
            rms_frames is not None
            and rms_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(rms_calc_end_time_rel)
        ):
            try:
                rms_mask = (rms_times_rel >= start_time_rel) & (
                    rms_times_rel < rms_calc_end_time_rel
                )  # Use potentially shorter end time
                if (
                    isinstance(rms_mask, np.ndarray)
                    and rms_mask.shape == rms_frames.shape
                ):
                    rms_vals = rms_frames[rms_mask]
                    time_vals = rms_times_rel[rms_mask]
                    finite_mask = np.isfinite(rms_vals) & np.isfinite(time_vals)
                    rms_vals = rms_vals[finite_mask]
                    time_vals = time_vals[finite_mask]
                    if rms_vals.size > 1:
                        rel_times = time_vals - time_vals[0]
                        slope, _, _, _, _ = scipy.stats.linregress(rel_times, rms_vals)
                        rms_trend = slope if np.isfinite(slope) else 0.0
                    elif rms_vals.size <= 1:
                        rms_trend = 0.0
                else:
                    print(f" -> WARNING: Section {i}: Invalid RMS mask for trend.")
            except ValueError as linreg_err:
                print(
                    f" -> Warning: Linregress failed for RMS trend in section {i}: {linreg_err}"
                )
                rms_trend = 0.0
            except Exception as e:
                print(
                    f" -> WARNING: Failed to calculate RMS trend for section {i}: {e}"
                )
                rms_trend = 0.0
        section_dict["rms_trend"] = rms_trend

        # Update previous values for next iteration's delta calculation
        prev_avg_rms = current_avg_rms
        prev_avg_centroid = current_avg_centroid

    # --- Calculate Label Proportion (after iterating through all sections) ---
    print(" -> Calculating Label Proportion...")
    total_sections = len(semantic_labels_list)
    if total_sections > 0:
        label_counts = Counter(semantic_labels_list)  # Use Counter for efficiency
        for i, section_dict in enumerate(section_features_list_of_dicts):
            current_label = (
                semantic_labels_list[i] if i < len(semantic_labels_list) else None
            )
            if current_label and isinstance(section_dict, dict):
                proportion = label_counts.get(current_label, 0) / total_sections
                section_dict["label_proportion"] = proportion
            elif isinstance(section_dict, dict):
                section_dict["label_proportion"] = (
                    np.nan
                )  # Assign NaN if label is missing
    else:
        # Assign NaN if there are no sections
        for section_dict in section_features_list_of_dicts:
            if isinstance(section_dict, dict):
                section_dict["label_proportion"] = np.nan

    print(f" -> DEBUG: Second Pass Complete.")
    first_valid_dict = next(
        (d for d in section_features_list_of_dicts if isinstance(d, dict)), None
    )
    if first_valid_dict:
        print(
            f" -> DEBUG: Keys in first section dict after update: {list(first_valid_dict.keys())}"
        )
    else:
        print(" -> DEBUG: No valid section dictionaries found after processing.")

    return section_features_list_of_dicts, semantic_labels_list


# Note: calculate_bar_features remains unchanged
def calculate_bar_features(track_data):
    """
    Calculates features averaged over each bar.
    Current features: Avg RMS, Avg Spectral Centroid.
    Returns features and corresponding labels per bar.
    NOTE: This might not be used by the section-based HMM trainer/predictor.
    """
    print("--- Running calculate_bar_features (NOTE: May be unused by section HMM) ---")
    # Retrieve necessary data
    bar_starts = track_data.get("bar_starts_absolute")  # Absolute times
    bar_rms = track_data.get("bar_rms_data")
    section_starts = track_data.get("section_starts")  # Absolute times
    semantic_labels = track_data.get("semantic_labels")
    spec_centroid = track_data.get("spectral_centroid_frames")  # Frame-based
    frame_times_abs = track_data.get(
        "times_absolute"
    )  # Absolute times for spectral features
    trim_offset = track_data.get("trim_offset_sec", 0)
    duration_processed = track_data.get("duration_processed", 0)

    # --- Validate Inputs ---
    required_data = {
        "bar_starts": bar_starts,
        "bar_rms": bar_rms,
        "section_starts": section_starts,
        "semantic_labels": semantic_labels,  # "spec_centroid": spec_centroid, # Optional now
        "frame_times_abs": frame_times_abs,
        "duration_processed": duration_processed,
        "trim_offset": trim_offset,
    }
    # Check if essential bar/section data is present
    if any(
        v is None
        for k, v in required_data.items()
        if k not in ["spec_centroid", "frame_times_abs"]
    ):
        missing = [
            k
            for k, v in required_data.items()
            if v is None and k not in ["spec_centroid", "frame_times_abs"]
        ]
        print(
            f"Warning: Missing essential data for bar feature calculation: {missing}. Skipping."
        )
        return [], []
    # Check lengths
    num_bars = len(bar_starts)
    if len(bar_rms) != num_bars:
        print(
            f"Warning: Mismatch between bar_starts ({num_bars}) and bar_rms ({len(bar_rms)}). Skipping."
        )
        return [], []
    # Check spectral data consistency only if present
    if spec_centroid is not None and (
        frame_times_abs is None or len(spec_centroid) != len(frame_times_abs)
    ):
        spec_len = len(spec_centroid) if spec_centroid is not None else "N/A"
        time_len = len(frame_times_abs) if frame_times_abs is not None else "N/A"
        print(
            f"Warning: Mismatch/Missing spec_centroid ({spec_len}) or frame_times ({time_len}). Skipping centroid calculation."
        )
        spec_centroid = None  # Disable centroid calculation if inconsistent

    # --- Calculate features per bar ---
    bar_features_list = []
    bar_labels_list = []
    current_section_idx = 0  # Track current section

    for i in range(num_bars):
        bar_start_time_abs = bar_starts[i]
        # Calculate bar end time (absolute)
        bar_end_time_abs = (
            bar_starts[i + 1] if i + 1 < num_bars else duration_processed + trim_offset
        )

        # --- Find the semantic label for the current bar ---
        # Advance section index while the bar start time is >= the next section start time
        if (
            section_starts is not None and semantic_labels is not None
        ):  # Check if section data exists
            while (
                current_section_idx + 1 < len(section_starts)
                and bar_start_time_abs >= section_starts[current_section_idx + 1]
            ):
                current_section_idx += 1
            # Get the label for the determined section index
            if current_section_idx >= len(semantic_labels):
                print(
                    f"Warning: Section index {current_section_idx} out of bounds for labels ({len(semantic_labels)}) at bar {i}. Assigning 'Unknown'."
                )
                current_label = "Unknown"
            else:
                current_label = semantic_labels[current_section_idx]
        else:
            current_label = "Unknown"  # Default if no section data

        # --- Calculate Features for the Bar ---
        # 1. Average RMS (already available per bar)
        avg_rms_bar = bar_rms[i] if np.isfinite(bar_rms[i]) else 0.0

        # 2. Average Spectral Centroid (if data available)
        avg_centroid_bar = 0.0  # Default
        if spec_centroid is not None and frame_times_abs is not None:
            # Find spectral frames within the bar's absolute time range
            frame_mask = (frame_times_abs >= bar_start_time_abs) & (
                frame_times_abs < bar_end_time_abs
            )
            frames_in_bar_centroid = spec_centroid[frame_mask]
            # Calculate mean only if valid frames exist
            if frames_in_bar_centroid.size > 0:
                finite_centroids = frames_in_bar_centroid[
                    np.isfinite(frames_in_bar_centroid)
                ]
                if finite_centroids.size > 0:
                    avg_centroid_bar = np.mean(finite_centroids)
            # Ensure result is finite, default to 0 otherwise
            if not np.isfinite(avg_centroid_bar):
                avg_centroid_bar = 0.0

        # --- Store features and label ---
        # Append the calculated features for this bar
        bar_features_list.append(
            [avg_rms_bar, avg_centroid_bar]
        )  # Add more features here if calculated
        # Append the corresponding label
        bar_labels_list.append(current_label)

    print(
        f"--- Finished calculate_bar_features: Processed {len(bar_features_list)} bars. ---"
    )
    return bar_features_list, bar_labels_list
