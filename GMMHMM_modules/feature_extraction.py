# =============================================================================
# FILE: GMMHMM_modules/feature_extraction.py
# Purpose: Calculate features specifically needed for GMMHMM training/prediction.
# ADDED: label_proportion feature calculation.
# =============================================================================

import numpy as np
import scipy.stats
import librosa
import traceback
from collections import Counter  # Added for counting labels


def extract_section_features(track_data):
    """
    Retrieves section features (as a list of dictionaries) and semantic labels
    from the track_data. Calculates and adds features like 'relative_position',
    'low_energy_norm', standard deviations, delta features, crest factor,
    spectral slope, 'relative_rms', 'rms_trend', 'position_context', AND
    'label_proportion'.

    Args:
        track_data (dict): Dictionary loaded from analysis file, expected
                           to contain base data like 'section_features',
                           'semantic_labels', 'rms', 'rms_times', etc.

    Returns:
        tuple: A tuple containing:
            - list[dict]: The list of section feature dictionaries, updated with calculated features.
            - list[str]: The list of corresponding semantic labels.
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
    times_absolute = track_data.get("times_absolute")

    # --- Initial Validation ---
    print(f"\n=== DEBUG: Inside GMMHMM feature_extraction.extract_section_features ===")
    valid_input = True
    if (
        not isinstance(semantic_labels_list, (list, np.ndarray))
        or not semantic_labels_list
    ):
        print(" -> ERROR: 'semantic_labels' missing, not a list/array, or empty.")
        return [], []  # Cannot proceed without labels

    total_sections = len(semantic_labels_list)
    print(f" -> Found {total_sections} initial labels.")

    if not isinstance(section_features_list_of_dicts, list):
        print(
            " -> WARNING: 'section_features' is not a list. Creating empty placeholders."
        )
        section_features_list_of_dicts = [{} for _ in range(total_sections)]
    elif len(section_features_list_of_dicts) != total_sections:
        print(
            f" -> WARNING: Initial 'section_features' length ({len(section_features_list_of_dicts)}) doesn't match labels ({total_sections}). Using placeholders."
        )
        section_features_list_of_dicts = [{} for _ in range(total_sections)]

    if duration_processed is None or duration_processed <= 0:
        print(" -> WARNING: 'duration_processed' missing or invalid.")
    if rms_frames is None:
        print(" -> FATAL ERROR: Frame-based 'rms' data is MISSING.")
        return [], []
    if rms_times is None:
        print(" -> FATAL ERROR: Frame-based 'rms_times' data is MISSING.")
        return [], []
    if not isinstance(rms_frames, np.ndarray) or not isinstance(rms_times, np.ndarray):
        print(f" -> FATAL ERROR: 'rms' or 'rms_times' is not a numpy array.")
        return [], []
    if rms_frames.shape != rms_times.shape:
        print(
            f" -> FATAL ERROR: Shape mismatch between 'rms' ({rms_frames.shape}) and 'rms_times' ({rms_times.shape})."
        )
        return [], []
    print(
        f" -> DEBUG: Found rms data (shape: {rms_frames.shape}) and rms_times (shape: {rms_times.shape}). Trim offset: {trim_offset_sec}"
    )

    if spectral_centroid_frames is None or times_absolute is None:
        print(
            " -> WARNING: Frame-based 'spectral_centroid_frames' or 'times_absolute' missing (needed for centroid features)."
        )

    if not valid_input:
        return [], []

    # --- Calculate Label Proportions (BEFORE loops) ---
    label_counts = Counter(semantic_labels_list)
    label_proportions = {
        label: count / total_sections for label, count in label_counts.items()
    }
    print(
        f" -> DEBUG: Calculated Label Proportions: { {k: f'{v:.2f}' for k, v in label_proportions.items()} }"
    )

    # Calculate relative times ONCE
    spec_times_rel, rms_times_rel = None, None
    if times_absolute is not None and isinstance(times_absolute, np.ndarray):
        try:
            spec_times_rel = times_absolute - trim_offset_sec
        except Exception as e:
            print(f" -> WARNING: Could not calculate spec_times_rel: {e}")
    if rms_times is not None:
        try:
            rms_times_rel = rms_times.astype(float) - float(trim_offset_sec)
        except Exception as e:
            print(f" -> WARNING: Could not calculate rms_times_rel: {e}")

    # --- First Pass: Calculate all avg_rms and find track maximum ---
    all_section_avg_rms = []
    max_track_rms = 0.0
    print(" -> DEBUG: Starting First Pass (Calculating avg_rms per section)...")
    section_starts_abs = track_data.get("section_starts")
    section_ends_abs = []
    if (
        section_starts_abs is not None
        and isinstance(section_starts_abs, (list, np.ndarray))
        and len(section_starts_abs) == total_sections
    ):
        track_end_time = (
            duration_processed + trim_offset_sec if duration_processed else float("inf")
        )
        for i in range(total_sections):
            section_ends_abs.append(
                section_starts_abs[i + 1] if i + 1 < total_sections else track_end_time
            )
    else:
        print(
            " -> WARNING: 'section_starts' missing or invalid length. Cannot reliably calculate section features."
        )
        section_starts_abs = None

    for i in range(total_sections):
        current_avg_rms, start_time_rel, end_time_rel = np.nan, np.nan, np.nan
        if section_starts_abs is not None:
            start_time_abs, end_time_abs = section_starts_abs[i], section_ends_abs[i]
            try:
                start_time_rel, end_time_rel = float(start_time_abs) - float(
                    trim_offset_sec
                ), float(end_time_abs) - float(trim_offset_sec)
            except (ValueError, TypeError):
                pass
        if (
            not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
            and rms_times_rel is not None
        ):
            try:
                rms_mask = (rms_times_rel >= start_time_rel) & (
                    rms_times_rel < end_time_rel
                )
                if (
                    isinstance(rms_mask, np.ndarray)
                    and rms_mask.shape == rms_frames.shape
                ):
                    finite_section_rms = rms_frames[rms_mask][
                        np.isfinite(rms_frames[rms_mask])
                    ]
                    if finite_section_rms.size > 0:
                        current_avg_rms = np.mean(finite_section_rms)
                        if (
                            np.isfinite(current_avg_rms)
                            and current_avg_rms > max_track_rms
                        ):
                            max_track_rms = current_avg_rms
            except Exception as e:
                print(f" -> ERROR calculating avg_rms for section {i}: {e}")
                current_avg_rms = np.nan
        all_section_avg_rms.append(current_avg_rms)

    print(
        f" -> DEBUG: First Pass Complete. Max Avg RMS found for track: {max_track_rms:.4f}"
    )
    if max_track_rms < 1e-9:
        print(" -> WARNING: Maximum track RMS is near zero.")
        max_track_rms = np.nan

    # --- Second Pass: Calculate all features ---
    print(" -> DEBUG: Starting Second Pass (Calculating all features)...")
    prev_avg_rms, prev_avg_centroid = np.nan, np.nan
    calculated_features_list = []

    for i in range(total_sections):
        section_dict = {}
        current_label = semantic_labels_list[i]
        current_avg_rms = all_section_avg_rms[i]
        section_dict["avg_rms"] = current_avg_rms

        # Relative RMS
        section_dict["relative_rms"] = (
            current_avg_rms / max_track_rms
            if not np.isnan(current_avg_rms) and not np.isnan(max_track_rms)
            else np.nan
        )

        # Get section times again
        start_time_abs, end_time_abs = None, None
        start_time_rel, end_time_rel = np.nan, np.nan
        if section_starts_abs is not None:
            start_time_abs, end_time_abs = section_starts_abs[i], section_ends_abs[i]
            try:
                start_time_rel, end_time_rel = float(start_time_abs) - float(
                    trim_offset_sec
                ), float(end_time_abs) - float(trim_offset_sec)
            except (ValueError, TypeError):
                pass

        # Get other pre-calculated values if available
        input_feature_dict = (
            section_features_list_of_dicts[i]
            if i < len(section_features_list_of_dicts)
            and isinstance(section_features_list_of_dicts[i], dict)
            else {}
        )
        current_peak_rms = input_feature_dict.get("peak_rms", np.nan)
        current_avg_centroid = input_feature_dict.get("spectral_centroid_avg", np.nan)
        if not isinstance(current_peak_rms, (int, float, np.number)):
            current_peak_rms = np.nan
        if not isinstance(current_avg_centroid, (int, float, np.number)):
            current_avg_centroid = np.nan

        # Relative Position
        relative_position = np.nan
        if (
            start_time_abs is not None
            and duration_processed is not None
            and duration_processed > 0
        ):
            try:
                relative_position = max(
                    0.0, min(float(start_time_abs) / float(duration_processed), 1.0)
                )
            except (ValueError, TypeError):
                pass
        section_dict["relative_position"] = relative_position

        # Position Context
        section_dict["position_context"] = (
            abs(relative_position - 0.5) * 2.0
            if not np.isnan(relative_position)
            else np.nan
        )

        # <<< Label Proportion (NEW FEATURE) >>>
        section_dict["label_proportion"] = label_proportions.get(
            current_label, 0.0
        )  # Get pre-calculated proportion

        # Low-End Energy Norm
        avg_low_energy = np.nan
        if (
            low_energy_norm is not None
            and low_energy_times is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
            and isinstance(low_energy_times, np.ndarray)
            and low_energy_times.shape == low_energy_norm.shape
        ):
            try:
                le_mask = (low_energy_times >= start_time_rel) & (
                    low_energy_times < end_time_rel
                )
                if (
                    isinstance(le_mask, np.ndarray)
                    and le_mask.shape == low_energy_norm.shape
                ):
                    finite_vals = low_energy_norm[le_mask][
                        np.isfinite(low_energy_norm[le_mask])
                    ]
                    if finite_vals.size > 0:
                        avg_low_energy = np.mean(finite_vals)
            except Exception as e:
                print(f" -> ERROR calculating low energy for section {i}: {e}")
        section_dict["low_energy_norm"] = avg_low_energy

        # RMS Std Dev
        std_dev_rms = np.nan
        if (
            rms_frames is not None
            and rms_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
        ):
            try:
                rms_mask = (rms_times_rel >= start_time_rel) & (
                    rms_times_rel < end_time_rel
                )
                if (
                    isinstance(rms_mask, np.ndarray)
                    and rms_mask.shape == rms_frames.shape
                ):
                    finite_section_rms = rms_frames[rms_mask][
                        np.isfinite(rms_frames[rms_mask])
                    ]
                    if finite_section_rms.size >= 2:
                        std_dev_rms = np.std(finite_section_rms)
                    elif finite_section_rms.size == 1:
                        std_dev_rms = 0.0
            except Exception as e:
                print(f" -> ERROR calculating RMS std dev for section {i}: {e}")
        section_dict["rms_std_dev_section"] = std_dev_rms

        # Centroid Std Dev
        std_dev_centroid = np.nan
        if (
            spectral_centroid_frames is not None
            and spec_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
            and spec_times_rel.shape == spectral_centroid_frames.shape
        ):
            try:
                centroid_mask = (spec_times_rel >= start_time_rel) & (
                    spec_times_rel < end_time_rel
                )
                if (
                    isinstance(centroid_mask, np.ndarray)
                    and centroid_mask.shape == spectral_centroid_frames.shape
                ):
                    finite_section_centroid = spectral_centroid_frames[centroid_mask][
                        np.isfinite(spectral_centroid_frames[centroid_mask])
                    ]
                    if finite_section_centroid.size >= 2:
                        std_dev_centroid = np.std(finite_section_centroid)
                    elif finite_section_centroid.size == 1:
                        std_dev_centroid = 0.0
            except Exception as e:
                print(f" -> ERROR calculating Centroid std dev for section {i}: {e}")
        section_dict["centroid_std_dev_section"] = std_dev_centroid

        # Delta RMS
        section_dict["delta_rms"] = (
            current_avg_rms - prev_avg_rms
            if i > 0 and not np.isnan(current_avg_rms) and not np.isnan(prev_avg_rms)
            else (0.0 if i == 0 else np.nan)
        )

        # Delta Centroid
        section_dict["delta_centroid"] = (
            current_avg_centroid - prev_avg_centroid
            if i > 0
            and not np.isnan(current_avg_centroid)
            and not np.isnan(prev_avg_centroid)
            else (0.0 if i == 0 else np.nan)
        )

        # Crest Factor
        section_dict["crest_factor"] = (
            current_peak_rms / current_avg_rms
            if not np.isnan(current_peak_rms)
            and not np.isnan(current_avg_rms)
            and current_avg_rms > 1e-9
            else (1.0 if not np.isnan(current_peak_rms) else np.nan)
        )

        # Spectral Centroid Slope
        spectral_centroid_slope = np.nan
        if (
            spectral_centroid_frames is not None
            and spec_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
            and spec_times_rel.shape == spectral_centroid_frames.shape
        ):
            try:
                centroid_mask = (spec_times_rel >= start_time_rel) & (
                    spec_times_rel < end_time_rel
                )
                if (
                    isinstance(centroid_mask, np.ndarray)
                    and centroid_mask.shape == spectral_centroid_frames.shape
                ):
                    centroid_vals, time_vals = (
                        spectral_centroid_frames[centroid_mask],
                        spec_times_rel[centroid_mask],
                    )
                    finite_mask = np.isfinite(centroid_vals) & np.isfinite(time_vals)
                    centroid_vals, time_vals = (
                        centroid_vals[finite_mask],
                        time_vals[finite_mask],
                    )
                    if centroid_vals.size > 1:
                        slope, _, _, _, _ = scipy.stats.linregress(
                            time_vals - time_vals[0], centroid_vals
                        )
                        spectral_centroid_slope = slope if np.isfinite(slope) else 0.0
                    else:
                        spectral_centroid_slope = 0.0
            except Exception as e:
                print(
                    f" -> WARNING: Failed to calculate spectral centroid slope for section {i}: {e}"
                )
                spectral_centroid_slope = 0.0
        section_dict["spectral_centroid_slope"] = spectral_centroid_slope

        # RMS Trend
        rms_trend = np.nan
        if (
            rms_frames is not None
            and rms_times_rel is not None
            and not np.isnan(start_time_rel)
            and not np.isnan(end_time_rel)
        ):
            try:
                rms_mask = (rms_times_rel >= start_time_rel) & (
                    rms_times_rel < end_time_rel
                )
                if (
                    isinstance(rms_mask, np.ndarray)
                    and rms_mask.shape == rms_frames.shape
                ):
                    rms_vals, time_vals = rms_frames[rms_mask], rms_times_rel[rms_mask]
                    finite_mask = np.isfinite(rms_vals) & np.isfinite(time_vals)
                    rms_vals, time_vals = rms_vals[finite_mask], time_vals[finite_mask]
                    if rms_vals.size > 1:
                        slope, _, _, _, _ = scipy.stats.linregress(
                            time_vals - time_vals[0], rms_vals
                        )
                        rms_trend = slope if np.isfinite(slope) else 0.0
                    else:
                        rms_trend = 0.0
            except Exception as e:
                print(
                    f" -> WARNING: Failed to calculate RMS trend for section {i}: {e}"
                )
                rms_trend = 0.0
        section_dict["rms_trend"] = rms_trend

        calculated_features_list.append(section_dict)
        prev_avg_rms, prev_avg_centroid = current_avg_rms, current_avg_centroid

    # --- Final Check and Return ---
    print(f" -> DEBUG: Second Pass Complete.")
    if calculated_features_list:
        first_valid_dict = next(
            (d for d in calculated_features_list if isinstance(d, dict)), None
        )
        if first_valid_dict:
            print(f" -> DEBUG: Keys calculated: {list(first_valid_dict.keys())}")
    return calculated_features_list, semantic_labels_list  # Return calculated features


# Note: calculate_bar_features remains unchanged.
def calculate_bar_features(track_data):
    """Calculates features averaged over each bar."""
    # ... (implementation remains the same) ...
    print("--- Running calculate_bar_features (NOTE: May be unused by section HMM) ---")
    bar_starts = track_data.get("bar_starts_absolute")
    bar_rms = track_data.get("bar_rms_data")
    section_starts = track_data.get("section_starts")
    semantic_labels = track_data.get("semantic_labels")
    spec_centroid = track_data.get("spectral_centroid_frames")
    frame_times_abs = track_data.get("times_absolute")
    trim_offset = track_data.get("trim_offset_sec", 0)
    duration_processed = track_data.get("duration_processed", 0)
    required_data = {
        "bar_starts": bar_starts,
        "bar_rms": bar_rms,
        "section_starts": section_starts,
        "semantic_labels": semantic_labels,
        "frame_times_abs": frame_times_abs,
        "duration_processed": duration_processed,
        "trim_offset": trim_offset,
    }
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
    num_bars = len(bar_starts)
    if len(bar_rms) != num_bars:
        print(
            f"Warning: Mismatch between bar_starts ({num_bars}) and bar_rms ({len(bar_rms)}). Skipping."
        )
        return [], []
    if spec_centroid is not None and (
        frame_times_abs is None or len(spec_centroid) != len(frame_times_abs)
    ):
        spec_centroid = None
        print(
            "Warning: Mismatch/Missing spec_centroid or frame_times. Skipping centroid calculation."
        )
    bar_features_list = []
    bar_labels_list = []
    current_section_idx = 0
    for i in range(num_bars):
        bar_start_time_abs = bar_starts[i]
        bar_end_time_abs = (
            bar_starts[i + 1] if i + 1 < num_bars else duration_processed + trim_offset
        )
        while (
            current_section_idx + 1 < len(section_starts)
            and bar_start_time_abs >= section_starts[current_section_idx + 1]
        ):
            current_section_idx += 1
        current_label = (
            semantic_labels[current_section_idx]
            if current_section_idx < len(semantic_labels)
            else "Unknown"
        )
        avg_rms_bar = bar_rms[i] if np.isfinite(bar_rms[i]) else 0.0
        avg_centroid_bar = 0.0
        if spec_centroid is not None and frame_times_abs is not None:
            frame_mask = (frame_times_abs >= bar_start_time_abs) & (
                frame_times_abs < bar_end_time_abs
            )
            frames_in_bar_centroid = spec_centroid[frame_mask]
            if frames_in_bar_centroid.size > 0:
                finite_centroids = frames_in_bar_centroid[
                    np.isfinite(frames_in_bar_centroid)
                ]
                avg_centroid_bar = (
                    np.mean(finite_centroids) if finite_centroids.size > 0 else 0.0
                )
            if not np.isfinite(avg_centroid_bar):
                avg_centroid_bar = 0.0
        bar_features_list.append([avg_rms_bar, avg_centroid_bar])
        bar_labels_list.append(current_label)
    print(
        f"--- Finished calculate_bar_features: Processed {len(bar_features_list)} bars. ---"
    )
    return bar_features_list, bar_labels_list
