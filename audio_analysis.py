# =============================================================================
# FILE: audio_analysis.py
# Contains functions for loading, preprocessing, and analyzing audio data.
# Uses "Body" instead of "Verse".
# Adds minimum duration check for Outro label.
# Adds check to relabel final short+quiet sections as "Fade Out".
# Uses RMS-based cluster colors (overriding Fade Out).
# Added calculate_bar_features function for HMM data prep/prediction.
# =============================================================================

import os
import traceback
from collections import defaultdict
import re # Keep re here if needed by analysis funcs, though mainly used in plotting/parsing

import librosa
import librosa.display # Keep if display features are used directly in analysis
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler # Needed for feature scaling
# from sklearn.decomposition import PCA # Optional: Could add PCA later
import scipy.stats # Needed for linregress (RMS trend)
import scipy.signal # Needed for filtering
import math
# Import messagebox here only if analysis functions MUST show GUI errors
# Ideally, they should raise exceptions handled by the GUI layer.
from tkinter import messagebox

# --- Constants ---
# Define thresholds used in labeling logic
MIN_OUTRO_BARS = 4
MIN_FADEOUT_BARS = 4 # Sections shorter than this at the end might be fade outs
FADEOUT_RMS_THRESHOLD = 0.05 # Avg RMS below this might indicate a fade out
# Define Fade Out color here for consistency
FADE_OUT_COLOR_HEX = '#8A2BE2' # BlueViolet/Purple
FALLBACK_COLOR_HEX = '#808080' # Grey for unknown labels
# Define specific colors for custom mapping
DARK_RED_HEX = '#8B0000'
RED_HEX = '#FF0000'


# --- Analysis Function Definitions ---

def load_and_preprocess(file_path, track_num, manual_bpm_override=None):
    """Loads audio, trims, calculates tempo (Median IBI or manual), and basic features."""
    print(f"Loading Track {track_num}: {os.path.basename(file_path)}...")
    y_processed = y_original = sr = bpm = None
    trim_offset_sec = duration = None
    hop_length = 256
    tempo_val = 120.0; detected_bpm = 120.0 # Initialize defaults

    try:
        # Load audio file
        y_original, sr = librosa.load(file_path, sr=None, mono=True)
        if y_original is None or len(y_original) == 0: raise ValueError("Audio empty.")
        if not np.all(np.isfinite(y_original)): y_original = np.nan_to_num(y_original); print("Warning: Non-finite values fixed in original audio.")
        if not np.all(np.isfinite(y_original)): raise ValueError("Audio still not finite after nan_to_num.")

        # Trim silence
        y_trimmed, index = librosa.effects.trim(y_original, top_db=55) # Keep stricter trim
        if index is None or len(index) != 2 or index[1] <= index[0]:
             print("Warning: No significant audio found after trimming (top_db=55), using original.")
             trim_offset_sec = 0.0; y_processed = y_original.copy()
        else: trim_offset_sec = float(index[0]) / sr; y_processed = y_trimmed; print(f" Trimmed {trim_offset_sec:.2f}s from start, duration after trim: {librosa.get_duration(y=y_processed, sr=sr):.2f}s")

        if not np.all(np.isfinite(y_processed)): y_processed = np.nan_to_num(y_processed); print("Warning: Non-finite values fixed after trimming.")
        if y_processed is None or not np.all(np.isfinite(y_processed)): raise ValueError("Audio not finite after trim/nan_to_num.")

        duration = librosa.get_duration(y=y_processed, sr=sr)
        if duration is None or not np.isfinite(duration) or duration <= 0: raise ValueError(f"Invalid duration calculated: {duration}")

        # --- Tempo & Beat Detection ---
        try:
            hop_length_tempo = 512; print(f" Running beat tracking for tempo estimate (Track {track_num})...")
            tempo_initial_raw, beat_frames_tracked = librosa.beat.beat_track(y=y_processed, sr=sr, hop_length=hop_length_tempo, units='frames')
            bpm_detected = 120.0; tempo_initial_float = 120.0
            if np.isscalar(tempo_initial_raw) and np.isfinite(tempo_initial_raw): tempo_initial_float = float(tempo_initial_raw); print(f" Initial tempo estimate from beat_track: {tempo_initial_float:.4f}")
            else: print(" Initial tempo estimate from beat_track is not a valid scalar.")
            if beat_frames_tracked is not None and beat_frames_tracked.size > 1:
                 beat_times = librosa.frames_to_time(beat_frames_tracked, sr=sr, hop_length=hop_length_tempo); ibis = np.diff(beat_times)
                 if ibis.size > 0:
                      median_ibi = np.median(ibis)
                      if median_ibi > 1e-6:
                           tempo_calc = 60.0 / median_ibi; print(f" Tempo calculated from median IBI: {tempo_calc:.4f}")
                           if np.isfinite(tempo_calc) and 30 < tempo_calc < 250: bpm_detected = tempo_calc
                           else: print(f" Median IBI unreasonable ({tempo_calc:.4f}). Using beat_track estimate."); bpm_detected = tempo_initial_float
                      else: print(" Warning: Median IBI zero/negative. Using beat_track estimate."); bpm_detected = tempo_initial_float
                 else: print(" Warning: Could not calculate IBIs. Using beat_track estimate."); bpm_detected = tempo_initial_float
            else: print(" Warning: Initial beat tracking failed/too few beats."); bpm_detected = tempo_initial_float
            if not (np.isfinite(bpm_detected) and 30 < bpm_detected < 300): print(f" Detected BPM unreasonable ({bpm_detected:.2f}). Defaulting to 120."); bpm_detected = 120.0
            tempo_val = bpm_detected; print(f" Detected Tempo Value (before override): {tempo_val:.4f}")
            if manual_bpm_override is not None: print(f" Manual BPM Override provided: {manual_bpm_override:.4f}. Overriding detected tempo."); tempo_val = float(manual_bpm_override)
            if not np.isfinite(tempo_val) or tempo_val < 30 or tempo_val > 300: print(f" Warning (Track {track_num}): Final tempo is unreasonable ({tempo_val:.4f}). Falling back to default 120 BPM."); tempo_val = 120.0
        except Exception as e: print(f"Tempo/Beat detection failed (Track {track_num}): {e}. Using default 120 BPM."); traceback.print_exc(); tempo_val = 120.0

        bpm = round(tempo_val); print(f"Final rounded BPM to be used (Track {track_num}): {bpm}")
        if sr is None or bpm is None: raise ValueError("Failed to determine sample rate or BPM.")
        if y_processed is None: raise ValueError("Processed audio is None before returning.")
        return {"y_processed": y_processed, "y_original": y_original, "sr": sr, "bpm": bpm, "trim_offset_sec": trim_offset_sec, "duration_processed": duration, "hop_length": hop_length, "file_path": file_path, "bpm_detected": round(bpm_detected)}
    except Exception as e: print(f"--- Error in load_and_preprocess (Track {track_num}) ---"); traceback.print_exc(); messagebox.showerror(f"Error Loading Track {track_num}", f"Failed to load or preprocess audio:\n{e}"); return None


def detect_sections(track_data):
    """Performs section detection based on RMS energy changes for one track."""
    bar_rms_data = track_data.get("bar_rms_data"); bar_starts = track_data.get("bar_starts_absolute")
    num_bars = len(bar_rms_data) if bar_rms_data is not None else 0; trim_offset = track_data.get("trim_offset_sec", 0)
    section_starts = []; section_labels = []; section_index = 1; drop_rms_reference = []
    if bar_rms_data is None or bar_starts is None or num_bars <= 0: print("Warning: Invalid input to detect_sections."); return [], []
    if len(bar_starts) != num_bars: print(f"Warning: Mismatch bar_starts/num_bars."); return [], []
    i = 0
    while i < num_bars:
        if i >= len(bar_rms_data): break
        current_rms = bar_rms_data[i]
        if np.isfinite(current_rms) and current_rms > 0.35: drop_rms_reference.append(current_rms)
        drop_avg = np.mean(drop_rms_reference) if drop_rms_reference else 0.35
        matched_fill = False
        for fill_length in [4, 3, 2, 1]:
            rebound_idx = i + fill_length
            if rebound_idx < num_bars:
                 fill_indices = range(i, i + fill_length)
                 is_fill = all(idx < len(bar_rms_data) and np.isfinite(bar_rms_data[idx]) and bar_rms_data[idx] < 0.65 * drop_avg for idx in fill_indices)
                 rebounds = rebound_idx < len(bar_rms_data) and np.isfinite(bar_rms_data[rebound_idx]) and bar_rms_data[rebound_idx] > 0.9 * drop_avg
                 if is_fill and rebounds:
                      section_starts.append(bar_starts[i]); section_labels.append("Fill")
                      section_starts.append(bar_starts[rebound_idx]); section_labels.append(f"S{section_index}"); section_index += 1
                      i += fill_length + 8; matched_fill = True; break
        if matched_fill: continue
        if i + 8 <= num_bars:
            slice1 = bar_rms_data[i : min(i + 4, num_bars)]; slice2 = bar_rms_data[min(i + 4, num_bars) : min(i + 8, num_bars)]
            slice1_rms = [val for val in slice1 if np.isfinite(val)]; slice2_rms = [val for val in slice2 if np.isfinite(val)]
            e1 = np.mean(slice1_rms) if slice1_rms else 0; e2 = np.mean(slice2_rms) if slice2_rms else 0
            delta = abs(e2 - e1) / e1 if e1 > 1e-9 else (1.0 if abs(e2) > 1e-9 else 0.0)
            if not section_starts or abs(bar_starts[i] - section_starts[-1]) > 1e-3: section_starts.append(bar_starts[i]); section_labels.append(f"S{section_index}"); section_index += 1
            if delta >= 0.5 and (i + 4 < num_bars):
                if not section_starts or abs(bar_starts[i+4] - section_starts[-1]) > 1e-3: section_starts.append(bar_starts[i + 4]); section_labels.append(f"S{section_index}"); section_index += 1
            i += 8
        else:
            if i < num_bars:
                 if not section_starts or abs(bar_starts[i] - section_starts[-1]) > 1e-3: section_starts.append(bar_starts[i]); section_labels.append(f"S{section_index}")
            break
    if section_starts:
        section_starts_np = np.array(section_starts); section_labels_np = np.array(section_labels); sorted_indices = np.argsort(section_starts_np)
        section_starts_sorted = section_starts_np[sorted_indices]; section_labels_sorted = section_labels_np[sorted_indices]
        unique_starts, unique_indices = np.unique(section_starts_sorted, return_index=True)
        section_starts = list(unique_starts); section_labels = list(section_labels_sorted[unique_indices])
    if not section_starts or section_starts[0] > trim_offset + 1e-3: section_starts.insert(0, trim_offset); section_labels.insert(0, "Start")
    elif section_starts and section_labels:
        if abs(section_starts[0] - trim_offset) < 1e-3: section_labels[0] = "Start"
    return section_starts, section_labels


def analyze_chroma_and_clusters(track_data):
    """Analyzes chroma, clusters sections, assigns colors and SEMANTIC labels (using 'Body'), adds duration check for Outro, relabels short final sections."""
    results = defaultdict(lambda: None); print("--- Starting NEW Cluster-Driven Labeling Approach ---")
    y = track_data.get("y_processed"); sr = track_data.get("sr"); bpm = track_data.get("bpm"); hop_length = track_data.get("hop_length")
    beat_frames = track_data.get("beat_frames"); rms_times = track_data.get("rms_times"); rms_frames = track_data.get("rms")
    trim_offset_sec = track_data.get("trim_offset_sec", 0); duration = track_data.get("duration_processed")
    section_starts = track_data.get("section_starts"); original_section_labels = track_data.get("section_labels"); seconds_per_bar = track_data.get("seconds_per_bar", 0)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']; results["note_names"] = note_names; fallback_color = FALLBACK_COLOR_HEX
    if (y is None or sr is None or bpm is None or hop_length is None or beat_frames is None or section_starts is None or original_section_labels is None or rms_times is None or rms_frames is None or duration is None or len(section_starts) == 0 or seconds_per_bar == 0):
        print("Warning: Missing/empty base data for enhanced labeling."); return results_on_failure(section_starts)
    num_sections = len(section_starts); section_times_abs = section_starts + [duration + trim_offset_sec]; total_duration_abs = duration + trim_offset_sec
    spec = track_data.get("spec"); freqs = track_data.get("freqs"); times_abs = track_data.get("times_absolute")
    if spec is None or freqs is None or times_abs is None:
        print("Calculating spectrogram within analyze_chroma_and_clusters..."); spec, freqs, times_abs, _, _ = _get_spec_data(y, sr, hop_length, trim_offset_sec)
        if spec is None: print("Warning: Spectrogram calculation failed.")
    else: print("Using existing spectrogram data.")
    if times_abs is not None: results["times_absolute"] = times_abs
    spectral_centroid_frames = None; spectral_bandwidth_frames = None; spectral_contrast_frames = None
    if spec is not None and freqs is not None:
        try:
            spectral_centroid_frames = librosa.feature.spectral_centroid(S=spec, freq=freqs)[0]; spectral_bandwidth_frames = librosa.feature.spectral_bandwidth(S=spec, freq=freqs)[0]
            spectral_contrast_frames = librosa.feature.spectral_contrast(S=spec, freq=freqs, n_bands=6); results["spectral_centroid_frames"] = spectral_centroid_frames
        except Exception as spec_feat_e: print(f"Warning: Could not calculate some frame-based spectral features: {spec_feat_e}")
    print("--- Phase 1: Enhanced Feature Extraction ---"); section_features = []; max_overall_avg_rms = 0
    for i in range(num_sections):
        features = {"index": i}; start_time_abs = section_starts[i]; end_time_abs = section_times_abs[i + 1]; start_time_rel = start_time_abs - trim_offset_sec; end_time_rel = end_time_abs - trim_offset_sec
        features["start_time"] = start_time_abs; features["end_time"] = end_time_abs; features["duration_sec"] = end_time_abs - start_time_abs
        features["duration_bars"] = round(features["duration_sec"] / seconds_per_bar) if seconds_per_bar > 0 else 0
        features["relative_position"] = start_time_abs / total_duration_abs if total_duration_abs > 0 else 0
        features["original_label"] = original_section_labels[i] if i < len(original_section_labels) else f"S{i+1}"
        rms_mask = (rms_times >= start_time_rel) & (rms_times < end_time_rel); section_rms_vals = rms_frames[rms_mask][np.isfinite(rms_frames[rms_mask])]; section_time_vals = rms_times[rms_mask][np.isfinite(rms_frames[rms_mask])]
        if section_rms_vals.size > 0:
            features["avg_rms"] = np.mean(section_rms_vals); features["peak_rms"] = np.max(section_rms_vals); features["rms_std_dev"] = np.std(section_rms_vals)
            max_overall_avg_rms = max(max_overall_avg_rms, features["avg_rms"])
            if section_rms_vals.size > 1:
                try: relative_time_vals = section_time_vals - section_time_vals[0]; slope, _, _, _, _ = scipy.stats.linregress(relative_time_vals, section_rms_vals); features["rms_trend"] = slope if np.isfinite(slope) else 0.0
                except ValueError as linreg_e: features["rms_trend"] = 0.0; print(f"Warning: Linregress failed section {i}: {linreg_e}")
            else: features["rms_trend"] = 0.0
        else: features["avg_rms"] = 0.0; features["peak_rms"] = 0.0; features["rms_std_dev"] = 0.0; features["rms_trend"] = 0.0
        if spec is not None and freqs is not None and times_abs is not None:
            spec_indices = np.where((times_abs >= start_time_abs) & (times_abs < end_time_abs))[0]
            if spec_indices.size > 0:
                section_spec = spec[:, spec_indices]; total_energy = np.sum(section_spec) + 1e-9; low_freq_mask = freqs < 150; high_freq_mask = freqs > 5000
                features["low_end_ratio"] = np.sum(section_spec[low_freq_mask,:]) / total_energy if np.any(low_freq_mask) else 0.0
                features["high_end_ratio"] = np.sum(section_spec[high_freq_mask,:]) / total_energy if np.any(high_freq_mask) else 0.0
                if spectral_centroid_frames is not None: section_centroid = spectral_centroid_frames[spec_indices]; features["spectral_centroid_avg"] = np.mean(section_centroid); features["spectral_centroid_std_dev"] = np.std(section_centroid)
                if spectral_bandwidth_frames is not None: features["spectral_bandwidth_avg"] = np.mean(spectral_bandwidth_frames[spec_indices])
                if spectral_contrast_frames is not None: features["spectral_contrast_avg"] = np.mean(np.mean(spectral_contrast_frames[:, spec_indices], axis=1))
            else: features["low_end_ratio"]=0.0; features["high_end_ratio"]=0.0
        features.setdefault("low_end_ratio", 0.0); features.setdefault("high_end_ratio", 0.0); features.setdefault("spectral_centroid_avg", 0.0); features.setdefault("spectral_centroid_std_dev", 0.0)
        features.setdefault("spectral_bandwidth_avg", 0.0); features.setdefault("spectral_contrast_avg", 0.0); features.setdefault("rms_trend", 0.0)
        section_features.append(features)
    print("--- Phase 2: Refined Clustering ---"); cluster_labels = list(range(num_sections)); clustering_successful = False; num_groups = 0
    feature_keys_for_clustering = ["avg_rms", "rms_std_dev", "rms_trend", "low_end_ratio", "high_end_ratio", "spectral_centroid_avg", "spectral_bandwidth_avg", "spectral_contrast_avg"]
    feature_matrix = []; valid_section_indices = []
    for i, f in enumerate(section_features):
        row = [f.get(key, 0.0) for key in feature_keys_for_clustering]
        if all(np.isfinite(val) for val in row): feature_matrix.append(row); valid_section_indices.append(i)
        else: print(f"Warning: Skipping section {i} in clustering due to non-finite features.")
    if len(feature_matrix) > 1:
        X_features = np.array(feature_matrix)
        try:
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_features); X_scaled = np.nan_to_num(X_scaled)
            num_groups = min(6, len(X_scaled)); num_groups = max(2, num_groups) if len(X_scaled) > 1 else 1
            if num_groups >= 2:
                print(f" Performing Agglomerative Clustering with {num_groups} clusters..."); clustering = AgglomerativeClustering(n_clusters=num_groups, linkage='ward').fit(X_scaled)
                new_labels = clustering.labels_; full_cluster_labels = np.full(num_sections, -1, dtype=int)
                for idx, label in zip(valid_section_indices, new_labels): full_cluster_labels[idx] = label
                cluster_labels = list(full_cluster_labels); clustering_successful = True; print(f" Clustering complete. New labels: {cluster_labels}")
            elif len(X_scaled) == 1: print("Warning: Only one valid section for refined clustering."); cluster_labels = [0] * num_sections; num_groups = 1; clustering_successful = True
            else: clustering_successful = False
        except Exception as cluster_e: print(f"Error during refined clustering: {cluster_e}"); traceback.print_exc(); print("Warning: Refined clustering failed."); cluster_labels = list(range(num_sections)); clustering_successful = False
    elif len(feature_matrix) == 1: print("Warning: Only one valid section for refined clustering."); cluster_labels = [0] * num_sections; num_groups = 1; clustering_successful = True
    else: print("Warning: Not enough valid sections for refined clustering."); clustering_successful = False
    for i in range(num_sections):
        if i < len(cluster_labels): section_features[i]['cluster_id'] = cluster_labels[i]
        else: section_features[i]['cluster_id'] = -1
    # Store cluster labels in results
    results["cluster_labels"] = list(cluster_labels) # Store the cluster IDs

    # *** Calculate initial colors based on cluster RMS ***
    rms_based_colors = [fallback_color] * num_sections # Initialize color list
    if clustering_successful and num_groups > 0:
        cluster_rms_totals = np.zeros(num_groups); cluster_counts = np.zeros(num_groups)
        for i, cid in enumerate(cluster_labels):
             avg_rms = section_features[i].get("avg_rms", 0.0)
             if 0 <= cid < num_groups: cluster_rms_totals[cid] += avg_rms; cluster_counts[cid] += 1
             elif cid != -1: print(f"Warning: Section {i} has invalid cluster ID {cid}.")
        cluster_rms_avg = np.divide(cluster_rms_totals, cluster_counts, out=np.zeros(num_groups), where=cluster_counts!=0); print(f"Recalculated Cluster RMS Averages: {cluster_rms_avg}")
        try:
            valid_cluster_indices = [idx for idx, count in enumerate(cluster_counts) if count > 0]
            if len(valid_cluster_indices) > 0:
                valid_cluster_rms_avg = cluster_rms_avg[valid_cluster_indices]; cluster_rms_pairs_for_color = list(zip(valid_cluster_indices, valid_cluster_rms_avg))
                sorted_clusters_for_color = sorted(cluster_rms_pairs_for_color, key=lambda item: item[1], reverse=True); sorted_cluster_ids = [cid for cid, rms in sorted_clusters_for_color]

                # *** Apply CUSTOM color mapping rule ***
                try: from section_editor import COLOR_NAME_MAP
                except ImportError: COLOR_NAME_MAP = {"Dark Red": DARK_RED_HEX, "Red": RED_HEX, "Orange": '#FFA500', "Dark Green": '#014421', "Light Green": '#7CCD7C', "Light Blue": '#ADD8E6', "Fade Out": FADE_OUT_COLOR_HEX}

                # Get the standard palette, excluding specific colors we'll manually assign
                palette = list(COLOR_NAME_MAP.values())
                manual_colors = [DARK_RED_HEX, RED_HEX, FADE_OUT_COLOR_HEX]
                remaining_palette = [c for c in palette if c not in manual_colors]
                while len(remaining_palette) < len(sorted_cluster_ids): # Pad if needed
                    remaining_palette.append(fallback_color)

                color_map = {}
                palette_idx = 0
                for j, cluster_id in enumerate(sorted_cluster_ids):
                    if j == 0: # Highest RMS
                        color_map[cluster_id] = DARK_RED_HEX
                        print(f"  Color Map: Cluster {cluster_id} (Rank 1) -> Dark Red")
                    elif j == 1: # Second Highest RMS
                        color_map[cluster_id] = DARK_RED_HEX # Assign Dark Red again
                        print(f"  Color Map: Cluster {cluster_id} (Rank 2) -> Dark Red")
                    elif j == 2: # Third Highest RMS
                        color_map[cluster_id] = RED_HEX # Assign Red
                        print(f"  Color Map: Cluster {cluster_id} (Rank 3) -> Red")
                    else: # Subsequent clusters
                        if palette_idx < len(remaining_palette):
                            color_map[cluster_id] = remaining_palette[palette_idx]
                            print(f"  Color Map: Cluster {cluster_id} (Rank {j+1}) -> {remaining_palette[palette_idx]}")
                            palette_idx += 1
                        else: # Should not happen if palette padded correctly
                            color_map[cluster_id] = fallback_color
                            print(f"  Color Map: Cluster {cluster_id} (Rank {j+1}) -> Fallback Grey (Palette Exhausted)")

                rms_based_colors = [color_map.get(cl, fallback_color) for cl in cluster_labels] # Store RMS-based colors
            else: print("Warning: No valid clusters found for color assignment.")
        except Exception as color_e: print(f"Error assigning initial colors based on clusters: {color_e}")
    # Store these initial colors temporarily in results, might be overwritten later
    # results["label_colors"] = list(rms_based_colors) # Assign initial colors here for now

    print("--- Phase 3: Simplified Cluster-Based Classification ---")
    semantic_labels = ["Body"] * num_sections; cluster_to_label_map = {}; sorted_clusters = [] # Default to Body
    if clustering_successful and len(cluster_rms_avg) > 0:
        valid_cluster_indices = [idx for idx, count in enumerate(cluster_counts) if count > 0]
        if len(valid_cluster_indices) > 0:
            valid_cluster_rms = cluster_rms_avg[valid_cluster_indices]; cluster_rms_pairs = list(zip(valid_cluster_indices, valid_cluster_rms))
            sorted_clusters = sorted(cluster_rms_pairs, key=lambda item: item[1], reverse=True); num_valid_clusters = len(sorted_clusters)
            if num_valid_clusters > 0: drop_cluster_id_1 = sorted_clusters[0][0]; cluster_to_label_map[drop_cluster_id_1] = "Drop"; print(f"  Mapping Cluster {drop_cluster_id_1} (Highest RMS) to: Drop")
            if num_valid_clusters > 1: drop_cluster_id_2 = sorted_clusters[1][0]; cluster_to_label_map[drop_cluster_id_2] = "Drop"; print(f"  Mapping Cluster {drop_cluster_id_2} (2nd Highest RMS) to: Drop")
            if num_valid_clusters > 2:
                build_cluster_id = sorted_clusters[2][0]
                if build_cluster_id not in cluster_to_label_map: cluster_to_label_map[build_cluster_id] = "Build"; print(f"  Mapping Cluster {build_cluster_id} (3rd Highest RMS) to: Build")
            if num_valid_clusters > 3:
                breakdown_cluster_id = sorted_clusters[-1][0]
                if breakdown_cluster_id not in cluster_to_label_map: cluster_to_label_map[breakdown_cluster_id] = "Breakdown"; print(f"  Mapping Cluster {breakdown_cluster_id} (Lowest RMS) to: Breakdown")
            elif num_valid_clusters == 3:
                 breakdown_cluster_id = sorted_clusters[-1][0]
                 if breakdown_cluster_id not in cluster_to_label_map: cluster_to_label_map[breakdown_cluster_id] = "Breakdown"; print(f"  Mapping Cluster {breakdown_cluster_id} (Lowest RMS, 3 clusters total) to: Breakdown")
            for cid, avg_rms in sorted_clusters:
                if cid not in cluster_to_label_map: cluster_to_label_map[cid] = "Body"; print(f"  Mapping Cluster {cid} (Other) to: Body")
            for i in range(num_sections): cid = section_features[i].get('cluster_id', -1); semantic_labels[i] = cluster_to_label_map.get(cid, "Body")
        else: print("Warning: No valid clusters to map labels from.")
    else: print("Warning: Clustering failed or no clusters found. Defaulting all to Body.")
    print(" Applying Overrides..."); outro_min_rel_pos = 0.85; fill_max_bars = 4; intro_forced = False; # MIN_OUTRO_BARS defined globally
    if num_sections > 0:
        semantic_labels[0] = "Intro"
        if num_sections > 1:
            cid_0 = section_features[0].get('cluster_id', -1); cid_1 = section_features[1].get('cluster_id', -1)
            if clustering_successful and cid_0 != -1 and cid_0 == cid_1: semantic_labels[1] = "Intro"; intro_forced = True
    for i in range(num_sections):
        f = section_features[i]
        if f["original_label"] == "Fill" and f["duration_bars"] <= fill_max_bars: semantic_labels[i] = "Fill"
        # *** Check duration BEFORE assigning Outro ***
        elif f["relative_position"] > outro_min_rel_pos and f["duration_bars"] >= MIN_OUTRO_BARS:
             if semantic_labels[i] not in ["Intro", "Fill", "Drop", "Build"]: semantic_labels[i] = "Outro"
    print(" Linking Intro/Outro by Cluster..."); intro_cluster_ids = set()
    if clustering_successful:
        for i in range(num_sections):
            cid = section_features[i].get('cluster_id', -1)
            if cid != -1 and semantic_labels[i] == "Intro": intro_cluster_ids.add(cid)
        print(f"  Identified Intro Clusters: {intro_cluster_ids}")
        if intro_cluster_ids:
            intro_max_rel_pos = 0.15
            for i in range(num_sections):
                 # Apply Outro link check ALSO considering duration
                 if section_features[i]['relative_position'] > outro_min_rel_pos and section_features[i]['duration_bars'] >= MIN_OUTRO_BARS:
                    cid = section_features[i].get('cluster_id', -1)
                    if cid in intro_cluster_ids:
                        if semantic_labels[i] != "Outro": print(f"  Relabeling section {i} ('{semantic_labels[i]}') as Outro based on Intro cluster match {cid}"); semantic_labels[i] = "Outro"
            intro_check_end_idx = 2 if intro_forced else (1 if semantic_labels[0] == "Intro" else 0)
            for i in range(intro_check_end_idx, num_sections):
                if section_features[i]['relative_position'] < intro_max_rel_pos:
                    cid = section_features[i].get('cluster_id', -1)
                    if cid in intro_cluster_ids:
                        if semantic_labels[i] != "Intro": print(f"  Relabeling section {i} ('{semantic_labels[i]}') as Intro based on shared cluster {cid}"); semantic_labels[i] = "Intro"
    labels_before_cleanup = list(semantic_labels); results["labels_before_cleanup"] = labels_before_cleanup
    print("--- Phase 4: Contextual Cleanup Rules ---"); final_labels = list(semantic_labels)
    print(" Applying Rule 1: Drop->Build->Body cleanup...")
    for i in range(num_sections - 2):
        if final_labels[i:i+3] == ["Drop", "Build", "Body"]: print(f"  Applying Rule 1 at index {i+1}: Changing Build to Breakdown."); final_labels[i+1] = "Breakdown"
    print(" Applying Rule 2: Build->Body->Drop cleanup...")
    for i in range(num_sections - 2):
         if final_labels[i:i+3] == ["Build", "Body", "Drop"]: print(f"  Applying Rule 2 at index {i+1}: Changing Body to Build."); final_labels[i+1] = "Build"
    print(" Applying Rule 3: Body->Body->Drop/Fill cleanup (Simplified)...")
    for i in range(num_sections - 2): # Check B -> B -> D
        if final_labels[i:i+3] == ["Body", "Body", "Drop"]: print(f"  Applying Rule 3 (BBD) at index {i+1}: Changing Body to Build."); final_labels[i+1] = "Build"
    for i in range(num_sections - 3): # Check B -> B -> F -> D
         if final_labels[i:i+4] == ["Body", "Body", "Fill", "Drop"]: print(f"  Applying Rule 3 (BBFD) at index {i+1}: Changing Body to Build."); final_labels[i+1] = "Build"
    print(" Applying Rule 4: Drop->Body->Drop cleanup...")
    for i in range(num_sections - 2): # Check D -> B -> D
         if final_labels[i:i+3] == ["Drop", "Body", "Drop"]: print(f"  Applying Rule 4 at index {i+1}: Changing Body to Build."); final_labels[i+1] = "Build"
    print(" Applying Rule 5: Merge short sections cleanup...")
    temp_labels_rule5 = list(final_labels); corrected_rule5 = False
    for i in range(1, num_sections - 1):
        if section_features[i].get('duration_bars', 0) == 4 and temp_labels_rule5[i] != "Fill":
            if temp_labels_rule5[i-1] == temp_labels_rule5[i+1] and temp_labels_rule5[i-1] != "Fill":
                 neighbor_label = temp_labels_rule5[i-1]
                 if temp_labels_rule5[i] != neighbor_label: print(f"  Applying Rule 5 at index {i}: Changing '{temp_labels_rule5[i]}' to '{neighbor_label}'."); temp_labels_rule5[i] = neighbor_label; corrected_rule5 = True
    if corrected_rule5: final_labels = temp_labels_rule5
    print(" Applying Rule 6: Consecutive Drops at End cleanup...")
    if num_sections >= 6 and all(l == "Drop" for l in final_labels[-6:]): print(f"  Applying Rule 6: Found 6+ consecutive Drops at end."); final_labels[-1] = "Outro"; final_labels[-2] = "Outro"
    elif num_sections >= 5 and all(l == "Drop" for l in final_labels[-5:]): print(f"  Applying Rule 6: Found 5 consecutive Drops at end."); final_labels[-1] = "Outro"; final_labels[-2] = "Outro"
    print(" Applying Rule 7: Drop->Body->Body with cluster change cleanup...")
    if clustering_successful:
        temp_labels_rule7 = list(final_labels); corrected_rule7 = False
        for i in range(num_sections - 2):
            if temp_labels_rule7[i:i+3] == ["Drop", "Body", "Body"]:
                cid1 = section_features[i+1].get('cluster_id', -1); cid2 = section_features[i+2].get('cluster_id', -1)
                if cid1 != -1 and cid2 != -1 and cid1 != cid2: print(f"  Applying Rule 7 at index {i+1}: Changing Body to Breakdown (D->B->B and Cluster {cid1} != {cid2})."); temp_labels_rule7[i+1] = "Breakdown"; corrected_rule7 = True
        if corrected_rule7: final_labels = temp_labels_rule7

    # *** Final Check: Relabel short, quiet final section as Fade Out ***
    num_final_sections = len(final_labels)
    if num_final_sections >= 1:
        last_idx = num_final_sections - 1
        last_section_duration = section_features[last_idx].get("duration_bars", 0)
        last_section_rms = section_features[last_idx].get("avg_rms", 0)
        if last_section_duration < MIN_FADEOUT_BARS and last_section_rms < FADEOUT_RMS_THRESHOLD:
            print(f"INFO: Relabeling final section {last_idx} ('{final_labels[last_idx]}') as Fade Out (Duration: {last_section_duration}, RMS: {last_section_rms:.3f})")
            final_labels[last_idx] = "Fade Out"

    # --- Prepare Final Output Lists ---
    final_semantic_labels = final_labels
    final_section_features = section_features # Features list corresponds to final_labels length
    final_cluster_labels = results["cluster_labels"] # Use the stored cluster labels
    final_labels_before_cleanup = results["labels_before_cleanup"] # Use the stored list
    final_section_starts = section_starts # Starts correspond to final_labels length

    # Ensure final list lengths are consistent before assigning colors
    min_len = len(final_semantic_labels)
    if not (len(final_section_features) == min_len and len(final_cluster_labels) == min_len and len(final_labels_before_cleanup) == min_len and len(final_section_starts) == min_len):
         print(f"!!! WARNING: Length mismatch in final lists before color assignment! Attempting to use length {min_len}.")
         final_section_features = final_section_features[:min_len]
         final_cluster_labels = final_cluster_labels[:min_len]
         final_labels_before_cleanup = final_labels_before_cleanup[:min_len]
         final_section_starts = final_section_starts[:min_len]
         # Also adjust rms_based_colors if its length doesn't match
         if len(rms_based_colors) != min_len:
             print(f"Adjusting rms_based_colors length from {len(rms_based_colors)} to {min_len}")
             rms_based_colors = (list(rms_based_colors) + [fallback_color] * min_len)[:min_len]


    # *** REVERTED COLOR ASSIGNMENT: Use RMS-based colors, override Fade Out ***
    final_label_colors = list(rms_based_colors) # Start with RMS-based colors

    # Ensure lists have the same length before override
    if len(final_label_colors) == len(final_semantic_labels):
        for i, label in enumerate(final_semantic_labels):
            if label == "Fade Out":
                final_label_colors[i] = FADE_OUT_COLOR_HEX # Assign specific color
        print("DEBUG: Applied Fade Out color override using RMS-based colors as base.")
    else:
        # Fallback if lengths mismatch (shouldn't happen after length correction, but keep as safety)
        print(f"!!! ERROR: Length mismatch persists between final labels ({len(final_semantic_labels)}) and RMS colors ({len(rms_based_colors)}). Using fallback grey.")
        final_label_colors = [fallback_color] * len(final_semantic_labels) # Fallback to grey


    # --- Store Final Results ---
    results["semantic_labels"] = final_semantic_labels
    results["section_features"] = final_section_features
    results["cluster_labels"] = final_cluster_labels
    results["label_colors"] = final_label_colors # Store the list using RMS-based + FadeOut override
    results["labels_before_cleanup"] = final_labels_before_cleanup
    results["section_starts"] = final_section_starts

    print("--- Simplified Cluster-Based Labeling Complete ---")
    return results


# --- Helper for failure cases ---
def results_on_failure(section_starts):
    """Returns default dict structure on analysis failure or skip."""
    num = len(section_starts) if section_starts else 0; fallback_color = FALLBACK_COLOR_HEX
    return defaultdict(lambda: None, {"semantic_labels": ["Error"] * num, "label_colors": [fallback_color] * num, "section_starts": section_starts if section_starts else [], "note_names": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], "cluster_labels": list(range(num)), "root_indices": None, "root_times_absolute": None, "chroma_sync": None, "low_energy_times": None, "low_energy_norm": None, "dyn_times_absolute": None, "dyn_range": None, "width_matrix": None, "times_absolute": None, "bands": None, "rms_time_absolute": None, "rms_harm": None, "rms_perc": None, "spec": None, "freqs": None, "section_features": [{} for _ in range(num)], "labels_before_cleanup": ["Error"] * num })

# --- Helper function moved from inside analyze_stereo_and_hpss ---
def _get_spec_data(y, sr, hop_length, trim_offset_sec, n_fft=4096):
    """Calculates and returns spectrogram data (magnitude, freqs, times, bands)."""
    spec = freqs = times_abs = min_frames = bands = None
    try:
        print(" Calculating Spectrogram..."); stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length); spec = np.abs(stft_result)
        if not np.all(np.isfinite(spec)): spec = np.nan_to_num(spec)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft); times_rel = librosa.frames_to_time(np.arange(spec.shape[1]), sr=sr, hop_length=hop_length)
        times_abs = times_rel + trim_offset_sec; min_frames = spec.shape[1]
        bands = {'Lows (<200 Hz)': freqs < 200, 'Low-Mids (200–500 Hz)': (freqs >= 200) & (freqs < 500), 'Mids (500–2000 Hz)': (freqs >= 500) & (freqs < 2000), 'High-Mids (2000–5000 Hz)': (freqs >= 2000) & (freqs < 5000), 'Highs (>5000 Hz)': freqs >= 5000}
    except Exception as spec_e: print(f"Error calculating spectrogram: {spec_e}"); return None, None, None, None, None
    return spec, freqs, times_abs, min_frames, bands


def analyze_stereo_and_hpss(track_data, calc_stereo=True, calc_hpss=True):
    """Calculates stereo width and/or harmonic/percussive separation based on flags."""
    results = defaultdict(lambda: None); y = track_data.get("y_processed"); sr = track_data.get("sr"); hop_length = track_data.get("hop_length")
    trim_offset_sec = track_data.get("trim_offset_sec", 0); file_path = track_data.get("file_path"); width_matrix = bands = spec = freqs = times_abs = rms_harm = rms_perc = rms_time_abs = None
    min_frames = None; n_fft = 4096
    try:
        if y is None: raise ValueError("Input audio 'y' is None.")
        spec = track_data.get("spec"); freqs = track_data.get("freqs"); times_abs = track_data.get("times_absolute"); bands = track_data.get("bands"); min_frames = spec.shape[1] if spec is not None else None
        if spec is None and (calc_stereo or calc_hpss):
             print("Calculating spectrogram within analyze_stereo_and_hpss..."); spec, freqs, times_abs, min_frames, bands = _get_spec_data(y, sr, hop_length, trim_offset_sec, n_fft)
             if spec is not None: results["spec"] = spec; results["freqs"] = freqs; results["times_absolute"] = times_abs; results["bands"] = bands
             else: calc_stereo = False; calc_hpss = False; print("Warning: Spectrogram calculation failed.")
        elif spec is not None: print("Using existing spectrogram data."); results["spec"] = spec; results["freqs"] = freqs; results["times_absolute"] = times_abs; results["bands"] = bands
        if calc_stereo:
            print(" Calculating stereo width...");
            if spec is None: print(" Skipping stereo width: Spectrogram failed."); results["width_matrix"] = None
            else:
                y_stereo = None
                try:
                    if file_path and os.path.exists(file_path): y_stereo, sr_stereo = librosa.load(file_path, sr=sr, mono=False);
                    if sr_stereo != sr: raise ValueError("Stereo sample rate mismatch.")
                    if y_stereo.ndim != 2 or y_stereo.shape[0] != 2: raise ValueError("Loaded audio is not stereo.")
                    # Removed warning about missing path, as it might not be needed if y_processed exists
                except Exception as e: print(f"Could not load valid stereo audio: {e}"); y_stereo = None
                if y_stereo is not None:
                    try:
                        spec_L = np.nan_to_num(np.abs(librosa.stft(y_stereo[0], n_fft=n_fft, hop_length=hop_length))); spec_R = np.nan_to_num(np.abs(librosa.stft(y_stereo[1], n_fft=n_fft, hop_length=hop_length)))
                        current_min_frames = spec.shape[1] if spec is not None else 0; c_min = min(spec_L.shape[1], spec_R.shape[1], current_min_frames) if current_min_frames > 0 else min(spec_L.shape[1], spec_R.shape[1])
                        if current_min_frames > 0 and c_min < current_min_frames: print(f"Warn: Stereo/Mono frame mismatch. Trimming mono spec/times to {c_min}."); results["spec"] = spec = spec[:, :c_min]; results["times_absolute"] = times_abs = times_abs[:c_min]; min_frames = c_min
                        elif current_min_frames == 0: min_frames = c_min
                        spec_L = spec_L[:, :c_min]; spec_R = spec_R[:, :c_min]; mid = (spec_L + spec_R) / 2; side = (spec_L - spec_R) / 2; results["width_matrix"] = np.nan_to_num(np.abs(side) / (np.abs(mid) + 1e-9))
                    except Exception as width_e: print(f"Error calculating width matrix: {width_e}"); results["width_matrix"] = None
                else: results["width_matrix"] = None
        else: results["width_matrix"] = None
        if calc_hpss:
             print(" Calculating HPSS...");
             try:
                 y_harm, y_perc = librosa.effects.hpss(y); rms_harm = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]; rms_perc = librosa.feature.rms(y=y_perc, hop_length=hop_length)[0]
                 rms_time_rel = librosa.frames_to_time(np.arange(len(rms_harm)), sr=sr, hop_length=hop_length); target_len = min_frames if min_frames is not None else len(rms_time_rel)
                 if len(rms_time_rel) != target_len: print(f"Warn: HPSS RMS time mismatch ({len(rms_time_rel)} vs {target_len}). Trimming RMS."); current_len = min(len(rms_time_rel), target_len); rms_time_rel = rms_time_rel[:current_len]; rms_harm = rms_harm[:current_len]; rms_perc = rms_perc[:current_len]
                 results["rms_harm"] = np.nan_to_num(rms_harm); results["rms_perc"] = np.nan_to_num(rms_perc); results["rms_time_absolute"] = rms_time_rel + trim_offset_sec
             except Exception as hpss_e: print(f"Error calculating HPSS: {hpss_e}"); results["rms_harm"] = None; results["rms_perc"] = None; results["rms_time_absolute"] = None
        else: results["rms_harm"] = None; results["rms_perc"] = None; results["rms_time_absolute"] = None
        return results
    except Exception as e: print("--- Error in analyze_stereo_and_hpss ---"); traceback.print_exc(); messagebox.showerror("Error", f"Stereo/HPSS Error:\n{e}"); results.setdefault("width_matrix", None); results.setdefault("rms_harm", None); results.setdefault("rms_perc", None); results.setdefault("rms_time_absolute", None); results.setdefault("spec", None); results.setdefault("freqs", None); results.setdefault("times_absolute", None); results.setdefault("bands", None); return results


# *** ADDED Function for HMM Feature Extraction ***
def calculate_bar_features(track_data):
    """
    Calculates features averaged over each bar for HMM input.
    Current features: Avg RMS, Avg Spectral Centroid.
    Returns features and corresponding labels per bar.
    """
    bar_starts = track_data.get("bar_starts_absolute")
    bar_rms = track_data.get("bar_rms_data")
    section_starts = track_data.get("section_starts")
    semantic_labels = track_data.get("semantic_labels")
    spec_centroid = track_data.get("spectral_centroid_frames")
    # spec_contrast = track_data.get("spectral_contrast_frames") # Uncomment if adding contrast
    frame_times_abs = track_data.get("times_absolute") # Absolute times for spectral features
    trim_offset = track_data.get("trim_offset_sec", 0)
    duration_processed = track_data.get("duration_processed", 0)

    # Add checks for all required data
    required_data = {
        "bar_starts": bar_starts, "bar_rms": bar_rms, "section_starts": section_starts,
        "semantic_labels": semantic_labels, "spec_centroid": spec_centroid,
        "frame_times_abs": frame_times_abs, "duration_processed": duration_processed,
        "trim_offset": trim_offset
        # "spec_contrast": spec_contrast # Add if using contrast
    }
    if any(v is None for v in required_data.values()):
        missing = [k for k, v in required_data.items() if v is None]
        print(f"Warning: Missing data for bar feature calculation: {missing}. Skipping track.")
        return [], []

    num_bars = len(bar_starts)
    if len(bar_rms) != num_bars:
        print(f"Warning: Mismatch between bar_starts ({num_bars}) and bar_rms ({len(bar_rms)}). Skipping track.")
        return [], []
    if spec_centroid is not None and len(spec_centroid) != len(frame_times_abs):
         print(f"Warning: Mismatch between spec_centroid ({len(spec_centroid)}) and frame_times ({len(frame_times_abs)}). Skipping track.")
         return [], []
    # Add contrast length check if using it

    bar_features_list = []
    bar_labels_list = []

    current_section_idx = 0
    for i in range(num_bars):
        bar_start_time = bar_starts[i]
        bar_end_time = bar_starts[i+1] if i + 1 < num_bars else duration_processed + trim_offset

        # Find the label for the current bar
        # Advance section index if bar start time passes the next section start
        while current_section_idx + 1 < len(section_starts) and bar_start_time >= section_starts[current_section_idx + 1]:
            current_section_idx += 1
        # Ensure index is valid
        if current_section_idx >= len(semantic_labels):
             print(f"Warning: Section index {current_section_idx} out of bounds for labels ({len(semantic_labels)}) at bar {i}. Assigning fallback.")
             current_label = "Unknown" # Assign a fallback or skip bar? Skipping might be safer.
             continue # Skip this bar if label is invalid
        else:
             current_label = semantic_labels[current_section_idx]


        # --- Calculate Features for the Bar ---
        # 1. Average RMS (already calculated)
        avg_rms_bar = bar_rms[i] if np.isfinite(bar_rms[i]) else 0.0

        # 2. Average Spectral Centroid
        avg_centroid_bar = 0.0 # Default
        if spec_centroid is not None:
            frame_mask = (frame_times_abs >= bar_start_time) & (frame_times_abs < bar_end_time)
            frames_in_bar_centroid = spec_centroid[frame_mask]
            if frames_in_bar_centroid.size > 0 and np.all(np.isfinite(frames_in_bar_centroid)):
                avg_centroid_bar = np.mean(frames_in_bar_centroid)
            if not np.isfinite(avg_centroid_bar): avg_centroid_bar = 0.0

        # 3. Average Spectral Contrast (Example - uncomment and adapt if needed)
        # avg_contrast_bar = 0.0
        # if spec_contrast is not None:
        #     frame_mask = (frame_times_abs >= bar_start_time) & (frame_times_abs < bar_end_time)
        #     frames_in_bar_contrast = spec_contrast[:, frame_mask] # Get contrast bands for frames in bar
        #     if frames_in_bar_contrast.size > 0:
        #         avg_contrast_bar_bands = np.mean(frames_in_bar_contrast, axis=1)
        #         avg_contrast_bar = np.mean(avg_contrast_bar_bands) # Overall average contrast
        #     if not np.isfinite(avg_contrast_bar): avg_contrast_bar = 0.0

        # Store features and label for this bar
        # Adjust feature list if adding more features
        bar_features_list.append([avg_rms_bar, avg_centroid_bar]) # Add avg_contrast_bar here if used
        bar_labels_list.append(current_label)

    return bar_features_list, bar_labels_list

