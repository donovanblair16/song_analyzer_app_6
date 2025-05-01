# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/chroma_analysis.py

"""
Functions for chroma analysis, clustering, and section labeling.
"""

import traceback
from collections import defaultdict
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.stats

# Import the shared constants
from .constants import (
    MIN_OUTRO_BARS, MIN_FADEOUT_BARS, FADEOUT_RMS_THRESHOLD,
    FADE_OUT_COLOR_HEX, FALLBACK_COLOR_HEX, DARK_RED_HEX, RED_HEX, NOTE_NAMES
)
# Import the spectrogram data function from spectral_analysis
from .spectral_analysis import _get_spec_data

def results_on_failure(section_starts):
    """Returns default dict structure on analysis failure or skip."""
    num = len(section_starts) if section_starts else 0
    fallback_color = FALLBACK_COLOR_HEX
    # Ensure essential keys expected by later steps exist, even if empty/None
    return defaultdict(lambda: None, {
        "semantic_labels": ["Error"] * num,
        "label_colors": [fallback_color] * num,
        "section_starts": section_starts if section_starts else [],
        "note_names": NOTE_NAMES,
        "cluster_labels": list(range(num)),  # Default cluster labels
        "root_indices": None,
        "root_times_absolute": None,
        "chroma_sync": None,
        "low_energy_times": None,  # Ensure these exist as None if calculation fails
        "low_energy_norm": None,
        "dyn_times_absolute": None,
        "dyn_range": None,
        "width_matrix": None,
        "times_absolute": None,  # Spectrogram times
        "bands": None,
        "rms_time_absolute": None,  # HPSS times
        "rms_harm": None,
        "rms_perc": None,
        "spec": None,  # Spectrogram data
        "freqs": None,
        "section_features": [{} for _ in range(num)],  # Empty feature dicts
        "labels_before_cleanup": ["Error"] * num
    })

def analyze_chroma_and_clusters(track_data):
    """
    Analyzes chroma features, performs clustering on sections, assigns semantic labels
    based on cluster properties and rules, calculates section features, and applies cleanup.

    Args:
        track_data (dict): Dictionary containing base analysis results including
                           'y_processed', 'sr', 'bpm', 'hop_length', 'beat_frames',
                           'section_starts', 'original_section_labels', 'rms_times', 'rms',
                           'trim_offset_sec', 'duration_processed', 'seconds_per_bar'.
                           May also contain 'spec', 'freqs', 'times_absolute'.

    Returns:
        dict: An updated dictionary containing 'semantic_labels', 'label_colors',
              'section_features' (list of dicts), 'cluster_labels', 'root_indices',
              'root_times_absolute', 'chroma_sync', 'labels_before_cleanup'.
              Returns default structure on failure.
    """
    results = defaultdict(lambda: None)
    print("--- Starting NEW Cluster-Driven Labeling Approach ---")

    # --- Retrieve necessary data from input ---
    y = track_data.get("y_processed")
    sr = track_data.get("sr")
    bpm = track_data.get("bpm")
    hop_length = track_data.get("hop_length")
    beat_frames = track_data.get("beat_frames")
    rms_times = track_data.get("rms_times")  # Relative times
    rms_frames = track_data.get("rms")
    trim_offset_sec = track_data.get("trim_offset_sec", 0)
    duration = track_data.get("duration_processed")
    section_starts = track_data.get("section_starts")  # Absolute times
    original_section_labels = track_data.get("section_labels")  # e.g., 'Start', 'S1', 'Fill'...
    seconds_per_bar = track_data.get("seconds_per_bar", 0)

    # Define note names and fallback color
    results["note_names"] = NOTE_NAMES
    fallback_color = FALLBACK_COLOR_HEX

    # --- Validate essential inputs ---
    if not all([y is not None, sr is not None, bpm is not None, hop_length is not None,
                beat_frames is not None, section_starts is not None, original_section_labels is not None,
                rms_times is not None, rms_frames is not None, duration is not None]) or \
       len(section_starts) == 0 or seconds_per_bar <= 0:
        print("Warning: Missing/empty base data required for enhanced labeling. Returning defaults.")
        return results_on_failure(section_starts)

    num_sections = len(section_starts)
    # Create absolute end times for sections
    section_times_abs = section_starts + [duration + trim_offset_sec]  # Add final end time
    total_duration_abs = duration + trim_offset_sec

    # --- Get or Calculate Spectrogram Data ---
    # Check if spectrogram data already exists from previous steps
    spec = track_data.get("spec")
    freqs = track_data.get("freqs")
    times_abs = track_data.get("times_absolute")  # Absolute times

    if spec is None or freqs is None or times_abs is None:
        print("Calculating spectrogram within analyze_chroma_and_clusters...")
        # Calculate spec if missing
        spec, freqs, times_abs, _, _ = _get_spec_data(y, sr, hop_length, trim_offset_sec)
        if spec is None:
            print("Warning: Spectrogram calculation failed within chroma analysis.")
            # Cannot proceed with spectral features if spec fails
            # Return default structure? Or try to continue with only RMS features?
            # Returning defaults for now.
            return results_on_failure(section_starts)
        # Store newly calculated spec data in results
        results["spec"] = spec
        results["freqs"] = freqs
        results["times_absolute"] = times_abs
    else:
        print("Using existing spectrogram data.")
        # Ensure existing data is stored in results for consistency
        results["spec"] = spec
        results["freqs"] = freqs
        results["times_absolute"] = times_abs

    # --- Calculate Frame-Based Spectral Features (if not already present) ---
    spectral_centroid_frames = track_data.get("spectral_centroid_frames")
    spectral_bandwidth_frames = track_data.get("spectral_bandwidth_frames")
    spectral_contrast_frames = track_data.get("spectral_contrast_frames")

    if spectral_centroid_frames is None or spectral_bandwidth_frames is None or spectral_contrast_frames is None:
        print("Calculating frame-based spectral features...")
        if spec is not None and freqs is not None:
            try:
                # Calculate features if missing
                if spectral_centroid_frames is None:
                    spectral_centroid_frames = librosa.feature.spectral_centroid(S=spec, freq=freqs)[0]
                    results["spectral_centroid_frames"] = spectral_centroid_frames
                if spectral_bandwidth_frames is None:
                    spectral_bandwidth_frames = librosa.feature.spectral_bandwidth(S=spec, freq=freqs)[0]
                    results["spectral_bandwidth_frames"] = spectral_bandwidth_frames
                if spectral_contrast_frames is None:
                    # Use 6 contrast bands by default
                    spectral_contrast_frames = librosa.feature.spectral_contrast(S=spec, sr=sr, hop_length=hop_length, freq=freqs, n_bands=6)
                    results["spectral_contrast_frames"] = spectral_contrast_frames
            except Exception as spec_feat_e:
                print(f"Warning: Could not calculate some frame-based spectral features: {spec_feat_e}")
                # Ensure defaults exist even if calculation fails partially
                results.setdefault("spectral_centroid_frames", None)
                results.setdefault("spectral_bandwidth_frames", None)
                results.setdefault("spectral_contrast_frames", None)
        else:
            print("Warning: Cannot calculate frame spectral features, spectrogram data missing.")

    # --- Phase 1: Enhanced Feature Extraction per Section ---
    print("--- Phase 1: Enhanced Feature Extraction per Section ---")
    section_features = []  # List to hold feature dictionaries for each section
    max_overall_avg_rms = 0  # Track max RMS for potential normalization/scaling later

    for i in range(num_sections):
        features = {"index": i}  # Initialize dict for the current section
        start_time_abs = section_starts[i]
        end_time_abs = section_times_abs[i + 1]  # Use pre-calculated end times
        # Calculate relative start/end times for indexing frame-based features
        start_time_rel = start_time_abs - trim_offset_sec
        end_time_rel = end_time_abs - trim_offset_sec

        # Store basic info
        features["start_time"] = start_time_abs
        features["end_time"] = end_time_abs
        features["duration_sec"] = end_time_abs - start_time_abs
        features["duration_bars"] = round(features["duration_sec"] / seconds_per_bar) if seconds_per_bar > 0 else 0
        features["relative_position"] = start_time_abs / total_duration_abs if total_duration_abs > 0 else 0
        features["original_label"] = original_section_labels[i] if i < len(original_section_labels) else f"S{i+1}"  # Initial label

        # --- Calculate RMS features for the section ---
        rms_mask = (rms_times >= start_time_rel) & (rms_times < end_time_rel)
        section_rms_vals = rms_frames[rms_mask][np.isfinite(rms_frames[rms_mask])]
        section_time_vals = rms_times[rms_mask][np.isfinite(rms_frames[rms_mask])]  # Corresponding relative times

        if section_rms_vals.size > 0:
            features["avg_rms"] = np.mean(section_rms_vals)
            features["peak_rms"] = np.max(section_rms_vals)
            features["rms_std_dev"] = np.std(section_rms_vals)
            
            # Calculate Crest Factor
            if features["avg_rms"] > 1e-9:  # Avoid division by zero
                features["crest_factor"] = features["peak_rms"] / features["avg_rms"]
            else:
                features["crest_factor"] = 1.0  # Default value if avg_rms is too small
            
            max_overall_avg_rms = max(max_overall_avg_rms, features["avg_rms"])  # Update max observed RMS
            # Calculate RMS trend (slope of linear regression)
            if section_rms_vals.size > 1:
                try:
                    relative_time_vals = section_time_vals - section_time_vals[0]  # Time relative to section start
                    slope, _, _, _, _ = scipy.stats.linregress(relative_time_vals, section_rms_vals)
                    features["rms_trend"] = slope if np.isfinite(slope) else 0.0
                except ValueError as linreg_e:  # Handle potential errors in linregress
                    features["rms_trend"] = 0.0
                    print(f"Warning: Linregress failed for RMS trend in section {i}: {linreg_e}")
            else:
                features["rms_trend"] = 0.0  # No trend for single point
        else:
            # Default RMS values if no valid frames found
            features["avg_rms"] = 0.0
            features["peak_rms"] = 0.0
            features["rms_std_dev"] = 0.0
            features["rms_trend"] = 0.0
            features["crest_factor"] = 1.0

        # --- Calculate Spectral features for the section ---
        if spec is not None and freqs is not None and times_abs is not None:
            # Find spectrogram frame indices corresponding to the section's absolute time
            spec_indices = np.where((times_abs >= start_time_abs) & (times_abs < end_time_abs))[0]
            if spec_indices.size > 0:
                section_spec = spec[:, spec_indices]
                total_energy = np.sum(section_spec) + 1e-9  # Add epsilon for stability

                # Low/High end energy ratios
                low_freq_mask = freqs < 150
                high_freq_mask = freqs > 5000
                features["low_end_ratio"] = np.sum(section_spec[low_freq_mask,:]) / total_energy if np.any(low_freq_mask) else 0.0
                features["high_end_ratio"] = np.sum(section_spec[high_freq_mask,:]) / total_energy if np.any(high_freq_mask) else 0.0

                # Average Spectral Centroid & Std Dev for the section
                if spectral_centroid_frames is not None and len(spectral_centroid_frames) == spec.shape[1]:
                    section_centroid = spectral_centroid_frames[spec_indices]
                    section_centroid_times = times_abs[spec_indices] - times_abs[spec_indices[0]] if spec_indices.size > 0 else []
                    
                    features["spectral_centroid_avg"] = np.mean(section_centroid)
                    features["spectral_centroid_std_dev"] = np.std(section_centroid)
                    
                    # Calculate Spectral Centroid Slope (similar to RMS trend)
                    if section_centroid.size > 1 and len(section_centroid_times) == len(section_centroid):
                        try:
                            sc_slope, _, _, _, _ = scipy.stats.linregress(section_centroid_times, section_centroid)
                            features["spectral_centroid_slope"] = sc_slope if np.isfinite(sc_slope) else 0.0
                        except Exception as sc_e:
                            features["spectral_centroid_slope"] = 0.0
                            print(f"Warning: Failed to calculate spectral centroid slope for section {i}: {sc_e}")
                    else:
                        features["spectral_centroid_slope"] = 0.0
                else: 
                    features["spectral_centroid_avg"] = 0.0
                    features["spectral_centroid_std_dev"] = 0.0
                    features["spectral_centroid_slope"] = 0.0

                # Average Spectral Bandwidth
                if spectral_bandwidth_frames is not None and len(spectral_bandwidth_frames) == spec.shape[1]:
                    features["spectral_bandwidth_avg"] = np.mean(spectral_bandwidth_frames[spec_indices])
                else: features["spectral_bandwidth_avg"]=0.0  # Default if missing

                # Average Spectral Contrast (average over bands first, then time)
                if spectral_contrast_frames is not None and spectral_contrast_frames.shape[1] == spec.shape[1]:
                    features["spectral_contrast_avg"] = np.mean(np.mean(spectral_contrast_frames[:, spec_indices], axis=1))
                else: features["spectral_contrast_avg"]=0.0  # Default if missing
            else:
                # Defaults if no spectrogram frames found for the section
                features["low_end_ratio"]=0.0; features["high_end_ratio"]=0.0
                features["spectral_centroid_avg"]=0.0; features["spectral_centroid_std_dev"]=0.0
                features["spectral_bandwidth_avg"]=0.0; features["spectral_contrast_avg"]=0.0
                features["spectral_centroid_slope"]=0.0  # Default value for missing feature
        else:
            # Defaults if base spectrogram data is missing
            features["low_end_ratio"]=0.0; features["high_end_ratio"]=0.0
            features["spectral_centroid_avg"]=0.0; features["spectral_centroid_std_dev"]=0.0
            features["spectral_bandwidth_avg"]=0.0; features["spectral_contrast_avg"]=0.0
            features["spectral_centroid_slope"]=0.0  # Default value for missing feature

        # Ensure all expected feature keys have a default value (0.0) if not calculated
        default_feature_keys = [
            "avg_rms", "peak_rms", "rms_std_dev", "rms_trend", "low_end_ratio",
            "high_end_ratio", "spectral_centroid_avg", "spectral_centroid_std_dev",
            "spectral_bandwidth_avg", "spectral_contrast_avg", "crest_factor", 
            "spectral_centroid_slope"
        ]
        for key in default_feature_keys:
            features.setdefault(key, 0.0)

        # Add the completed feature dictionary to the list
        section_features.append(features)

    # --- Phase 2: Refined Clustering based on extracted features ---
    print("--- Phase 2: Refined Clustering ---")
    cluster_labels = list(range(num_sections))  # Default: each section is its own cluster
    clustering_successful = False
    num_groups = 0  # Number of clusters found

    # Define features to use for clustering
    feature_keys_for_clustering = [
        "avg_rms", "rms_std_dev", "rms_trend", "low_end_ratio", "high_end_ratio",
        "spectral_centroid_avg", "spectral_bandwidth_avg", "spectral_contrast_avg"
    ]
    # Create feature matrix, handling potential non-finite values
    feature_matrix = []
    valid_section_indices = []  # Keep track of sections included in clustering
    for i, f_dict in enumerate(section_features):
        row = [f_dict.get(key, 0.0) for key in feature_keys_for_clustering]
        # Check if all values in the row are finite
        if all(np.isfinite(val) for val in row):
            feature_matrix.append(row)
            valid_section_indices.append(i)
        else:
            print(f"Warning: Skipping section {i} ('{section_features[i].get('original_label')}') in clustering due to non-finite features.")

    # Perform clustering only if enough valid sections exist
    if len(feature_matrix) > 1:
        X_features = np.array(feature_matrix)
        try:
            # Scale features before clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
            X_scaled = np.nan_to_num(X_scaled)  # Ensure no NaNs remain after scaling

            # Determine number of clusters (e.g., fixed or adaptive)
            num_groups = min(6, len(X_scaled))  # Limit to max 6 clusters, or fewer if fewer sections
            num_groups = max(2, num_groups) if len(X_scaled) > 1 else 1  # Need at least 2 clusters if possible

            if num_groups >= 2:
                print(f" Performing Agglomerative Clustering with {num_groups} clusters...")
                # Fit clustering model
                clustering = AgglomerativeClustering(n_clusters=num_groups, linkage='ward').fit(X_scaled)
                # Get cluster labels for the valid sections
                new_labels_for_valid_sections = clustering.labels_
                # Create a full list of cluster labels, mapping back to original section indices
                full_cluster_labels = np.full(num_sections, -1, dtype=int)  # Initialize with -1 (unclustered)
                for valid_idx, cluster_label in zip(valid_section_indices, new_labels_for_valid_sections):
                    full_cluster_labels[valid_idx] = cluster_label
                cluster_labels = list(full_cluster_labels)  # Final list of cluster IDs for all sections
                clustering_successful = True
                print(f" Clustering complete. Cluster labels assigned: {cluster_labels}")
            elif len(X_scaled) == 1:
                # Handle case with only one valid section
                print("Warning: Only one valid section for refined clustering.")
                cluster_labels = [0] * num_sections  # Assign all to cluster 0
                num_groups = 1
                clustering_successful = True
            else:
                # Should not happen if len(feature_matrix) > 1, but safety check
                clustering_successful = False
        except Exception as cluster_e:
            # Handle errors during clustering
            print(f"Error during refined clustering: {cluster_e}")
            traceback.print_exc()
            print("Warning: Refined clustering failed. Using default sequential labels.")
            cluster_labels = list(range(num_sections))  # Fallback
            clustering_successful = False
    elif len(feature_matrix) == 1:
        # Handle case with only one valid section from the start
        print("Warning: Only one valid section available for refined clustering.")
        cluster_labels = [0] * num_sections
        num_groups = 1
        clustering_successful = True
    else:
        # Handle case with no valid sections for clustering
        print("Warning: Not enough valid sections for refined clustering.")
        clustering_successful = False

    # Add cluster ID to each section's feature dictionary
    for i in range(num_sections):
        if i < len(cluster_labels):  # Check bounds
            section_features[i]['cluster_id'] = cluster_labels[i]
        else:
            section_features[i]['cluster_id'] = -1  # Assign default if something went wrong

    # Store cluster labels in results dictionary
    results["cluster_labels"] = list(cluster_labels)  # Store the final cluster IDs

    # --- Calculate initial colors based on cluster RMS (Used for initial mapping/debug) ---
    rms_based_colors = [fallback_color] * num_sections  # Initialize color list
    cluster_rms_avg = np.zeros(num_groups) if num_groups > 0 else np.array([])  # Initialize average RMS array
    if clustering_successful and num_groups > 0:
        cluster_rms_totals = np.zeros(num_groups)
        cluster_counts = np.zeros(num_groups)
        # Sum RMS and count sections per cluster
        for i, cid in enumerate(cluster_labels):
             avg_rms = section_features[i].get("avg_rms", 0.0)
             if 0 <= cid < num_groups:  # Check if cluster ID is valid
                 cluster_rms_totals[cid] += avg_rms
                 cluster_counts[cid] += 1
             elif cid != -1:  # Ignore sections not clustered (-1)
                 print(f"Warning: Section {i} has invalid cluster ID {cid} during RMS calculation.")
        # Calculate average RMS per cluster, handling division by zero
        cluster_rms_avg = np.divide(cluster_rms_totals, cluster_counts, out=np.zeros(num_groups), where=cluster_counts!=0)
        print(f"Recalculated Cluster RMS Averages: {cluster_rms_avg}")

        # Assign colors based on RMS rank (for debugging/initial viz)
        try:
            valid_cluster_indices = [idx for idx, count in enumerate(cluster_counts) if count > 0]
            if len(valid_cluster_indices) > 0:
                valid_cluster_rms_avg = cluster_rms_avg[valid_cluster_indices]
                # Pair valid cluster IDs with their average RMS
                cluster_rms_pairs_for_color = list(zip(valid_cluster_indices, valid_cluster_rms_avg))
                # Sort clusters by average RMS (descending)
                sorted_clusters_for_color = sorted(cluster_rms_pairs_for_color, key=lambda item: item[1], reverse=True)
                sorted_cluster_ids = [cid for cid, rms in sorted_clusters_for_color]

                # --- Apply CUSTOM color mapping rule based on RMS rank ---
                try:
                    # Try importing from section_editor, otherwise use defaults
                    from section_editor import COLOR_NAME_MAP
                except ImportError:
                    # Define default map if import fails
                    COLOR_NAME_MAP = {"Dark Red": DARK_RED_HEX, "Red": RED_HEX, "Orange": '#FFA500',
                                      "Dark Green": '#014421', "Light Green": '#7CCD7C',
                                      "Light Blue": '#ADD8E6', "Fade Out": FADE_OUT_COLOR_HEX}

                # Get standard palette, excluding colors used for specific ranks
                palette = list(COLOR_NAME_MAP.values())
                manual_colors = [DARK_RED_HEX, RED_HEX, FADE_OUT_COLOR_HEX]  # Colors assigned manually
                remaining_palette = [c for c in palette if c not in manual_colors]
                # Ensure palette is long enough, pad with fallback if needed
                while len(remaining_palette) < len(sorted_cluster_ids):
                    remaining_palette.append(fallback_color)

                color_map = {}  # Dictionary to map cluster ID -> color hex
                palette_idx = 0  # Index for iterating through remaining_palette
                # Assign colors based on RMS rank
                for j, cluster_id in enumerate(sorted_cluster_ids):
                    if j == 0:  # Highest RMS cluster
                        color_map[cluster_id] = DARK_RED_HEX
                        print(f"  Color Map (RMS Rank): Cluster {cluster_id} (Rank 1) -> Dark Red")
                    elif j == 1:  # Second Highest RMS cluster
                        color_map[cluster_id] = DARK_RED_HEX  # Assign Dark Red again
                        print(f"  Color Map (RMS Rank): Cluster {cluster_id} (Rank 2) -> Dark Red")
                    elif j == 2:  # Third Highest RMS cluster
                        color_map[cluster_id] = RED_HEX  # Assign Red
                        print(f"  Color Map (RMS Rank): Cluster {cluster_id} (Rank 3) -> Red")
                    else:  # Subsequent clusters get colors from the remaining palette
                        if palette_idx < len(remaining_palette):
                            assigned_color = remaining_palette[palette_idx]
                            color_map[cluster_id] = assigned_color
                            print(f"  Color Map (RMS Rank): Cluster {cluster_id} (Rank {j+1}) -> {assigned_color}")
                            palette_idx += 1
                        else:  # Fallback if palette somehow exhausted
                            color_map[cluster_id] = fallback_color
                            print(f"  Color Map (RMS Rank): Cluster {cluster_id} (Rank {j+1}) -> Fallback Grey (Palette Exhausted)")

                # Create the list of colors based on each section's cluster ID
                rms_based_colors = [color_map.get(cl_id, fallback_color) for cl_id in cluster_labels]
            else:
                print("Warning: No valid clusters found for RMS-based color assignment.")
        except Exception as color_e:
            print(f"Error assigning initial colors based on clusters: {color_e}")
            traceback.print_exc()
    # Note: These rms_based_colors are not directly used later; final colors are assigned based on final labels.

    # --- Phase 3: Simplified Cluster-Based Classification (Label Assignment) ---
    print("--- Phase 3: Simplified Cluster-Based Classification ---")
    semantic_labels = ["Body"] * num_sections  # Default all sections to 'Body' initially
    cluster_to_label_map = {}  # Dictionary to map cluster ID -> semantic label
    sorted_clusters = []  # Will hold (cluster_id, avg_rms) sorted pairs

    if clustering_successful and len(cluster_rms_avg) > 0:
        valid_cluster_indices = [idx for idx, count in enumerate(cluster_counts) if count > 0]
        if len(valid_cluster_indices) > 0:
            valid_cluster_rms = cluster_rms_avg[valid_cluster_indices]
            cluster_rms_pairs = list(zip(valid_cluster_indices, valid_cluster_rms))
            # Sort clusters by average RMS (descending)
            sorted_clusters = sorted(cluster_rms_pairs, key=lambda item: item[1], reverse=True)
            num_valid_clusters = len(sorted_clusters)

            # Map highest RMS clusters to 'Drop'
            if num_valid_clusters > 0:
                drop_cluster_id_1 = sorted_clusters[0][0]
                cluster_to_label_map[drop_cluster_id_1] = "Drop"
                print(f"  Mapping Cluster {drop_cluster_id_1} (Highest RMS) to: Drop")
            if num_valid_clusters > 1:
                drop_cluster_id_2 = sorted_clusters[1][0]
                # Avoid mapping the same cluster twice if only one cluster exists
                if drop_cluster_id_2 != drop_cluster_id_1:
                     cluster_to_label_map[drop_cluster_id_2] = "Drop"
                     print(f"  Mapping Cluster {drop_cluster_id_2} (2nd Highest RMS) to: Drop")

            # Map 3rd highest RMS cluster to 'Build'
            if num_valid_clusters > 2:
                build_cluster_id = sorted_clusters[2][0]
                if build_cluster_id not in cluster_to_label_map:  # Check if not already mapped
                    cluster_to_label_map[build_cluster_id] = "Build"
                    print(f"  Mapping Cluster {build_cluster_id} (3rd Highest RMS) to: Build")

            # Map lowest RMS cluster to 'Breakdown'
            if num_valid_clusters > 0:  # Check if there are any valid clusters
                 # Use the last cluster in the sorted list (lowest RMS)
                 breakdown_cluster_id = sorted_clusters[-1][0]
                 if breakdown_cluster_id not in cluster_to_label_map:  # Check if not already mapped
                     cluster_to_label_map[breakdown_cluster_id] = "Breakdown"
                     print(f"  Mapping Cluster {breakdown_cluster_id} (Lowest RMS) to: Breakdown")

            # Map any remaining unassigned clusters to 'Body'
            for cid, avg_rms in sorted_clusters:
                if cid not in cluster_to_label_map:
                    cluster_to_label_map[cid] = "Body"
                    print(f"  Mapping Cluster {cid} (Other) to: Body")

            # Assign labels to sections based on their cluster ID
            for i in range(num_sections):
                cid = section_features[i].get('cluster_id', -1)
                # Use mapped label, default to 'Body' if cluster was invalid (-1) or not mapped
                semantic_labels[i] = cluster_to_label_map.get(cid, "Body")
        else:
            print("Warning: No valid clusters to map labels from. All sections remain 'Body'.")
    else:
        print("Warning: Clustering failed or no clusters found. Defaulting all to 'Body'.")

    # --- Apply Overrides and Contextual Logic ---
    print(" Applying Overrides and Contextual Logic...")
    outro_min_rel_pos = 0.85  # Threshold for considering a section as potentially Outro
    fill_max_bars = 4  # Max duration for a section to be considered a Fill override
    intro_forced = False  # Flag if second section was forced to Intro

    # Force first section to Intro
    if num_sections > 0:
        semantic_labels[0] = "Intro"
        # If first two sections belong to the same cluster, force second to Intro too
        if num_sections > 1:
            cid_0 = section_features[0].get('cluster_id', -1)
            cid_1 = section_features[1].get('cluster_id', -1)
            if clustering_successful and cid_0 != -1 and cid_0 == cid_1:
                semantic_labels[1] = "Intro"
                intro_forced = True

    # Apply Fill override and initial Outro check
    for i in range(num_sections):
        f = section_features[i]
        # Override with Fill if original label was Fill and duration is short
        if f["original_label"] == "Fill" and f["duration_bars"] <= fill_max_bars:
            semantic_labels[i] = "Fill"
        # Initial Outro check: late position, long enough duration, not already critical label
        elif f["relative_position"] > outro_min_rel_pos and f["duration_bars"] >= MIN_OUTRO_BARS:
             if semantic_labels[i] not in ["Intro", "Fill", "Drop", "Build"]:  # Don't override these yet
                 semantic_labels[i] = "Outro"

    # Link Intro/Outro based on shared cluster IDs
    print(" Linking Intro/Outro by Cluster...")
    intro_cluster_ids = set()
    if clustering_successful:
        # Identify clusters associated with Intro sections
        for i in range(num_sections):
            cid = section_features[i].get('cluster_id', -1)
            if cid != -1 and semantic_labels[i] == "Intro":
                intro_cluster_ids.add(cid)
        print(f"  Identified Intro Clusters: {intro_cluster_ids}")

        if intro_cluster_ids:
            intro_max_rel_pos = 0.15  # Max position to be considered potentially Intro
            # Relabel late sections matching Intro clusters as Outro
            for i in range(num_sections):
                 # Check position, duration, and cluster match
                 if section_features[i]['relative_position'] > outro_min_rel_pos and \
                    section_features[i]['duration_bars'] >= MIN_OUTRO_BARS:
                    cid = section_features[i].get('cluster_id', -1)
                    if cid in intro_cluster_ids:
                        if semantic_labels[i] != "Outro":
                            print(f"  Relabeling section {i} ('{semantic_labels[i]}') as Outro based on Intro cluster match {cid}")
                            semantic_labels[i] = "Outro"
            # Relabel early sections matching Intro clusters as Intro
            # Determine how far to check based on whether second section was forced Intro
            intro_check_end_idx = 2 if intro_forced else (1 if semantic_labels[0] == "Intro" else 0)
            for i in range(intro_check_end_idx, num_sections):
                if section_features[i]['relative_position'] < intro_max_rel_pos:
                    cid = section_features[i].get('cluster_id', -1)
                    if cid in intro_cluster_ids:
                        if semantic_labels[i] != "Intro":
                            print(f"  Relabeling section {i} ('{semantic_labels[i]}') as Intro based on shared cluster {cid}")
                            semantic_labels[i] = "Intro"

    # Store labels before cleanup rules for comparison/debug
    labels_before_cleanup = list(semantic_labels)
    results["labels_before_cleanup"] = labels_before_cleanup

    # --- Phase 4: Contextual Cleanup Rules ---
    print("--- Phase 4: Contextual Cleanup Rules ---")
    final_labels = list(semantic_labels)  # Work on a copy

    # Rule 1: Drop -> Build -> Body => Drop -> Breakdown -> Body
    print(" Applying Rule 1: Drop->Build->Body cleanup...")
    for i in range(num_sections - 2):
        if final_labels[i:i+3] == ["Drop", "Build", "Body"]:
            print(f"  Applying Rule 1 at index {i+1}: Changing Build to Breakdown.")
            final_labels[i+1] = "Breakdown"

    # Rule 2: Build -> Body -> Drop => Build -> Build -> Drop
    print(" Applying Rule 2: Build->Body->Drop cleanup...")
    for i in range(num_sections - 2):
         if final_labels[i:i+3] == ["Build", "Body", "Drop"]:
             print(f"  Applying Rule 2 at index {i+1}: Changing Body to Build.")
             final_labels[i+1] = "Build"

    # Rule 3: Body -> Body -> Drop/Fill => Body -> Build -> Drop/Fill
    print(" Applying Rule 3: Body->Body->Drop/Fill cleanup (Simplified)...")
    for i in range(num_sections - 2):  # Check B -> B -> D
        if final_labels[i:i+3] == ["Body", "Body", "Drop"]:
            print(f"  Applying Rule 3 (BBD) at index {i+1}: Changing Body to Build.")
            final_labels[i+1] = "Build"
    for i in range(num_sections - 3):  # Check B -> B -> F -> D
         if final_labels[i:i+4] == ["Body", "Body", "Fill", "Drop"]:
             print(f"  Applying Rule 3 (BBFD) at index {i+1}: Changing Body to Build.")
             final_labels[i+1] = "Build"

    # Rule 4: Drop -> Body -> Drop => Drop -> Build -> Drop
    print(" Applying Rule 4: Drop->Body->Drop cleanup...")
    for i in range(num_sections - 2):  # Check D -> B -> D
         if final_labels[i:i+3] == ["Drop", "Body", "Drop"]:
             print(f"  Applying Rule 4 at index {i+1}: Changing Body to Build.")
             final_labels[i+1] = "Build"

    # Rule 5: Merge short (4-bar) sections surrounded by same label
    print(" Applying Rule 5: Merge short sections cleanup...")
    temp_labels_rule5 = list(final_labels)  # Work on a temp list for this rule
    corrected_rule5 = False
    for i in range(1, num_sections - 1):
        # Check if section i is short and not a Fill
        if section_features[i].get('duration_bars', 0) == 4 and temp_labels_rule5[i] != "Fill":
            # Check if neighbors have the same label (and not Fill)
            if temp_labels_rule5[i-1] == temp_labels_rule5[i+1] and temp_labels_rule5[i-1] != "Fill":
                 neighbor_label = temp_labels_rule5[i-1]
                 # If section i has a different label, change it to match neighbors
                 if temp_labels_rule5[i] != neighbor_label:
                     print(f"  Applying Rule 5 at index {i}: Changing '{temp_labels_rule5[i]}' to '{neighbor_label}'.")
                     temp_labels_rule5[i] = neighbor_label
                     corrected_rule5 = True
    # Apply changes if any were made
    if corrected_rule5: final_labels = temp_labels_rule5

    # Rule 6: Handle long sequences of Drops at the end
    print(" Applying Rule 6: Consecutive Drops at End cleanup...")
    # Check for 5 or 6 consecutive Drops ending the track
    if num_sections >= 6 and all(l == "Drop" for l in final_labels[-6:]):
        print(f"  Applying Rule 6: Found 6+ consecutive Drops at end.")
        final_labels[-1] = "Outro"; final_labels[-2] = "Outro"  # Label last two as Outro
    elif num_sections >= 5 and all(l == "Drop" for l in final_labels[-5:]):
        print(f"  Applying Rule 6: Found 5 consecutive Drops at end.")
        final_labels[-1] = "Outro"; final_labels[-2] = "Outro"  # Label last two as Outro

    # Rule 7: Drop -> Body -> Body with different clusters => Drop -> Breakdown -> Body
    print(" Applying Rule 7: Drop->Body->Body with cluster change cleanup...")
    if clustering_successful:
        temp_labels_rule7 = list(final_labels)  # Work on temp list
        corrected_rule7 = False
        for i in range(num_sections - 2):
            if temp_labels_rule7[i:i+3] == ["Drop", "Body", "Body"]:
                # Get cluster IDs for the two Body sections
                cid1 = section_features[i+1].get('cluster_id', -1)
                cid2 = section_features[i+2].get('cluster_id', -1)
                # If clusters are valid and different, change first Body to Breakdown
                if cid1 != -1 and cid2 != -1 and cid1 != cid2:
                    print(f"  Applying Rule 7 at index {i+1}: Changing Body to Breakdown (D->B->B and Cluster {cid1} != {cid2}).")
                    temp_labels_rule7[i+1] = "Breakdown"
                    corrected_rule7 = True
        # Apply changes if any were made
        if corrected_rule7: final_labels = temp_labels_rule7

    # --- Final Check: Relabel potential Fade Out ---
    # Check the very last section
    num_final_sections = len(final_labels)
    if num_final_sections >= 1:
        last_idx = num_final_sections - 1
        last_section_duration = section_features[last_idx].get("duration_bars", 0)
        last_section_rms = section_features[last_idx].get("avg_rms", 0)
        # Check if duration is short AND RMS is low
        if last_section_duration < MIN_FADEOUT_BARS and last_section_rms < FADEOUT_RMS_THRESHOLD:
            print(f"INFO: Relabeling final section {last_idx} ('{final_labels[last_idx]}') as Fade Out (Duration: {last_section_duration}, RMS: {last_section_rms:.3f})")
            final_labels[last_idx] = "Fade Out"

    # --- Prepare Final Output Lists ---
    final_semantic_labels = final_labels
    # Ensure other lists match the length of the final labels
    # This assumes section_features, cluster_labels etc. were not modified in length by cleanup rules
    # If rules involved merging/deleting, those lists need updating too (handled in merge logic now)
    final_section_features = section_features[:len(final_semantic_labels)]
    final_cluster_labels = results["cluster_labels"][:len(final_semantic_labels)]
    final_labels_before_cleanup = results["labels_before_cleanup"][:len(final_semantic_labels)]
    final_section_starts = section_starts[:len(final_semantic_labels)]

    # --- Assign Final Colors Based on Final Labels ---
    # Use the fixed color scheme defined globally or imported
    final_label_colors = []
    for i, label in enumerate(final_semantic_labels):
        if label == "Intro" or label == "Outro":
            final_label_colors.append(RED_HEX)
        elif label == "Drop":
            final_label_colors.append(DARK_RED_HEX)
        elif label == "Body":
            final_label_colors.append('#014421')  # Dark Green
        elif label == "Breakdown":
            final_label_colors.append('#ADD8E6')  # Light Blue
        elif label == "Build":
            final_label_colors.append('#FFA500')  # Orange
        elif label == "Fade Out":
            final_label_colors.append(FADE_OUT_COLOR_HEX)  # Purple
        elif label == "Fill":
            final_label_colors.append('#8A2BE2')  # Purple (Same as Fade Out for now)
        else:  # Unknown or other labels
            final_label_colors.append(fallback_color)
    print("DEBUG: Applied fixed color scheme based on final semantic labels.")

    # --- Store Final Results ---
    results["semantic_labels"] = final_semantic_labels
    results["section_features"] = final_section_features  # Store the feature dicts list
    results["cluster_labels"] = final_cluster_labels
    results["label_colors"] = final_label_colors  # Store the final colors
    results["labels_before_cleanup"] = final_labels_before_cleanup
    results["section_starts"] = final_section_starts

    # --- Calculate Chroma and Root Note ---
    # (Keep this part if needed, otherwise remove)
    try:
        print(" Calculating Chroma Features...")
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        # Sync chroma features to beats
        chroma_sync = librosa.util.sync(chroma_stft, beat_frames, aggregate=np.median)
        results["chroma_sync"] = chroma_sync  # Store beat-synchronous chroma

        # Estimate root note for each beat frame
        root_indices = np.argmax(chroma_sync, axis=0)
        results["root_indices"] = root_indices
        # Get absolute times for root notes (using beat times)
        beat_times_abs = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length) + trim_offset_sec
        results["root_times_absolute"] = beat_times_abs
        print(" Chroma and Root Note analysis complete.")
    except Exception as chroma_e:
        print(f"Error during Chroma/Root Note analysis: {chroma_e}")
        results["chroma_sync"] = None
        results["root_indices"] = None
        results["root_times_absolute"] = None


    print("--- Cluster-Driven Labeling and Feature Extraction Complete ---")
    return results  # Return the updated results dictionary