# =============================================================================
# FILE: audio_plotting.py
# Contains functions for creating matplotlib plots of audio features.
# Includes helpers to add section context overlays (lines, labels, secondary axis).
# Default vertical grid lines removed; explicit section lines added.
# FIXED: Added align='edge' to ax.bar in plot_hmm_posteriors to fix visual offset.
# FIXED: Corrected x-axis label setting in plot_emission_probabilities heatmap (v2).
# FIXED: Improved normalization for Section-State Distance heatmap to handle NaNs/outliers.
# UPDATED: Changed plot_feature_importance (ax2) to show section-centric view.
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Import ticker for FuncFormatter
import numpy as np
from scipy.ndimage import uniform_filter1d # For smoothing plots
import re # For parsing frequency strings
import traceback # Keep for debugging prints if needed
import math # Needed for bar number calculation
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar placement

# --- Utility Functions ---

def parse_freq_string(k):
    """Parses frequency band strings like '<200 Hz', '(2000–5000 Hz)', '>5000 Hz' for sorting."""
    try:
        k = k.lower().strip() # Normalize string
        base_val = 0 # Base value for sorting (used to put '>' bands last)
        num_str = '' # String containing the number
        match = re.search(r'([\d.]+k?)', k)
        if match:
            num_str = match.group(1)
            if '>' in k: base_val = 100000
            multiplier = 1000 if num_str.endswith('k') else 1
            num_str = num_str[:-1] if num_str.endswith('k') else num_str
            val = float(num_str) * multiplier
            return base_val + val
        else:
             match = re.search(r'\(?([\d.]+k?)\s*[-–]', k)
             if match:
                 num_str = match.group(1)
                 multiplier = 1000 if num_str.endswith('k') else 1
                 num_str = num_str[:-1] if num_str.endswith('k') else num_str
                 val = float(num_str) * multiplier
                 return base_val + val
             else: raise ValueError(f"No parsable number found in '{k}'")
    except Exception as e:
        print(f"Warning: Could not parse frequency string for sorting: '{k}'. Error: {e}")
        return float('inf')

# --- X-Axis Formatter Function (Fallback) ---
def time_to_bar_formatter(time_sec, pos, trim_offset, sec_per_bar):
    """Matplotlib FuncFormatter to convert time in seconds to bar number."""
    if sec_per_bar is None or sec_per_bar <= 1e-6: return f"{time_sec:.1f}s"
    try:
        relative_time = time_sec - trim_offset
        bar_num = math.floor(relative_time / sec_per_bar + 1e-6) + 1
        return f"{bar_num}"
    except Exception: return f"{time_sec:.1f}s"

# --- Function to Generate Structure Summary ---
def generate_structure_summary(labels, features):
    """Generates a summarized structure string (e.g., Intro 16 Bars - Drop 32 Bars)."""
    if not labels or not features or len(labels) != len(features): return "Error: Label/feature mismatch or empty data."
    summary_parts = [];
    if not labels: return ""
    current_label = labels[0]; current_bar_count = features[0].get('duration_bars', 0); num_sections = len(labels)
    for i in range(1, num_sections):
        label = labels[i]; bars = features[i].get('duration_bars', 0)
        if label == current_label: current_bar_count += bars
        else: part_str = f"{current_label} {current_bar_count} Bars"; summary_parts.append(part_str); current_label = label; current_bar_count = bars
    part_str = f"{current_label} {current_bar_count} Bars"; summary_parts.append(part_str)
    return " - ".join(summary_parts)

# --- Helper Functions for Structure Overlays ---

def _add_section_lines(ax, data, use_hmm_starts=False):
    """
    Adds vertical lines at section boundaries.
    Uses 'hmm_section_starts' if use_hmm_starts is True and available,
    otherwise defaults to 'section_starts'.
    """
    section_starts = None
    if use_hmm_starts and "hmm_section_starts" in data and data["hmm_section_starts"]:
        section_starts = data.get("hmm_section_starts")
        # print("DEBUG Plotting: Adding HMM section lines.") # Less verbose
    else:
        section_starts = data.get("section_starts")
        # print("DEBUG Plotting: Adding Original section lines.") # Less verbose

    if section_starts is None or len(section_starts) == 0: return
    # print(f"DEBUG Plotting: Adding {len(section_starts)} section lines.") # Less verbose
    for start_time in section_starts:
        # Draw lines behind plot elements (low zorder)
        ax.axvline(x=start_time, color='grey', linestyle=':', alpha=0.6, linewidth=0.8, zorder=1)

def _add_section_labels_text(ax, data, location='top', use_hmm_labels=False):
    """
    Adds section text labels above or below the plot area.
    Uses HMM starts/labels if use_hmm_labels is True and available,
    otherwise defaults to original starts/labels.
    """
    section_starts = None
    semantic_labels = None
    view_type = "Original"

    if use_hmm_labels and "hmm_section_starts" in data and data["hmm_section_starts"] and \
       "hmm_semantic_labels" in data and data["hmm_semantic_labels"]:
        section_starts = data.get("hmm_section_starts")
        semantic_labels = data.get("hmm_semantic_labels")
        view_type = "HMM"
    else:
        section_starts = data.get("section_starts")
        semantic_labels = data.get("semantic_labels")

    # print(f"DEBUG Plotting: Adding {view_type} section text labels at {location}.") # Less verbose

    sec_per_bar = data.get("seconds_per_bar", 0) # Needed for slight x offset

    has_sections = (section_starts is not None and
                    semantic_labels is not None and
                    len(section_starts) == len(semantic_labels) and
                    len(section_starts) > 0)

    if not has_sections: return
    # print(f"DEBUG Plotting: Adding {len(section_starts)} section text labels at {location}.") # Less verbose

    if location == 'top': y_pos_text = 1.02; va = 'bottom'
    else: y_pos_text = -0.25; va = 'top' # Below primary x-axis

    if location == 'top':
        current_ylim = ax.get_ylim()
        new_top_lim = current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.08
        if np.isfinite(current_ylim[0]) and np.isfinite(new_top_lim): ax.set_ylim(current_ylim[0], new_top_lim)
        else: print(f"Warning: Cannot adjust ylim for labels: {current_ylim}")

    for i, start_time in enumerate(section_starts):
        # Ensure label index is valid
        if i < len(semantic_labels):
            label_text = semantic_labels[i]
            x_offset = (sec_per_bar * 0.1) if sec_per_bar > 0 else 0.1
            ax.text(start_time + x_offset, y_pos_text, label_text,
                    transform=ax.get_xaxis_transform(), # X is data, Y is axes fraction
                    ha='left', va=va, fontsize=7, color='dimgrey', clip_on=False)
        else:
            print(f"Warning: Mismatch between section_starts and semantic_labels at index {i}")



def _add_bar_axis(ax, data, location='top'):
    """Adds a secondary x-axis showing bar numbers at the specified location ('top' or 'bottom')."""
    # Bar axis typically relates to the original structure, so we use original starts
    section_starts = data.get("section_starts")
    trim_offset = data.get("trim_offset_sec", 0)
    sec_per_bar = data.get("seconds_per_bar", 0)

    has_sections = (section_starts is not None and len(section_starts) > 0)
    if not has_sections or sec_per_bar <= 1e-6: return

    # print(f"DEBUG Plotting: Adding bar axis at {location}.") # Less verbose
    start_bars = [str(round((start_time - trim_offset) / sec_per_bar) + 1) for start_time in section_starts]

    secax = ax.secondary_xaxis(location=location)
    secax.set_xticks(section_starts)
    secax.set_xticklabels(start_bars, fontsize=7)
    secax.set_xlabel("Approx. Bar at Section Start", fontsize=8)
    if location == 'top': secax.tick_params(axis='x', direction='in', labelrotation=30); secax.xaxis.set_label_position('top')
    else: secax.tick_params(axis='x', direction='out', labelrotation=30, pad=-15); plt.setp(secax.get_xticklabels(), ha='right')
    secax.spines[[location]].set_visible(False)


# --- Generic Helper for Section Feature Step Plots ---
def _plot_feature_per_section(ax, data, feature_key, y_label, title_part):
    """Helper to draw a step plot of a feature calculated per section."""
    # This plot uses original section data
    section_starts = data.get("section_starts")
    section_features = data.get("section_features")
    duration_processed = data.get("duration_processed", 0)
    trim_offset = data.get("trim_offset_sec", 0)
    absolute_end_time = duration_processed + trim_offset

    if section_starts is None or section_features is None or len(section_starts) != len(section_features) or len(section_starts) == 0:
        ax.text(0.5, 0.5, f"{title_part} Data N/A", ha='center', va='center', color='grey'); ax.set_ylabel(y_label, fontsize=8); return

    try:
        values = [f.get(feature_key, np.nan) for f in section_features]
        section_times_abs = section_starts + [absolute_end_time]; x_coords_step = section_times_abs[:-1]
        valid_mask = np.isfinite(values)
        if not np.any(valid_mask):
             ax.text(0.5, 0.5, f"{title_part} Data Invalid", ha='center', va='center', color='orange'); ax.set_ylabel(y_label, fontsize=8); return

        ax.step(x_coords_step, values, where='post', color='darkcyan', lw=1.0)
        valid_values = np.array(values)[valid_mask]; min_val = np.min(valid_values); max_val = np.max(valid_values)
        padding = (max_val - min_val) * 0.1; padding = max(padding, abs(min_val)*0.1 if min_val != 0 else 0.01); padding = max(padding, abs(max_val)*0.1 if max_val != 0 else 0.01); padding = max(padding, 0.01)
        ax.set_ylim(min_val - padding, max_val + padding); ax.set_xlim(section_starts[0], absolute_end_time)
    except Exception as e:
        print(f"Error plotting feature '{feature_key}': {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, f"{title_part} Plot Error", ha='center', va='center', color='red')

    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.tick_params(axis='y', labelsize=7); ax.tick_params(axis='x', labelbottom=False)
    ax.margins(x=0.01)
    # Add section lines (original)
    _add_section_lines(ax, data, use_hmm_starts=False)


# --- Individual Feature Plotting Functions ---

def plot_waveform(ax, data, title):
    """
    Plots waveform with coloring/labels and spectral centroid overlay.
    Uses specific bar number ticks/labels on primary x-axis.
    Determines colors/labels based on whether HMM data is present and requested.
    """
    # Extract data
    y = data.get("y_processed"); sr = data.get("sr"); hop_length = data.get("hop_length", 256)
    trim_offset = data.get("trim_offset_sec", 0); duration_processed = data.get("duration_processed", 0)
    seconds_per_bar = data.get("seconds_per_bar")
    spec_centroid = data.get("spectral_centroid_frames"); spec_times = data.get("times_absolute")

    # Determine which section data to use (HMM or Original)
    # This assumes 'data' dict might contain HMM results if they are being displayed
    # We need a flag or check to know which view is active. Let's assume 'use_hmm_view' exists in 'data' if applicable.
    use_hmm_view = data.get("use_hmm_view", False) # Default to original view

    section_starts = None
    label_colors = None
    semantic_labels = None
    view_type = "Original"

    if use_hmm_view and "hmm_section_starts" in data and data["hmm_section_starts"]:
        section_starts = data.get("hmm_section_starts")
        label_colors = data.get("hmm_label_colors")
        semantic_labels = data.get("hmm_semantic_labels")
        view_type = "HMM"
        # print("DEBUG Waveform Plot: Using HMM section data.") # Less verbose
    else:
        section_starts = data.get("section_starts")
        label_colors = data.get("label_colors") # Original cleaned colors
        semantic_labels = data.get("semantic_labels") # Original cleaned labels
        # print("DEBUG Waveform Plot: Using Original section data.") # Less verbose


    # Validation
    if y is None or sr is None:
        ax.text(0.5, 0.5, "Waveform Missing", ha='center', va='center', color='red')
        ax.set_title(title); return

    has_sections = (section_starts is not None and
                    label_colors is not None and
                    semantic_labels is not None and
                    len(section_starts) == len(label_colors) == len(semantic_labels))

    # --- Plot Waveform ---
    absolute_end_time = duration_processed + trim_offset
    time_axis_waveform = np.linspace(trim_offset, absolute_end_time, num=len(y))
    ax.set_facecolor('#F0F0F0')
    max_amp = np.max(np.abs(y)) if len(y) > 0 else 1.0; max_amp = max(max_amp, 0.1)
    y_limit_top = max_amp * 1.25; ax.set_ylim(-max_amp * 1.05, y_limit_top)

    start_bars = [] # To store bar numbers for x-axis labels (based on original starts)
    original_starts = data.get("section_starts", []) # Always use original starts for bar axis

    if has_sections:
        section_times_abs = section_starts + [absolute_end_time]
        for i in range(len(section_starts)):
            start_time = section_starts[i]; end_time = min(section_times_abs[i + 1], absolute_end_time)
            start_sample = max(0, int(np.round((start_time - trim_offset) * sr))); end_sample = min(len(y), int(np.round((end_time - trim_offset) * sr)))

            # Calculate bar number based on ORIGINAL start time for consistency
            if i < len(original_starts) and seconds_per_bar and seconds_per_bar > 0:
                 bar_num = round((original_starts[i] - trim_offset) / seconds_per_bar) + 1
                 start_bars.append(bar_num)
            else:
                 start_bars.append(f"{start_time:.1f}s") # Fallback to time

            if start_sample < end_sample:
                 segment_time = np.linspace(start_time, end_time, num=(end_sample - start_sample), endpoint=False)
                 segment_y = y[start_sample:end_sample]
                 plot_color = label_colors[i] if i < len(label_colors) else '#808080'
                 label_text = semantic_labels[i] if i < len(semantic_labels) else f"S{i+1}"
                 ax.plot(segment_time, segment_y, color=plot_color, alpha=0.9, linewidth=0.7)
            # Draw VLines and Labels ABOVE waveform
            if absolute_end_time - start_time > 1.0:
                ax.axvline(x=start_time, color='black', linestyle=':', alpha=0.6, linewidth=0.8, zorder=1) # Added zorder
                label_y_pos = max_amp * 1.05
                ax.text(start_time + 0.01, label_y_pos, label_text, ha='left', va='bottom', fontsize=8, color=plot_color, weight='bold', bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))
            # else: print(f"Skipping vline/label for section start near end: {start_time:.2f}s") # Less verbose
    else: # If no section data
        ax.plot(time_axis_waveform, y, color='black', alpha=0.9, linewidth=0.7)
        status_msg = "Section data unavailable"; ax.text(0.02, 0.95, status_msg, transform=ax.transAxes, fontsize=8, color='gray', va='top')

    # Add Spectral Centroid Overlay
    if spec_centroid is not None and spec_times is not None and spec_centroid.size == spec_times.size and spec_centroid.size > 0:
        try:
            # print(" Plotting Spectral Centroid Overlay..."); # Less verbose
            smooth_size = max(1, int(spec_times.size * 0.04))
            centroid_smoothed = uniform_filter1d(spec_centroid, size=smooth_size); max_c = np.max(centroid_smoothed)
            centroid_norm = centroid_smoothed / max_c if max_c > 1e-6 else np.zeros_like(centroid_smoothed)
            ax2 = ax.twinx(); ax2.plot(spec_times, centroid_norm, color='#FFFFFF', linestyle='-', linewidth=1.2, alpha=0.9)
            ax2.set_ylabel("Norm. Brightness\n(Spec. Centroid)", fontsize=7, color='white')
            ax2.set_ylim(0, 1.05); ax2.tick_params(axis='y', labelcolor='white', labelsize=6); ax2.spines['right'].set_color('white'); ax2.grid(False)
        except Exception as e: print(f"Error plotting spectral centroid overlay: {e}"); traceback.print_exc()
    # else: print(" Skipping Spectral Centroid Overlay (data missing or invalid).") # Less verbose

    # Final styling for waveform plot
    ax.set_title(title, fontsize=10); ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(axis='y', labelsize=7); ax.grid(True, axis='y', linestyle=':', alpha=0.5) # Only horizontal grid
    ax.margins(x=0.01)

    # Modify X-axis Ticks and Labels (Use original starts for bar numbers)
    if original_starts and len(start_bars) == len(original_starts) and seconds_per_bar and seconds_per_bar > 0:
        ax.set_xticks(original_starts); ax.set_xticklabels(start_bars, fontsize=7, rotation=30, ha='right')
        ax.set_xlabel("Approx. Bar Number at Section Start", fontsize=9); plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    else: ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8)

    # Set x-axis limits
    if time_axis_waveform.size > 0 : ax.set_xlim(time_axis_waveform[0], time_axis_waveform[-1])
    else: ax.set_xlim(trim_offset, absolute_end_time)


# --- NEW Plotting Functions for Feature Tabs ---

def plot_energy_balance_features(axes, data, title):
    """Plots Energy & Balance features, each on its own subplot."""
    # print("DEBUG: Plotting Energy/Balance Features...") # Less verbose
    features_to_plot = [
        ('avg_rms', "Avg RMS"),
        ('rms_std_dev', "RMS Std Dev"),
        ('rms_trend', "RMS Trend (Sect)"),
        ('low_end_ratio', "Low End Ratio"),
        ('high_end_ratio', "High End Ratio")
    ]
    if len(axes) != len(features_to_plot):
        print(f"Error: Expected {len(features_to_plot)} axes for Energy/Balance, got {len(axes)}.")
        axes[0].text(0.5, 0.5, "Axes Mismatch", ha='center', va='center', color='red'); return

    # Add section labels and bar axis (original) above the TOP plot
    _add_section_labels_text(axes[0], data, location='top', use_hmm_labels=False)
    _add_bar_axis(axes[0], data, location='top')

    for i, (key, label) in enumerate(features_to_plot):
         _plot_feature_per_section(axes[i], data, key, label, label) # Uses original data
         if i < len(axes) - 1: # Hide x-labels on all but bottom plot
              axes[i].tick_params(axis='x', labelbottom=False)
              axes[i].set_xlabel("") # Explicitly remove label
         else: # Set final x-label on bottom plot
              axes[i].set_xlabel("Time (s)", fontsize=9) # Use Time(s) for primary axis
              axes[i].tick_params(axis='x', labelsize=8, labelbottom=True)


def plot_timbre_texture_features(axes, data, title):
    """Plots Timbre & Texture features, each on its own subplot."""
    # print("DEBUG: Plotting Timbre/Texture Features...") # Less verbose
    features_to_plot = [
        ('spectral_centroid_avg', "Spec Centroid Avg"),
        ('spectral_bandwidth_avg', "Spec Bandwidth Avg"),
        ('spectral_contrast_avg', "Spec Contrast Avg")
    ]
    if len(axes) != len(features_to_plot):
        print(f"Error: Expected {len(features_to_plot)} axes for Timbre/Texture, got {len(axes)}.")
        axes[0].text(0.5, 0.5, "Axes Mismatch", ha='center', va='center', color='red'); return

    # Add section labels and bar axis (original) above the TOP plot
    _add_section_labels_text(axes[0], data, location='top', use_hmm_labels=False)
    _add_bar_axis(axes[0], data, location='top')

    for i, (key, label) in enumerate(features_to_plot):
         _plot_feature_per_section(axes[i], data, key, label, label) # Uses original data
         if i < len(axes) - 1: # Hide x-labels on all but bottom plot
              axes[i].tick_params(axis='x', labelbottom=False)
              axes[i].set_xlabel("") # Explicitly remove label
         else: # Set final x-label on bottom plot
              axes[i].set_xlabel("Time (s)", fontsize=9) # Use Time(s) for primary axis
              axes[i].tick_params(axis='x', labelsize=8, labelbottom=True)

# --- Other Plotting Functions (Modified to add lines/labels) ---

def plot_low_energy(ax, data, title):
    """Plots normalized low-frequency energy (<150 Hz) over time."""
    times = data.get("low_energy_times"); low_energy_norm = data.get("low_energy_norm"); ylabel = "Low-End Energy\n(<150 Hz)"
    if times is None or low_energy_norm is None or times.size != low_energy_norm.size: ax.text(0.5, 0.5, "Low Energy Data Missing/Mismatch", ha='center', va='center', color='red')
    else: ax.plot(times, low_energy_norm, color='black', linewidth=0.8); ax.fill_between(times, 0, low_energy_norm, color='black', alpha=0.1); ax.set_xlim(times[0], times[-1]) if times.size > 0 else None
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.tick_params(axis='y', labelsize=7)
    # Add section context (original)
    _add_section_lines(ax, data, use_hmm_starts=False)
    _add_section_labels_text(ax, data, location='top', use_hmm_labels=False)
    ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8, labelbottom=True)

def plot_dynamic_range(ax, data, title):
    """Plots dynamic range (RMS / Peak) over time."""
    dyn_times = data.get("dyn_times_absolute"); dyn_range = data.get("dyn_range"); ylabel = "Dynamic Range\n(RMS / Peak)"
    if dyn_times is None or dyn_range is None or dyn_times.size != dyn_range.size: ax.text(0.5, 0.5, "Dyn Range Data Missing/Mismatch", ha='center', va='center', color='red')
    else: ax.plot(dyn_times, dyn_range, color='black', linewidth=0.8); ax.fill_between(dyn_times, 0, dyn_range, color='black', alpha=0.1); ax.set_xlim(dyn_times[0], dyn_times[-1]) if dyn_times.size > 0 else None
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.tick_params(axis='y', labelsize=7)
    # Add section context (original)
    _add_section_lines(ax, data, use_hmm_starts=False)
    _add_section_labels_text(ax, data, location='top', use_hmm_labels=False)
    ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8, labelbottom=True)

def plot_stereo_width_overlay(ax, data, title):
    """Plots smoothed, normalized stereo width per frequency band on the same axes."""
    width_matrix = data.get("width_matrix"); times = data.get("times_absolute"); bands = data.get("bands"); ylabel = "Stereo Width\n(Smoothed)"
    if (width_matrix is None or times is None or bands is None or not isinstance(bands, dict) or width_matrix.shape[1] != times.size): ax.text(0.5, 0.5, "Stereo Width Data Missing/Mismatch", ha='center', va='center', color='red'); ax.set_title(title, fontsize=10); return
    color_map = plt.get_cmap('viridis', len(bands));
    try: sorted_band_keys = sorted(bands.keys(), key=parse_freq_string); sorted_band_items = [(k, bands[k]) for k in sorted_band_keys]
    except Exception as sort_e: print(f"Error sorting bands: {sort_e}"); sorted_band_items = list(bands.items())
    plotted = False; handles = []
    for i, (band_name, mask) in enumerate(sorted_band_items):
        if isinstance(mask, np.ndarray) and mask.dtype == bool and mask.size == width_matrix.shape[0] and np.any(mask):
            width_band = np.mean(width_matrix[mask, :], axis=0); norm = np.max(width_band)
            if norm > 1e-6: width_norm = width_band / norm; smooth_size = max(1, int(times.size * 0.02)); width_smoothed = uniform_filter1d(width_norm, size=smooth_size); line, = ax.plot(times, width_smoothed, label=band_name.split('(')[0].strip(), color=color_map(i / max(1, len(bands)-1)), lw=1.0); handles.append(line); plotted = True
            else: line, = ax.plot(times, np.zeros_like(times), label=band_name.split('(')[0].strip(), color=color_map(i / max(1, len(bands)-1)), lw=1.0, ls=':'); handles.append(line)
        elif not (isinstance(mask, np.ndarray) and mask.dtype == bool and mask.size == width_matrix.shape[0]): print(f"Plot Warn (Stereo): Invalid mask for band '{band_name}'")
    if plotted: ax.legend(handles=handles, fontsize=6, loc='upper right', ncol=min(3, len(handles)))
    if times.size > 0: ax.set_xlim(times[0], times[-1])
    elif not plotted: ax.text(0.5, 0.5, "Stereo Width Data Empty", ha='center', va='center')
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.tick_params(axis='y', labelsize=7)
    # Add section context (original)
    _add_section_lines(ax, data, use_hmm_starts=False)
    _add_section_labels_text(ax, data, location='top', use_hmm_labels=False)
    ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8, labelbottom=True)

def plot_hpss(ax, data, title):
    """Plots smoothed, normalized Harmonic and Percussive RMS energy over time."""
    rms_time = data.get("rms_time_absolute"); rms_harm = data.get("rms_harm"); rms_perc = data.get("rms_perc"); ylabel = "Harm/Perc\nEnergy"
    if (rms_time is None or rms_harm is None or rms_perc is None or not (rms_time.size == rms_harm.size == rms_perc.size)): ax.text(0.5, 0.5, "HPSS Data Missing/Mismatch", ha='center', va='center', color='red')
    else:
        norm_h = np.max(rms_harm); norm_p = np.max(rms_perc); rms_h_norm = rms_harm / (norm_h + 1e-9); rms_p_norm = rms_perc / (norm_p + 1e-9)
        smooth_size = max(1, int(rms_time.size * 0.01)); rms_h_smooth = uniform_filter1d(rms_h_norm, size=smooth_size); rms_p_smooth = uniform_filter1d(rms_p_norm, size=smooth_size)
        ax.plot(rms_time, rms_h_smooth, label='Harmonic', color='purple', lw=1.0); ax.plot(rms_time, rms_p_smooth, label='Percussive', color='teal', lw=1.0)
        ax.fill_between(rms_time, 0, rms_h_smooth, color='purple', alpha=0.1); ax.fill_between(rms_time, 0, rms_p_smooth, color='teal', alpha=0.1)
        ax.legend(loc='upper right', fontsize=7); ax.set_xlim(rms_time[0], rms_time[-1]) if rms_time.size > 0 else None
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.tick_params(axis='y', labelsize=7)
    # Add section context (original)
    _add_section_lines(ax, data, use_hmm_starts=False)
    _add_section_labels_text(ax, data, location='top', use_hmm_labels=False)
    ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8, labelbottom=True)

def plot_root_note(ax, data, title):
    """Plots the estimated root note over time as a step plot."""
    root_times = data.get("root_times_absolute"); root_indices = data.get("root_indices"); note_names = data.get("note_names"); ylabel = "Est. Root Note"
    if root_times is None or root_indices is None or note_names is None: ax.text(0.5, 0.5, "Root Note Missing", ha='center', va='center', color='red')
    elif root_times.size != root_indices.size: ax.text(0.5, 0.5, "Root Note Mismatch", ha='center', va='center', color='orange')
    elif root_times.size == 0: ax.text(0.5, 0.5, "Root Note Empty", ha='center', va='center', color='grey')
    else: ax.step(root_times, root_indices, where='post', color='darkblue', lw=0.8); ax.set_yticks(range(len(note_names))); ax.set_yticklabels(note_names, fontsize=7); ax.set_ylim(-0.5, len(note_names) - 0.5); ax.set_xlim(root_times[0], root_times[-1]) if root_times.size > 0 else None
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    # Add section context (original)
    _add_section_lines(ax, data, use_hmm_starts=False)
    _add_section_labels_text(ax, data, location='top', use_hmm_labels=False)
    ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8, labelbottom=True)


# --- Plot Creation Helper Functions ---

def create_comparison_plot(data1, data2, plot_func, title, track1_name="Track 1", track2_name="Track 2"):
    """Creates a figure with two subplots for comparing a feature between two tracks."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, squeeze=False)
    axes = axes.flatten(); fig.suptitle(title, fontsize=14); max_xlim = 0

    ax1_title = f"{track1_name}: {title.split('Comparison')[0].strip()}"
    if data1:
        try: plot_func(axes[0], data1, ax1_title); xlim = axes[0].get_xlim(); max_xlim = max(max_xlim, xlim[1]) if xlim[1] > xlim[0] else max_xlim
        except Exception as e: print(f"Error plotting T1 '{title}': {e}"); traceback.print_exc(); axes[0].text(0.5, 0.5, f"Error T1", ha='center', va='center', color='red')
    else: axes[0].text(0.5, 0.5, f"{track1_name}: N/A", ha='center', va='center')
    axes[0].set_title(ax1_title, fontsize=10)
    axes[0].set_xlabel("") # Remove x label from top plot

    ax2_title = f"{track2_name}: {title.split('Comparison')[0].strip()}"
    if data2:
        try: plot_func(axes[1], data2, ax2_title); xlim = axes[1].get_xlim(); max_xlim = max(max_xlim, xlim[1]) if xlim[1] > xlim[0] else max_xlim
        except Exception as e: print(f"Error plotting T2 '{title}': {e}"); traceback.print_exc(); axes[1].text(0.5, 0.5, f"Error T2", ha='center', va='center', color='red')
    else: axes[1].text(0.5, 0.5, f"{track2_name}: N/A", ha='center', va='center')
    axes[1].set_title(ax2_title, fontsize=10)

    if max_xlim > 0: [ax.set_xlim(0, max_xlim) for ax in axes]
    else: t1_end = (data1.get("trim_offset_sec", 0) + data1.get("duration_processed", 0)) if data1 else 0; t2_end = (data2.get("trim_offset_sec", 0) + data2.get("duration_processed", 0)) if data2 else 0; max_end = max(t1_end, t2_end); [ax.set_xlim(0, max_end) for ax in axes] if max_end > 0 else None
    # X-label is set by the plot_func called on axes[1]
    try: fig.align_ylabels(axes)
    except Exception: pass
    # Adjust bottom margin for potential secondary axis + labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.subplots_adjust(hspace=0.15, bottom=0.15, top=0.9) # Adjust top/bottom
    return fig


def create_single_track_plot(data, plot_func, title, track_name="Track", num_rows=1, **kwargs):
    """Creates a figure with subplots for displaying features of one track."""
    figsize_h = 4 if num_rows == 1 else 2.5 * num_rows # Adjusted height slightly
    figsize_w = 12
    gridspec_kw = None

    # Determine if sharing x-axis is appropriate
    # Don't share if the plot_func is plot_emission_probabilities or plot_feature_importance
    share_x_flag = True
    if plot_func.__name__ in ['plot_emission_probabilities', 'plot_feature_importance']:
        share_x_flag = False
        print(f"DEBUG create_single_track_plot: Disabling sharex for {plot_func.__name__}")

    fig, axes = plt.subplots(num_rows, 1, figsize=(figsize_w, figsize_h), sharex=share_x_flag, squeeze=False)
    axes = axes.flatten() # Always return a flat array of axes
    plot_full_title = f"{track_name}: {title}" # Title only used if single plot

    if data:
        try:
            # Pass kwargs (like use_hmm_view for waveform) to the plot function
            plot_func(axes[0] if num_rows == 1 else axes, data, plot_full_title if num_rows == 1 else title, **kwargs)
            if num_rows > 1: # Set overall title if multiple subplots
                 fig.suptitle(f"{track_name}: {title}", fontsize=12, y=0.98) # Nudge title up slightly
        except Exception as e:
            print(f"Error plotting single track '{title}': {e}"); traceback.print_exc()
            axes[0].text(0.5, 0.5, f"Error", ha='center', va='center', color='red')
    else: # If no data
        axes[0].text(0.5, 0.5, f"{track_name}: N/A", ha='center', va='center')

    # --- Final Figure Adjustments ---
    if num_rows == 1: axes[0].set_title(plot_full_title, fontsize=10) # Set title only if single plot

    # Ensure bottom x-axis ticks are visible (label set by plot func or helper)
    # Only do this if sharex was True
    if share_x_flag:
        ax_bottom = axes[-1]
        ax_bottom.tick_params(axis='x', labelbottom=True, labelsize=8)

    if axes.size > 1:
        try: fig.align_ylabels(axes);
        except Exception: pass

    # Adjust margins for potential top/bottom labels/axis
    plt.tight_layout(rect=[0, 0.03, 1, 0.95 if num_rows == 1 else 0.92]) # Lower top slightly more for multi
    plt.subplots_adjust(hspace=0.25, bottom=0.15, top=0.88 if num_rows>1 else 0.95) # Adjust spacing (increased top margin for multi)

    return fig


def create_band_analysis_plot(data, track_name="Track"):
    """Creates a multi-panel plot showing energy and stereo width for each frequency band."""
    bands = data.get("bands"); spec = data.get("spec"); times = data.get("times_absolute"); width_matrix = data.get("width_matrix")
    if not bands or spec is None or times is None or not (times.size == spec.shape[1]): fig, ax = plt.subplots(1, 1, figsize=(12, 2)); ax.text(0.5, 0.5, "Band Data Missing/Mismatch", ha='center', va='center', color='red'); ax.set_title(f"{track_name}: Band Analysis"); return fig
    num_bands = len(bands); fig, axes = plt.subplots(num_bands, 1, figsize=(12, 2 * num_bands), sharex=True); axes = [axes] if num_bands == 1 else axes; fig.suptitle(f"{track_name}: Band Energy & Stereo Width", fontsize=14)
    try: sorted_band_keys = sorted(bands.keys(), key=parse_freq_string); sorted_band_items = [(k, bands[k]) for k in sorted_band_keys]
    except Exception as sort_e: print(f"Error sorting bands: {sort_e}"); sorted_band_items = list(bands.items())
    cmap_w = plt.get_cmap('coolwarm'); valid_w = width_matrix is not None and width_matrix.shape[1] == times.size; plotted_any_energy = False

    # Add section labels and bar axis (original) above the TOP plot
    _add_section_labels_text(axes[0], data, location='top', use_hmm_labels=False)
    _add_bar_axis(axes[0], data, location='top')

    for i, (ax, (band_name, mask)) in enumerate(zip(reversed(axes), reversed(sorted_band_items))):
        is_valid_mask = isinstance(mask, np.ndarray) and mask.dtype == bool and mask.size == spec.shape[0]; ax2 = None
        if is_valid_mask and np.any(mask):
            # Plot Energy
            energy = np.sum(spec[mask, :], axis=0); norm = np.max(energy); e_norm = energy / norm if norm > 1e-6 else np.zeros_like(energy); ax.plot(times, e_norm, label='Energy', color='black', lw=1.0); ax.fill_between(times, 0, e_norm, color='black', alpha=0.1); plotted_any_energy = True
            # Plot Width
            if valid_w:
                ax2 = ax.twinx(); w_band = np.mean(width_matrix[mask, :], axis=0); norm_w = np.max(w_band)
                if norm_w > 1e-6: w_norm = w_band / norm_w; w_smooth = uniform_filter1d(w_norm, size=max(1, int(times.size * 0.02))); avg_w = np.mean(w_smooth); c = cmap_w(avg_w); ax2.plot(times, w_smooth, color=c, lw=1.5, alpha=0.8, label='Width')
                else: ax2.plot(times, np.zeros_like(times), color='gray', lw=1.0, ls=':', alpha=0.5)
                ax2.set_ylim(0, 1.05); ax2.set_ylabel("Norm. Width", fontsize=8, color='gray'); ax2.tick_params(axis='y', labelcolor='gray', labelsize=8); ax2.grid(False)
            else: ax.text(0.98, 0.85, 'Width N/A', ha='right', transform=ax.transAxes, color='gray', fontsize=8)
        else: ax.text(0.5, 0.5, 'Data N/A', ha='center', va='center', transform=ax.transAxes)
        # Styling
        ax.set_ylabel(band_name, fontsize=9); ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
        ax.tick_params(axis='y', labelsize=8)
        if ax2: ax.tick_params(axis='y', labelleft=False)
        # Add section lines (original) to this axis
        _add_section_lines(ax, data, use_hmm_starts=False)
        # Hide x-axis labels on all but bottom plot
        if i < num_bands - 1: ax.tick_params(axis='x', labelbottom=False)

    # Final adjustments for bottom axis
    ax_bottom = axes[-1]
    if plotted_any_energy and times.size > 0: ax_bottom.set_xlim(times[0], times[-1])
    ax_bottom.set_xlabel("Time (s)", fontsize=9) # Primary axis is Time
    ax_bottom.tick_params(axis='x', labelsize=8, labelbottom=True)

    try: fig.align_ylabels(axes)
    except Exception: pass
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]); plt.subplots_adjust(hspace=0.15, bottom=0.1, top=0.88) # Adjust top/bottom more
    return fig


def create_band_comparison_plot(data1, data2, track1_name="Track 1", track2_name="Track 2"):
    """Compares band energy & stereo width for two tracks in a multi-panel plot."""
    # (This plot type doesn't easily support the secondary structure axis from one track)
    # (Keeping original implementation for now)
    ref_data = data1 if data1 and data1.get("bands") else data2
    if not ref_data or not ref_data.get("bands"): fig, ax = plt.subplots(1, 1, figsize=(12, 2)); ax.text(0.5, 0.5, "No Band Data Available", ha='center', va='center'); return fig
    bands = ref_data.get("bands"); num_bands = len(bands); fig, axes = plt.subplots(num_bands, 1, figsize=(12, 2 * num_bands), sharex=True); axes = [axes] if num_bands == 1 else axes; fig.suptitle("Band Energy & Stereo Width Comparison", fontsize=14)
    try: sorted_band_keys = sorted(bands.keys(), key=parse_freq_string); sorted_band_items = [(k, bands[k]) for k in sorted_band_keys]
    except Exception as sort_e: print(f"Error sorting bands: {sort_e}"); sorted_band_items = list(bands.items())
    cmap_w = plt.get_cmap('coolwarm'); max_dur = 0
    for data in [data1, data2]:
        if data and data.get("times_absolute") is not None and data["times_absolute"].size > 0: max_dur = max(max_dur, data["times_absolute"][-1])
    for i, (ax, (band_name, _)) in enumerate(zip(reversed(axes), reversed(sorted_band_items))):
        ax.set_ylabel(band_name, fontsize=9); ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
        ax.tick_params(axis='y', labelsize=8); ax2 = ax.twinx(); ax2.set_ylim(0, 1.05); ax2.set_ylabel("Norm. Width", fontsize=8, color='gray'); ax2.tick_params(axis='y', labelcolor='gray', labelsize=8); ax2.grid(False)
        handles = []; plotted_t1 = False; plotted_t2 = False
        if data1: s1 = data1.get("spec"); t1 = data1.get("times_absolute"); w1 = data1.get("width_matrix"); m1 = data1.get("bands", {}).get(band_name); v_s1 = s1 is not None and t1 is not None and m1 is not None and m1.size == s1.shape[0] and t1.size == s1.shape[1]; v_w1 = w1 is not None and t1 is not None and w1.shape[1] == t1.size
        if v_s1 and np.any(m1): e1 = np.sum(s1[m1,:], axis=0); n1 = np.max(e1); en1 = e1 / n1 if n1 > 1e-6 else np.zeros_like(e1); p1, = ax.plot(t1, en1, color='black', lw=1.0, label=f"{track1_name} E"); handles.append(p1); plotted_t1 = True
        if v_w1: wb1 = np.mean(w1[m1,:], axis=0); nw1 = np.max(wb1)
        if nw1 > 1e-6: wn1 = wb1 / nw1; ws1 = uniform_filter1d(wn1, size=max(1, int(t1.size * 0.02))); aw1 = np.mean(ws1); c1 = cmap_w(aw1 * 0.4 + 0.6); pw1, = ax2.plot(t1, ws1, color=c1, lw=1.2, alpha=0.8, label=f"{track1_name} W"); handles.append(pw1)
        if data2: s2 = data2.get("spec"); t2 = data2.get("times_absolute"); w2 = data2.get("width_matrix"); m2 = data2.get("bands", {}).get(band_name); v_s2 = s2 is not None and t2 is not None and m2 is not None and m2.size == s2.shape[0] and t2.size == s2.shape[1]; v_w2 = w2 is not None and t2 is not None and w2.shape[1] == t2.size
        if v_s2 and np.any(m2): e2 = np.sum(s2[m2,:], axis=0); n2 = np.max(e2); en2 = e2 / n2 if n2 > 1e-6 else np.zeros_like(e2); p2, = ax.plot(t2, en2, color='blue', lw=1.0, ls='--', label=f"{track2_name} E"); handles.append(p2); plotted_t2 = True
        if v_w2: wb2 = np.mean(w2[m2,:], axis=0); nw2 = np.max(wb2)
        if nw2 > 1e-6: wn2 = wb2 / nw2; ws2 = uniform_filter1d(wn2, size=max(1, int(t2.size * 0.02))); aw2 = np.mean(ws2); c2 = cmap_w(aw2 * 0.4 + 0.1); pw2, = ax2.plot(t2, ws2, color=c2, lw=1.2, alpha=0.8, ls='--', label=f"{track2_name} W"); handles.append(pw2)
        if not plotted_t1: ax.text(0.5, 0.6, f"{track1_name}: N/A", transform=ax.transAxes, ha='center', va='center', color='grey', fontsize=8)
        if not plotted_t2: ax.text(0.5, 0.4, f"{track2_name}: N/A", transform=ax.transAxes, ha='center', va='center', color='grey', fontsize=8)
        if i == num_bands - 1 and handles: ax.legend(handles=handles, fontsize=7, loc='upper right', ncol=2)
        if ax2: ax.tick_params(axis='y', labelleft=False)
        # Hide x-axis labels on all but bottom plot
        if i < num_bands - 1: ax.tick_params(axis='x', labelbottom=False)

    # Set label only on bottom axis for comparison plot
    ax_bottom = axes[-1]
    if max_dur > 0: ax_bottom.set_xlim(0, max_dur)
    ax_bottom.set_xlabel("Time (s)", fontsize=9)
    ax_bottom.tick_params(axis='x', labelsize=8, labelbottom=True)

    if axes.size > 0:
        try: fig.align_ylabels(axes);
        except Exception: pass
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.subplots_adjust(hspace=0.15)
    return fig

def plot_hmm_posteriors(ax, data, title):
    """
    Plots the posterior probabilities for each HMM state as a stacked bar plot.
    Each state has a different color, showing how confident the model is in its
    state assignments for each section. Uses HMM section starts for alignment.
    """
    print("DEBUG Plotting: Plotting HMM Posteriors...")
    # --- Get HMM specific data ---
    hmm_posteriors = data.get("hmm_posteriors")
    hmm_section_starts = data.get("hmm_section_starts")
    hmm_semantic_labels = data.get("hmm_semantic_labels") # Predicted labels
    int_to_label = data.get("hmm_int_to_label")  # Mapping from model aux data

    # --- Validation ---
    if (hmm_posteriors is None or hmm_section_starts is None or
        not isinstance(hmm_posteriors, np.ndarray) or hmm_posteriors.ndim != 2 or
        len(hmm_section_starts) != hmm_posteriors.shape[0]): # Check length consistency
        ax.text(0.5, 0.5, "HMM Posteriors Missing or Mismatched", ha='center', va='center', color='red')
        ax.set_title(title, fontsize=10)
        print(" -> HMM Posteriors plot aborted due to missing/mismatched data.")
        return

    try:
        n_sections, n_states = hmm_posteriors.shape
        print(f" -> Plotting {n_sections} sections, {n_states} states.")

        # --- Create a colormap for the states ---
        # Use a perceptually uniform map if many states, otherwise tab10/tab20
        cmap_name = 'viridis' if n_states > 20 else ('tab20' if n_states > 10 else 'tab10')
        cmap = plt.cm.get_cmap(cmap_name, n_states)
        colors = [cmap(i) for i in range(n_states)]

        # --- Determine state labels for the legend ---
        state_labels_for_legend = []
        if int_to_label and isinstance(int_to_label, dict) and len(int_to_label) == n_states:
            # Use the mapping directly if it matches the number of states
            state_labels_for_legend = [int_to_label.get(i, f"State {i}") for i in range(n_states)]
            # print(f" -> Using int_to_label map for legend: {state_labels_for_legend}") # Less verbose
        else:
            # Fallback: Generate generic state labels
            state_labels_for_legend = [f"State {i}" for i in range(n_states)]
            # print(f" -> Using generic State labels for legend (int_to_label map missing or mismatch).") # Less verbose


        # --- Calculate section end times needed for bar widths ---
        section_end_times = []
        if n_sections > 0:
            # End times are the start times of the *next* section
            section_end_times = list(hmm_section_starts[1:])
            # Estimate the end time of the *last* section
            last_section_start = hmm_section_starts[-1]
            estimated_end = data.get("duration_processed", 0) + data.get("trim_offset_sec", 0) # Use track end
            # If last start is beyond estimated end, use a small duration
            if last_section_start >= estimated_end:
                # Estimate duration based on second-to-last section if possible
                if n_sections > 1:
                    estimated_duration = hmm_section_starts[-1] - hmm_section_starts[-2]
                else:
                    estimated_duration = 10.0 # Arbitrary if only one section
                section_end_times.append(last_section_start + max(0.1, estimated_duration)) # Ensure positive width
            else:
                section_end_times.append(estimated_end) # Append estimated end of track

        if len(section_end_times) != n_sections:
            print(f"ERROR: Mismatch calculating section end times ({len(section_end_times)} vs {n_sections}). Aborting plot.")
            ax.text(0.5, 0.5, "Error calculating section ends", ha='center', va='center', color='red')
            return

        # --- Plot stacked bars for each section ---
        for i in range(n_sections):
            x_start = hmm_section_starts[i]
            x_end = section_end_times[i]
            width = x_end - x_start

            if width <= 0: # Skip sections with zero or negative width
                print(f"Warning: Skipping section {i} at time {x_start:.2f} due to non-positive width ({width:.2f}).")
                continue

            bottom = 0
            for j in range(n_states): # Iterate through states (columns of posteriors)
                prob = hmm_posteriors[i, j]
                if prob > 1e-6: # Only plot if probability is non-negligible
                    # *** THE FIX: Add align='edge' ***
                    ax.bar(x_start, prob, width=width, bottom=bottom, color=colors[j],
                           edgecolor='white', linewidth=0.2, alpha=0.8, align='edge')

                    # Add text label inside the bar if probability and width are sufficient
                    if prob > 0.15 and width > 1.0: # Adjust thresholds as needed
                        text_y = bottom + prob / 2
                        # Use state label from legend list
                        bar_label_text = state_labels_for_legend[j]
                        # Determine text color based on background brightness (heuristic)
                        bg_lum = 0.299*colors[j][0] + 0.587*colors[j][1] + 0.114*colors[j][2]
                        text_color = 'white' if bg_lum < 0.5 else 'black'
                        ax.text(x_start + width / 2, text_y, bar_label_text,
                                ha='center', va='center', fontsize=7, color=text_color,
                                clip_on=True) # Clip text to axes bounds

                bottom += prob # Increment bottom for the next stack

        # --- Add Legend ---
        handles = [plt.Rectangle((0,0), 1, 1, color=colors[i]) for i in range(n_states)]
        ax.legend(handles, state_labels_for_legend, loc='upper right', fontsize=7, ncol=min(3, n_states))

    except Exception as e:
        print(f"Error plotting HMM posteriors: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error plotting posteriors: {str(e)[:50]}...",
                ha='center', va='center', color='red')

    # --- Set labels and title ---
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("State Probability", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8, labelbottom=True)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.margins(x=0.01) # Add slight margin to x-axis

    # --- Add section lines and labels (using HMM data for this plot) ---
    _add_section_lines(ax, data, use_hmm_starts=True)
    _add_section_labels_text(ax, data, location='top', use_hmm_labels=True)

    # print(" -> HMM Posteriors plot generation complete.") # Less verbose


def plot_feature_importance(axes, data, title):
    """
    Plots feature importance and contribution for each section.
    Shows which features contributed most to state assignments.

    Args:
        axes: List of matplotlib axes for plotting (expects 2 axes)
        data: Dictionary containing track data with HMM results
        title: Plot title string
    """
    print("DEBUG: Plotting Feature Importance...")

    # Check if we have enough axes
    if len(axes) < 2:
        print(
            f"Error: Expected at least 2 axes for Feature Importance, got {len(axes)}."
        )
        axes[0].text(0.5, 0.5, "Axes Mismatch", ha="center", va="center", color="red")
        return

    # Get HMM feature importance data
    importance_data = data.get("hmm_feature_importance")
    if not importance_data:
        axes[0].text(
            0.5,
            0.5,
            "No Feature Importance Data Available",
            ha="center",
            va="center",
            color="red",
        )
        axes[1].text(
            0.5,
            0.5,
            "Run HMM prediction with feature importance enabled",
            ha="center",
            va="center",
            color="red",
        )
        return

    # Get relevant data from importance_data
    feature_names = importance_data.get("feature_names", [])
    state_means = importance_data.get("state_means", [])
    state_feature_importance = importance_data.get("state_feature_importance", [])
    section_feature_contribution = importance_data.get(
        "section_feature_contribution", []
    )
    state_to_label_map = importance_data.get("state_to_label_map", {})

    if not feature_names or not state_means or not state_feature_importance:
        axes[0].text(
            0.5,
            0.5,
            "Invalid Feature Importance Data",
            ha="center",
            va="center",
            color="red",
        )
        return

    # ---- Plot 1: Overall Feature Importance per State ----
    ax1 = axes[0]
    n_states = len(state_feature_importance)
    n_features = len(feature_names)

    # Debug info about data structure
    # print(f"DEBUG: Number of states: {n_states}") # Less verbose
    # print(f"DEBUG: Number of features: {n_features}") # Less verbose

    # Create color map for states with more distinctive colors
    def get_distinct_colors(n, cmap_name='tab20'): # Default to tab20
        """Generate highly distinct colors for better visualization"""
        cmap = plt.cm.get_cmap(cmap_name)
        # Handle cases where n > number of colors in map
        num_colors_in_map = len(cmap.colors)
        colors = [cmap(i % num_colors_in_map) for i in range(n)]
        return colors


    state_colors = get_distinct_colors(n_states)

    # Process feature names for better display
    # Make feature names more readable
    short_feature_names = []
    for name in feature_names:
        # Replace underscores and title-case
        processed_name = name.replace("_", " ").title()
        # Abbreviate long names
        if len(processed_name) > 15:
            processed_name = processed_name[:15] + "..."
        short_feature_names.append(processed_name)

    # Plot feature importance for each state as a grouped bar chart
    bar_width = 0.8 / n_states
    bar_positions = np.arange(n_features)

    for i, (state_imp, color) in enumerate(zip(state_feature_importance, state_colors)):
        label = state_to_label_map.get(i, f"State {i}")
        x_positions = bar_positions + (i - n_states / 2 + 0.5) * bar_width

        # Ensure state_imp is a 1D array of the right length
        state_imp_arr = np.array(state_imp, dtype=float)

        # Add bars for this state
        ax1.bar(
            x_positions,
            state_imp_arr,
            width=bar_width,
            color=color,
            alpha=0.7,
            label=label,
        )

    # Customize the plot
    ax1.set_ylabel("Feature Importance", fontsize=9)
    ax1.set_title("Overall Feature Importance per State", fontsize=10)
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(short_feature_names, rotation=45, ha="right", fontsize=8)

    # Improve legend layout and visibility
    ax1.legend(
        fontsize=8,
        loc="upper right",
        ncol=min(n_states, 3),
        framealpha=0.7,  # Semi-transparent background for better readability
    )

    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)


    # ---- Plot 2: Feature Contribution per Section (NEW VISUALIZATION) ----
    ax2 = axes[1]

    # Get HMM state assignments and section starts
    hmm_semantic_labels = data.get("hmm_semantic_labels", [])

    if (
        not hmm_semantic_labels
        or not section_feature_contribution
        or len(hmm_semantic_labels) != len(section_feature_contribution)
    ):
        ax2.text(
            0.5,
            0.5,
            "No/Mismatched Section Feature Contribution Data",
            ha="center",
            va="center",
            color="orange",
        )
        return

    # Prepare data: Calculate absolute contributions and normalize per section
    try:
        # Convert contributions to a NumPy array, handle potential None values
        contrib_list_of_lists = []
        valid_section_indices = []
        original_labels_for_valid = []
        for i, contrib in enumerate(section_feature_contribution):
            if contrib is not None and i < len(hmm_semantic_labels):
                 # Ensure inner list/array has the correct number of features
                 contrib_arr = np.array(contrib, dtype=float)
                 if len(contrib_arr) == n_features:
                     contrib_list_of_lists.append(contrib_arr)
                     valid_section_indices.append(i)
                     original_labels_for_valid.append(hmm_semantic_labels[i])
                 else:
                     print(f"Warning: Section {i} contribution length ({len(contrib_arr)}) != n_features ({n_features}). Skipping.")
            else:
                 print(f"Warning: Contribution data missing or label index out of bounds for section {i}. Skipping.")

        if not contrib_list_of_lists:
             raise ValueError("No valid contribution vectors found.")

        contrib_matrix = np.array(contrib_list_of_lists) # Shape (n_valid_sections, n_features)
        abs_contrib_matrix = np.abs(contrib_matrix)

        # Normalize row-wise (per section)
        row_sums = abs_contrib_matrix.sum(axis=1, keepdims=True)
        # Add epsilon to avoid division by zero if a section has zero total contribution
        normalized_matrix = abs_contrib_matrix / (row_sums + 1e-9)

        n_valid_sections = normalized_matrix.shape[0]

    except Exception as e:
        print(f"Error processing contribution data: {e}")
        traceback.print_exc()
        ax2.text(0.5, 0.5, "Error Processing Contributions", ha='center', va='center', color='red')
        return


    # Create color map for features
    feature_colors = get_distinct_colors(n_features, cmap_name='tab10') # Use a different map

    # Plot stacked bars: sections on x-axis, features stacked
    section_indices = np.arange(n_valid_sections)
    bottom = np.zeros(n_valid_sections)

    for j in range(n_features): # Iterate through features
        feature_name = short_feature_names[j] # Use the processed names
        contributions = normalized_matrix[:, j] # Contributions of this feature across all sections
        ax2.bar(
            section_indices,
            contributions,
            bottom=bottom,
            label=feature_name,
            color=feature_colors[j],
            alpha=0.8
        )
        bottom += contributions # Add to the bottom for the next feature stack

    # Customize the plot
    ax2.set_ylabel("Normalized Feature Contribution (%)", fontsize=9)
    ax2.set_title("Feature Contribution Breakdown per Section", fontsize=10)

    # Create section labels for x-axis
    section_tick_labels = [f"Sec {i+1}\n({lbl})" for i, lbl in zip(valid_section_indices, original_labels_for_valid)]
    ax2.set_xticks(section_indices)
    ax2.set_xticklabels(section_tick_labels, rotation=45, ha="right", fontsize=7) # Smaller font for potentially many labels

    # Improve legend (for features)
    ax2.legend(
        title="Features",
        fontsize=7,
        loc='center left', # Place legend outside plot
        bbox_to_anchor=(1, 0.5), # Position to the right
        ncol=1 # Single column legend
    )

    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax2.set_ylim(0, 1.05) # Y-axis represents percentage (0 to 1)
    ax2.margins(x=0.02) # Add small margin to x-axis

    # Adjust layout slightly to make space for legend
    plt.subplots_adjust(right=0.85) # Make space on the right


def plot_emission_probabilities(axes, data, title):
    """
    Plots emission probability distributions and Mahalanobis distances.
    Shows how well sections match expected state distributions.

    Args:
        axes: List of matplotlib axes for plotting (expects 2 axes)
        data: Dictionary containing track data with HMM results
        title: Plot title string
    """
    print("DEBUG: Plotting Emission Probabilities...")

    # Check if we have enough axes
    if len(axes) < 2:
        print(
            f"Error: Expected at least 2 axes for Emission Probabilities, got {len(axes)}."
        )
        axes[0].text(0.5, 0.5, "Axes Mismatch", ha="center", va="center", color="red")
        return

    # Get HMM feature importance data (which contains emission probabilities)
    importance_data = data.get("hmm_feature_importance")
    if not importance_data:
        axes[0].text(
            0.5,
            0.5,
            "No Emission Probability Data Available",
            ha="center",
            va="center",
            color="red",
        )
        axes[1].text(
            0.5,
            0.5,
            "Run HMM prediction with emission probabilities enabled",
            ha="center",
            va="center",
            color="red",
        )
        return

    # Get relevant data from importance_data
    feature_names = importance_data.get("feature_names", [])
    state_means = importance_data.get("state_means", [])
    state_variances = importance_data.get("state_variances", [])
    state_to_label_map = importance_data.get("state_to_label_map", {})
    section_mahalanobis_distances = importance_data.get(
        "section_mahalanobis_distances", []
    )

    if not feature_names or not state_means or not state_variances:
        axes[0].text(
            0.5,
            0.5,
            "Invalid Emission Probability Data",
            ha="center",
            va="center",
            color="red",
        )
        return

    # ---- Plot 1: State-Feature Heatmap ----
    ax1 = axes[0]
    n_states = len(state_means)
    n_features = len(feature_names)

    # Debug info about data structures
    # print(f"DEBUG: Number of states: {n_states}") # Less verbose
    # print(f"DEBUG: Number of features: {n_features}") # Less verbose
    # print(f"DEBUG: Shape of state means: {[np.array(means).shape for means in state_means]}") # Less verbose
    # print(f"DEBUG: Shape of state variances: {[np.array(var).shape for var in state_variances]}") # Less verbose

    if n_states == 0 or n_features == 0:
        ax1.text(
            0.5, 0.5, "No State-Feature Data", ha="center", va="center", color="orange"
        )
        return

    # Create heatmap data matrix
    # We'll use a normalized version of means and variances
    heatmap_data = np.zeros((n_states, n_features))

    # Transform state means through the feature weights to show importance
    for i, means in enumerate(state_means):
        if i < len(state_variances):
            # Convert means and variances to flat arrays if they're not already
            means_array = np.atleast_1d(np.array(means).flatten())
            variances_array = np.atleast_1d(np.array(state_variances[i]).flatten())

            # Ensure both arrays are of expected length
            if len(means_array) != n_features or len(variances_array) != n_features:
                print(f"WARNING: State {i} has mismatched feature dimensions:")
                # print(f"  - Means shape: {means_array.shape}, expected {n_features}") # Less verbose
                # print(f"  - Variances shape: {variances_array.shape}, expected {n_features}") # Less verbose

                # Trim or pad to match expected dimensions
                if len(means_array) > n_features:
                    means_array = means_array[:n_features]
                else:
                    means_array = np.pad(
                        means_array, (0, n_features - len(means_array)), "constant"
                    )

                if len(variances_array) > n_features:
                    variances_array = variances_array[:n_features]
                else:
                    variances_array = np.pad(
                        variances_array,
                        (0, n_features - len(variances_array)),
                        "constant",
                    )

            # Fix for feature_discriminative_power calculation
            feature_discriminative_power = np.zeros_like(means_array)
            valid_var = variances_array > 1e-10  # Only calculate for positive variances
            if np.any(valid_var):
                feature_discriminative_power[valid_var] = np.abs(
                    means_array[valid_var]
                ) / np.sqrt(variances_array[valid_var])
            else:
                # If no valid variances, just use the absolute means directly
                feature_discriminative_power = np.abs(means_array)

            heatmap_data[i, :] = feature_discriminative_power

    # Normalize across all features to get values between 0-1
    max_val = np.max(heatmap_data) if np.max(heatmap_data) > 0 else 1.0
    heatmap_data = heatmap_data / max_val

    # Prepare feature names for display
    # Create better feature names from the actual feature names
    if feature_names and len(feature_names) == n_features:
        # Use the actual feature names from the model
        short_feature_names = [name.replace("_", " ").title() for name in feature_names]
        # Further abbreviate long names
        short_feature_names = [
            name[:15] + "..." if len(name) > 15 else name
            for name in short_feature_names
        ]
    else:
        # Fallback if no meaningful names are available
        short_feature_names = [f"Feature {i+1}" for i in range(n_features)]

    # --- DEBUG PRINT for feature names ---
    print(f"DEBUG Emission Plot: Feature names for X-axis: {short_feature_names}")
    # --- END DEBUG ---

    # State labels for y-axis
    state_labels = [state_to_label_map.get(i, f"State {i}") for i in range(n_states)]

    # Create the heatmap
    im = ax1.imshow(heatmap_data, cmap="plasma", aspect="auto")

    # --- FIX V2: Explicitly set ax1 as current axis before setting labels ---
    plt.sca(ax1) # Set current axis to ax1

    # Y-axis: States
    ax1.set_yticks(np.arange(n_states))
    ax1.set_yticklabels(state_labels, fontsize=8)

    # X-axis: Features - Set ticks and labels clearly
    ax1.set_xticks(np.arange(n_features))
    ax1.set_xticklabels(short_feature_names, rotation=45, ha="right", fontsize=8)

    # Add title and axis labels AFTER setting ticks/labels
    ax1.set_title("State-Feature Importance Heatmap", fontsize=10)
    ax1.set_ylabel("HMM States", fontsize=9)
    ax1.set_xlabel("Features", fontsize=9, labelpad=10) # Keep axis title

    # Ensure ticks are visible (redundant after set_xticklabels, but safe)
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.tick_params(axis='y', labelleft=True)
    # --- END FIX V2 ---

    # Add colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # Add explanatory text
    ax1.text(
        0.98,
        0.02,
        "Brighter = Feature more important\nfor identifying this state",
        ha="right",
        va="bottom",
        fontsize=7,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, pad=2),
        transform=ax1.transAxes,
    )

    # Add light gridlines to match feature boundaries
    ax1.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n_states, 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.2)

    # ---- Plot 2: Mahalanobis Distances by Section ----
    ax2 = axes[1]

    # Get HMM section data
    hmm_semantic_labels = data.get("hmm_semantic_labels", [])
    hmm_section_starts = data.get("hmm_section_starts", [])

    if (
        not hmm_semantic_labels
        or not hmm_section_starts
        or not section_mahalanobis_distances
    ):
        ax2.text(
            0.5,
            0.5,
            "No Section-State Distance Data",
            ha="center",
            va="center",
            color="orange",
        )
        return

    # Create list of valid sections with distance data
    valid_sections = []
    for i, distances in enumerate(section_mahalanobis_distances):
        if i < len(hmm_semantic_labels) and distances is not None:
            # Ensure distances is a 1D array of correct length
            distances_array = np.atleast_1d(np.array(distances).flatten())
            if len(distances_array) != n_states:
                print(
                    f"WARNING: Section {i} distances has {len(distances_array)} values but expected {n_states}"
                )
                # Reshape to match expected state count
                if len(distances_array) > n_states:
                    distances_array = distances_array[:n_states]
                else:
                    distances_array = np.pad(
                        distances_array,
                        (0, n_states - len(distances_array)),
                        "constant",
                        constant_values=np.nan # Pad with NaN to handle later
                    )
            valid_sections.append((i, hmm_semantic_labels[i], distances_array))

    if not valid_sections:
        ax2.text(
            0.5, 0.5, "No Valid Distance Data", ha="center", va="center", color="orange"
        )
        return

    # Plot a heatmap of Mahalanobis distances for each section to each state
    # Lower distance = better match (more likely)
    n_valid_sections = len(valid_sections)
    distance_matrix = np.full((n_valid_sections, n_states), np.nan) # Initialize with NaN
    section_labels = []

    for i, (sec_idx, label, distances) in enumerate(valid_sections):
        distance_matrix[i, :] = distances # Fill with potentially NaN-padded data
        section_labels.append(f"Sec {sec_idx+1} ({label})")

    # --- FIX V3: Improved NaN handling and Normalization ---
    # Calculate percentiles IGNORING NaNs
    try:
        # Check if there are any finite values at all
        if np.all(np.isnan(distance_matrix)):
            print("WARNING: All Mahalanobis distances are NaN. Cannot plot heatmap.")
            ax2.text(0.5, 0.5, "All Distances are NaN", ha="center", va="center", color="red")
            return

        min_val = np.nanpercentile(distance_matrix, 5)  # 5th percentile of valid distances
        max_val = np.nanpercentile(distance_matrix, 95) # 95th percentile of valid distances
        print(f"DEBUG: Calculated nanpercentiles - min_val (5th): {min_val:.2f}, max_val (95th): {max_val:.2f}")

        # Handle edge case where min and max are the same (or very close)
        if np.isclose(min_val, max_val):
            # If all valid values are the same, use a small range around it
            if np.isfinite(min_val):
                 max_val = min_val + 1.0 # Add a small range
                 min_val = min_val - 1.0
            else: # If min_val is somehow still not finite (shouldn't happen if not all NaN)
                 min_val = 0.0
                 max_val = 1.0
            print(f"DEBUG: Adjusted range due to close percentiles: min={min_val:.2f}, max={max_val:.2f}")

    except Exception as e:
        print(f"ERROR calculating nanpercentiles: {e}. Defaulting range.")
        min_val = 0.0
        max_val = 10.0 # Arbitrary default range on error

    # Replace NaNs with a value corresponding to the worst match (max_val) AFTER calculating range
    distance_matrix_filled = np.nan_to_num(distance_matrix, nan=max_val)

    # Clip values to the calculated percentile range
    capped_distances = np.clip(distance_matrix_filled, min_val, max_val)

    # Calculate normalized values - invert so smaller distances (better matches) are brighter
    # Ensure denominator is not zero
    range_val = max_val - min_val
    if range_val < 1e-9: # If range is effectively zero
        print("DEBUG: Distance range is near zero. Setting normalized distances uniformly.")
        # Set all to 0.5 (neutral) or 1.0 (perfect match) depending on interpretation
        normalized_distances = np.full_like(capped_distances, 0.5)
    else:
        normalized_distances = 1.0 - ((capped_distances - min_val) / range_val)

    # Clip again just to be sure normalized values are within [0, 1]
    normalized_distances = np.clip(normalized_distances, 0.0, 1.0)
    # --- END FIX V3 ---


    # print(f"DEBUG: Normalized matrix min: {np.min(normalized_distances)}, max: {np.max(normalized_distances)}") # Less verbose

    # Create the heatmap with a higher contrast colormap like 'plasma' or 'magma'
    im2 = ax2.imshow(
        normalized_distances, cmap="plasma", aspect="auto", vmin=0, vmax=1
    )

    # Customize the plot
    # Set state labels using the list derived earlier
    state_labels_for_x = [state_to_label_map.get(i, f"State {i}") for i in range(n_states)]

    # --- ADD DEBUG PRINT ---
    print(f"DEBUG Emission Plot (ax2): Setting X-tick labels: {state_labels_for_x}")
    # --- END DEBUG ---

    ax2.set_yticks(np.arange(n_valid_sections))
    ax2.set_yticklabels(section_labels, fontsize=8)
    ax2.set_xticks(np.arange(n_states))
    ax2.set_xticklabels(state_labels_for_x, rotation=45, ha="right", fontsize=8) # Use state labels for X axis
    ax2.set_title("Section-State Distance Map (Brighter = Better Match)", fontsize=10)
    ax2.set_ylabel("Track Sections", fontsize=9)
    ax2.set_xlabel("HMM States", fontsize=9)

    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im2, cax=cax2, ticks=[0, 0.25, 0.5, 0.75, 1.0])

    # Add light gridlines
    ax2.set_xticks(np.arange(-0.5, n_states, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n_valid_sections, 1), minor=True)
    ax2.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.2)

    print(" -> Emission Probabilities plot generation complete.")
