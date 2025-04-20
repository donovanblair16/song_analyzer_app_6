# =============================================================================
# FILE: audio_plotting.py
# Contains functions for creating matplotlib plots of audio features.
# Includes helpers to add section context overlays (lines, labels, secondary axis).
# Default vertical grid lines removed; explicit section lines added.
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Import ticker for FuncFormatter
import numpy as np
from scipy.ndimage import uniform_filter1d # For smoothing plots
import re # For parsing frequency strings
import traceback # Keep for debugging prints if needed
import math # Needed for bar number calculation

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

# *** RE-INTRODUCED HELPER TO DRAW SECTION LINES ***
def _add_section_lines(ax, data):
    """Adds vertical lines at section boundaries."""
    section_starts = data.get("section_starts")
    if section_starts is None or len(section_starts) == 0: return
    print(f"DEBUG Plotting: Adding {len(section_starts)} section lines.")
    for start_time in section_starts:
        # Draw lines behind plot elements (low zorder)
        ax.axvline(x=start_time, color='grey', linestyle=':', alpha=0.6, linewidth=0.8, zorder=1)

def _add_section_labels_text(ax, data, location='top'):
    """Adds section text labels above or below the plot area."""
    section_starts = data.get("section_starts")
    semantic_labels = data.get("semantic_labels")
    sec_per_bar = data.get("seconds_per_bar", 0) # Needed for slight x offset

    has_sections = (section_starts is not None and
                    semantic_labels is not None and
                    len(section_starts) == len(semantic_labels) and
                    len(section_starts) > 0)

    if not has_sections: return
    print(f"DEBUG Plotting: Adding {len(section_starts)} section text labels at {location}.")

    if location == 'top': y_pos_text = 1.02; va = 'bottom'
    else: y_pos_text = -0.25; va = 'top' # Below primary x-axis

    if location == 'top':
        current_ylim = ax.get_ylim()
        new_top_lim = current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.08
        if np.isfinite(current_ylim[0]) and np.isfinite(new_top_lim): ax.set_ylim(current_ylim[0], new_top_lim)
        else: print(f"Warning: Cannot adjust ylim for labels: {current_ylim}")

    for i, start_time in enumerate(section_starts):
        label_text = semantic_labels[i]
        x_offset = (sec_per_bar * 0.1) if sec_per_bar > 0 else 0.1
        ax.text(start_time + x_offset, y_pos_text, label_text,
                transform=ax.get_xaxis_transform(), # X is data, Y is axes fraction
                ha='left', va=va, fontsize=7, color='dimgrey', clip_on=False)


def _add_bar_axis(ax, data, location='top'):
    """Adds a secondary x-axis showing bar numbers at the specified location ('top' or 'bottom')."""
    section_starts = data.get("section_starts")
    trim_offset = data.get("trim_offset_sec", 0)
    sec_per_bar = data.get("seconds_per_bar", 0)

    has_sections = (section_starts is not None and len(section_starts) > 0)
    if not has_sections or sec_per_bar <= 1e-6: return

    print(f"DEBUG Plotting: Adding bar axis at {location}.")
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
    # *** ADDED call to _add_section_lines ***
    _add_section_lines(ax, data)


# --- Individual Feature Plotting Functions ---

def plot_waveform(ax, data, title):
    """
    Plots waveform with coloring/labels and spectral centroid overlay.
    Uses specific bar number ticks/labels on primary x-axis.
    (Remains unchanged from user's preferred version - handles own lines/labels)
    """
    # Extract data
    y = data.get("y_processed"); sr = data.get("sr"); hop_length = data.get("hop_length", 256)
    section_starts = data.get("section_starts"); original_label_colors = data.get("label_colors")
    semantic_labels = data.get("semantic_labels"); original_section_labels = data.get("section_labels")
    trim_offset = data.get("trim_offset_sec", 0); duration_processed = data.get("duration_processed", 0)
    seconds_per_bar = data.get("seconds_per_bar")
    spec_centroid = data.get("spectral_centroid_frames"); spec_times = data.get("times_absolute")

    # Validation
    if y is None or sr is None:
        ax.text(0.5, 0.5, "Waveform Missing", ha='center', va='center', color='red')
        ax.set_title(title); return
    use_orig_colors = original_label_colors is not None and section_starts is not None and len(section_starts) == len(original_label_colors)
    use_sem_labels = semantic_labels is not None and section_starts is not None and len(section_starts) == len(semantic_labels)
    has_sections = section_starts is not None and (use_sem_labels or (original_section_labels is not None and len(section_starts) == len(original_section_labels)))

    # --- Plot Waveform ---
    absolute_end_time = duration_processed + trim_offset
    time_axis_waveform = np.linspace(trim_offset, absolute_end_time, num=len(y))
    ax.set_facecolor('#F0F0F0')
    max_amp = np.max(np.abs(y)) if len(y) > 0 else 1.0; max_amp = max(max_amp, 0.1)
    y_limit_top = max_amp * 1.25; ax.set_ylim(-max_amp * 1.05, y_limit_top)

    start_bars = [] # To store bar numbers for x-axis labels
    if has_sections:
        section_times_abs = section_starts + [absolute_end_time]
        for i in range(len(section_starts)):
            start_time = section_starts[i]; end_time = min(section_times_abs[i + 1], absolute_end_time)
            start_sample = max(0, int(np.round((start_time - trim_offset) * sr))); end_sample = min(len(y), int(np.round((end_time - trim_offset) * sr)))
            if seconds_per_bar and seconds_per_bar > 0: bar_num = round((start_time - trim_offset) / seconds_per_bar) + 1; start_bars.append(bar_num)
            else: start_bars.append(f"{start_time:.1f}s")
            if start_sample < end_sample:
                 segment_time = np.linspace(start_time, end_time, num=(end_sample - start_sample), endpoint=False)
                 segment_y = y[start_sample:end_sample]
                 plot_color = original_label_colors[i] if use_orig_colors and i < len(original_label_colors) else '#808080'
                 label_text = semantic_labels[i] if use_sem_labels and i < len(semantic_labels) else (original_section_labels[i] if original_section_labels and i<len(original_section_labels) else f"S{i+1}")
                 ax.plot(segment_time, segment_y, color=plot_color, alpha=0.9, linewidth=0.7)
            # Draw VLines and Labels ABOVE waveform
            if absolute_end_time - start_time > 1.0:
                ax.axvline(x=start_time, color='black', linestyle=':', alpha=0.6, linewidth=0.8, zorder=1) # Added zorder
                label_y_pos = max_amp * 1.05
                ax.text(start_time + 0.01, label_y_pos, label_text, ha='left', va='bottom', fontsize=8, color=plot_color, weight='bold', bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))
            else: print(f"Skipping vline/label for section start near end: {start_time:.2f}s")
    else: # If no section data
        ax.plot(time_axis_waveform, y, color='black', alpha=0.9, linewidth=0.7)
        status_msg = "Section data unavailable"; ax.text(0.02, 0.95, status_msg, transform=ax.transAxes, fontsize=8, color='gray', va='top')

    # Add Spectral Centroid Overlay
    if spec_centroid is not None and spec_times is not None and spec_centroid.size == spec_times.size and spec_centroid.size > 0:
        try:
            print(" Plotting Spectral Centroid Overlay..."); smooth_size = max(1, int(spec_times.size * 0.04))
            centroid_smoothed = uniform_filter1d(spec_centroid, size=smooth_size); max_c = np.max(centroid_smoothed)
            centroid_norm = centroid_smoothed / max_c if max_c > 1e-6 else np.zeros_like(centroid_smoothed)
            ax2 = ax.twinx(); ax2.plot(spec_times, centroid_norm, color='#FFFFFF', linestyle='-', linewidth=1.2, alpha=0.9)
            ax2.set_ylabel("Norm. Brightness\n(Spec. Centroid)", fontsize=7, color='white')
            ax2.set_ylim(0, 1.05); ax2.tick_params(axis='y', labelcolor='white', labelsize=6); ax2.spines['right'].set_color('white'); ax2.grid(False)
        except Exception as e: print(f"Error plotting spectral centroid overlay: {e}"); traceback.print_exc()
    else: print(" Skipping Spectral Centroid Overlay (data missing or invalid).")

    # Final styling for waveform plot
    ax.set_title(title, fontsize=10); ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(axis='y', labelsize=7); ax.grid(True, axis='y', linestyle=':', alpha=0.5) # Only horizontal grid
    ax.margins(x=0.01)

    # Modify X-axis Ticks and Labels (Keep specific bar logic for this plot)
    if has_sections and len(start_bars) == len(section_starts) and seconds_per_bar and seconds_per_bar > 0:
        ax.set_xticks(section_starts); ax.set_xticklabels(start_bars, fontsize=7, rotation=30, ha='right')
        ax.set_xlabel("Approx. Bar Number at Section Start", fontsize=9); plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    else: ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8)

    # Set x-axis limits
    if time_axis_waveform.size > 0 : ax.set_xlim(time_axis_waveform[0], time_axis_waveform[-1])
    else: ax.set_xlim(trim_offset, absolute_end_time)


# --- NEW Plotting Functions for Feature Tabs ---

def plot_energy_balance_features(axes, data, title):
    """Plots Energy & Balance features, each on its own subplot."""
    print("DEBUG: Plotting Energy/Balance Features...")
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

    # Add section labels and bar axis above the TOP plot
    _add_section_labels_text(axes[0], data, location='top')
    _add_bar_axis(axes[0], data, location='top')

    for i, (key, label) in enumerate(features_to_plot):
         _plot_feature_per_section(axes[i], data, key, label, label) # Draws lines internally now
         if i < len(axes) - 1: # Hide x-labels on all but bottom plot
              axes[i].tick_params(axis='x', labelbottom=False)
              axes[i].set_xlabel("") # Explicitly remove label
         else: # Set final x-label on bottom plot
              axes[i].set_xlabel("Time (s)", fontsize=9) # Use Time(s) for primary axis
              axes[i].tick_params(axis='x', labelsize=8, labelbottom=True)


def plot_timbre_texture_features(axes, data, title):
    """Plots Timbre & Texture features, each on its own subplot."""
    print("DEBUG: Plotting Timbre/Texture Features...")
    features_to_plot = [
        ('spectral_centroid_avg', "Spec Centroid Avg"),
        ('spectral_bandwidth_avg', "Spec Bandwidth Avg"),
        ('spectral_contrast_avg', "Spec Contrast Avg")
    ]
    if len(axes) != len(features_to_plot):
        print(f"Error: Expected {len(features_to_plot)} axes for Timbre/Texture, got {len(axes)}.")
        axes[0].text(0.5, 0.5, "Axes Mismatch", ha='center', va='center', color='red'); return

    # Add section labels and bar axis above the TOP plot
    _add_section_labels_text(axes[0], data, location='top')
    _add_bar_axis(axes[0], data, location='top')

    for i, (key, label) in enumerate(features_to_plot):
         _plot_feature_per_section(axes[i], data, key, label, label) # Draws lines internally now
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
    # Add section context
    _add_section_lines(ax, data) # Call helper to add lines
    _add_section_labels_text(ax, data, location='top') # Add labels above
    ax.set_xlabel("Time (s)", fontsize=9); ax.tick_params(axis='x', labelsize=8, labelbottom=True)

def plot_dynamic_range(ax, data, title):
    """Plots dynamic range (RMS / Peak) over time."""
    dyn_times = data.get("dyn_times_absolute"); dyn_range = data.get("dyn_range"); ylabel = "Dynamic Range\n(RMS / Peak)"
    if dyn_times is None or dyn_range is None or dyn_times.size != dyn_range.size: ax.text(0.5, 0.5, "Dyn Range Data Missing/Mismatch", ha='center', va='center', color='red')
    else: ax.plot(dyn_times, dyn_range, color='black', linewidth=0.8); ax.fill_between(dyn_times, 0, dyn_range, color='black', alpha=0.1); ax.set_xlim(dyn_times[0], dyn_times[-1]) if dyn_times.size > 0 else None
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4) # Only horizontal grid
    ax.tick_params(axis='y', labelsize=7)
    # Add section context
    _add_section_lines(ax, data)
    _add_section_labels_text(ax, data, location='top')
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
    # Add section context
    _add_section_lines(ax, data)
    _add_section_labels_text(ax, data, location='top')
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
    # Add section context
    _add_section_lines(ax, data)
    _add_section_labels_text(ax, data, location='top')
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
    # Add section context
    _add_section_lines(ax, data)
    _add_section_labels_text(ax, data, location='top')
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


def create_single_track_plot(data, plot_func, title, track_name="Track", num_rows=1):
    """Creates a figure with subplots for displaying features of one track."""
    figsize_h = 4 if num_rows == 1 else 2.5 * num_rows # Adjusted height slightly
    figsize_w = 12
    gridspec_kw = None

    fig, axes = plt.subplots(num_rows, 1, figsize=(figsize_w, figsize_h), sharex=True, squeeze=False)
    axes = axes.flatten() # Always return a flat array of axes
    plot_full_title = f"{track_name}: {title}" # Title only used if single plot

    if data:
        try:
            plot_func(axes[0] if num_rows == 1 else axes, data, plot_full_title if num_rows == 1 else title)
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

    # Add section labels and bar axis above the TOP plot
    _add_section_labels_text(axes[0], data, location='top')
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
        # Add section lines to this axis
        _add_section_lines(ax, data) # Call helper to add lines
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
