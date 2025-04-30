# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/spectral_analysis.py

"""
Functions for spectral analysis, including stereo width and HPSS decomposition.
"""

import os
import traceback
from collections import defaultdict
import numpy as np
import librosa
from tkinter import messagebox


def _get_spec_data(y, sr, hop_length, trim_offset_sec, n_fft=4096):
    """Calculates and returns spectrogram data (magnitude, freqs, times, bands)."""
    spec = freqs = times_abs = min_frames = bands = None
    try:
        print(" Calculating Spectrogram...")
        # Calculate Short-Time Fourier Transform (STFT)
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        # Get magnitude spectrogram
        spec = np.abs(stft_result)
        # Ensure finite values
        if not np.all(np.isfinite(spec)):
            spec = np.nan_to_num(spec)
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        # Get relative time frames
        times_rel = librosa.frames_to_time(
            np.arange(spec.shape[1]), sr=sr, hop_length=hop_length
        )
        # Calculate absolute time frames
        times_abs = times_rel + trim_offset_sec
        min_frames = spec.shape[1]  # Number of time frames
        # Define frequency bands (example)
        bands = {
            "Lows (<200 Hz)": freqs < 200,
            "Low-Mids (200–500 Hz)": (freqs >= 200) & (freqs < 500),
            "Mids (500–2000 Hz)": (freqs >= 500) & (freqs < 2000),
            "High-Mids (2000–5000 Hz)": (freqs >= 2000) & (freqs < 5000),
            "Highs (>5000 Hz)": freqs >= 5000,
        }
    except Exception as spec_e:
        print(f"Error calculating spectrogram: {spec_e}")
        # Return Nones if calculation fails
        return None, None, None, None, None
    # Return calculated data
    return spec, freqs, times_abs, min_frames, bands


def analyze_stereo_and_hpss(track_data, calc_stereo=True, calc_hpss=True):
    """
    Calculates stereo width and/or harmonic/percussive separation.
    Ensures spectrogram data ('spec', 'freqs', 'times_absolute') is calculated
    and stored in the results if needed.
    """
    results = defaultdict(lambda: None)  # Initialize results dict
    # Retrieve necessary data from input track_data
    y = track_data.get("y_processed")
    sr = track_data.get("sr")
    hop_length = track_data.get("hop_length")
    trim_offset_sec = track_data.get("trim_offset_sec", 0)
    file_path = track_data.get("file_path")  # Original file path for stereo loading
    # Initialize output variables
    width_matrix = bands = spec = freqs = times_abs = rms_harm = rms_perc = (
        rms_time_abs
    ) = None
    min_frames = None
    n_fft = 4096  # FFT window size

    try:
        if y is None:
            raise ValueError("Input audio 'y_processed' is None.")

        # --- Ensure Spectrogram Data Exists ---
        # Check if spectrogram data was already calculated and passed in track_data
        spec = track_data.get("spec")
        freqs = track_data.get("freqs")
        times_abs = track_data.get("times_absolute")
        bands = track_data.get("bands")
        min_frames = spec.shape[1] if spec is not None else None

        # Calculate spectrogram if it's missing AND needed for stereo/hpss
        if spec is None and (calc_stereo or calc_hpss):
            print("Calculating spectrogram within analyze_stereo_and_hpss...")
            spec, freqs, times_abs, min_frames, bands = _get_spec_data(
                y, sr, hop_length, trim_offset_sec, n_fft
            )
            if spec is None:
                # If spec calculation fails, cannot proceed with stereo/hpss
                calc_stereo = False
                calc_hpss = False
                print("Warning: Spectrogram calculation failed. Skipping Stereo/HPSS.")
        # Store the spectrogram data (either existing or newly calculated) in results
        if spec is not None:
            print("Storing/Confirming Spectrogram data in results.")
            results["spec"] = spec
            results["freqs"] = freqs
            results["times_absolute"] = times_abs  # Store absolute times
            results["bands"] = bands
        # --- End Spectrogram Handling ---

        # --- Calculate Stereo Width ---
        if calc_stereo:
            print(" Calculating stereo width...")
            if spec is None:  # Check again in case calculation failed above
                print(" Skipping stereo width: Spectrogram data unavailable.")
                results["width_matrix"] = None
            else:
                y_stereo = None  # Initialize stereo audio variable
                try:
                    # Load stereo audio if file path exists
                    if file_path and os.path.exists(file_path):
                        y_stereo, sr_stereo = librosa.load(
                            file_path, sr=sr, mono=False
                        )  # Load as stereo
                        # Validate loaded stereo audio
                        if sr_stereo != sr:
                            raise ValueError("Stereo sample rate mismatch.")
                        if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
                            raise ValueError("Loaded audio is not stereo.")
                    else:
                        print(
                            "Warning: Original file path missing or invalid. Cannot calculate stereo width."
                        )
                except Exception as e:
                    print(f"Could not load valid stereo audio: {e}")
                    y_stereo = None  # Ensure it's None if loading fails

                if y_stereo is not None:
                    try:
                        # Calculate STFT for Left and Right channels
                        spec_L = np.nan_to_num(
                            np.abs(
                                librosa.stft(
                                    y_stereo[0], n_fft=n_fft, hop_length=hop_length
                                )
                            )
                        )
                        spec_R = np.nan_to_num(
                            np.abs(
                                librosa.stft(
                                    y_stereo[1], n_fft=n_fft, hop_length=hop_length
                                )
                            )
                        )
                        # Ensure consistent number of frames between mono spec and L/R specs
                        current_min_frames = spec.shape[1]  # Frames in mono spec
                        c_min = min(
                            spec_L.shape[1], spec_R.shape[1], current_min_frames
                        )
                        # Trim if necessary
                        if c_min < current_min_frames:
                            print(
                                f"Warn: Stereo/Mono frame mismatch. Trimming mono spec/times to {c_min}."
                            )
                            results["spec"] = spec = spec[:, :c_min]
                            results["times_absolute"] = times_abs = times_abs[:c_min]
                            min_frames = c_min  # Update frame count
                        spec_L = spec_L[:, :c_min]
                        spec_R = spec_R[:, :c_min]
                        # Calculate Mid/Side signals
                        mid = (spec_L + spec_R) / 2
                        side = (spec_L - spec_R) / 2
                        # Calculate width matrix (Side / Mid ratio), handle division by zero
                        results["width_matrix"] = np.nan_to_num(
                            np.abs(side) / (np.abs(mid) + 1e-9)
                        )
                    except Exception as width_e:
                        print(f"Error calculating width matrix: {width_e}")
                        results["width_matrix"] = None
                else:
                    # Set width to None if stereo audio couldn't be loaded
                    results["width_matrix"] = None
        else:
            # Set width to None if calculation was skipped
            results["width_matrix"] = None

        # --- Calculate HPSS ---
        if calc_hpss:
            print(" Calculating HPSS...")
            try:
                # Perform Harmonic-Percussive Source Separation
                y_harm, y_perc = librosa.effects.hpss(y)
                # Calculate RMS for harmonic and percussive components
                rms_harm = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
                rms_perc = librosa.feature.rms(y=y_perc, hop_length=hop_length)[0]
                # Calculate corresponding time vector (relative)
                rms_time_rel = librosa.frames_to_time(
                    np.arange(len(rms_harm)), sr=sr, hop_length=hop_length
                )
                # Ensure RMS vectors match the number of spectrogram frames if available
                target_len = min_frames if min_frames is not None else len(rms_time_rel)
                if len(rms_time_rel) != target_len:
                    print(
                        f"Warn: HPSS RMS time mismatch ({len(rms_time_rel)} vs {target_len}). Trimming RMS."
                    )
                    current_len = min(len(rms_time_rel), target_len)
                    rms_time_rel = rms_time_rel[:current_len]
                    rms_harm = rms_harm[:current_len]
                    rms_perc = rms_perc[:current_len]
                # Store results, converting times to absolute
                results["rms_harm"] = np.nan_to_num(rms_harm)
                results["rms_perc"] = np.nan_to_num(rms_perc)
                results["rms_time_absolute"] = (
                    rms_time_rel + trim_offset_sec
                )  # Store absolute times
            except Exception as hpss_e:
                # Handle errors during HPSS calculation
                print(f"Error calculating HPSS: {hpss_e}")
                results["rms_harm"] = None
                results["rms_perc"] = None
                results["rms_time_absolute"] = None
        else:
            # Set HPSS results to None if calculation was skipped
            results["rms_harm"] = None
            results["rms_perc"] = None
            results["rms_time_absolute"] = None

        # Return the results dictionary (potentially updated with spec, width, hpss)
        return results

    except Exception as e:
        # Catch-all for errors in this function
        print("--- Error in analyze_stereo_and_hpss ---")
        traceback.print_exc()
        messagebox.showerror("Error", f"Stereo/HPSS Analysis Error:\n{e}")
        # Ensure default keys exist on failure
        results.setdefault("width_matrix", None)
        results.setdefault("rms_harm", None)
        results.setdefault("rms_perc", None)
        results.setdefault("rms_time_absolute", None)
        results.setdefault("spec", None)
        results.setdefault("freqs", None)
        results.setdefault("times_absolute", None)
        results.setdefault("bands", None)
        return results
