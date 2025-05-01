# /Users/donovanblair/Desktop/song_analyzer_app_6/audio_analysis_modules/loading.py

"""
Audio loading and preprocessing functions.
"""

import os
import traceback
import numpy as np
import librosa
from tkinter import messagebox


def load_and_preprocess(file_path, track_num, manual_bpm_override=None):
    """
    Loads audio, trims leading/trailing silence, calculates tempo
    (using Median IBI fallback or manual override), and returns basic features.

    Args:
        file_path (str): Path to the audio file.
        track_num (int): Identifier for the track (e.g., 1 or 2).
        manual_bpm_override (float, optional): If provided, overrides detected BPM.

    Returns:
        dict or None: A dictionary containing processed audio ('y_processed'),
                      original audio ('y_original'), sample rate ('sr'),
                      beats per minute ('bpm'), detected bpm ('bpm_detected'),
                      trim offset ('trim_offset_sec'), processed duration ('duration_processed'),
                      hop length ('hop_length'), and original file path ('file_path').
                      Returns None on failure.
    """
    print(f"Loading Track {track_num}: {os.path.basename(file_path)}...")
    y_processed = y_original = sr = bpm = None
    trim_offset_sec = 0.0  # Default trim offset
    duration = None
    hop_length = 256  # Default hop length, ensure consistency if changed elsewhere
    tempo_val = 120.0
    detected_bpm = 120.0  # Initialize defaults

    try:
        # Load audio file using librosa
        y_original, sr = librosa.load(file_path, sr=None, mono=True)
        # Validate loaded audio
        if y_original is None or len(y_original) == 0:
            raise ValueError("Audio data is empty after loading.")
        # Ensure audio data is finite (no NaNs or Infs)
        if not np.all(np.isfinite(y_original)):
            y_original = np.nan_to_num(y_original)
            print("Warning: Non-finite values found and replaced in original audio.")
            # Double-check after replacement
            if not np.all(np.isfinite(y_original)):
                raise ValueError(
                    "Audio still contains non-finite values after nan_to_num."
                )

        # Trim leading/trailing silence using a threshold (top_db)
        # top_db=55 means silence is below 55 dB from the peak
        y_trimmed, index = librosa.effects.trim(y_original, top_db=55)
        # Check if trimming actually occurred
        if index is None or len(index) != 2 or index[1] <= index[0]:
            print(
                "Warning: No significant audio found after trimming (top_db=55), using original audio."
            )
            trim_offset_sec = 0.0
            y_processed = y_original.copy()  # Use a copy to avoid modifying original
        else:
            # Calculate the trimmed duration and offset
            trim_offset_sec = float(index[0]) / sr
            y_processed = y_trimmed
            print(
                f" Trimmed {trim_offset_sec:.2f}s from start, duration after trim: {librosa.get_duration(y=y_processed, sr=sr):.2f}s"
            )

        # Ensure processed audio is finite after trimming
        if not np.all(np.isfinite(y_processed)):
            y_processed = np.nan_to_num(y_processed)
            print("Warning: Non-finite values found and replaced after trimming.")
            if y_processed is None or not np.all(np.isfinite(y_processed)):
                raise ValueError("Audio became non-finite after trim/nan_to_num.")

        # Calculate duration of the processed audio
        duration = librosa.get_duration(y=y_processed, sr=sr)
        if duration is None or not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid processed audio duration calculated: {duration}")

        # --- Tempo & Beat Detection ---
        try:
            hop_length_tempo = (
                512  # Use a slightly larger hop for tempo detection potentially
            )
            print(f" Running beat tracking for tempo estimate (Track {track_num})...")
            # Get initial tempo estimate and beat frame locations
            tempo_initial_raw, beat_frames_tracked = librosa.beat.beat_track(
                y=y_processed, sr=sr, hop_length=hop_length_tempo, units="frames"
            )

            bpm_detected = 120.0  # Default
            tempo_initial_float = 120.0  # Default
            # Validate initial tempo estimate
            if np.isscalar(tempo_initial_raw) and np.isfinite(tempo_initial_raw):
                tempo_initial_float = float(tempo_initial_raw)
                print(
                    f" Initial tempo estimate from beat_track: {tempo_initial_float:.4f}"
                )
            else:
                print(" Initial tempo estimate from beat_track is not a valid scalar.")

            # Refine tempo using Inter-Beat Intervals (IBI) if possible
            if beat_frames_tracked is not None and beat_frames_tracked.size > 1:
                beat_times = librosa.frames_to_time(
                    beat_frames_tracked, sr=sr, hop_length=hop_length_tempo
                )
                ibis = np.diff(beat_times)  # Calculate differences between beat times
                if ibis.size > 0:
                    median_ibi = np.median(ibis)  # Use median IBI for robustness
                    if median_ibi > 1e-6:  # Avoid division by zero
                        tempo_calc = 60.0 / median_ibi
                        print(f" Tempo calculated from median IBI: {tempo_calc:.4f}")
                        # Use calculated tempo if it's within a reasonable range
                        if np.isfinite(tempo_calc) and 30 < tempo_calc < 250:
                            bpm_detected = tempo_calc
                        else:
                            print(
                                f" Median IBI tempo unreasonable ({tempo_calc:.4f}). Using beat_track estimate."
                            )
                            bpm_detected = tempo_initial_float
                    else:
                        print(
                            " Warning: Median IBI zero/negative. Using beat_track estimate."
                        )
                        bpm_detected = tempo_initial_float
                else:
                    print(
                        " Warning: Could not calculate IBIs. Using beat_track estimate."
                    )
                    bpm_detected = tempo_initial_float
            else:
                print(
                    " Warning: Initial beat tracking failed or too few beats found. Using initial estimate/default."
                )
                bpm_detected = tempo_initial_float

            # Final check on detected BPM reasonableness
            if not (np.isfinite(bpm_detected) and 30 < bpm_detected < 300):
                print(
                    f" Detected BPM unreasonable ({bpm_detected:.2f}). Defaulting to 120."
                )
                bpm_detected = 120.0

            # Use detected BPM unless overridden
            tempo_val = bpm_detected
            print(f" Detected Tempo Value (before override): {tempo_val:.4f}")
            if manual_bpm_override is not None:
                print(
                    f" Manual BPM Override provided: {manual_bpm_override:.4f}. Overriding detected tempo."
                )
                tempo_val = float(manual_bpm_override)

            # Final validation of the tempo value to be used
            if not np.isfinite(tempo_val) or tempo_val < 30 or tempo_val > 300:
                print(
                    f" Warning (Track {track_num}): Final tempo is unreasonable ({tempo_val:.4f}). Falling back to default 120 BPM."
                )
                tempo_val = 120.0

        except Exception as e:
            # Handle errors during tempo detection
            print(
                f"Tempo/Beat detection failed (Track {track_num}): {e}. Using default 120 BPM."
            )
            traceback.print_exc()
            tempo_val = 120.0

        # Round final BPM to integer
        bpm = round(tempo_val)
        print(f"Final rounded BPM to be used (Track {track_num}): {bpm}")

        # Final checks before returning
        if sr is None or bpm is None:
            raise ValueError("Failed to determine sample rate or BPM.")
        if y_processed is None:
            raise ValueError("Processed audio is None before returning.")

        # Return dictionary of results
        return {
            "y_processed": y_processed,
            "y_original": y_original,
            "sr": sr,
            "bpm": bpm,
            "trim_offset_sec": trim_offset_sec,
            "duration_processed": duration,
            "hop_length": hop_length,
            "file_path": file_path,
            "bpm_detected": round(
                bpm_detected
            ),  # Store the originally detected BPM too
        }
    except Exception as e:
        # Handle any errors during loading or preprocessing
        print(f"--- Error in load_and_preprocess (Track {track_num}) ---")
        traceback.print_exc()
        messagebox.showerror(
            f"Error Loading Track {track_num}",
            f"Failed to load or preprocess audio:\n{e}",
        )
        return None  # Indicate failure
