# Song Analyzer App - Code Context Documentation

This document provides comprehensive information about the Song Analyzer App codebase, designed to give AI assistants like Claude sufficient context to understand and extend the application functionality without requiring every file to be uploaded.

## Table of Contents

1. [Application Overview](#application-overview)
2. [Audio Analysis System](#audio-analysis-system)
   - [Architecture](#audio-analysis-architecture)
   - [Audio Analysis Module Files](#audio-analysis-module-files-and-their-purposes)
   - [Audio Analysis Data Flow](#audio-analysis-data-flow)
   - [Key Algorithms and Techniques](#key-algorithms-and-techniques)
   - [Important Data Structures](#important-data-structures)
   - [Dependencies and Libraries](#dependencies-and-libraries)
   - [Error Handling and Robustness](#error-handling-and-robustness)
   - [Extension Points](#extension-points)
   - [Coding Conventions](#coding-conventions-and-patterns)
   - [Performance Considerations](#performance-considerations)
   - [Key Constants](#key-constants-and-configuration-values)
   - [Testing and Debugging](#testing-and-debugging-techniques)
   - [Known Limitations](#known-limitations)
   - [Future Enhancement Opportunities](#future-enhancement-opportunities)
3. [GUI Builder Module](#gui-builder-module)
4. [Section Editor Module](#section-editor-module)
5. [Plot Manager Module](#plot-manager-module)
6. [Playback Manager Module](#playback-manager-module)
7. [File Manager Module](#file-manager-module)
8. [HMM Predictor Module](#hmm-predictor-module)
9. [Main Application Module](#main-application-module) 
10. [Transition Matrix Extraction](#transition-matrix-extraction)
11. [GMMHMM Trainer](#gmmhmm-trainer)

## Application Overview

The Song Analyzer App is a Python-based desktop application for analyzing music files, detecting song sections (intro, verse, chorus, etc.), extracting audio features, and visualizing audio characteristics. It uses machine learning (specifically GMM-HMM models) for pattern recognition in audio.

The application provides a comprehensive set of tools for understanding song structure, energy distribution, and spectral characteristics, allowing users to gain insights into audio composition and production techniques.

## Application Architecture

![Dependency Graph](images/dependency_graph.png)

## Audio Analysis System

The audio analysis system forms the core functionality of the application, providing sophisticated audio processing capabilities through a modular architecture. The system is designed to be flexible, allowing for different levels of analysis (from basic to comprehensive) and supporting both single-track and comparative analysis workflows.

### Audio Analysis Architecture

The audio analysis system follows a modular architecture:

1. **Audio Analysis Wrapper**: Exposes all audio analysis functions through a unified interface
2. **Audio Analysis Modules**:
   - **Loading Module**: Handles audio file loading and preprocessing
   - **Section Detection Module**: Identifies section boundaries in audio
   - **Chroma Analysis Module**: Performs harmonic analysis and section labeling
   - **Spectral Analysis Module**: Calculates spectrograms and stereo width information
   - **Feature Extraction Module**: Extracts numerical features from audio sections
3. **Shared Constants**: Central definition of thresholds and color values
4. **Hierarchical Processing Pipeline**: Sequential stages of analysis from basic audio loading to sophisticated feature extraction

### Audio Analysis Module Files and Their Purposes

#### Audio Analysis Wrapper

- **audio_analysis_wrapper.py**:
  - Main wrapper for audio analysis functions
  - Re-exports functions from the audio_analysis_modules package
  - Maintains backward compatibility with existing code
  - Provides a unified entry point for all audio analysis functionality
  - Also re-exports important constants for use by other modules

#### Audio Analysis Modules

- **audio_analysis_modules/__init__.py**:
  - Defines the audio analysis modules package
  - Imports and re-exports key functions from submodules
  - Makes functions available at the package level
  - Creates a clean API for the package

- **audio_analysis_modules/loading.py**:
  - Contains `load_and_preprocess()` function
  - Loads audio files using librosa
  - Trims leading/trailing silence
  - Estimates tempo using either:
    - Beat tracking with librosa
    - Inter-Beat Interval (IBI) calculation
    - Manual BPM override if provided
  - Includes robust error handling and fallback mechanisms
  - Returns a dictionary with processed audio and metadata

- **audio_analysis_modules/section_detection.py**:
  - Contains `detect_sections()` function
  - Detects section boundaries based on RMS energy changes
  - Implements fill detection logic for identifying transitional sections
  - Uses 8-bar window analysis for standard section detection
  - Handles post-processing of detected sections
  - Returns section start times and initial labels

- **audio_analysis_modules/chroma_analysis.py**:
  - Contains `analyze_chroma_and_clusters()` function
  - Performs advanced section labeling using clustering
  - Implements a multi-phase approach:
    1. Enhanced feature extraction per section
    2. Refined clustering using scikit-learn
    3. Cluster-based classification
    4. Contextual cleanup rules
  - Returns semantic labels, label colors, section features, and more
  - Contains `results_on_failure()` for graceful fallback on errors

- **audio_analysis_modules/spectral_analysis.py**:
  - Contains `analyze_stereo_and_hpss()` function
  - Calculates spectrograms via internal `_get_spec_data()` function
  - Performs stereo width analysis (if requested)
  - Implements Harmonic-Percussive Source Separation (HPSS) analysis
  - Handles time synchronization between different analysis types
  - Returns a comprehensive dictionary of spectral analysis results

- **audio_analysis_modules/feature_extraction.py**:
  - Contains `extract_section_features()` function
  - Calculates features for each detected section, including:
    - Average RMS (loudness)
    - Relative RMS
    - Relative position
    - Position context
    - Low-end energy
    - RMS standard deviation
    - Spectral centroid standard deviation
    - Delta features (changes from previous section)
    - Crest factor (peak-to-average ratio)
    - Spectral centroid slope (brightness trend)
    - RMS trend (loudness trajectory)
  - Also includes `calculate_bar_features()` for bar-level analysis
  - Returns processed feature dictionaries for each section

- **audio_analysis_modules/constants.py**:
  - Defines shared constants used throughout the audio analysis modules
  - Contains thresholds for labeling logic
  - Defines color hex codes for visualization consistency
  - Provides note names for chroma analysis

### Audio Analysis Data Flow

The audio analysis system processes audio files through a sequential pipeline, with data flowing between modules as follows:

1. **Loading and Preprocessing** (`loading.py`):
   - Takes a file path as input
   - Loads the audio file using librosa
   - Trims silence from the beginning and end
   - Estimates tempo (BPM) or applies manual override
   - Returns a dictionary containing the processed audio and metadata:
     - `y_processed`: The processed audio samples
     - `sr`: Sample rate
     - `bpm`: Detected or overridden tempo
     - `trim_offset_sec`: Amount trimmed from the start
     - Other metadata (file_path, duration, etc.)

2. **Section Detection** (`section_detection.py`):
   - Takes the processed track data dictionary
   - Analyzes the audio in 8-bar windows to detect energy changes
   - Identifies "Fill" sections based on energy patterns
   - Returns a list of section start times and initial labels (e.g., "Start", "S1", "S2", "Fill")

3. **Spectrogram and HPSS Analysis** (`spectral_analysis.py`):
   - Takes the processed track data dictionary (optionally with previous analysis results)
   - Calculates spectrogram data (magnitude, frequencies, times)
   - Performs HPSS (Harmonic-Percussive Source Separation) if requested
   - Calculates stereo width matrix if requested and if file is stereo
   - Returns spectrogram data and HPSS results to be added to the track data dictionary

4. **Chroma Analysis and Labeling** (`chroma_analysis.py`):
   - Takes the track data dictionary with sections and spectrogram info
   - Extracts features for each section (RMS, spectral features, etc.)
   - Performs clustering on sections using these features
   - Maps clusters to semantic labels ("Intro", "Drop", "Build", etc.)
   - Applies cleanup rules based on musical structure knowledge
   - Returns updated track data with semantic labels, colors, and features

5. **Feature Extraction** (`feature_extraction.py`):
   - Takes the track data dictionary with labeled sections
   - Calculates detailed features for each section
   - Calculates relative and differential features (changes between sections)
   - Returns updated section feature dictionaries

### Key Algorithms and Techniques

#### Audio Loading and Preprocessing
- **Silence Trimming**: Uses librosa's `effects.trim()` with a 55dB threshold to remove leading/trailing silence
- **Tempo Detection**: Multi-stage approach that:
  1. Uses librosa's `beat.beat_track()` for initial tempo estimate
  2. Calculates Inter-Beat Intervals (IBIs) for more robust tempo
  3. Takes the median IBI to avoid outliers
  4. Converts to BPM (beats per minute)
  5. Supports manual BPM override for correction

#### Section Detection
- **8-Bar Window Analysis**: Examines energy changes across 8-bar windows
- **Fill Detection**: Special logic to identify short "Fill" sections:
  1. Looks for sections with low RMS energy (below 65% of "Drop" average)
  2. Checks for energy rebound after potential fill
  3. Variable fill length detection (1-4 bars)
- **Energy Change Detection**: Identifies significant changes (>=50% change) in RMS energy between 4-bar segments

#### Chroma Analysis and Labeling
- **Feature Extraction**: Calculates per-section features (RMS, spectral features, etc.)
- **Clustering**: Agglomerative clustering with scikit-learn to group similar sections
- **Cluster-Based Classification**: Maps clusters to labels based on audio characteristics:
  1. Highest RMS clusters → "Drop"
  2. Third highest RMS cluster → "Build"
  3. Lowest RMS cluster → "Breakdown"
  4. Other clusters → "Body"
- **Contextual Cleanup Rules**: Applies music theory knowledge to fix labeling:
  1. Force first section to "Intro"
  2. Identify "Outro" sections based on position and shared clusters with "Intro"
  3. Apply pattern-based rules (e.g., Drop→Build→Body to Drop→Breakdown→Body)
  4. Special "Fade Out" detection for very short, low-energy final sections

#### Spectral Analysis
- **Spectrogram Generation**: FFT-based spectrogram calculation
- **Stereo Width Analysis**: Calculates Mid/Side signals from stereo channels to measure stereo width
- **HPSS**: Separates harmonic and percussive components using librosa's `effects.hpss()`

#### Feature Extraction
- **RMS-Based Features**: Loudness (avg_rms), relative_rms, crest_factor, rms_trend
- **Spectral Features**: spectral_centroid, spectral_bandwidth, spectral_contrast
- **Position Features**: relative_position, position_context
- **Low-End Energy**: Measures bass content below 150Hz
- **Variation Features**: Standard deviations of RMS and spectral centroid
- **Delta Features**: Changes from previous section (delta_rms, delta_centroid)
- **Trend Features**: Linear regression slopes for RMS and spectral centroid

### Important Data Structures

#### track_data Dictionary
The `track_data` dictionary is the central data structure in the audio analysis system. It starts with basic audio data and is progressively enriched through the analysis pipeline:

```python
track_data = {
    # Audio Data (from loading.py)
    "y_processed": ndarray,        # Processed audio samples (mono)
    "y_original": ndarray,         # Original audio samples
    "sr": int,                     # Sample rate (e.g., 44100)
    "bpm": float,                  # Beats per minute
    "trim_offset_sec": float,      # Seconds trimmed from start
    "duration_processed": float,   # Duration after trimming
    "hop_length": int,             # Hop length for feature extraction
    "file_path": str,              # Original file path
    
    # Bar-Level Analysis
    "bar_rms_data": list,          # RMS energy per bar
    "bar_starts_absolute": list,   # Bar start times (seconds)
    "seconds_per_bar": float,      # Duration of one bar
    
    # Beat-Level Analysis
    "beat_frames": ndarray,        # Beat positions in frames
    
    # Frame-Level Analysis
    "rms": ndarray,                # RMS energy per frame
    "rms_times": ndarray,          # Times for RMS frames (relative)
    
    # Section Analysis (from section_detection.py)
    "section_starts": list,        # Section start times (absolute)
    "section_labels": list,        # Initial labels ("Start", "S1", "S2", etc.)
    
    # Spectrogram Analysis (from spectral_analysis.py)
    "spec": ndarray,               # Magnitude spectrogram (freqs × frames)
    "freqs": ndarray,              # Frequency bins
    "times_absolute": ndarray,     # Times for spectrogram frames (absolute)
    "width_matrix": ndarray,       # Stereo width matrix
    "bands": dict,                 # Frequency band definitions
    
    # HPSS Analysis
    "rms_harm": ndarray,           # RMS of harmonic component
    "rms_perc": ndarray,           # RMS of percussive component
    "rms_time_absolute": ndarray,  # Times for HPSS frames (absolute)
    
    # Spectral Features
    "spectral_centroid_frames": ndarray,  # Spectral centroid per frame
    "spectral_bandwidth_frames": ndarray, # Spectral bandwidth per frame
    "spectral_contrast_frames": ndarray,  # Spectral contrast per frame
    
    # Chroma Analysis (from chroma_analysis.py)
    "semantic_labels": list,       # Semantic section labels (Intro, Drop, etc.)
    "label_colors": list,          # Color hex codes for sections
    "cluster_labels": list,        # Cluster IDs for each section
    "section_features": list,      # List of feature dictionaries (see below)
    "labels_before_cleanup": list, # Labels before applying cleanup rules
    
    # Chroma Features
    "chroma_sync": ndarray,        # Beat-synchronized chroma features
    "root_indices": ndarray,       # Root note indices per beat
    "root_times_absolute": ndarray,# Times for root notes (absolute)
    
    # Low-End Features
    "low_energy_norm": ndarray,    # Normalized low-end energy
    "low_energy_times": ndarray,   # Times for low-end energy (relative)
    
    # Dynamic Range Features
    "dyn_range": ndarray,          # Dynamic range per frame
    "dyn_times_absolute": ndarray, # Times for dynamic range (absolute)
}
```

#### section_features List
The `section_features` list contains a dictionary for each detected section:

```python
section_features = [
    {
        # Basic Section Info
        "index": int,              # Section index (0-based)
        "start_time": float,       # Start time (absolute seconds)
        "end_time": float,         # End time (absolute seconds)
        "duration_sec": float,     # Duration in seconds
        "duration_bars": int,      # Duration in bars
        "original_label": str,     # Original label (from section detection)
        
        # Audio Features
        "avg_rms": float,          # Average RMS (loudness)
        "peak_rms": float,         # Peak RMS
        "relative_rms": float,     # RMS relative to track maximum
        "relative_position": float,# Position in track (0-1)
        "position_context": float, # Position emphasis (0-1)
        "low_end_ratio": float,    # Ratio of low-frequency energy
        "high_end_ratio": float,   # Ratio of high-frequency energy
        
        # Spectral Features
        "spectral_centroid_avg": float,    # Average spectral centroid
        "spectral_bandwidth_avg": float,   # Average spectral bandwidth
        "spectral_contrast_avg": float,    # Average spectral contrast
        
        # Variation Features
        "rms_std_dev": float,              # RMS standard deviation
        "rms_std_dev_section": float,      # Per-section RMS std dev
        "spectral_centroid_std_dev": float,# Centroid standard deviation
        "centroid_std_dev_section": float, # Per-section centroid std dev
        
        # Delta Features
        "delta_rms": float,        # RMS change from previous section
        "delta_centroid": float,   # Centroid change from previous section
        
        # Trend Features
        "rms_trend": float,        # RMS slope (rising/falling)
        "spectral_centroid_slope": float,  # Centroid slope (rising/falling)
        
        # Other Features
        "crest_factor": float,     # Peak-to-average ratio
        "low_energy_norm": float,  # Normalized low-end energy
        
        # Cluster Info
        "cluster_id": int,         # Cluster ID from clustering
    },
    # ... one dictionary per section
]
```

#### Section Labels and Colors
The audio analysis system uses a predefined set of section labels:

- **"Intro"**: Beginning of the track
- **"Drop"**: High-energy sections with strong bass
- **"Build"**: Transitional sections leading to drops
- **"Breakdown"**: Lower-energy sections
- **"Body"**: Medium-energy sections
- **"Outro"**: End section of the track
- **"Fill"**: Short transitional sections
- **"Fade Out"**: Short, fading end section

Each label has an associated color for visualization:
- "Drop": Dark Red (`#8B0000`)
- "Intro"/"Outro": Red (`#FF0000`)
- "Body": Dark Green (`#014421`)
- "Breakdown": Light Blue (`#ADD8E6`)
- "Build": Orange (`#FFA500`)
- "Fill"/"Fade Out": Purple (`#8A2BE2`)
- Fallback: Grey (`#808080`)

### Dependencies and Libraries

The audio analysis system relies on several Python libraries:

#### Core Libraries
- **numpy**: Fundamental numerical processing; used throughout for array operations
- **scipy**: Scientific computing; used for regression analysis (linregress) and stats
- **librosa**: Audio processing library; central to all audio analysis functions:
  - `librosa.load()`: Loading audio files
  - `librosa.effects.trim()`: Trimming silence
  - `librosa.beat.beat_track()`: Tempo detection
  - `librosa.stft()`: Spectrogram calculation
  - `librosa.feature.rms()`: RMS energy calculation
  - `librosa.feature.spectral_centroid()`: Spectral centroid
  - `librosa.feature.chroma_stft()`: Chroma feature extraction
  - `librosa.effects.hpss()`: Harmonic-percussive separation
  - `librosa.frames_to_time()`: Convert frame indices to times
  - Various utility functions for audio analysis

#### Machine Learning
- **sklearn.cluster.AgglomerativeClustering**: Used for clustering sections
- **sklearn.preprocessing.StandardScaler**: Used for feature scaling

#### Other Libraries
- **traceback**: Used for detailed error reporting
- **os**: Used for file path operations
- **collections.defaultdict**: Used for flexible dictionary structures
- **tkinter.messagebox**: Used for displaying error messages in the GUI

### Error Handling and Robustness

The audio analysis system implements several strategies for error handling and robustness:

#### Graceful Degradation
- **Modular Analysis Options**: Individual analysis components can be enabled/disabled
- **results_on_failure()**: Returns default structure on analysis failure
- **Parameter Validation**: Each function validates its inputs before processing
- **defaultdict Usage**: Ensures dictionary access doesn't fail on missing keys

#### Error Recovery Patterns
1. **Try-Except Patterns**: Extensive use of exception handling:
   ```python
   try:
       # Operation that may fail
       result = some_operation()
   except Exception as e:
       print(f"Error during operation: {e}")
       traceback.print_exc()
       result = fallback_value  # Use fallback
   ```

2. **NaN Handling**: Special handling for non-finite values:
   ```python
   # Check for NaN/Inf values
   if not np.all(np.isfinite(array)):
       array = np.nan_to_num(array)  # Replace NaN/Inf with finite values
   
   # Handle NaN during calculations
   finite_values = array[np.isfinite(array)]  # Filter out non-finite values
   if finite_values.size > 0:
       result = np.mean(finite_values)  # Calculate only on finite values
   else:
       result = fallback_value  # Use fallback if no finite values
   ```

3. **Division by Zero Prevention**:
   ```python
   # Safe division
   result = numerator / (denominator + epsilon)  # Add small value to avoid division by zero
   
   # Value checking before division
   if denominator > threshold:
       result = numerator / denominator
   else:
       result = fallback_value
   ```

4. **Data Consistency Checks**:
   ```python
   # Check array shapes match
   if array1.shape != array2.shape:
       print("Warning: Shape mismatch")
       # Handle mismatch (trim, pad, or skip)
   
   # Check for valid array lengths
   if len(array) < minimum_required:
       print("Warning: Not enough data points")
       # Use fallback algorithm or return default
   ```

5. **Informative Error Messages**:
   ```python
   if failed_condition:
       print(f"WARNING: Failed at stage X with parameters {params}")
       messagebox.showerror("Analysis Error", "Detailed error message for user")
   ```

#### Fallback Mechanisms
- **Default BPM**: Falls back to 120 BPM if tempo detection fails
- **Default Labels**: Uses "Error" label if section labeling fails
- **Skip Advanced Features**: Skips calculation of advanced features if prerequisites fail
- **Original Audio Fallback**: Uses original audio if trimming fails

### Extension Points

The audio analysis system is designed for extensibility. Here are the key points for adding new functionality:

#### 1. Adding New Audio Features

To add a new audio feature to the analysis:

1. **Identify the appropriate module**:
   - Frame-level features → Add to existing spectral analysis
   - Section-level features → Add to `feature_extraction.py`
   - Label-related features → Add to `chroma_analysis.py`

2. **Define the feature calculation**:
   - For a new section-level feature in `feature_extraction.py`:
   ```python
   # Example: adding a new feature called "spectral_flux"
   def extract_section_features(track_data):
       # Existing code...
       
       for i, section_dict in enumerate(section_features_list_of_dicts):
           # Existing feature calculations...
           
           # New feature calculation
           spectral_flux = np.nan  # Default to NaN
           if spec is not None and spec_times_rel is not None:
               try:
                   # Calculate flux between consecutive frames within the section
                   spec_mask = (spec_times_rel >= start_time_rel) & (spec_times_rel < end_time_rel)
                   section_spec = spec[:, spec_mask]
                   if section_spec.shape[1] > 1:  # Need at least 2 frames
                       # Calculate frame-to-frame differences
                       diffs = np.diff(section_spec, axis=1)
                       # Sum across frequencies and average
                       spectral_flux = np.mean(np.sum(np.abs(diffs), axis=0))
               except Exception as e:
                   print(f" -> ERROR calculating spectral flux: {e}")
           
           section_dict["spectral_flux"] = spectral_flux
       
       # Rest of the function...
   ```

3. **Update data structures**:
   - Add the feature to the appropriate section in the track_data documentation
   - Ensure the feature is propagated to any dependent modules

4. **Add to constants if needed**:
   - If the feature has associated thresholds or values, add them to `constants.py`

#### 2. Enhancing Section Detection

To improve section detection:

1. **Modify the detection algorithm in `section_detection.py`**:
   ```python
   def detect_sections(track_data):
       # Existing code...
       
       # Add new detection approach, e.g., using spectral features
       if spectral_features_available:
           # Integrate spectral information into section detection
           # ...
           
       # Existing code...
   ```

2. **Extend the section labeling in `chroma_analysis.py`**:
   ```python
   # Add a new cleanup rule
   print(" Applying Rule 8: New Pattern Recognition...")
   for i in range(num_sections - 2):
       if final_labels[i:i+3] == ["Pattern1", "Pattern2", "Pattern3"]:
           print(f"  Applying Rule 8 at index {i+1}")
           final_labels[i+1] = "NewLabel"
   ```

#### 3. Adding New Analysis Types

To add an entirely new type of audio analysis:

1. **Create a new module in `audio_analysis_modules/`**:
   ```python
   # new_analysis.py
   def perform_new_analysis(track_data):
       """Performs a new type of audio analysis."""
       results = {}
       # ... calculation code ...
       return results
   ```

2. **Update `__init__.py` to expose the function**:
   ```python
   # In __init__.py
   from .new_analysis import perform_new_analysis
   
   __all__ = [
       # Existing functions...
       "perform_new_analysis",
   ]
   ```

3. **Update the wrapper to re-export the function**:
   ```python
   # In audio_analysis_wrapper.py
   from audio_analysis_modules import perform_new_analysis
   
   __all__ = [
       # Existing functions...
       "perform_new_analysis",
   ]
   ```

#### 4. Improving Robustness

To enhance error handling and robustness:

1. **Add more validation checks**:
   ```python
   # Validate input data more thoroughly
   if not isinstance(data, np.ndarray) or data.ndim != 2:
       print("Warning: Expected 2D array for spectral data")
       return fallback_result
   ```

2. **Enhance recovery mechanisms**:
   ```python
   # More sophisticated fallback
   if primary_method_failed:
       try:
           # Try alternative approach
           result = alternative_calculation()
       except Exception as e:
           # If that also fails, use simple fallback
           print(f"Both primary and alternative methods failed: {e}")
           result = simple_fallback
   ```

3. **Add detailed logging**:
   ```python
   print(f"DEBUG: Processing section {i}: position={pos:.2f}, duration={dur:.2f}s")
   ```

### Coding Conventions and Patterns

#### Naming Conventions
- **Function Names**: snake_case (e.g., `detect_sections`, `analyze_chroma_and_clusters`)
- **Constants**: UPPER_CASE (e.g., `MIN_OUTRO_BARS`, `FADE_OUT_COLOR_HEX`)
- **File Names**: snake_case (e.g., `feature_extraction.py`, `chroma_analysis.py`)
- **Variables**: snake_case, with descriptive names (e.g., `section_features_list`, `spectral_centroid_frames`)
- **Boolean Variables**: Often prefixed with `is_` or similar (e.g., `is_valid`, `has_data`, `clustering_successful`)

#### Documentation Style
- **Module Docstrings**: At the top of each file, describing the module's purpose
  ```python
  """
  Functions for chroma analysis, clustering, and section labeling.
  """
  ```

- **Function Docstrings**: Google style with Args and Returns sections
  ```python
  def function_name(arg1, arg2):
      """Short description of function purpose.
      
      Detailed description if needed.
      
      Args:
          arg1 (type): Description of arg1.
          arg2 (type): Description of arg2.
      
      Returns:
          type: Description of return value.
      """
  ```

#### Common Code Patterns

1. **Function Structure Pattern**: Most functions follow this pattern:
   ```python
   def analyze_something(track_data):
       # 1. Retrieve necessary data from input
       y = track_data.get("y_processed")
       sr = track_data.get("sr")
       
       # 2. Validate inputs
       if y is None or sr is None:
           print("Warning: Missing required data.")
           return fallback_result
       
       # 3. Initialize results structure
       results = defaultdict(lambda: None)
       
       # 4. Perform main calculation with error handling
       try:
           # Core calculation logic...
       except Exception as e:
           print(f"Error during calculation: {e}")
           # Handle error, use fallbacks...
       
       # 5. Store results
       results["key1"] = value1
       results["key2"] = value2
       
       # 6. Return results
       return results
   ```

2. **Data Access Pattern**: Using `.get()` with default values for safety
   ```python
   value = track_data.get("key", default_value)
   ```

3. **Validation Pattern**: Early validation and return on invalid data
   ```python
   if not data or not isinstance(data, expected_type):
       print("Warning: Invalid data.")
       return None
   ```

4. **Phase-Based Approach**: Breaking complex analyses into labeled phases
   ```python
   print("--- Phase 1: Feature Extraction ---")
   # Phase 1 code...
   
   print("--- Phase 2: Clustering ---")
   # Phase 2 code...
   ```

5. **Debug Printing Pattern**: Structured debug output
   ```python
   print(f" -> DEBUG: Processing section {i}: {key_info}")
   ```

6. **For-Loop With Exception Handling**: Safely iterating with error isolation
   ```python
   for i, item in enumerate(collection):
       try:
           # Process item
       except Exception as e:
           print(f"Error processing item {i}: {e}")
           # Continue with next item
   ```

7. **NaN Handling Pattern**: Explicit handling of NaN/Inf values
   ```python
   # Use np.nan as default, then check before calculations
   if not np.isnan(value) and np.isfinite(value):
       # Perform calculation with value
   
   # Or replace NaN/Inf values
   array = np.nan_to_num(array)
   ```

### Performance Considerations

#### Computational Complexity

The audio analysis system handles several computationally intensive tasks:

1. **Spectrogram Calculation** (`_get_spec_data` in `spectral_analysis.py`):
   - Uses FFT (Fast Fourier Transform)
   - Complexity: O(N log N) where N is the number of audio samples
   - Memory-intensive for long audio files
   - Uses `n_fft=4096` for frequency resolution

2. **HPSS** (Harmonic-Percussive Source Separation):
   - Involves multiple STFT operations and filtering
   - High memory usage and computation time
   - Can be optionally disabled for faster analysis

3. **Section Feature Calculation** (`extract_section_features`):
   - Iterates through all sections and calculates multiple features
   - Efficiency depends on number of sections and their duration
   - Uses multiple loops over audio frames within each section

#### Optimization Strategies

The code implements several optimizations to improve performance:

1. **Selective Analysis**: Only calculate what's requested
   ```python
   if calc_stereo:
       # Perform stereo width calculation
   else:
       # Skip computation, set result to None
   ```

2. **Array Masking**: Use NumPy boolean masking instead of loops
   ```python
   # Efficient: Use boolean mask to select frames within a time range
   mask = (times >= start_time) & (times < end_time)
   values_in_section = data[mask]
   
   # Instead of:
   values_in_section = []
   for i, t in enumerate(times):
       if start_time <= t < end_time:
           values_in_section.append(data[i])
   ```

3. **Calculation Reuse**: Store and reuse intermediate calculations
   ```python
   # Calculate spectrogram once and reuse for multiple analyses
   if spec is None:  # Only calculate if not already done
       spec, freqs, times = _get_spec_data(...)
   ```

4. **Early Termination**: Stop processing when conditions aren't met
   ```python
   if not valid_input:
       return [], []  # Return early if inputs are invalid
   ```

5. **Finite Value Filtering**: Filter out NaN/Inf before calculations
   ```python
   # Only use finite values for statistics
   finite_values = values[np.isfinite(values)]
   if finite_values.size > 0:
       mean_val = np.mean(finite_values)
   ```

#### Memory Management

The system manages memory by:

1. **Selective Data Retention**: Only store necessary data in results
   ```python
   # Only store specific results, not intermediate calculations
   results["key"] = final_value
   # Temporary variables are garbage collected
   ```

2. **Avoiding Deep Copies**: Use references when possible
   ```python
   # Reference existing data instead of copying
   results["spec"] = spec  # Just store reference
   ```

3. **Using Appropriate Data Types**: Choose efficient types
   ```python
   # Use NumPy arrays for numerical data
   # Use lists for collection of heterogeneous items
   # Use dictionaries for named attributes
   ```

4. **Frame-by-Frame Processing**: Avoid loading all data at once when possible
   ```python
   # Process in chunks or frames rather than whole file at once
   ```

### Key Constants and Configuration Values

The audio analysis system defines several important constants that control its behavior:

#### Labeling Thresholds (in `constants.py`)
```python
# Thresholds used in labeling logic
MIN_OUTRO_BARS = 4        # Minimum bars for a section to be labeled Outro
MIN_FADEOUT_BARS = 4      # Sections shorter than this at the end might be fade outs
FADEOUT_RMS_THRESHOLD = 0.05  # Avg RMS below this might indicate a fade out
```

#### Visualization Colors (in `constants.py`)
```python
# Color hex codes for consistency
FADE_OUT_COLOR_HEX = "#8A2BE2"  # BlueViolet/Purple
FALLBACK_COLOR_HEX = "#808080"  # Grey for unknown labels
DARK_RED_HEX = "#8B0000"        # Dark Red for Drop sections
RED_HEX = "#FF0000"             # Red for Intro/Outro sections
```

#### Musical Constants (in `constants.py`)
```python
# Note names for chroma analysis
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
```

#### Analysis Parameters

Various functions in the audio analysis modules use specific parameter values:

1. **Audio Loading** (`loading.py`):
   ```python
   top_db = 55  # Threshold for silence trimming (dB)
   ```

2. **Spectrogram** (`spectral_analysis.py`):
   ```python
   n_fft = 4096  # FFT window size for spectrogram
   ```

3. **Section Detection** (`section_detection.py`):
   ```python
   # Relative thresholds
   fill_threshold = 0.65  # Fill RMS threshold relative to drop average
   rebound_threshold = 0.9  # Rebound threshold after fill
   section_change_threshold = 0.5  # Relative energy change to mark section boundary
   ```

4. **Feature Extraction** (`feature_extraction.py`):
   ```python
   # Feature-specific parameters
   low_freq_threshold = 150  # Threshold for low frequency (Hz)
   high_freq_threshold = 5000  # Threshold for high frequency (Hz)
   intro_max_rel_pos = 0.15  # Max position to consider as potential Intro
   outro_min_rel_pos = 0.85  # Min position to consider as potential Outro
   ```

### Testing and Debugging Techniques

The audio analysis system includes several patterns for testing and debugging:

#### Debug Print Statements

Debug print statements are used throughout the code to trace execution and inspect values:

```python
print(f"\n=== DEBUG: Inside extract_section_features ===")
print(f" -> DEBUG: Found rms data (shape: {rms_frames.shape}) and rms_times (shape: {rms_times.shape})")
print(f" -> DEBUG: Starting First Pass (Calculating avg_rms per section)...")
print(f" -> DEBUG: First Pass Complete. Max Avg RMS found for track: {max_track_rms:.4f}")
```

These print statements follow a hierarchical pattern:
1. `===` for major function boundaries
2. `---` for phases or sections
3. ` ->` for nested details

#### Section Labeling

Section phase labeling helps with tracking progress through complex algorithms:

```python
print("--- Phase 1: Enhanced Feature Extraction per Section ---")
# Phase 1 code...

print("--- Phase 2: Refined Clustering ---")
# Phase 2 code...

print("--- Phase 3: Simplified Cluster-Based Classification ---")
# Phase 3 code...

print("--- Phase 4: Contextual Cleanup Rules ---")
# Phase 4 code...
```

#### Rule Application Tracing

Detailed tracing of rule applications in section labeling:

```python
print(" Applying Rule 1: Drop->Build->Body cleanup...")
for i in range(num_sections - 2):
    if final_labels[i:i+3] == ["Drop", "Build", "Body"]:
        print(f"  Applying Rule 1 at index {i+1}: Changing Build to Breakdown.")
        final_labels[i+1] = "Breakdown"
```

#### Debugging Tools

The code relies on several approaches for debugging:

1. **Verbose Printing**: Detailed information about processing stages
   ```python
   print(f" -> Original start time: {original_start_time:.3f}, Split time: {split_time:.3f}, End time: {original_end_time:.3f}")
   ```

2. **Traceback Printing**: Full stack traces for errors
   ```python
   except Exception as e:
       print(f"ERROR: Unexpected error during split: {e}")
       traceback.print_exc()
   ```

3. **Parameter Logging**: Recording parameters for debugging
   ```python
   print(f" -> Detected BPM: {bpm_detected:.2f}, Manual override: {manual_bpm_override}")
   ```

4. **Process Announcement**: Clear indication of process stages
   ```python
   print("--- Starting NEW Cluster-Driven Labeling Approach ---")
   ```

5. **Warning/Error Levels**: Different indicators for severity
   ```python
   print("Warning: No valid clusters to map labels from. All sections remain 'Body'.")
   print("ERROR: Missing essential data for bar feature calculation.")
   ```

#### Testing Patterns

The code contains patterns that support testing:

1. **Default Returns**: Consistent return values on failure
   ```python
   return [], []  # Empty lists on failure
   ```

2. **Fallback Results Structure**: Default structure via `results_on_failure()`
   ```python
   return results_on_failure(section_starts)
   ```

3. **Input Validation**: Clear validation with descriptive messages
   ```python
   if bar_rms_data is None or bar_starts is None:
       print("Warning: Missing bar RMS or start times for section detection.")
       return [], []
   ```

4. **Alternative Code Paths**: Conditional execution for testing
   ```python
   if calc_stereo:
       # Calculate stereo width
   else:
       # Skip calculation
   ```

### Known Limitations

The audio analysis system has several known limitations:

#### Audio Processing Limitations

1. **Mono Processing**: Most analysis is performed on mono (single-channel) audio
   - Stereo information is only used for width calculation
   - Multi-channel audio (>2 channels) may not be properly handled

2. **Fixed Hop Length**: Uses predefined hop lengths
   - Default hop_length=256 samples
   - May not be optimal for all audio types or sample rates

3. **FFT Window Size**: Fixed n_fft=4096
   - Good compromise between frequency resolution and time resolution
   - May be suboptimal for very short or very high-frequency content

4. **Tempo Constraints**: Tempo detection has limits
   - Designed for music in 30-300 BPM range
   - Assumes 4/4 time signature for bar calculations
   - May struggle with complex rhythm patterns or tempo changes

#### Algorithm Limitations

1. **Section Detection**: 
   - Based primarily on RMS energy changes
   - Fixed 8-bar window approach may miss some boundaries
   - Fill detection uses simplistic energy patterns
   - No detection of gradual transitions

2. **Semantic Labeling**:
   - Relies heavily on energy (RMS) characteristics
   - Limited set of predefined labels
   - Contextual rules are fixed and may not work for all music styles
   - Clustering approach assumes distinct section types

3. **Feature Extraction**:
   - Many features depend on accurate section boundaries
   - Some features may be sensitive to noise or recording quality
   - Limited spectral resolution for low frequencies

#### Technical Limitations

1. **Error Handling**:
   - Some error handlers use print statements instead of proper logging
   - Error messages may not be visible to end users in GUI mode
   - Some error conditions may lead to inconsistent state

2. **Code Structure**:
   - Some functions are quite long and complex
   - Duplication of validation code across modules
   - Limited use of object-oriented design

3. **Testing**:
   - No formal unit tests
   - Relies on print-based debugging
   - Limited validation of complex calculations

### Future Enhancement Opportunities

Based on the current audio analysis system, several enhancement opportunities exist:

#### Algorithm Improvements

1. **Advanced Section Detection**:
   - Use more sophisticated change-point detection algorithms
   - Incorporate spectral features in addition to RMS
   - Add machine learning-based boundary detection
   - Support variable time signatures and tempo changes

2. **Enhanced Labeling**:
   - Add more section types (e.g., "Pre-chorus", "Post-chorus")
   - Implement genre-specific labeling rules
   - Use transfer learning from pre-trained audio models
   - Add confidence scores for section labels

3. **Feature Expansion**:
   - Add more perceptual features (roughness, inharmonicity)
   - Include vocal detection features
   - Add rhythm complexity metrics
   - Calculate tonal/modal features

#### Technical Enhancements

1. **Performance Optimization**:
   - Implement parallel processing for long files
   - Add caching of intermediate results
   - Optimize FFT calculations
   - Use GPU acceleration for spectral processing

2. **Code Quality**:
   - Improve error handling with proper logging
   - Add comprehensive unit tests
   - Refactor complex functions into smaller ones
   - Use more object-oriented design patterns

3. **Additional Analysis Types**:
   - Add vocal detection and analysis
   - Incorporate beat and downbeat tracking
   - Add chord progression analysis
   - Implement music style classification

## GUI Builder Module

The `gui_builder.py` module handles the construction of the application's graphical user interface components. It creates a cohesive UI structure that integrates all the app's functionality in an organized way.

### Module Overview

The GUI Builder module:
- Constructs the main application's UI components
- Creates the control panel with all input controls and options
- Builds the tabbed notebook interface for displaying plots
- Sets up the section editor interface for waveform analysis
- Manages initial UI state and placeholder content

### Key Functions

#### `build_gui(app)`
The main entry point function that builds the complete application UI by calling helper functions.

#### `_build_control_panel(app)`
Creates the top control frame containing:
- Status/action area with toggle buttons and playback controls
- File selection area with track selection buttons
- Analysis and HMM buttons for processing
- Mode selection radio buttons (single track vs. compare)
- Manual BPM input controls
- Analysis option checkboxes

#### `_build_plot_notebook(app)`
Creates the tabbed notebook interface that holds all the visualization tabs:
- Creates tabs for each visualization type ('Waveform', 'Energy/Balance', etc.)
- Initializes the section editor in the 'Waveform' tab
- Sets up placeholder labels in each tab
- Initializes plot widget dictionaries for each tab

### Important Design Patterns

1. **App Reference Pattern**: Functions take a reference to the main app instance to access its methods and attributes instead of using `self`
2. **Modular Tab Creation**: Each tab is created in a standard way but can house specialized content
3. **UI Component Organization**: Controls are grouped by function in labeled frames
4. **External Component Integration**: SectionEditor is instantiated and embedded in the Waveform tab
5. **Error Handling for Imports**: Graceful degradation if components like SectionEditor cannot be imported

### UI Component Hierarchy

```
master (root window)
├── control_frame
│   ├── action_status_frame
│   │   ├── status_label
│   │   ├── toggles_frame
│   │   │   ├── toggle_labels_button
│   │   │   └── show_hmm_button
│   │   └── playback_frame
│   │       ├── play_pause_button
│   │       └── stop_button
│   ├── file_frame
│   │   ├── select_button1 & file_label1
│   │   ├── select_button2 & file_label2
│   │   ├── analysis_buttons_frame
│   │   │   ├── analyze_button
│   │   │   └── hmm_predict_button
│   │   └── save_load_frame
│   │       ├── load_button
│   │       └── save_button
│   ├── mode_frame
│   │   ├── single track radiobutton
│   │   └── compare tracks radiobutton
│   ├── manual_bpm_frame
│   │   ├── manual_bpm_check
│   │   ├── manual_bpm_entry
│   │   └── update_bpm_button
│   └── options_frame
│       └── multiple analysis checkboxes
└── notebook
    ├── Waveform tab
    │   └── section_editor (embedded)
    ├── Energy/Balance tab
    ├── Timbre/Texture tab
    ├── Low-End tab
    ├── Dynamic Range tab
    ├── Stereo Width tab
    ├── HPSS tab
    ├── Root Note tab
    └── Band Analysis tab
```

### Key Implementation Details

1. **Command Binding**:
   - Commands are properly bound to main app methods using lambda or direct references
   - HMM button connects to the plot manager's display method
   - File operation buttons connect to file manager methods

2. **Section Editor Integration**:
   - SectionEditor is instantiated with a reference to the main application
   - SectionEditor receives a callback for applying edits

3. **Placeholder Management**:
   - Each tab initially contains a placeholder label
   - Plot manager handles replacing placeholders with actual content

### Dependencies and Limitations

1. **Dependencies**:
   - Tkinter/ttk for UI components
   - SectionEditor component from section_editor.py
   - Main app's manager instances (plot_manager, file_manager, etc.)

2. **Limitations**:
   - No dynamic reconfiguration of UI based on screen size
   - Fixed tab order and names
   - Limited error handling for UI creation failures

## Section Editor Module

The `section_editor.py` module provides a specialized UI component for viewing and editing section labels and colors. It's embedded in the Waveform tab and allows users to modify the automatically detected sections.

### Module Overview

The Section Editor module:
- Presents section data in a Treeview widget
- Enables right-click editing via a pop-up dialog
- Displays section information including bar numbers, durations, labels, and colors
- Uses visual color indicators in the Treeview
- Provides an update button to apply section edits

### Constants and Definitions

```python
COLOR_NAME_MAP = {
    "Dark Red":   '#8B0000',
    "Red":        '#FF0000',
    "Orange":     '#FFA500',
    "Dark Green": '#014421',
    "Light Green":'#7CCD7C',
    "Light Blue": '#ADD8E6',
    "Fade Out":   '#8A2BE2' # BlueViolet for Fade Out
}
HEX_TO_COLOR_NAME = {v: k for k, v in COLOR_NAME_MAP.items()}
ALLOWED_LABELS = ["Intro", "Body", "Breakdown", "Build", "Drop", "Outro", "Fill", "Fade Out"]
```

### SectionEditor Class

#### Initialization
```python
def __init__(self, master, apply_callback=None, app_ref=None, **kwargs):
```
- Takes a parent widget, callback function, and app reference
- Initializes a Tkinter frame with section editing controls

#### Methods

##### `_setup_widgets()`
- Creates the Treeview with columns for color indicator, section number, start bar, duration, label, and color
- Sets up scrollbars and visual styles
- Configures color tags for the Treeview
- Binds right-click events to edit cells
- Creates the update button

##### `populate(track_data)`
- Fills the Treeview with data from the track_data dictionary
- Formats section data into rows
- Calculates bar numbers from time values
- Applies color tags to rows

##### `clear()`
- Removes all rows from the Treeview
- Disables the update button

##### `_on_cell_edit_start(event)`
- Handles right-click events on the Treeview
- Identifies the clicked section and delegates to the main app
- Calls the main app's popup method

##### `_commit_popup_edit(popup, item_id, label_combo, color_combo)`
- Updates the Treeview with new label and color values
- Applies appropriate color tags
- Destroys the pop-up dialog
- Called by the main app as a callback

##### `_trigger_apply_edits()`
- Reads current values from all rows
- Constructs lists of new labels and colors
- Validates labels against allowed values
- Calls the apply_callback with the new data

### Data Flow

1. **Population Flow**:
   - Main application runs audio analysis
   - Analysis results are passed to SectionEditor.populate()
   - Treeview is populated with formatted section data
   - User can view and edit sections

2. **Edit Flow**:
   - User right-clicks a section
   - SectionEditor._on_cell_edit_start() is called
   - Main app shows a popup dialog
   - Changes are committed via _commit_popup_edit()
   - Treeview is updated with new values

3. **Update Flow**:
   - User clicks the "Update Sections" button
   - SectionEditor._trigger_apply_edits() is called
   - New labels and colors are collected from Treeview
   - Changes are passed to main app via apply_callback
   - Main app updates data and regenerates plots

### Error Handling and Robustness

1. **Data Validation**:
   - Checks for required keys in track_data
   - Verifies data list lengths match
   - Validates label values against allowed labels
   - Provides default values for missing or invalid data

2. **UI Robustness**:
   - Handles widget creation failures
   - Manages exceptions during population and updates
   - Uses try-except blocks for critical operations
   - Provides clear debug output

## Plot Manager Module

The `plot_manager.py` module handles the creation, embedding, updating, and clearing of plots within the application's tabbed notebook interface. It acts as a bridge between the data analysis results and the visual representation.

### Module Overview

The Plot Manager module:
- Manages plot creation and display across all visualization tabs
- Handles placeholder content for empty tabs
- Supports both single-track and comparison plotting modes
- Coordinates with other modules to determine which plots to generate
- Manages clearing and updating plots as the application state changes
- Handles the display of both original analysis and HMM prediction results

### PlotManager Class

#### Initialization
```python
def __init__(self, app_instance):
```
- Takes a reference to the main application instance
- Stores this reference for accessing application state and methods

#### Methods

##### `add_placeholder_label(tab_name, message)`
- Adds a placeholder label to a specified tab
- Ensures that the section editor is preserved when adding placeholders
- Updates the plot_widgets dictionary to track the placeholder
- Handles cases where the tab already has content

##### `add_placeholder_labels()`
- Adds appropriate placeholder labels to all tabs
- Checks analysis selection state to determine appropriate message
- Shows different messages based on mode and required analysis types
- Special handling for the Waveform tab which contains the section editor

##### `clear_plots(keep_editor=False)`
- Removes existing plots, toolbars, and placeholders
- Optionally preserves the section editor in the Waveform tab
- Manages widget destruction safely
- Clears tracking dictionaries and variables

##### `embed_plot(fig, tab_name)`
- Clears the specified tab (except for the editor if needed)
- Creates a new frame for the plot
- Embeds a Matplotlib figure into the tab
- Sets up a navigation toolbar for the plot
- Creates summary labels and playhead lines for waveform plots
- Connects event handlers for waveform interaction
- Stores references to created widgets

##### `display_analysis_results()`
- Generates and embeds plots based on current mode and data
- Handles both single-track and comparison modes
- Supports toggling between original analysis and HMM prediction views
- Selectively generates plots based on analysis options
- Updates section editor with the original data (regardless of view)
- Manages summary text and feature display
- Uses appropriate plotting functions from the audio_plotting module

### Plot Generation Process

1. **Data Preparation**:
   - Copies and potentially modifies track data for display
   - Swaps in HMM prediction data if that view is selected
   - Manages toggle button states based on view

2. **Plot Configuration**:
   - Uses a configuration list to define what plots to generate for each tab
   - Each configuration includes tab name, plotting functions, and requirements
   - Checks analysis selections against requirements
   - Verifies data availability

3. **Plot Creation**:
   - Calls appropriate plotting functions from audio_plotting
   - Uses different functions for single track vs. comparison
   - Passes the correct display data to plotting functions
   - Embeds resulting figures in tabs

4. **Summary Generation**:
   - Updates the waveform summary label with structure information
   - Adjusts wrapping based on available width
   - Handles structure summary for original view only

### Error Handling and Debug Features

1. **Error Isolation**:
   - Each plot is generated in its own try-except block
   - Failures in one plot do not affect others
   - Detailed exception information is displayed
   - Plot errors show in the relevant tab

2. **Debug Output**:
   - Extensive debug printing throughout the module
   - Tracks method entries and exits
   - Reports widget creation and destruction
   - Logs data state and plot decisions

3. **UI Robustness**:
   - Checks if widgets exist before destroying
   - Verifies tab existence before operations
   - Handles figure cleanup properly
   - Maintains UI state consistency

### Dependencies and Interfaces

1. **Dependencies**:
   - Tkinter and ttk for UI components
   - Matplotlib (plt) for creating figures
   - FigureCanvasTkAgg and NavigationToolbar2Tk for embedding plots
   - audio_plotting module (imported as ap) for plot generation functions

2. **Interfaces**:
   - Receives track_data dictionaries from main app
   - Gets analysis selection state from app.analysis_vars
   - Interfaces with section_editor for populating editor with data
   - Sets app.waveform_summary_label and app.playhead_line
   - Modifies app.plot_widgets dictionary

## Playback Manager Module

The `playback_manager.py` module handles audio playback functionality, including playing, pausing, stopping, and seeking within audio files. It uses the sounddevice library to manage audio streams and provides a robust interface between the audio subsystem and the main application.

### Module Overview

The Playback Manager module:
- Manages audio playback state and controls
- Handles loading and preparing audio data
- Provides threading-safe audio callbacks
- Coordinates GUI updates with playback position
- Manages playhead visualization on waveforms
- Implements seeking within audio files
- Ensures clean resource management

### PlaybackManager Class

#### Initialization
```python
def __init__(self, master, on_state_change=None, on_position_update=None):
```
- Takes references to the Tkinter master window and callback functions
- Sets up initial state variables and threading primitives
- Prepares for communication between audio and GUI threads

#### Core Methods

##### `set_audio(audio_data, sample_rate)`
- Loads new audio data for playback
- Stops any current playback
- Converts stereo to mono if needed
- Ensures correct data type (float32)
- Resets position and notifies state change

##### `toggle_play_pause()`
- Toggles between play and pause states
- Calls either play() or pause() based on current state

##### `play()`
- Starts or resumes audio playback
- Handles both new stream creation and resuming paused streams
- Sets up sounddevice output stream with appropriate callbacks
- Starts the playhead update loop
- Notifies state change to GUI

##### `pause()`
- Pauses audio playback without resetting position
- Stops the sounddevice stream
- Stops playhead updates
- Updates playback state

##### `stop()`
- Stops audio playback and resets position
- Cleans up resources (stream, queue)
- Resets state variables
- Notifies GUI of state change and position reset

##### `seek(target_frame)`
- Sets the playback position to a specific frame
- Clamps target to valid range
- Handles seeking during different playback states
- Provides immediate visual feedback

#### Callback Methods

##### `_audio_callback(outdata, frames, time_info, status)`
- Called by sounddevice to fill the audio buffer
- Runs in the audio thread
- Manages current frame position with thread safety
- Handles end-of-file conditions
- Communicates position updates via queue

##### `_stream_finished_callback()`
- Called when the audio stream finishes or is stopped
- Schedules GUI thread updates

##### `_handle_stream_finished_gui()`
- Runs in the GUI thread after stream finishes
- Updates playback state
- Manages position reset if needed
- Notifies GUI

#### Playhead Update Methods

##### `_start_playhead_updates()`
- Starts the loop to update the playhead position
- Schedules the first update

##### `_stop_playhead_updates()`
- Stops the scheduled playhead update loop
- Cancels any pending after callbacks

##### `_update_playhead()`
- Gets position from the queue
- Calls the GUI callback with current position
- Handles 'finished' signals
- Schedules the next update if still playing

##### `cleanup()`
- Stops playback and releases resources
- Called during application shutdown

### Threading and Synchronization

The PlaybackManager implements a robust approach to thread safety:

1. **Thread Separation**:
   - Audio processing runs in sounddevice's audio thread
   - UI updates run in Tkinter's main thread
   - Communication between threads is carefully managed

2. **Synchronization Mechanisms**:
   - Lock: Protects access to shared state (current_frame)
   - Queue: Passes position updates from audio thread to GUI thread
   - after(): Schedules GUI updates in the main thread

3. **State Management**:
   - Explicit flags track playing and paused states
   - State transitions are handled atomically
   - Callbacks are coordinated to maintain consistency

### Error Handling and Robustness

1. **Audio Thread Safety**:
   - Exceptions in audio callback are caught and logged
   - Stream is stopped cleanly on errors
   - Silent audio is produced if processing fails

2. **GUI Thread Safety**:
   - Checks if master window still exists before scheduling
   - Handles widget destruction gracefully
   - Manages after_id cancellation properly

3. **Resource Management**:
   - Ensures streams are properly closed
   - Clears queues to prevent memory leaks
   - Manages references to prevent circular dependencies

## File Manager Module

The `file_manager.py` module handles all file-related operations for the Song Analyzer application, including file selection, loading, and saving of analysis data. It provides a clean interface for file operations and ensures data persistence.

### Module Overview

The File Manager module:
- Handles audio file selection for single-track and comparison modes
- Manages saving analysis results to disk
- Provides loading functionality for previously saved analyses
- Implements file dialog interactions
- Ensures proper data serialization and deserialization
- Manages file paths and naming conventions
- Provides error handling for file operations

### Constants and Configuration

```python
PERFECT_SUBFOLDER = "Perfect"
WIP_SUBFOLDER = "WIP"
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"
```

- `PERFECT_SUBFOLDER`: Directory for storing fully validated analysis files (training data)
- `WIP_SUBFOLDER`: Directory for work-in-progress analyses
- `ANALYSIS_BASE_FOLDER`: Base directory for all saved analyses

### FileManager Class

#### Initialization
```python
def __init__(self, app_instance):
```
- Takes a reference to the main application instance
- Stores the analysis base folder path

#### File Selection Methods

##### `select_file(track_num)`
- Opens a file dialog for selecting audio files
- Updates application state with selected file
- Updates UI labels with filenames
- Clears previous analysis data
- Updates button states
- Stops any current playback
- Clears plots and adds placeholders

#### Analysis Saving Methods

##### `_ask_save_status(parent)`
- Creates a modal dialog asking user to classify the save status (Perfect or WIP)
- Centers the dialog on the parent window
- Returns the selected status or empty string if cancelled

##### `save_analysis()`
- Validates that analysis data exists for Track 1 and is in single-track mode
- Asks for save status (Perfect or WIP) via dialog
- Creates appropriate directory if needed
- Derives safe filename from audio filename
- Handles confirmation for overwriting existing files
- Converts any defaultdict to standard dict before saving
- Uses joblib.dump() for serialization
- Shows appropriate success/error messages

#### Analysis Loading Methods

##### `load_analysis()`
- Opens a file dialog for selecting saved analysis files
- Loads the selected file using joblib.load()
- Verifies it contains a valid analysis dictionary
- Converts loaded data to standard dict
- Validates required keys are present
- Checks if original audio file still exists at stored path
- Updates application state with loaded data
- Updates UI to reflect loaded state
- Shows appropriate success/error messages

### Serialization and Data Management

1. **File Formats**:
   - Uses joblib for Python object serialization
   - Saves with compression level 3
   - Includes full track_data dictionary in saved files
   - Stores original audio file path for reference

2. **Data Validation**:
   - Checks for essential keys in loaded data
   - Verifies data types
   - Handles defaultdict conversion for storage safety
   - Provides detailed error messages for loading issues

3. **File Organization**:
   - Categorizes saved analyses as "Perfect" (for training) or "WIP"
   - Uses standardized file naming (base_audio_name.analysis.joblib)
   - Sanitizes filenames to ensure safe paths

### Error Handling and Robustness

1. **File Operation Safety**:
   - Handles file dialog cancellations
   - Checks directory creation success
   - Verifies file existence before operations
   - Uses try-except blocks for all file operations

2. **Data Corruption Prevention**:
   - Ensures proper dict conversion before saving
   - Validates loaded data structure
   - Provides fallbacks for missing audio files
   - Maintains application state on load failures

3. **Feedback and Reporting**:
   - Shows informative message boxes for operations
   - Updates status label with operation results
   - Displays appropriate warnings for file issues
   - Provides detailed debug output

### Integration with Main Application

1. **Application State Updates**:
   - Updates app.file_path
   - Updates app.track_names
   - Sets app.track_data
   - Updates file labels

2. **UI Coordination**:
   - Clears plots via plot_manager
   - Sets placeholder labels via plot_manager
   - Updates playback manager audio
   - Updates button states via main app methods

3. **Mode Handling**:
   - Forces single-track mode when loading
   - Clears track 2 data when in single mode
   - Updates UI for mode changes

## HMM Predictor Module

The `hmm_predictor.py` module implements a Hidden Markov Model (HMM) predictor for automatically labeling song sections based on their audio features. It loads a pre-trained model and auxiliary data, and provides prediction functionality that integrates with the main application.

### Module Overview

The HMM Predictor module:
- Loads trained Gaussian HMM or GMM-HMM models
- Loads associated auxiliary data (feature keys, scaling parameters, etc.)
- Extracts features from audio sections
- Processes features for prediction (scaling, weighting)
- Predicts section labels using the trained model
- Maps numeric states to semantic labels
- Provides colors for visualization
- Aligns predictions with original section starts

### Constants and Configuration

```python
HMM_PURPLE_COLOR = "#8A2BE2"  # Define Purple locally for clarity
HMM_RESULT_COLOR_MAP = {
    "Intro": "#FF0000",  # Red
    "Outro": "#FF0000",  # Red
    "Body": "#014421",  # Dark Green
    "Build": "#FFA500",  # Orange
    "Drop": "#8B0000",  # Dark Red
    "Breakdown": "#ADD8E6",  # Light Blue
    "Fill": HMM_PURPLE_COLOR,  # Purple
    "Fade Out": HMM_PURPLE_COLOR,  # Purple
    "Unknown": HMM_PURPLE_COLOR,  # Purple
}
HMM_FALLBACK_COLOR = HMM_PURPLE_COLOR  # Purple for unexpected labels
```

### HMMPredictor Class

#### Initialization
```python
def __init__(self, model_path, aux_data_path):
```
- Takes paths to the saved model and auxiliary data files
- Initializes attributes for model, scaler, mapping, and feature configuration
- Sets initial loaded state to False

#### Methods

##### `load_model()`
- Loads the HMM model from the specified file using joblib
- Loads auxiliary data including:
  - `int_to_label`: Mapping from state indices to label names
  - `scaler_multi`: StandardScaler for multiple features
  - `feature_keys`: List of feature keys to extract
  - `feature_weights_vector`: Weights for each feature
  - `labels_ignored`: Labels to ignore during prediction
- Validates that all required components were loaded
- Checks compatibility of dimensions and types
- Sets loaded state to True on success

##### `predict(track_data)`
- Validates that model is loaded
- Retrieves section feature dictionaries via extract_section_features()
- Creates a feature vector for each section by extracting keys defined in feature_keys
- Skips sections with ignored labels
- Filters out sections with missing or non-finite features
- Scales features using the loaded scaler
- Applies feature weights
- Runs the HMM prediction using model.predict()
- Maps predicted state indices to semantic labels
- Assigns colors based on the label map
- Aligns predictions with original section starts
- Returns section starts, labels, and colors on success

### Prediction Process

1. **Data Preparation**:
   - Extracts section features from track_data
   - Selects only the features defined in feature_keys
   - Filters out sections with ignored labels
   - Validates feature values (checks for None, NaN, Inf)
   - Tracks indices of kept sections for later alignment

2. **Feature Processing**:
   - Converts feature list to NumPy array
   - Scales features using the pre-trained StandardScaler
   - Applies feature weights via element-wise multiplication
   - Prepares final feature matrix for prediction

3. **HMM Prediction**:
   - Calls model.predict() with processed features
   - Gets state indices for each section
   - Maps indices to semantic labels using int_to_label mapping
   - Assigns colors based on predicted labels

4. **Alignment and Return**:
   - Aligns predictions with original section starts
   - Handles potential length mismatches
   - Returns tuple of (section_starts, labels, colors)
   - Returns empty lists if no valid sections found
   - Returns None on errors

### Error Handling and Robustness

1. **Model Loading Validation**:
   - Checks file existence
   - Validates model type and attributes
   - Verifies all auxiliary components are present
   - Confirms dimensional compatibility
   - Shows detailed error messages on failure

## Main Application Module

The `main_app.py` file serves as the core of the Song Analyzer App, containing the `AudioAnalyzerApp` class which orchestrates all other components and implements the main application logic. This module integrates the GUI, audio analysis, plotting, playback, file operations, and HMM prediction functionality into a cohesive desktop application.

### Module Overview

The Main Application module:
- Initializes the Tkinter application root and window
- Creates instances of all manager classes (Plot, Playback, File, HMM)
- Sets up application state variables and Tkinter control variables
- Defines the audio analysis orchestration pipeline
- Implements section editing, merging, and splitting functionality
- Manages UI state updates based on user actions and application mode
- Provides callbacks for UI elements and user interactions
- Coordinates data flow between different components

### Constants and Configuration

```python
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"
PROJECT_BASE_FOLDER = os.path.dirname(ANALYSIS_BASE_FOLDER)
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")

# HMM model configuration
N_FEATURES_EXPECTED = 7 
N_MIXTURES_EXPECTED = 1
CLEANING_FLAGS_EXPECTED = "out_sh_con"
base_filename = f"gmmhmm_{N_FEATURES_EXPECTED}f_{N_MIXTURES_EXPECTED}m_{CLEANING_FLAGS_EXPECTED}"
GMMHMM_MODEL_PATH = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_model.joblib")
GMMHMM_AUX_PATH = os.path.join(HMM_OUTPUT_FOLDER, f"{base_filename}_aux.joblib")

# Section editing constraints
MIN_SPLIT_SECTION_DURATION_SEC = 1.0
```

### AudioAnalyzerApp Class

#### Initialization and Structure

```python
def __init__(self, master):
    # Master window setup
    self.master = master
    master.title("Audio Analyzer Tool")
    master.geometry("1100x900")
    
    # Application state variables
    self.file_path = {1: None, 2: None}
    self.track_data = {1: None, 2: None}
    self.track_names = {1: "Track 1", 2: "Track 2"}
    self.plot_widgets = {}
    
    # Manager instances
    self.playback_manager = PlaybackManager(...)
    self.plot_manager = PlotManager(self)
    self.hmm_predictor = HMMPredictor(GMMHMM_MODEL_PATH, GMMHMM_AUX_PATH)
    self.file_manager = FileManager(self)
    
    # Tkinter control variables
    self.mode = tk.StringVar(value="single")
    self.analysis_vars = {...}
    self.use_manual_bpm = tk.BooleanVar(value=False)
    self.manual_bpm_entry_var = tk.StringVar()
    self.show_pre_cleanup_labels_var = tk.BooleanVar(value=False)
    self.show_hmm_var = tk.BooleanVar(value=False)
    
    # Build GUI and set initial state
    build_gui(self)
    self.ui_manager.update_ui_for_mode()
    self.ui_manager.update_manual_bpm_state()
```

#### UI State Management Methods

The class includes several methods to manage UI state based on user actions and application mode:

- `update_manual_bpm_state()`: Enables/disables BPM entry based on checkbox state and data availability
- `update_ui_for_mode()`: Updates UI elements when switching between single/compare modes
- `update_analyze_button_state()`: Enables/disables analyze button based on file selection
- `_update_save_button_state()`: Enables save button when track data exists in single mode
- `_update_hmm_button_state()`: Manages HMM buttons based on model availability and prediction state
- `_update_toggle_button_state()`: Enables/disables label toggle button based on availability
- `_update_playback_buttons_state_from_manager()`: Updates playback UI based on playback state

#### Analysis Orchestration

The core analysis pipeline is implemented through two key methods:

##### `run_analysis(self, is_update=False, manual_bpm_val=None)`
- Entry point for analysis, called when "Analyze" button is clicked
- Handles both initial analysis and BPM updates
- Manages UI state during analysis (status updates, button disabling)
- Calls `_analyze_single_track()` for each track based on mode
- Updates playback with processed audio
- Triggers plot display on completion

##### `_analyze_single_track(self, track_num, manual_bpm_override=None)`
- Implements the complete analysis pipeline for a single track
- Loads and preprocesses audio with librosa
- Calculates intermediate features (bar RMS, beat frames)
- Conditionally calls section detection based on analysis options
- Calls chroma analysis and labeling if enabled
- Performs spectral analysis, HPSS, and stereo width calculation if enabled
- Calculates low-end energy features if enabled
- Computes dynamic range metrics if enabled
- Ensures all feature dictionaries are properly populated
- Returns the complete track_data dictionary

#### Section Editing Functionality

The application supports advanced section editing through several mechanisms:

1. **Label/Color Editing**:
   - `_show_section_edit_popup(section_index, event)`: Creates popup dialog for editing
   - `_apply_section_edits_from_editor(new_labels, new_colors_hex)`: Applies edits from the SectionEditor

2. **Section Merging**:
   - `_trigger_merge(popup, section_index, direction)`: Handles merge button clicks
   - `_merge_section(remove_boundary_index)`: Merges two adjacent sections by removing boundary

3. **Section Splitting**:
   - `_show_split_section_popup(section_index, clicked_time)`: Shows popup for split configuration
   - `_commit_split_by_bar(popup, section_index, bar_var, seconds_per_bar, trim_offset)`: Validates and executes split
   - `_split_section(section_index, split_time)`: Splits section at specified time point

4. **Feature Recalculation**:
   - `_recalculate_section_features(track_data, section_idx)`: Helper method to update features after edits

#### Section Shifting Functionality

The application provides a feature to shift multiple section boundaries simultaneously, useful for correcting timing offsets across a portion of the track.

- **Trigger:** A dedicated "Shift Sections..." button located in the main control panel (near playback controls). This button is enabled only in single-track mode when analysis data is present.
- **Interaction:** Clicking the button opens a dialog box prompting the user for:
    - **Start Bar #:** The 1-based bar number from which the shift should begin. All section boundaries *at or after* the start of this bar will be affected.
    - **Shift Amount (bars, +/-):** The number of bars to shift the boundaries. Positive values shift boundaries later in time, while negative values shift them earlier.
- **Backend Process (`_apply_section_shift` in `main_app.py`):**
    1.  The specified `shift_bars` value is converted into `shift_seconds` using the track's `seconds_per_bar`.
    2.  The `section_starts` list within `track_data` is modified by adding `shift_seconds` to all boundary times from the calculated `start_index` onwards.
    3.  Validation checks prevent boundaries from being shifted to negative times.
    4.  The `start_time` and `end_time` values within the corresponding dictionaries in the `section_features` list are updated based on the new `section_starts`. Durations (`duration_sec`, `duration_bars`) are also recalculated.
    5.  The `_recalculate_section_features` method is called for each section whose boundaries were modified to update its acoustic features based on the new timing.
    6.  Any existing HMM prediction results are cleared, as the shift invalidates them.
    7.  The plots (`PlotManager`) and the Section Editor display (`SectionEditor.populate`) are refreshed to reflect the changes.


#### Event Handling

The class implements several event handlers for user interactions:

- `_on_waveform_click(event)`: Handles mouse clicks on waveform for seeking, editing, and splitting
- `_toggle_label_view()`: Switches between final and pre-cleanup labels
- `_trigger_hmm_prediction()`: Runs the HMM prediction process
- `update_plots_with_manual_bpm()`: Re-analyzes with user-provided BPM value
- `_update_playhead_display(frame)`: Updates visual playhead position during playback
- `_on_closing()`: Handles application shutdown

### Analysis Data Flow

The overall data flow in the application follows this pattern:

1. **File Selection**:
   - User selects audio file(s) via FileManager
   - Paths stored in `self.file_path` dictionary

2. **Analysis Initialization**:
   - User selects analysis options and clicks "Analyze"
   - `run_analysis()` checks inputs and prepares UI state
   - `track_data` dictionary created to store results

3. **Analysis Pipeline**:
   - `_analyze_single_track()` orchestrates processing
   - Audio loaded from file path using librosa
   - Sequential processing through analysis modules
   - Results progressively added to `track_data`

4. **Post-Processing**:
   - Final track_data dictionary passed to PlotManager for visualization
   - SectionEditor populated with section data
   - Processed audio sent to PlaybackManager
   - UI state updated based on results

5. **User Interaction**:
   - User can edit, merge, or split sections via GUI
   - Modifications update track_data
   - Plots and editor refreshed to reflect changes
   - Optional HMM prediction adds alternative labeling

6. **Persistence**:
   - User can save analysis results via FileManager
   - Can reload previous analyses for continued work

### Integration with Other Modules

The main application integrates with other modules in specific ways:

1. **GUI Builder**:
   - Calls `build_gui(self)` to construct the interface
   - Provides widget references for updates

2. **Plot Manager**:
   - Creates PlotManager instance with app reference
   - Calls `display_analysis_results()` to show plots
   - Uses `clear_plots()` and `add_placeholder_labels()`

3. **Playback Manager**:
   - Creates PlaybackManager with callbacks
   - Calls `set_audio()` with processed audio
   - Provides event handling for seeking
   - Updates UI based on playback state

4. **File Manager**:
   - Creates FileManager with app reference
   - Delegates file operations (select, load, save)
   - Updates UI based on file operations

5. **HMM Predictor**:
   - Creates HMMPredictor with model paths
   - Calls `load_model()` and `predict()`
   - Stores results in track_data

6. **Section Editor**:
   - Instantiated by GUI Builder
   - Receives app reference for callbacks
   - Provides editing interface for sections
   - Returns edits to main app for processing

7. **Audio Analysis**:
   - Directly calls analysis functions from imported modules
   - Orchestrates sequential processing
   - Handles error conditions and validation

### Error Handling and Robustness

The main application implements extensive error handling:

1. **Import Safety**:
   - Catches ImportError during module imports
   - Shows helpful error message box
   - Gracefully exits if critical modules missing

2. **Analysis Error Handling**:
   - Try-except blocks around critical operations
   - Detailed error messages using traceback
   - Graceful degradation when possible
   - User-friendly error messagebox on failure

3. **UI State Consistency**:
   - Widget existence checks before configuration
   - Logical dependencies between UI states
   - Safe handling of deleted widgets

4. **Data Validation**:
   - Input checking before operations
   - Validation of time/bar ranges for edits
   - Array bounds and data type verification
   - Handling of missing or invalid data

5. **Recovery Mechanisms**:
   - Graceful failure for individual plots
   - Safe handling of playback issues
   - Preservation of UI state on errors

## Transition Matrix Extraction

The `extract_transitions.py` script is responsible for generating a transition probability matrix from analyzed tracks. This matrix serves as a prior for HMM training, capturing the likelihood of transitions between different section types in music.

### Module Overview

The Transition Matrix Extraction module:
- Processes "Perfect" analysis files to extract state transition patterns
- Counts occurrences of transitions between different section types
- Calculates probability matrices from raw counts
- Applies smoothing to avoid zero probabilities
- Can incorporate prior musical knowledge to adjust probabilities
- Visualizes transition matrices as heatmaps
- Saves the resulting matrix for use in HMM training

### Constants and Configuration

```python
ANALYSIS_BASE_FOLDER = "/Users/donovanblair/Desktop/song_analyzer_app_6/completed_analyses"
PERFECT_FOLDER_PATH = os.path.join(ANALYSIS_BASE_FOLDER, "Perfect")
PROJECT_BASE_FOLDER = os.path.dirname(os.path.dirname(ANALYSIS_BASE_FOLDER))
HMM_OUTPUT_FOLDER = os.path.join(PROJECT_BASE_FOLDER, "hmm_model")

LABEL_TO_IDX = {
    "Body": 0,
    "Breakdown": 1,
    "Build": 2,
    "Drop": 3,
    "Intro": 4,
    "Outro": 5,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
N_STATES = len(LABEL_TO_IDX)

SMOOTHING_ALPHA = 0  # Laplace smoothing parameter (0 = disabled)
OUTPUT_MATRIX_FILENAME = "transition_matrix.npy"
```

### Key Functions

#### `load_perfect_analyses(folder_path)`
- Loads all .joblib analysis files from the "Perfect" subfolder
- Validates that each file contains semantic_labels
- Returns a list of loaded track data dictionaries

#### `extract_transition_counts(all_data)`
- Extracts raw transition counts from labeled tracks
- Only considers labels defined in LABEL_TO_IDX
- Returns a transition count matrix and state occurrence counts

#### `create_transition_matrix(transition_counts)`
- Converts raw counts into a probability matrix
- Handles states with zero outgoing transitions by assigning uniform probability
- Ensures each row sums to 1.0 (proper probability distribution)

#### `apply_smoothing(transition_probs, alpha)`
- Applies Laplace (additive) smoothing to avoid zero probabilities
- Uses formula: P_smooth = (1-α) * P_original + α / N_states
- Ensures rows sum to 1 after smoothing
- Skipped if alpha is 0 or less

#### `incorporate_prior_knowledge(transition_probs)`
- Adjusts transition probabilities based on domain knowledge
- Multiplies specific transitions by enhancement/reduction factors
- Ensures rows are renormalized afterwards
- Tracks and reports probability changes

#### `visualize_transition_matrix(transition_probs, title)`
- Creates a heatmap visualization of the transition matrix
- Includes probability values in each cell
- Labels axes with state names
- Saves the visualization to a file

#### `save_transition_matrix(transition_matrix, filename)`
- Saves the NumPy transition matrix to the specified file
- Used by the HMM training process as a prior

### Matrix Generation Process

1. **Data Loading**:
   - Loads analysis files that have been marked as "Perfect"
   - Validates the presence of semantic_labels in each file
   - Reports the number of valid files found

2. **Transition Counting**:
   - Extracts section labels from each track
   - Counts transitions between consecutive sections
   - Maintains a count matrix and individual state counts
   - Reports total transitions counted

3. **Probability Calculation**:
   - Converts raw counts to probabilities (P = count / row_sum)
   - Handles zero-count rows with uniform distribution
   - Ensures valid probability distributions (rows sum to 1)

4. **Smoothing (Optional)**:
   - Applies Laplace smoothing if alpha > 0
   - Prevents zero probabilities in the matrix
   - Useful for HMM training to avoid numerical issues

5. **Enhancement (Calculated but not saved)**:
   - Incorporates domain knowledge about typical transitions
   - Enhances probabilities for common patterns (e.g., Build→Drop)
   - Reduces probabilities for rare transitions (e.g., Outro→Intro)
   - Renormalizes after adjustments

6. **Visualization**:
   - Creates heatmaps of the initial, smoothed, and enhanced matrices
   - Includes numerical values and state labels
   - Saves visualizations to the HMM output folder

7. **Matrix Saving**:
   - Saves the smoothed (non-enhanced) matrix to a .npy file
   - This matrix can be used as a prior for HMM training

### Integration with HMM Training

- The saved transition matrix serves as a prior for HMM training
- In GMMHMM training, it can be used to initialize the transition matrix
- Improves HMM convergence by providing a musically informed starting point
- Can be bypassed if the user prefers standard HMM initialization

## GMMHMM Trainer

The `run_gmmhmm_trainer.py` script orchestrates the training of a Gaussian Mixture Model Hidden Markov Model (GMMHMM) for automatic section labeling. It serves as the main entry point for the training process, coordinating various modules to prepare data, configure and train the model, and save the results.

### Module Overview

The GMMHMM Trainer module:
- Provides a GUI interface for configuring training parameters
- Loads and preprocesses "Perfect" analysis files
- Prepares multi-feature data for HMM training
- Optionally loads a transition matrix prior
- Trains a GMMHMM model with specified parameters
- Analyzes and visualizes the trained model's behavior
- Saves the model and auxiliary data for later use by the predictor
- Exports detailed analysis results to JSON

### Training Configuration Options

1. **Feature Selection and Weighting**:
   - The user can select which audio features to use for training
   - Each feature can be assigned a weight to control its importance
   - Feature selection impacts what section characteristics the model learns
   - ALL_FEATURE_KEYS = [
    "avg_rms",
    "relative_rms",
    "relative_position",
    "position_context",
    "label_proportion",  # <-- ADDED
    "low_energy_norm",
    "rms_std_dev_section",
    "centroid_std_dev_section",
    "delta_rms",
    "delta_centroid",
    "rms_trend",
    "crest_factor",
    "spectral_centroid_slope",
   ]


2. **Model Parameters**:
   - Number of Gaussian mixtures per state
   - Minimum covariance value (to prevent numerical issues)
   - Covariance type (diagonal, full, tied, spherical)

3. **Data Cleaning Options**:
   - Outlier removal: Removes sections with feature values that are statistical outliers
   - Short section removal: Excludes sections shorter than a specified number of bars
   - Consistency filtering: Removes tracks with inconsistent labeling patterns

4. **Transition Prior Usage**:
   - Option to use or ignore the pre-computed transition matrix
   - Allows comparing HMM with learned vs. pre-defined transition probabilities

### Training Process Flow

1. **Configuration Dialog**:
   - Shows a GUI dialog to select features, weights, and model parameters
   - Validates user inputs and constructs a configuration dictionary
   - Determines base filename for outputs based on selected options

2. **Data Loading**:
   - Loads all "Perfect" analysis files using the `load_perfect_analyses` function
   - Reports the number of valid tracks found

3. **Multi-Feature Data Preparation**:
   - Calls `prepare_multi_feature_data` with selected features and cleaning settings
   - Extracts features from each section of each track
   - Applies selected data cleaning procedures
   - Scales features using StandardScaler to normalize ranges
   - Creates mapping between numeric states and semantic labels
   - Tracks sequence lengths for proper HMM segmentation

4. **Transition Matrix Loading (Optional)**:
   - Checks if user enabled transition prior usage
   - If enabled, loads the transition matrix generated by `extract_transitions.py`
   - Validates matrix shape and row sums
   - Prepares for use as HMM initialization

5. **GMMHMM Training**:
   - Calls `train_gmmhmm` with prepared data and parameters
   - Initializes GMMHMM with selected number of mixtures and other parameters
   - Sets initial transition matrix with loaded prior (if enabled)
   - Trains the model using EM algorithm
   - Returns the trained model and feature weight vector

6. **Model Analysis and Visualization**:
   - Analyzes the trained model using `analyze_trained_gmmhmm`
   - Visualizes emission distributions, predicted sequences, and transition matrix
   - Shows a detailed analysis dialog with performance metrics
   - Allows the user to evaluate model quality before saving

7. **Model and Auxiliary Data Saving**:
   - Saves the trained model to a .joblib file
   - Creates an auxiliary data dictionary with:
     - Label mappings (int_to_label, label_to_int)
     - Feature scaler for normalization
     - Feature keys and weights
     - Labels to ignore during prediction
     - Model parameters and configuration
   - Saves auxiliary data to a separate .joblib file

8. **Analysis Export**:
   - Exports detailed analysis results to a JSON file
   - Includes model parameters, cleaning settings, feature information
   - Records convergence information and model performance metrics
   - Provides a record of training configuration for future reference

### Key Implementation Details

1. **Dynamic Configuration**:
   - Filename generation based on selected options (features, mixtures, cleaning)
   - Flag for transition prior usage reflected in filename
   - Comprehensive configuration summary printed for reference

2. **Error Handling**:
   - Extensive try-except blocks for robust operation
   - Detailed error reporting with traceback for debugging
   - Graceful failure handling with appropriate exit codes

3. **Configuration Tracking**:
   - Records all user selections and configurations
   - Saves cleaning settings and preprocessing parameters
   - Preserves feature keys and weights for prediction
   - Tracks whether transition prior was actually used

4. **Integration with HMM Predictor**:
   - Saves model and auxiliary data in format expected by predictor
   - Ensures all necessary data for prediction is preserved
   - Compatible with the file paths expected by the main application

### Dependencies and Integration

1. **Dependencies**:
   - GMMHMM_modules package for the actual implementation
   - Tkinter for GUI dialogs
   - joblib for model serialization
   - numpy for numerical operations

2. **Integration with Main Application**:
   - Saves models where the main app expects to find them
   - Uses same label mappings as extract_transitions.py
   - Compatible with the HMMPredictor module for inference
   - Preserves all data needed for accurate prediction

2. **Feature Extraction Safety**:
   - Checks feature existence in dictionaries
   - Explicitly handles None, NaN, and Inf values
   - Validates feature vector dimensions
   - Skips sections with invalid features
   - Handles empty feature set case

3. **Prediction Robustness**:
   - Checks model loading state before prediction
   - Validates input data availability
   - Handles alignment issues between predictions and starts
   - Uses try-except blocks for all critical operations
   - Provides detailed error reporting

### Dependencies and Integration

1. **Dependencies**:
   - joblib for model loading
   - numpy for numerical processing
   - traceback for detailed error reporting
   - tkinter.messagebox for error display
   - extract_section_features from audio_analysis_wrapper

2. **Integration Points**:
   - Returns data in format ready for main application
   - Provides consistent color mapping for visualization
   - Handles errors through both console output and message boxes
   - Follows same pattern as original analysis for section representation

### Implementation Details

1. **Feature Weighting**:
   - Uses element-wise multiplication to apply weights to scaled features
   - Allows controlling the importance of different features in prediction
   - Example: Position features might be weighted higher than variation features

2. **State Mapping**:
   - Maps numeric HMM states to semantic labels
   - Uses a dictionary loaded from the auxiliary data
   - Provides "Unknown" fallback for unmapped states

3. **Ignoring Labels**:
   - Skips sections with labels in the `labels_ignored` list
   - Typical ignored labels might include "Fill" or "Fade Out"
   - Improves prediction accuracy by focusing on main structural sections

4. **Fixes for Common Issues**:
   - Handles defaultdict conversion for storage safety
   - Uses correct key names when loading auxiliary data (feature_weights_vector)
   - Implements proper checks for missing or invalid features
   - Ensures proper color mapping for visualization consistency

### Use in Application Flow

1. **Model Loading**:
   - Main application initializes HMMPredictor with paths to model files
   - Calls load_model() to prepare the predictor
   - Manages UI to indicate model loading status

2. **Prediction Triggering**:
   - User clicks "Run HMM Prediction" button
   - Main application calls HMMPredictor.predict() with current track_data
   - Results are stored in track_data as hmm_semantic_labels, hmm_label_colors, and hmm_section_starts

3. **Visualization**:
   - User toggles between original analysis and HMM prediction
   - Plot manager handles display switching based on toggle state
   - HMM results are visually distinguished with consistent colors

4. **Results Application**:
   - HMM predictions can be viewed alongside original analysis
   - User can compare automated predictions with original analysis
   - Visual distinction helps identify differences in labeling approaches
