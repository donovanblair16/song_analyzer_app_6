# =============================================================================
# FILE: playback_manager.py
# Contains the PlaybackManager class for handling audio playback,
# seeking, pausing, and state management using sounddevice.
# =============================================================================

import sounddevice as sd
import numpy as np
import threading
import queue
import traceback

class PlaybackManager:
    """
    Manages audio playback using sounddevice, including state,
    seeking, and communication with the main GUI thread.
    """
    def __init__(self, master, on_state_change=None, on_position_update=None):
        """
        Initializes the PlaybackManager.

        Args:
            master: The root Tkinter window (for scheduling GUI updates).
            on_state_change (callable): Callback function to notify the GUI
                                        about playback state changes
                                        (e.g., 'playing', 'paused', 'stopped').
            on_position_update (callable): Callback function to notify the GUI
                                           about playback position updates
                                           (passes current_frame).
        """
        self.master = master # Reference to main Tkinter window for scheduling
        self.on_state_change = on_state_change
        self.on_position_update = on_position_update

        self.stream = None
        self.audio_data = None
        self.sample_rate = None
        self.current_frame = 0
        self.is_playing = False
        self.is_paused = False
        self.lock = threading.Lock()
        self.queue = queue.Queue() # For callback -> playhead update loop communication
        self.playhead_after_id = None

    def set_audio(self, audio_data, sample_rate):
        """Loads new audio data for playback."""
        print("DEBUG PlaybackManager: Setting audio data.")
        self.stop() # Stop any current playback
        with self.lock:
            # Ensure audio data is float32 and mono (N,)
            if audio_data is not None and audio_data.ndim > 1:
                 # Convert to mono by averaging if stereo
                 if audio_data.shape[1] == 2:
                      print("DEBUG PlaybackManager: Converting stereo audio to mono for playback.")
                      audio_data = np.mean(audio_data, axis=1)
                 else: # Handle other multi-channel cases if necessary
                      print(f"Warning PlaybackManager: Unsupported channel count {audio_data.shape[1]}. Using first channel.")
                      audio_data = audio_data[:, 0]

            # Ensure correct dtype
            if audio_data is not None and audio_data.dtype != np.float32:
                 print(f"DEBUG PlaybackManager: Converting audio dtype to float32.")
                 audio_data = audio_data.astype(np.float32)


            self.audio_data = audio_data
            self.sample_rate = sample_rate
            self.current_frame = 0
            print(f"DEBUG PlaybackManager: Audio set. Length: {len(self.audio_data) if self.audio_data is not None else 'None'} frames, SR: {self.sample_rate}")
        self.is_playing = False
        self.is_paused = False
        if self.on_state_change:
            self.on_state_change('stopped') # Notify GUI state changed

    def toggle_play_pause(self):
        """Toggles playback between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Starts or resumes audio playback."""
        if self.audio_data is None or self.sample_rate is None:
            print("Playback Error: No audio data loaded.")
            # Optionally show a message box? Might be better handled in GUI layer
            return
        if self.is_playing:
            print("DEBUG PlaybackManager: Already playing.")
            return

        try:
            if self.is_paused and self.stream and not self.stream.closed:
                print("DEBUG PlaybackManager: Resuming audio stream...")
                self.stream.start()
                self.is_paused = False
                print("DEBUG PlaybackManager: Stream resumed.")
            else: # Start new stream
                with self.lock:
                    start_frame = self.current_frame
                    if start_frame >= len(self.audio_data): start_frame = 0
                    self.current_frame = start_frame # Ensure updated if clamped
                print(f"DEBUG PlaybackManager: Starting new audio stream from frame {start_frame}...");
                if self.stream and not self.stream.closed:
                    self.stream.stop()
                    self.stream.close()
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1, # Assuming mono
                    callback=self._audio_callback,
                    blocksize=4096, # Buffer size
                    finished_callback=self._stream_finished_callback # Internal callback
                )
                self.stream.start()
                self.is_paused = False
                print("DEBUG PlaybackManager: New stream started.")

            self.is_playing = True
            if self.on_state_change: self.on_state_change('playing')
            self._start_playhead_updates() # Start internal loop for position updates

        except Exception as e:
            print(f"Playback Error: Failed to start audio stream: {e}")
            traceback.print_exc()
            self.stop() # Ensure clean state on error

    def pause(self):
        """Pauses audio playback."""
        if not self.is_playing or not self.stream:
            print("DEBUG PlaybackManager: Not playing or no stream, cannot pause.")
            return
        try:
            print("DEBUG PlaybackManager: Pausing audio stream...")
            if self.stream and not self.stream.stopped:
                 self.stream.stop()
                 print("DEBUG PlaybackManager: Stream stopped.")
            else:
                 print("DEBUG PlaybackManager: Stream already stopped or doesn't exist.")
            self.is_playing = False
            self.is_paused = True
            self._stop_playhead_updates() # Stop position updates
            if self.on_state_change: self.on_state_change('paused')
            print("DEBUG PlaybackManager: Pause complete.")
        except Exception as e:
            print(f"Playback Error: Failed to pause audio stream: {e}")
            traceback.print_exc()

    def stop(self):
        """Stops audio playback and resets position."""
        print("DEBUG PlaybackManager: Stopping audio stream...")
        self._stop_playhead_updates() # Stop position updates first
        stream_to_close = self.stream
        self.stream = None # Clear ref early
        if stream_to_close:
            try:
                if not stream_to_close.closed:
                    print("DEBUG PlaybackManager: Calling stream.stop()...")
                    stream_to_close.stop()
                    print("DEBUG PlaybackManager: Calling stream.close()...")
                    stream_to_close.close()
                    print("DEBUG PlaybackManager: Stream stopped and closed.")
            except Exception as e: print(f"Error stopping/closing stream: {e}")

        # Clear the queue *after* stopping stream
        print("DEBUG PlaybackManager: Clearing playback queue...")
        while not self.queue.empty():
            try: self.queue.get_nowait();
            except queue.Empty: break

        print("DEBUG PlaybackManager: Resetting playback state...")
        with self.lock: # Lock before resetting frame
            self.current_frame = 0;
            print(f"DEBUG PlaybackManager: current_frame reset to 0 (inside lock).")
        self.is_playing = False
        self.is_paused = False
        if self.on_state_change: self.on_state_change('stopped')
        # Notify GUI to hide playhead (position 0)
        if self.on_position_update: self.on_position_update(0)
        print("DEBUG PlaybackManager: Stop audio finished.")

    def seek(self, target_frame):
        """Sets the playback position to the target frame."""
        if self.audio_data is None: return

        audio_length = len(self.audio_data)
        target_frame = max(0, min(int(target_frame), audio_length - 1)) # Clamp

        print(f"DEBUG PlaybackManager: Seek requested to frame {target_frame}")

        with self.lock:
            self.current_frame = target_frame
            print(f"DEBUG PlaybackManager: Set current_frame to {self.current_frame} (inside lock)")

        # If paused, seeking should unpause and start playing from new spot
        if self.is_paused:
            print("DEBUG PlaybackManager: Was paused during seek, restarting play.")
            self.is_playing = False # Ensure play() starts fresh stream state
            self.is_paused = False
            self.play() # Restart playback from the new frame
        elif self.is_playing:
            # If already playing, the callback will pick up the new frame.
            # We just need to ensure the playhead update loop is running.
             print("DEBUG PlaybackManager: Was playing during seek, callback will handle.")
             self._start_playhead_updates() # Make sure loop is active
             # Send immediate update for visual feedback
             if self.on_position_update: self.on_position_update(target_frame)
        else: # Was stopped
             print("DEBUG PlaybackManager: Was stopped during seek.")
             # Update state for buttons, playback will start here if user hits play
             if self.on_state_change: self.on_state_change('stopped')
             # Send immediate update for visual feedback
             if self.on_position_update: self.on_position_update(target_frame)


    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback function for sounddevice stream (runs in audio thread)."""
        if status: print(f"DEBUG Callback Status: {status}", flush=True)
        try:
            with self.lock: # Acquire lock to safely access/update current_frame
                chunk_start = self.current_frame
                chunk_end = chunk_start + frames
                remaining_frames = len(self.audio_data) - chunk_start

                if remaining_frames <= 0:
                    chunk = None; self.queue.put('finished'); should_stop = True
                elif remaining_frames < frames:
                    chunk = self.audio_data[chunk_start : chunk_start + remaining_frames]
                    self.current_frame += remaining_frames
                    self.queue.put('finished'); should_stop = True
                else:
                    chunk = self.audio_data[chunk_start : chunk_end]
                    self.current_frame += frames
                    self.queue.put(self.current_frame); should_stop = False

            # Process output data outside the lock
            if chunk is not None:
                chunk_2d = chunk.reshape(-1, 1) # Ensure 2D for sounddevice
                outdata[:len(chunk_2d)] = chunk_2d
                if len(chunk_2d) < frames: outdata[len(chunk_2d):].fill(0)
            else:
                outdata.fill(0)

            if should_stop: raise sd.CallbackStop
        except Exception as e:
            print(f"!!! EXCEPTION IN AUDIO CALLBACK: {e}")
            traceback.print_exc()
            outdata.fill(0) # Fill with silence on error
            self.queue.put('finished') # Try to signal end
            raise sd.CallbackStop # Stop the stream on error


    def _stream_finished_callback(self):
        """Internal callback when sounddevice stream finishes or is stopped."""
        # This runs in the audio thread
        print("DEBUG PlaybackManager: Stream Finished Callback Triggered.")
        # Use master.after(0) to schedule the GUI update in the main thread
        # Check if master exists and hasn't been destroyed
        if self.master and self.master.winfo_exists():
            self.master.after(0, self._handle_stream_finished_gui)
        else:
            print("DEBUG PlaybackManager: Master window not available for stream finished handling.")


    def _handle_stream_finished_gui(self):
        """Handles GUI updates after stream finishes (runs in main thread)."""
        print("DEBUG PlaybackManager: Handling Stream Finished in GUI Thread.")
        # Check if it finished naturally while playing, or was stopped/paused
        reset_frame = self.is_playing and not self.is_paused
        self.is_playing = False; self.is_paused = False # Always reset flags
        if reset_frame:
             print("DEBUG PlaybackManager: Stream finished naturally, resetting frame.")
             with self.lock:
                 self.current_frame = 0
        else:
             print("DEBUG PlaybackManager: Stream stopped/paused, not resetting frame.")

        self._stop_playhead_updates() # Ensure updates are stopped
        # Notify main app about the state change
        if self.on_state_change: self.on_state_change('stopped')
        # Notify main app to hide playhead (position 0) if reset
        if reset_frame and self.on_position_update: self.on_position_update(0)

        # Don't set self.stream = None here, stop() or play() handles it
        print("DEBUG PlaybackManager: Stream finished GUI handling complete.")

    # --- Playhead Update Loop (Internal to Manager) ---
    def _start_playhead_updates(self):
        """Starts the loop to update the playhead position."""
        if not self.playhead_after_id:
             print("DEBUG PlaybackManager: Starting playhead update loop.")
             self._update_playhead() # Start the first update
        else:
             print("DEBUG PlaybackManager: Playhead update loop already running.")

    def _stop_playhead_updates(self):
        """Stops the scheduled playhead update loop."""
        if self.playhead_after_id:
            print("DEBUG PlaybackManager: Stopping playhead update loop.")
            # Check if master exists before cancelling
            if self.master and self.master.winfo_exists():
                 try:
                      self.master.after_cancel(self.playhead_after_id)
                 except Exception as e:
                      print(f"Error cancelling after_id: {e}")
            self.playhead_after_id = None

    def _update_playhead(self):
        """Gets position from queue and calls GUI callback."""
        # print("DEBUG PlaybackManager: _update_playhead tick") # Too verbose
        current_pos_frame = None
        try:
            # Get the latest frame from the queue without blocking
            while not self.queue.empty():
                current_pos_frame = self.queue.get_nowait()
                if current_pos_frame == 'finished':
                    print("DEBUG PlaybackManager: Playhead received 'finished' signal in update loop.");
                    self._stop_playhead_updates() # Stop loop on finished signal
                    return

            # If we got a valid frame number, notify the GUI
            if current_pos_frame is not None and self.on_position_update:
                # print(f"DEBUG PlaybackManager: Notifying GUI of frame {current_pos_frame}") # Verbose
                self.on_position_update(current_pos_frame)

        except queue.Empty:
             pass # No update from callback yet, just wait for next cycle
        except Exception as e:
            print(f"Error in PlaybackManager _update_playhead: {e}")
            traceback.print_exc()
            self._stop_playhead_updates() # Stop updates on error

        # Schedule the next update only if playing AND not paused
        if self.is_playing and not self.is_paused:
             # Check if master exists before scheduling
             if self.master and self.master.winfo_exists():
                  self.playhead_after_id = self.master.after(50, self._update_playhead) # Update ~20 times/sec
             else:
                  self.playhead_after_id = None # Stop if master is gone
        else: # Ensure loop stops if paused or stopped externally
             self._stop_playhead_updates()

    # --- Cleanup ---
    def cleanup(self):
        """Stops audio and cleans up resources."""
        print("DEBUG PlaybackManager: Cleaning up...")
        self.stop()

