import wave
import numpy as np
from scipy.io import wavfile
import os

class SoundProcessor:
    def __init__(self):
        self.min_chunk_duration = 0.2  # seconds
        self.silence_threshold = 0.1  # 10% of mean amplitude
        self.min_silence_duration = 0.1  # seconds
        self.max_silence_duration = 2.0  # seconds
        
    def chop_recording(self, filename):
        """Chop a recording into chunks based on silence detection"""
        print(f"Processing file: {filename}")
        # Read the wav file
        rate, data = wavfile.read(filename)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Normalize data
        data = data / np.max(np.abs(data))
        
        # Detect silence
        is_silence = np.abs(data) < self.silence_threshold
        print(f"Found {np.sum(~is_silence)} non-silent samples out of {len(data)}")
        print(f"Silence threshold: {self.silence_threshold}")
        
        # Find silence boundaries
        silence_starts = []
        silence_ends = []
        current_silence_start = None
        
        for i in range(len(is_silence)):
            if is_silence[i] and current_silence_start is None:
                current_silence_start = i
            elif not is_silence[i] and current_silence_start is not None:
                silence_duration = (i - current_silence_start) / rate
                if self.min_silence_duration <= silence_duration <= self.max_silence_duration:
                    silence_starts.append(current_silence_start)
                    silence_ends.append(i)
                    print(f"Found silence: {silence_duration:.2f}s")
                current_silence_start = None
        
        print(f"Found {len(silence_starts)} valid silences")
        
        # Create chunks based on silences
        chunk_starts = []
        chunk_ends = []
        
        # If we have no silences at all and the file is long enough, treat it as one chunk
        if not silence_starts and len(data)/rate > self.min_chunk_duration:
            chunk_starts = [0]
            chunk_ends = [len(data)]
        else:
            # Only start with beginning of file if there's a silence after it
            if silence_starts:
                chunk_starts.append(0)

            # Process middle chunks
            for start, end in zip(silence_starts, silence_ends):
                chunk_ends.append(start)
                chunk_starts.append(end)

            # Only include last chunk if there was a silence before it
            if silence_ends:
                chunk_ends.append(len(data))
        
        # Create chunks
        chunk_files = []
        for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
            duration = (end - start) / rate
            print(f"Chunk {i}: duration = {duration:.2f}s")
            # Only keep chunks longer than min_duration
            if duration > self.min_chunk_duration:
                chunk_filename = filename.replace('.wav', f'_chunk_{i}.wav')
                self._save_chunk(data[start:end], rate, chunk_filename)
                chunk_files.append(chunk_filename)
            else:
                print(f"Rejecting chunk {i}: too short ({duration:.2f}s < {self.min_chunk_duration}s)")
        
        return chunk_files
    
    def _save_chunk(self, data, rate, filename):
        """Save a chunk of audio as a wav file"""
        # Denormalize data back to int16 range
        data = data * 32767
        wavfile.write(filename, rate, data.astype(np.int16))

# Make sure the class is available for import
__all__ = ['SoundProcessor'] 