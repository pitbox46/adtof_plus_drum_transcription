"""Core drum transcription functionality."""

from mdx23c import demix_kit_from_mix, demix_stems_from_kit
from adtof_pytorch import transcribe_to_midi
import tempfile
from pathlib import Path
import os
import essentia.standard as es
import math
import pretty_midi as pm
import numpy as np
import tqdm


def map_energy_to_velocity(energy, max_db, min_velocity=0, max_velocity=127, use_log=False):
    """Map energy values to MIDI velocity values."""
    # Normalize energy to range [0, 1]
    energy = np.clip(np.array(energy), -90, max_db) + 90
    energy_min = 0
    energy_max = max_db + 90

    normalized_energy = (energy - energy_min) / (energy_max - energy_min)

    if use_log:
        normalized_energy = np.clip(normalized_energy, 0.33, 1)
        normalized_energy = normalized_energy ** 1.75

    return np.interp(normalized_energy, [0, 1], [min_velocity, max_velocity]).astype(int)


def get_loudness(path, threshold=-float("inf"), activity_threshold=None):
    """Extract loudness values from audio file."""
    loader = es.EqloudLoader(filename=str(path), replayGain=-3)
    audio = loader()

    frame_size = 1024
    hop_size = 441
    w = es.Windowing(type="hann")
    loudness = es.Loudness()

    loudness_values = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        windowed = w(frame)
        loudness_values.append(loudness(windowed))

    # convert loudness values to decibels, ensuring no non-positive values
    loudness_values = [10 * math.log10(value) if value > 0 else -90 for value in loudness_values]

    max_db = max(loudness_values)

    # set anything below threshold to -90
    if threshold is not None:
        loudness_values = [-90 if value < threshold else value for value in loudness_values]

    if activity_threshold is not None:
        if not any(value > activity_threshold for value in loudness_values):
            return -90, [-90 for _ in loudness_values]

    return max_db, loudness_values


def is_monotonic_neighbour(x, n, neighbour):
    """Check if values around index n are monotonic."""
    monotonic = True

    for i in range(neighbour):
        if x[n - i] < x[n - i - 1]:
            monotonic = False
        if x[n + i] < x[n + i + 1]:
            monotonic = False

    return monotonic


def nearest_onset_time(onsets, onset_time, use_shift=False, threshold=float("-inf")):
    """Find the nearest onset time with optional time shifting."""
    idx = int(onset_time * 100)

    neighbour = 2

    # limit window to 2 samples either side of idx
    window = onsets[max(0, idx - neighbour):min(len(onsets), idx + neighbour)]
    window_onset_idx = np.argmax(window)

    onset_idx = int(max(0, idx - neighbour) + window_onset_idx)

    if onset_idx < 0 or onset_idx >= len(onsets):
        return None, None
    
    if onsets[onset_idx] <= threshold:
        return None, None

    if not use_shift:
        return onset_idx, onset_idx / 100

    if onsets[onset_idx] > -55 and is_monotonic_neighbour(onsets, onset_idx, neighbour):
        """See Section III-D in [1] for deduction.
        [1] Q. Kong, et al., High-resolution Piano Transcription 
        with Pedals by Regressing Onsets and Offsets Times, 2020."""
        if onsets[onset_idx - 1] > onsets[onset_idx + 1]:
            shift = (onsets[onset_idx + 1] - onsets[onset_idx - 1]) / (onsets[onset_idx] - onsets[onset_idx + 1]) / 2
        else:
            shift = (onsets[onset_idx + 1] - onsets[onset_idx - 1]) / (onsets[onset_idx] - onsets[onset_idx - 1]) / 2
    else:
        shift = 0
        
    refined_onset_time = (onset_idx + shift) / 100
    
    return int(onset_idx), refined_onset_time


def transcribe_drums(audio_path, output_path, input_is_mix=True, default_threshold=-float("inf")):
    """
    Transcribe drums from audio file to MIDI.
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str): Path for output MIDI file
        input_is_mix (bool): Whether input is a full mix (needs drum separation)
        default_threshold (float): Threshold for onset detection
    """
    combined_mid = pm.PrettyMIDI()
    combined_mid.instruments.append(pm.Instrument(program=0, is_drum=True))

    hihat_loudness = None
    cymbals_loudness = None
    label_note_lookup = {
        "kick": 35,
        "snare": 38,
        "hihat": 42,
        "cymbals": 49,
        "toms": 47
    }

    note_label_lookup = {v: k for k, v in label_note_lookup.items()}

    # create a temporary folder for the output directory using tempfile
    with tempfile.TemporaryDirectory() as temp_dir:

        # optional - get kit from mix
        if input_is_mix:
            demix_kit_from_mix(audio_path, output_dir=temp_dir)
            audio_path = os.path.join(temp_dir, Path(audio_path).stem + "_drums.wav")

        demix_stems_from_kit(audio_path, output_dir=temp_dir)
        kick_path = os.path.join(temp_dir, Path(audio_path).stem + "_kick.wav")
        snare_path = os.path.join(temp_dir, Path(audio_path).stem + "_snare.wav")
        hihat_path = os.path.join(temp_dir, Path(audio_path).stem + "_hh.wav")
        cymbals_path = os.path.join(temp_dir, Path(audio_path).stem + "_cymbals.wav")
        toms_path = os.path.join(temp_dir, Path(audio_path).stem + "_toms.wav")

        for label, path in tqdm.tqdm([("kick", kick_path), ("snare", snare_path), ("hihat", hihat_path), ("cymbals", cymbals_path), ("toms", toms_path)], desc="Transcribing MIDI from stems"):
            mid_out_path = os.path.join(temp_dir, f"{label}.mid")
            transcribe_to_midi(path, mid_out_path)
            print(f"Transcribed {label} to {mid_out_path}")
            mid = pm.PrettyMIDI(mid_out_path)

            max_db, loudness_values = get_loudness(path)
            use_log = True if label in ["kick", "snare"] else False
            velocities = map_energy_to_velocity(loudness_values, max_db, use_log=use_log)

            if label == "hihat":
                hihat_loudness = loudness_values
            if label == "cymbals":
                cymbals_loudness = loudness_values

            for n in mid.instruments[0].notes:
                if n.pitch in [35, 38, 42, 49]:  # kick, snare and hh
                    # Only transcribe kicks from the kick stem, etc.
                    if label != note_label_lookup[n.pitch]:
                        continue

                    onset_idx, onset_time = nearest_onset_time(loudness_values, n.start, threshold=default_threshold)
                    if onset_idx is not None:
                        n.end = n.start + 0.01

                        n.velocity = max(velocities[max(0, onset_idx-1):min(len(velocities), onset_idx+2)])

                        if n.velocity == 0:
                            n.velocity = 40
                elif n.pitch == 47:  # toms
                    onset_idx, onset_time = nearest_onset_time(loudness_values, n.start, threshold=default_threshold)

                    if onset_idx is None:
                        # possibly a snare
                        n.pitch = 38
                        n.end = n.start + 0.01
                        n.velocity = 40
                    else:
                        n.end = n.start + 0.01
                        n.velocity = max(velocities[max(0, onset_idx-1):min(len(velocities), onset_idx+2)])
                            
                        if n.velocity == 0:
                            n.velocity = 40

                combined_mid.instruments[0].notes.append(n)
        
        combined_mid.instruments[0].notes.sort(key=lambda x: x.start)

        hihat_notes = [n for n in combined_mid.instruments[0].notes if n.pitch == 42]
        for i in range(len(hihat_notes)):
            if i < len(hihat_notes) - 1:
                interval_start = hihat_notes[i].start
                interval_end = min(interval_start + 0.15, hihat_notes[i+1].start)

                interval_loudness = hihat_loudness[int(interval_start * 100):int(interval_end * 100)]
                if len(interval_loudness) == 0:
                    continue

                interval_loudness = [l + 90 for l in interval_loudness]
                if min(interval_loudness) > (0.75 * max(interval_loudness)):
                    # hihat is probably open
                    hihat_notes[i].pitch = 46
        
        # Process cymbals notes (crash vs ride detection) if we have cymbals loudness data
        if cymbals_loudness is not None:
            cymbals_notes = [n for n in combined_mid.instruments[0].notes if n.pitch == 49]
            for i in range(len(cymbals_notes)):
                if i < len(cymbals_notes) - 1:
                    interval_start = cymbals_notes[i].start
                    interval_end = min(interval_start + 0.15, cymbals_notes[i+1].start)

                    interval_loudness = cymbals_loudness[int(interval_start * 100):int(interval_end * 100)]
                    if len(interval_loudness) == 0:
                        continue

                    interval_loudness = [l + 90 for l in interval_loudness]
                    if min(interval_loudness) > (0.75 * max(interval_loudness)):
                        # cymbals is probably a crash
                        cymbals_notes[i].pitch = 49
                    else:
                        # cymbals is probably a ride
                        cymbals_notes[i].pitch = 51

        combined_mid.write(output_path)


def transcribe_audio_file(audio_path, output_path, input_is_mix=True, default_threshold=-float("inf")):
    """
    Convenience function to transcribe a single audio file.
    
    Args:
        audio_path (str or Path): Path to input audio file
        output_path (str or Path): Path for output MIDI file
        input_is_mix (bool): Whether input is a full mix (needs drum separation)
        default_threshold (float): Threshold for onset detection
    """
    audio_path = str(audio_path)
    output_path = str(output_path)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    transcribe_drums(audio_path, output_path, input_is_mix, default_threshold)
