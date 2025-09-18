# ADTOF Plus Drum Transcription

A Python package for drum transcription using ADTOF and MDX23C models. This tool can automatically transcribe drum patterns from audio files and output them as MIDI files.

## Credits

The MDX23c model was developed by [Kuielab](https://github.com/kuielab)

The drum stem variant was published by Jarredou here: https://github.com/jarredou/models

The ADTOF transcription model was published by Zehren et al here: https://github.com/MZehren/ADTOF.

My Pytorch port of their weights is here: https://github.com/xavriley/ADTOF-pytorch

## Features

- **Source Separation**: Automatically separates drums from full mixes using MDX23C
- **Drum Stem Separation**: Separates drum kit into individual components (kick, snare, hi-hat, cymbals, toms)
- **MIDI Transcription**: Converts drum audio to MIDI using ADTOF models
- **Velocity Mapping**: Maps audio energy to MIDI velocities
- **Hi-hat Detection**: Distinguishes between open and closed hi-hat
- **Batch Processing**: Process multiple audio files at once
- **CLI Interface**: Easy-to-use command-line interface

## Installation

### From PyPI (when published)
```bash
pip install adtof-plus-drum-transcription
```

### From Source
```bash
git clone <repository-url>
cd adtof-plus-drum-transcription
pip install -e .
```

### Dependencies

This package depends on:
- [MDX23C](https://github.com/xavriley/mdx23c-drum-separation) for drum separation
- [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) for drum transcription
- Essential audio processing libraries (essentia, numpy, scipy)

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Transcribe a single audio file
adtof-transcribe --audio_path song.wav --output_path drums.mid

# Process all audio files in a directory
adtof-transcribe --audio_path /path/to/songs/ --output_path /path/to/output/
```

#### Options
- `--audio_path`: Path to audio file or directory (supports .wav, .mp3, .flac)
- `--output_path`: Output MIDI file or directory
- `--input_is_mix`: Input is a full mix requiring drum separation (default: True)
- `--no-input_is_mix`: Input is isolated drums, no separation needed
- `--default_threshold`: Threshold for onset detection (default: -inf)

#### Examples

```bash
# Process isolated drums (skip source separation)
adtof-transcribe --audio_path drums.wav --output_path drums.mid --no-input_is_mix

# Use custom threshold for onset detection
adtof-transcribe --audio_path song.wav --output_path drums.mid --default_threshold -30

# Batch process all files in a folder
adtof-transcribe --audio_path ./audio_files/ --output_path ./midi_output/
```

### Python API

```python
from adtof_plus_drum_transcription import transcribe_audio_file

# Transcribe a single file
transcribe_audio_file(
    audio_path="song.wav",
    output_path="drums.mid", 
    input_is_mix=True,
    default_threshold=-float('inf')
)
```

## How It Works

1. **Source Separation** (if `input_is_mix=True`): Uses MDX23C to extract drums from full mix
2. **Stem Separation**: Separates drum kit into kick, snare, hi-hat, cymbals, and toms
3. **Transcription**: Uses ADTOF models to detect drum onsets and convert to MIDI
4. **Velocity Mapping**: Maps audio energy to MIDI velocities with different curves for different drums
5. **Hi-hat Processing**: Analyzes sustain patterns to distinguish open vs closed hi-hat
6. **MIDI Export**: Combines all drum parts into a single MIDI file

## Supported Formats

- **Input**: WAV, MP3, FLAC
- **Output**: MIDI (.mid)

## Requirements

- Python 3.8+
- 64-bit system (required for some dependencies)
- Sufficient RAM for audio processing (8GB+ recommended)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [ADTOF](https://github.com/xavriley/ADTOF-pytorch) for drum transcription models
- [MDX23C](https://github.com/xavriley/mdx23c-drum-separation) for drum separation
- [Essentia](https://essentia.upf.edu/) for audio analysis
