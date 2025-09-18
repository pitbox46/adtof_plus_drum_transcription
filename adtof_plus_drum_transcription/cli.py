"""Command-line interface for ADTOF Plus Drum Transcription."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

from .core import transcribe_audio_file


def get_audio_files(path: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """
    Get audio files from a path (file or directory).
    
    Args:
        path: Path to file or directory
        extensions: List of supported extensions (default: wav, mp3, flac)
    
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac']
    
    path = Path(path)
    
    if path.is_file():
        if path.suffix.lower() in extensions:
            return [path]
        else:
            raise ValueError(f"File {path} has unsupported extension. Supported: {extensions}")
    
    elif path.is_dir():
        audio_files = []
        for ext in extensions:
            audio_files.extend(path.glob(f"*{ext}"))
            audio_files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(audio_files)
    
    else:
        raise FileNotFoundError(f"Path {path} does not exist")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe drums from audio files to MIDI using ADTOF and MDX23C models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file
  adtof-transcribe --audio_path song.wav --output_path drums.mid

  # Transcribe all audio files in a folder
  adtof-transcribe --audio_path /path/to/songs/ --output_path /path/to/output/

  # Process isolated drums (not a full mix)
  adtof-transcribe --audio_path drums.wav --output_path drums.mid --no-input_is_mix

  # Use custom threshold
  adtof-transcribe --audio_path song.wav --output_path drums.mid --default_threshold -30
        """
    )
    
    parser.add_argument(
        "--audio_path",
        required=True,
        help="Path to audio file or directory containing audio files (wav, mp3, flac)"
    )
    
    parser.add_argument(
        "--output_path", 
        required=True,
        help="Path for output MIDI file or directory (when processing multiple files)"
    )
    
    parser.add_argument(
        "--input_is_mix",
        action="store_true",
        default=True,
        help="Input is a full mix requiring drum separation (default: True)"
    )
    
    parser.add_argument(
        "--no-input_is_mix",
        dest="input_is_mix",
        action="store_false",
        help="Input is isolated drums, no separation needed"
    )
    
    parser.add_argument(
        "--default_threshold",
        type=float,
        default=float('-inf'),
        help="Threshold for onset detection (default: -inf, no threshold)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()
    
    try:
        # Get audio files
        audio_files = get_audio_files(args.audio_path)
        
        if not audio_files:
            print(f"No audio files found in {args.audio_path}")
            sys.exit(1)
        
        print(f"Found {len(audio_files)} audio file(s) to process")
        
        # Determine output handling
        output_path = Path(args.output_path)
        
        if len(audio_files) == 1:
            # Single file processing
            audio_file = audio_files[0]
            
            if output_path.is_dir():
                # Output directory specified, create MIDI filename
                midi_filename = audio_file.stem + ".mid"
                final_output_path = output_path / midi_filename
            else:
                # Output file specified
                final_output_path = output_path
            
            print(f"Processing: {audio_file} -> {final_output_path}")
            transcribe_audio_file(
                audio_file, 
                final_output_path, 
                args.input_is_mix, 
                args.default_threshold
            )
            print("✓ Transcription completed!")
            
        else:
            # Multiple files processing
            if not output_path.is_dir():
                # Create output directory if it doesn't exist
                output_path.mkdir(parents=True, exist_ok=True)
            
            for i, audio_file in enumerate(audio_files, 1):
                midi_filename = audio_file.stem + ".mid"
                final_output_path = output_path / midi_filename
                
                print(f"Processing ({i}/{len(audio_files)}): {audio_file} -> {final_output_path}")
                
                try:
                    transcribe_audio_file(
                        audio_file,
                        final_output_path,
                        args.input_is_mix,
                        args.default_threshold
                    )
                    print(f"✓ Completed: {audio_file}")
                
                except Exception as e:
                    print(f"✗ Failed to process {audio_file}: {e}")
                    continue
            
            print(f"✓ Batch transcription completed! {len(audio_files)} files processed.")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
