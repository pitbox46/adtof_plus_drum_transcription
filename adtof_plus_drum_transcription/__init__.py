"""ADTOF Plus Drum Transcription Package

A Python package for drum transcription using ADTOF and MDX23C models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import transcribe_drums, transcribe_audio_file

__all__ = ['transcribe_drums', 'transcribe_audio_file']
