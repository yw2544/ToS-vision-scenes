"""
Multi-Room Generator Package
============================

A comprehensive system for generating objects in multi-room environments
based on mask analysis, with camera positioning and door handling.

Modules:
- room_analyzer: Analyzes room layout from masks
- object_generator: Generates objects in rooms based on size constraints
- door_handler: Manages door positions, colors, and photography
- camera_capture: Handles all camera operations and image capture
- metadata_manager: Manages all metadata for rooms, objects, and doors
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .room_analyzer import RoomAnalyzer
from .object_generator import ObjectGenerator
from .door_handler import DoorHandler
from .camera_capture import CameraCapture
from .metadata_manager import MetadataManager

__all__ = [
    'RoomAnalyzer',
    'ObjectGenerator', 
    'DoorHandler',
    'CameraCapture',
    'MetadataManager'
] 