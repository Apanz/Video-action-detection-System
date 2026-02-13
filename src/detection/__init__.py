"""
Detection module for real-time video processing
"""

from .pipeline import DetectionPipeline
from .human_detector import HumanDetector
from .action_classifier import ActionClassifier, load_classifier
from .temporal_processor import TemporalProcessor
from .video_writer import VideoWriter, FrameOverlay
from .result_collector import ResultCollector
from .model_metadata import ModelMetadata

__all__ = [
    'DetectionPipeline',
    'HumanDetector',
    'ActionClassifier',
    'load_classifier',
    'TemporalProcessor',
    'VideoWriter',
    'FrameOverlay',
    'ResultCollector',
    'ModelMetadata'
]
