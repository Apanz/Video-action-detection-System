"""
GUI module for video action detection
"""

from .main_window import MainWindow
from .detection_tab import DetectionTab
from .results_tab import ResultsTab
from .model_management_tab import ModelManagementTab

__all__ = ['MainWindow', 'DetectionTab', 'ResultsTab', 'ModelManagementTab']
