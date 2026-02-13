"""
Human Detection Module using YOLO
Detects multiple persons in video frames for action recognition
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


class HumanDetector:
    """
    YOLO-based human detector for multi-person tracking
    """

    def __init__(self, model_path: str = 'yolov5s.pt', confidence: float = 0.5,
                 iou_threshold: float = 0.45, device: str = 'auto'):
        """
        Initialize human detector

        Args:
            model_path: Path to YOLO model (or 'yolov5s.pt' to download)
            confidence: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            device: 'cpu', 'cuda', or 'auto' (auto-detect)
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading YOLO model on {self.device}...")

        # Load YOLO model
        try:
            # Try ultralytics YOLO (supports both v5 and v8)
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.is_ultralytics = True
        except ImportError:
            try:
                # Fallback to yolov5
                import yolov5
                self.model = yolov5.load(model_path, device=self.device)
                self.is_ultralytics = False
            except ImportError:
                raise ImportError("Please install YOLO: pip install ultralytics")

        # COCO class names (YOLO default)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

        # Person class ID in COCO
        self.person_class_id = 0

        # For tracking
        self.detection_history = []
        self.track_id_counter = 0
        self.last_detection = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect humans in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections with person info
        """
        detections = []

        try:
            # Run inference
            if self.is_ultralytics:
                # Ultralytics YOLO
                results = self.model(frame, conf=self.confidence, iou=self.iou_threshold)
                detections = self._parse_ultralytics_results(results)
            else:
                # YOLOv5
                results = self.model(frame)
                detections = self._parse_yolov5_results(results)

        except Exception as e:
            print(f"Detection error: {e}")

        return detections

    def _parse_ultralytics_results(self, results) -> List[Dict]:
        """Parse ultralytics YOLO results"""
        detections = []

        if results:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID
                        class_id = int(box.cls[0])

                        # Only detect persons
                        if class_id == self.person_class_id:
                            # Get bounding box
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])

                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': 'person'
                            }
                            detections.append(detection)

        return detections

    def _parse_yolov5_results(self, results) -> List[Dict]:
        """Parse YOLOv5 results"""
        detections = []

        for result in results:
            # Get detections
            if hasattr(result, 'xyxy'):
                boxes = result.xyxy[0]
                confidences = result.conf[0]
                class_ids = result.cls[0]

                # Convert to numpy
                boxes = boxes.cpu().numpy()
                confidences = confidences.cpu().numpy()
                class_ids = class_ids.cpu().numpy()

                for i in range(len(boxes)):
                    class_id = int(class_ids[i])

                    # Only detect persons
                    if class_id == self.person_class_id:
                        x1, y1, x2, y2 = boxes[i]
                        confidence = float(confidences[i])

                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': 'person'
                        }
                        detections.append(detection)

        return detections

    def get_best_detection(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Get best detection based on confidence score
        For single-person detection, we use detection with highest confidence

        Args:
            detections: List of detection results

        Returns:
            Best detection or None if no detection
        """
        if not detections:
            return None

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        # Return best detection
        return sorted_detections[0]

    def crop_person(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Crop person region from frame

        Args:
            frame: Input frame
            detection: Detection dictionary with bbox

        Returns:
            Cropped person region
        """
        x1, y1, x2, y2 = detection['bbox']

        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        # Crop person
        person_crop = frame[y1:y2, x1:x2]



        return person_crop

    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                       action_label: str = None, confidence: float = None,
                       colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw detection boxes on frame with support for multiple people

        Args:
            frame: Input frame
            detections: List of detections
            action_label: Action classification label (for single detection mode)
            confidence: Classification confidence (for single detection mode)
            colors: List of BGR colors for each detection (cycles if needed)

        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()

        # Default colors (green, blue, red, cyan, magenta)
        if colors is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        if detections:
            # If action_label is provided, use single-person mode (backward compatibility)
            if action_label is not None:
                best_detection = self.get_best_detection(detections)
                if best_detection:
                    self._draw_single_detection(output_frame, best_detection,
                                               action_label, confidence, colors[0])
            else:
                # Multi-person mode: draw all detections
                for idx, detection in enumerate(detections):
                    color = colors[idx % len(colors)]
                    self._draw_single_detection(output_frame, detection,
                                               None, None, color, idx + 1)

        return output_frame

    def _draw_single_detection(self, frame: np.ndarray, detection: Dict,
                              action_label: str = None, confidence: float = None,
                              color: Tuple[int, int, int] = (0, 255, 0),
                              person_id: int = None):
        """
        Draw a single detection box with labels

        Args:
            frame: Input frame
            detection: Detection dictionary
            action_label: Action classification label
            confidence: Classification confidence
            color: BGR color for bounding box
            person_id: Person ID for labeling
        """
        x1, y1, x2, y2 = detection['bbox']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1
        line_height = 18

        # Prepare labels
        labels = []
        if person_id is not None:
            labels.append(f"Person {person_id}: {detection['confidence']:.2f}")
        else:
            labels.append(f"Person: {detection['confidence']:.2f}")

        if action_label and confidence is not None:
            labels.append(f"Action: {action_label}")
            labels.append(f"Conf: {confidence:.2f}")

        # Draw labels with background for better visibility
        for i, label in enumerate(labels):
            y_pos = y1 - 10 - (len(labels) - i - 1) * line_height

            # Ensure labels don't go off screen
            y_pos = max(y_pos, line_height)

            # Get text size for background rectangle
            (text_width, text_height) = cv2.getTextSize(label, font, font_scale, text_thickness)[0]

            # Draw background rectangle
            cv2.rectangle(frame,
                         (x1, y_pos - text_height - 2),
                         (x1 + text_width + 4, y_pos + 2),
                         color, -1)

            # Draw text
            cv2.putText(frame, label, (x1 + 2, y_pos),
                       font, font_scale, (255, 255, 255), text_thickness)
