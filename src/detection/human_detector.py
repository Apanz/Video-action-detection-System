"""
使用YOLO的人体检测模块
检测视频帧中的多个人用于动作识别
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


class HumanDetector:
    """
    基于YOLO的人体检测器，用于多人跟踪
    """

    def __init__(self, model_path: str = 'yolov5s.pt', confidence: float = 0.5,
                 iou_threshold: float = 0.45, device: str = 'auto'):
        """
        初始化人体检测器

        Args:
            model_path: YOLO模型路径（或'yolov5s.pt'以下载）
            confidence: 检测置信度阈值
            iou_threshold: NMS的IoU阈值
            device: 'cpu'、'cuda' 或 'auto'（自动检测）
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading YOLO model on {self.device}...")

        # 加载YOLO模型
        try:
            # 尝试ultralytics YOLO（同时支持v5和v8）
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.is_ultralytics = True
        except ImportError:
            try:
                # 回退到yolov5
                import yolov5
                self.model = yolov5.load(model_path, device=self.device)
                self.is_ultralytics = False
            except ImportError:
                raise ImportError("Please install YOLO: pip install ultralytics")

        # COCO类别名称（YOLO默认）
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

        # COCO中的人员类别ID
        self.person_class_id = 0

        # 用于跟踪
        self.detection_history = []
        self.track_id_counter = 0
        self.last_detection = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        在帧中检测人体

        Args:
            frame: 输入帧（BGR格式）

        Returns:
            包含人员信息的检测列表
        """
        detections = []

        try:
            # 运行推理
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
        """解析ultralytics YOLO结果"""
        detections = []

        if results:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取类别ID
                        class_id = int(box.cls[0])

                        # 仅检测人员
                        if class_id == self.person_class_id:
                            # 获取边界框
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
        """解析YOLOv5结果"""
        detections = []

        for result in results:
            # 获取检测
            if hasattr(result, 'xyxy'):
                boxes = result.xyxy[0]
                confidences = result.conf[0]
                class_ids = result.cls[0]

                # 转换为numpy
                boxes = boxes.cpu().numpy()
                confidences = confidences.cpu().numpy()
                class_ids = class_ids.cpu().numpy()

                for i in range(len(boxes)):
                    class_id = int(class_ids[i])

                    # 仅检测人员
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
        基于置信度分数获取最佳检测
        对于单人检测，我们使用置信度最高的检测

        Args:
            detections: 检测结果列表

        Returns:
            最佳检测或如果没有检测则为None
        """
        if not detections:
            return None

        # 按置信度排序
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        # 返回最佳检测
        return sorted_detections[0]

    def crop_person(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        从帧中裁剪人员区域

        Args:
            frame: 输入帧
            detection: 包含bbox的检测字典

        Returns:
            裁剪的人员区域
        """
        x1, y1, x2, y2 = detection['bbox']

        # 添加填充
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
        在帧上绘制检测框，支持多人检测

        Args:
            frame: 输入帧
            detections: 检测列表
            action_label: 动作分类标签（用于单人检测模式）
            confidence: 分类置信度（用于单人检测模式）
            colors: 每个检测的BGR颜色列表（需要时循环使用）

        Returns:
            绘制了检测的帧
        """
        output_frame = frame.copy()

        # 默认颜色（绿色、蓝色、红色、青色、品红色）
        if colors is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        if detections:
            # 如果提供了action_label，使用单人模式（向后兼容）
            if action_label is not None:
                best_detection = self.get_best_detection(detections)
                if best_detection:
                    self._draw_single_detection(output_frame, best_detection,
                                               action_label, confidence, colors[0])
            else:
                # 多人模式：绘制所有检测
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
        绘制单个检测框和标签

        Args:
            frame: 输入帧
            detection: 检测字典
            action_label: 动作分类标签
            confidence: 分类置信度
            color: 边界框的BGR颜色
            person_id: 用于标签的人员ID
        """
        x1, y1, x2, y2 = detection['bbox']

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 文本设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1
        line_height = 18

        # 准备标签
        labels = []
        if person_id is not None:
            labels.append(f"Person {person_id}: {detection['confidence']:.2f}")
        else:
            labels.append(f"Person: {detection['confidence']:.2f}")

        if action_label and confidence is not None:
            labels.append(f"Action: {action_label}")
            labels.append(f"Conf: {confidence:.2f}")

        # 绘制带背景的标签以提高可见性
        for i, label in enumerate(labels):
            y_pos = y1 - 10 - (len(labels) - i - 1) * line_height

            # 确保标签不会超出屏幕
            y_pos = max(y_pos, line_height)

            # 获取背景矩形的文本大小
            (text_width, text_height) = cv2.getTextSize(label, font, font_scale, text_thickness)[0]

            # 绘制背景矩形
            cv2.rectangle(frame,
                         (x1, y_pos - text_height - 2),
                         (x1 + text_width + 4, y_pos + 2),
                         color, -1)

            # 绘制文本
            cv2.putText(frame, label, (x1 + 2, y_pos),
                       font, font_scale, (255, 255, 255), text_thickness)
