"""
Real-time detection tab for the GUI
Provides controls and display for video/webcam detection
"""

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFileDialog, QGridLayout, QCheckBox,
                             QProgressBar, QTextEdit, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, QSize, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np

from .video_thread import VideoProcessingThread
from core.config import DetectionConfig


class LoadingOverlay(QWidget):
    """Loading animation overlay widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setFixedSize(100, 100)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 150);
                border-radius: 50px;
                border: 3px solid rgba(255, 255, 255, 100);
            }
        """)

        # Loading label
        self.label = QLabel("...", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                background: transparent;
            }
        """)
        self.label.setGeometry(10, 10, 80, 80)

    def show_loading(self, message="加载中..."):
        """Show loading overlay with message"""
        self.label.setText(message)
        self.setVisible(True)
        self.raise_()  # Bring to front

    def hide_loading(self):
        """Hide loading overlay"""
        self.setVisible(False)

    def update_message(self, message):
        """Update loading message"""
        self.label.setText(message)


class DetectionTab(QWidget):
    """
    Tab for real-time video detection
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = None
        self.video_thread = None
        self.loading_overlay = None
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel - Controls (fixed minimum width to ensure complete display)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable horizontal scroll
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        # Add margins to scrollbar area to prevent overlap with content
        controls_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(200, 200, 200, 50);
                width: 14px;
                margin: 0px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: rgba(163, 204, 218, 150);
                min-height: 30px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(163, 204, 218, 200);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        controls_widget = self.create_controls_panel()
        controls_widget.setMinimumWidth(360)  # Optimized width for better space balance
        controls_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(360)  # Ensure scroll area maintains minimum width

        main_layout.addWidget(controls_scroll, 0)  # No stretch factor - fixed width

        # Right panel - Video display (expands to fill remaining space)
        self.video_panel = self.create_video_panel()
        self.video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_panel, 1)  # Stretch factor 1 for video - takes remaining space

        self.setLayout(main_layout)

    def create_controls_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 20, 5)  # Increased right margin to avoid scrollbar overlap
        layout.setSpacing(8)
        panel.setLayout(layout)

        # Title with icon
        title = QLabel("🔍 检测控制")
        title.setFont(QFont("Hiragino Sans GB", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Mode selection
        mode_group = QGroupBox("📹 检测模式")
        mode_layout = QGridLayout()
        mode_layout.setHorizontalSpacing(10)
        mode_layout.setVerticalSpacing(6)
        mode_layout.setContentsMargins(10, 15, 12, 10)  # Increased right margin

        mode_label = QLabel("模式：")
        mode_label.setMinimumWidth(60)
        mode_layout.addWidget(mode_label, 0, 0)

        self.mode_combo = QComboBox()
        self.mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_combo.addItems(["摄像头", "视频文件"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo, 0, 1)

        camera_label = QLabel("摄像头索引：")
        camera_label.setMinimumWidth(70)
        mode_layout.addWidget(camera_label, 1, 0)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.setValue(0)
        self.camera_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mode_layout.addWidget(self.camera_spin, 1, 1)

        self.video_path_edit = QPushButton("选择视频文件...")
        self.video_path_edit.clicked.connect(self.select_video_file)
        self.video_path_edit.setEnabled(False)
        self.selected_video_path = None
        self.video_path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mode_layout.addWidget(self.video_path_edit, 2, 0, 1, 2)  # Span both columns

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Model selection
        model_group = QGroupBox("⚙️ 模型设置")
        model_layout = QGridLayout()
        model_layout.setHorizontalSpacing(10)
        model_layout.setVerticalSpacing(6)
        model_layout.setContentsMargins(10, 15, 12, 10)  # Increased right margin

        model_layout.addWidget(QLabel("检查点："), 0, 0)
        model_layout.itemAtPosition(0, 0).widget().setMinimumWidth(65)

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.checkpoint_combo.addItems(["UCF101 (默认)", "HMDB51 (默认)", "自定义模型"])  # Updated text
        self.checkpoint_combo.currentIndexChanged.connect(self.on_checkpoint_changed)
        model_layout.addWidget(self.checkpoint_combo, 0, 1)

        # Custom checkpoint selection button
        self.custom_checkpoint_button = QPushButton("选择模型文件...")
        self.custom_checkpoint_button.setEnabled(False)
        self.custom_checkpoint_button.clicked.connect(self.select_custom_checkpoint)
        self.custom_checkpoint_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.custom_checkpoint_button, 1, 0, 1, 2)
        self.custom_checkpoint_path = None

        model_layout.addWidget(QLabel("YOLO模型："), 2, 0)
        model_layout.itemAtPosition(2, 0).widget().setMinimumWidth(65)

        self.yolo_combo = QComboBox()
        self.yolo_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.yolo_combo.addItems(["yolov5s", "yolov8n", "yolov8s", "yolov8m"])
        model_layout.addWidget(self.yolo_combo, 2, 1)

        model_layout.addWidget(QLabel("置信度："), 3, 0)
        model_layout.itemAtPosition(3, 0).widget().setMinimumWidth(65)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.confidence_spin, 3, 1)

        model_layout.addWidget(QLabel("输出帧率："), 4, 0)
        model_layout.itemAtPosition(4, 0).widget().setMinimumWidth(65)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 60.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.fps_spin, 4, 1)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Output options
        output_group = QGroupBox("📤 输出选项")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(6)
        output_layout.setContentsMargins(10, 15, 12, 10)  # Increased right margin

        self.save_video_check = QCheckBox("保存输出视频")
        output_layout.addWidget(self.save_video_check)

        self.show_overlay_check = QCheckBox("显示检测叠加")
        self.show_overlay_check.setChecked(True)
        output_layout.addWidget(self.show_overlay_check)

        self.record_results_check = QCheckBox("记录检测结果")
        self.record_results_check.setChecked(False)
        self.record_results_check.setToolTip("启用后会收集检测结果并按动作类别归类（每类最多保存10帧）")
        output_layout.addWidget(self.record_results_check)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Control buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        self.start_button = QPushButton("▶️ 开始检测")
        self.start_button.setMinimumHeight(45)  # Increased height for better touch
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #5CB85C, stop:1 #4CAF50);
                transform: scale(1.02);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #3d8b40, stop:1 #388e3c);
                transform: scale(0.98);
            }
            QPushButton:disabled {
                background: #c0c0c0;
                color: #808080;
            }
        """)
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("⏹️ 停止检测")
        self.stop_button.setMinimumHeight(45)  # Increased height for better touch
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f44336, stop:1 #da190b);
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #ff6b6b, stop:1 #f44336);
                transform: scale(1.02);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #c0392b, stop:1 #a4190a);
                transform: scale(0.98);
            }
            QPushButton:disabled {
                background: #c0c0c0;
                color: #808080;
            }
        """)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Status display
        status_group = QGroupBox("📊 状态")
        status_layout = QGridLayout()
        status_layout.setHorizontalSpacing(10)
        status_layout.setVerticalSpacing(5)
        status_layout.setContentsMargins(10, 15, 12, 10)  # Increased right margin

        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label, 0, 0, 1, 2)

        self.fps_label = QLabel("帧率：0.0")
        status_layout.addWidget(self.fps_label, 1, 0)

        self.action_label = QLabel("动作：-")
        status_layout.addWidget(self.action_label, 1, 1)

        self.confidence_label = QLabel("置信度：0%")
        status_layout.addWidget(self.confidence_label, 2, 0, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bbb;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                min-height: 22px;
                font-size: 13px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.progress_bar, 3, 0, 1, 2)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Log output
        log_group = QGroupBox("📝 日志")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(6)
        log_layout.setContentsMargins(10, 15, 12, 10)  # Increased right margin

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)  # Optimized height
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Don't add stretch since we're using scroll area
        return panel

    def create_video_panel(self):
        """Create video display panel"""
        panel = QGroupBox("🎥 视频显示")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel.setMinimumHeight(400)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        panel.setLayout(layout)
        panel.setFont(QFont("Hiragino Sans GB", 16, QFont.Bold))

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setObjectName("video_label")
        self.video_label.setText("无视频")
        self.video_label.setFont(QFont("Arial", 18))  # Increased font size

        # Container for video label to center it
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addStretch()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        video_layout.addStretch()

        layout.addWidget(video_container)

        # Loading overlay (initially hidden)
        self.loading_overlay = LoadingOverlay(self.video_label)
        self.loading_overlay.hide()

        # Info labels
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)

        self.resolution_label = QLabel("分辨率：-")
        self.resolution_label.setStyleSheet("color: #666; font-weight: 500;")
        info_layout.addWidget(self.resolution_label)

        self.device_label = QLabel("设备：-")
        self.device_label.setStyleSheet("color: #666; font-weight: 500;")
        info_layout.addWidget(self.device_label)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        return panel

    def resizeEvent(self, event):
        """Handle window resize - update video display size for 16:9 ratio"""
        super().resizeEvent(event)

        # Get the video panel size
        panel_width = self.video_panel.width()
        panel_height = self.video_panel.height()

        # Calculate 16:9 video size (accounting for padding and info section)
        available_height = panel_height - 80  # Reserve space for info labels
        target_width = panel_width - 40  # Reserve space for padding

        # Calculate height based on 16:9 ratio
        target_height = int(target_width * 9 / 16)

        # If calculated height exceeds available space, recalculate based on available height
        if target_height > available_height:
            target_height = available_height
            target_width = int(target_height * 16 / 9)

        # Update video label size
        self.video_label.setFixedSize(target_width, target_height)

        event.accept()

    def select_video_file(self):
        """Select video file for processing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if file_path:
            self.selected_video_path = file_path
            self.video_path_edit.setText(os.path.basename(file_path))
            self.log(f"已选择视频：{file_path}")

    def on_checkpoint_changed(self, index):
        """Handle checkpoint selection change"""
        # Enable custom checkpoint button only when "自定义模型" is selected
        is_custom = (self.checkpoint_combo.currentText() == "自定义模型")
        self.custom_checkpoint_button.setEnabled(is_custom)

    def on_mode_changed(self, index):
        """Handle mode selection change"""
        # Enable video file button only when video mode is selected (index 1)
        self.video_path_edit.setEnabled(index == 1)

    def select_custom_checkpoint(self):
        """Select custom checkpoint file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择检查点文件",
            "",
            "检查点文件 (*.pth *.pt)"
        )
        if file_path:
            self.custom_checkpoint_path = file_path
            self.custom_checkpoint_button.setText(os.path.basename(file_path))
            self.log(f"已选择自定义检查点：{file_path}")

    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def start_detection(self):
        """Start detection"""
        try:
            from detection import DetectionPipeline
            from core.config import DetectionConfig

            # Show loading overlay
            if self.loading_overlay:
                self.loading_overlay.show_loading("初始化中...")

            # Get settings
            mode_text = self.mode_combo.currentText()
            mode = "webcam" if "摄像头" in mode_text else "video"
            camera_index = self.camera_spin.value()

            # Get checkpoint path
            checkpoint_text = self.checkpoint_combo.currentText()
            if "自定义模型" in checkpoint_text:
                if not self.custom_checkpoint_path:
                    self.log("错误：请先选择自定义检查点文件")
                    if self.loading_overlay:
                        self.loading_overlay.hide_loading()
                    return
                checkpoint = self.custom_checkpoint_path
            elif "UCF101" in checkpoint_text:
                checkpoint = DetectionConfig.DEFAULT_UCF101_CHECKPOINT
            else:  # HMDB51
                checkpoint = DetectionConfig.DEFAULT_HMDB51_CHECKPOINT

            # Verify checkpoint exists
            if not os.path.exists(checkpoint):
                self.log(f"错误：检查点文件不存在：{checkpoint}")
                if self.loading_overlay:
                    self.loading_overlay.hide_loading()
                return

            if mode == "video" and not self.selected_video_path:
                self.log("错误：请先选择视频文件")
                if self.loading_overlay:
                    self.loading_overlay.hide_loading()
                return

            # Get output path
            output_path = None
            if self.save_video_check.isChecked():
                if mode == "video":
                    base_name = os.path.splitext(os.path.basename(self.selected_video_path))[0]
                    output_path = f"outputs/videos/{base_name}_output.mp4"
                else:
                    output_path = f"outputs/videos/webcam_output.mp4"

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create pipeline
            self.log("初始化检测管道...")
            self.log(f"使用检查点：{checkpoint}")

            # Get YOLO model path from config
            yolo_model_key = self.yolo_combo.currentText()
            yolo_model_path = DetectionConfig.DEFAULT_YOLO_MODELS.get(
                yolo_model_key,
                yolo_model_key + '.pt'
            )

            self.pipeline = DetectionPipeline(
                checkpoint_path=checkpoint,
                yolo_model=yolo_model_path,
                output_path=output_path,
                fps=self.fps_spin.value(),
                show_display=False,  # We'll handle display ourselves
                save_video=self.save_video_check.isChecked(),
                enable_result_collection=self.record_results_check.isChecked()
            )

            # Create video thread
            self.video_thread = VideoProcessingThread(
                pipeline=self.pipeline,
                mode=mode,
                video_path=self.selected_video_path,
                camera_index=camera_index
            )

            # Connect signals
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.processing_finished.connect(self.on_processing_finished)
            self.video_thread.error_occurred.connect(self.on_error)

            # Connect result collection signal
            if self.record_results_check.isChecked():
                self.video_thread.result_ready.connect(self.on_result_ready)
                # Also set video source for result collector
                if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
                    source_desc = f"{mode}" + (f": {os.path.basename(self.selected_video_path)}" if mode == "video" else "")
                    self.pipeline.result_collector.set_video_source(source_desc)

            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.mode_combo.setEnabled(False)
            self.camera_spin.setEnabled(False)
            self.video_path_edit.setEnabled(False)
            self.checkpoint_combo.setEnabled(False)
            self.yolo_combo.setEnabled(False)

            # Start thread
            self.video_thread.start()
            self.log("检测已开始")
            if self.loading_overlay:
                self.loading_overlay.hide_loading()

            if mode == "video":
                self.progress_bar.setVisible(True)

        except Exception as e:
            self.log(f"启动检测时出错：{str(e)}")
            if self.loading_overlay:
                self.loading_overlay.hide_loading()
            self.reset_ui_state()

    def stop_detection(self):
        """Stop detection"""
        if self.video_thread and self.video_thread.isRunning():
            self.log("停止检测...")
            self.video_thread.stop()
        self.reset_ui_state()

    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.camera_spin.setEnabled(True)
        # Video button state is now handled by on_mode_changed signal
        self.on_mode_changed(self.mode_combo.currentIndex())
        self.checkpoint_combo.setEnabled(True)
        self.yolo_combo.setEnabled(True)
        self.custom_checkpoint_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
        self.video_label.setText("无视频")

    def update_frame(self, frame, info):
        """Update video display"""
        # Hide loading overlay when first frame arrives
        if self.loading_overlay and self.loading_overlay.isVisible():
            self.loading_overlay.hide_loading()

        # Convert BGR to RGB for display
        if frame is not None and len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create QImage
            h, w, c = rgb_frame.shape
            bytes_per_line = c * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            # Update info labels
            self.status_label.setText("处理中...")
            self.fps_label.setText(f"帧率：{info.get('fps', 0):.1f}")
            self.action_label.setText(f"动作：{info.get('action', '-')}")
            self.confidence_label.setText(f"置信度：{info.get('confidence', 0)*100:.1f}%")

            # Update resolution and device info
            self.resolution_label.setText(f"分辨率：{info.get('resolution', f'{w}x{h}')}")
            self.device_label.setText(f"设备：{info.get('device', '-')}")

            # Update progress bar (for video file mode)
            if 'progress' in info:
                self.progress_bar.setValue(int(info['progress'] * 100))

    def on_processing_finished(self, stats):
        """Handle processing completion"""
        self.log("检测完成！")
        self.log(f"处理帧数：{stats.get('frames_processed', 0)}")
        self.log(f"平均帧率：{stats.get('average_fps', 0):.1f}")

        # Emit final results if collection was enabled
        if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
            final_stats = self.pipeline.result_collector.get_statistics()
            self.on_result_ready(final_stats)

        self.reset_ui_state()

    def on_result_ready(self, stats):
        """Handle detection results update"""
        # Update results tab if it exists
        parent = self.parent()
        while parent and not hasattr(parent, 'results_tab'):
            parent = parent.parent()

        if parent and hasattr(parent, 'results_tab'):
            parent.results_tab.update_results(stats)

    def on_error(self, error_message):
        """Handle error"""
        self.log(f"错误：{error_message}")
        self.status_label.setText("错误")
        self.reset_ui_state()

    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()
