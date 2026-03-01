"""
GUI的实时检测标签页
为视频/摄像头检测提供控制和显示
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
    """加载动画叠加小部件"""

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

        # 加载标签
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
        """显示带消息的加载叠加"""
        self.label.setText(message)
        self.setVisible(True)
        self.raise_()  # Bring to front

    def hide_loading(self):
        """隐藏加载叠加"""
        self.setVisible(False)

    def update_message(self, message):
        """更新加载消息"""
        self.label.setText(message)


class DetectionTab(QWidget):
    """
    实时视频检测标签页
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = None
        self.video_thread = None
        self.loading_overlay = None
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧面板 - 控制器（固定最小宽度以确保完整显示）
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        # 向滚动条区域添加边距以防止与内容重叠
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
        controls_widget.setMinimumWidth(360)  # 为更好的空间平衡优化宽度
        controls_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(360)  # 确保滚动区域保持最小宽度

        main_layout.addWidget(controls_scroll, 0)  # 无拉伸因子 - 固定宽度

        # 右侧面板 - 视频显示（扩展以填充剩余空间）
        self.video_panel = self.create_video_panel()
        self.video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_panel, 1)  # 视频的拉伸因子为1 - 占据剩余空间

        self.setLayout(main_layout)

    def create_controls_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 20, 5)  # 增加右边距以避免滚动条重叠
        layout.setSpacing(8)
        panel.setLayout(layout)

        # 带图标的标题
        title = QLabel("🔍 检测控制")
        title.setFont(QFont("Hiragino Sans GB", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 模式选择
        mode_group = QGroupBox("📹 检测模式")
        mode_layout = QGridLayout()
        mode_layout.setHorizontalSpacing(10)
        mode_layout.setVerticalSpacing(6)
        mode_layout.setContentsMargins(10, 15, 12, 10)  # 增加右边距

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
        mode_layout.addWidget(self.video_path_edit, 2, 0, 1, 2)  # 跨越两列

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 模型选择
        model_group = QGroupBox("⚙️ 模型设置")
        model_layout = QGridLayout()
        model_layout.setHorizontalSpacing(10)
        model_layout.setVerticalSpacing(6)
        model_layout.setContentsMargins(10, 15, 12, 10)  # 增加右边距

        model_layout.addWidget(QLabel("检查点："), 0, 0)
        model_layout.itemAtPosition(0, 0).widget().setMinimumWidth(65)

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.checkpoint_combo.addItems(["UCF101 (默认)", "HMDB51 (默认)", "自定义模型"])  # 更新文本
        self.checkpoint_combo.currentIndexChanged.connect(self.on_checkpoint_changed)
        model_layout.addWidget(self.checkpoint_combo, 0, 1)

        # Custom checkpoint selection button
        self.custom_checkpoint_button = QPushButton("选择模型文件...")
        self.custom_checkpoint_button.setEnabled(False)
        self.custom_checkpoint_button.clicked.connect(self.select_custom_checkpoint)
        self.custom_checkpoint_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.custom_checkpoint_button, 1, 0, 1, 2)
        self.custom_checkpoint_path = None

        # Label file selection button
        self.label_file_button = QPushButton("选择标签文件...")
        self.label_file_button.setEnabled(False)
        self.label_file_button.setToolTip("为自定义模型指定类别标签文件（classInd.txt格式）")
        self.label_file_button.clicked.connect(self.select_label_file)
        self.label_file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.label_file_button, 2, 0, 1, 2)

        self.label_file_path = None
        self.label_file_label = QLabel("")  # 显示已选择的文件名
        self.label_file_label.setStyleSheet("color: #666; font-size: 11px;")
        model_layout.addWidget(self.label_file_label, 3, 0, 1, 2)

        model_layout.addWidget(QLabel("YOLO模型："), 4, 0)
        model_layout.itemAtPosition(4, 0).widget().setMinimumWidth(65)

        self.yolo_combo = QComboBox()
        self.yolo_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.yolo_combo.addItems(["yolov5s", "yolov8n", "yolov8s", "yolov8m"])
        model_layout.addWidget(self.yolo_combo, 4, 1)

        model_layout.addWidget(QLabel("置信度："), 5, 0)
        model_layout.itemAtPosition(5, 0).widget().setMinimumWidth(65)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.confidence_spin, 5, 1)

        model_layout.addWidget(QLabel("输出帧率："), 6, 0)
        model_layout.itemAtPosition(6, 0).widget().setMinimumWidth(65)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 60.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.fps_spin, 6, 1)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 输出选项
        output_group = QGroupBox("📤 输出选项")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(6)
        output_layout.setContentsMargins(10, 15, 12, 10)  # 增加右边距

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

        # 控制按钮
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        self.start_button = QPushButton("▶️ 开始检测")
        self.start_button.setMinimumHeight(45)  # 增加高度以改善触摸体验
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
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #3d8b40, stop:1 #388e3c);
            }
            QPushButton:disabled {
                background: #c0c0c0;
                color: #808080;
            }
        """)
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("⏹️ 停止检测")
        self.stop_button.setMinimumHeight(45)  # 增加高度以改善触摸体验
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
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #c0392b, stop:1 #a4190a);
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

        # 状态显示
        status_group = QGroupBox("📊 状态")
        status_layout = QGridLayout()
        status_layout.setHorizontalSpacing(10)
        status_layout.setVerticalSpacing(5)
        status_layout.setContentsMargins(10, 15, 12, 10)  # 增加右边距

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

        # 日志输出
        log_group = QGroupBox("📝 日志")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(6)
        log_layout.setContentsMargins(10, 15, 12, 10)  # 增加右边距

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)  # 优化高度
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

        # 不要添加弹性空间，因为我们使用滚动区域
        return panel

    def create_video_panel(self):
        """创建视频显示面板"""
        panel = QGroupBox("🎥 视频显示")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel.setMinimumHeight(400)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        panel.setLayout(layout)
        panel.setFont(QFont("Hiragino Sans GB", 16, QFont.Bold))

        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setObjectName("video_label")
        self.video_label.setText("无视频")
        self.video_label.setFont(QFont("Arial", 18))  # 增加字体大小

        # 视频标签的容器以将其居中
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addStretch()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        video_layout.addStretch()

        layout.addWidget(video_container)

        # 加载叠加（最初隐藏）
        self.loading_overlay = LoadingOverlay(self.video_label)
        self.loading_overlay.hide()

        # 信息标签
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
        """处理窗口大小调整 - 更新16:9比例的视频显示大小"""
        super().resizeEvent(event)

        # 获取视频面板大小
        panel_width = self.video_panel.width()
        panel_height = self.video_panel.height()

        # 计算16:9视频大小（考虑填充和信息部分）
        available_height = panel_height - 80  # 为信息标签预留空间
        target_width = panel_width - 40  # 为填充预留空间

        # 基于16:9比例计算高度
        target_height = int(target_width * 9 / 16)

        # 如果计算的高度超出可用空间，基于可用高度重新计算
        if target_height > available_height:
            target_height = available_height
            target_width = int(target_height * 16 / 9)

        # 更新视频标签大小
        self.video_label.setFixedSize(target_width, target_height)

        event.accept()

    def select_video_file(self):
        """选择要处理的视频文件"""
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
        """处理检查点选择更改"""
        # 仅在选择"自定义模型"时启用自定义检查点按钮和标签文件按钮
        is_custom = (self.checkpoint_combo.currentText() == "自定义模型")
        self.custom_checkpoint_button.setEnabled(is_custom)
        self.label_file_button.setEnabled(is_custom)
        if not is_custom:
            # 清除标签文件选择
            self.label_file_path = None
            self.label_file_label.setText("")

    def on_mode_changed(self, index):
        """处理模式选择更改"""
        # 仅在选择视频模式时启用视频文件按钮（索引1）
        self.video_path_edit.setEnabled(index == 1)

    def select_custom_checkpoint(self):
        """选择自定义检查点文件"""
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

    def select_label_file(self):
        """选择标签文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标签文件",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            self.label_file_path = file_path
            # 显示文件名（不带路径）
            file_name = os.path.basename(file_path)
            self.label_file_label.setText(f"已选择: {file_name}")
            self.log(f"已选择标签文件：{file_path}")

    def log(self, message):
        """向日志添加消息"""
        self.log_text.append(message)
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def start_detection(self):
        """开始检测"""
        try:
            from detection import DetectionPipeline
            from core.config import DetectionConfig

            # 显示加载叠加
            if self.loading_overlay:
                self.loading_overlay.show_loading("初始化中...")

            # 获取设置
            mode_text = self.mode_combo.currentText()
            mode = "webcam" if "摄像头" in mode_text else "video"
            camera_index = self.camera_spin.value()

            # 获取检查点路径
            checkpoint_text = self.checkpoint_combo.currentText()
            if "自定义模型" in checkpoint_text:
                if not self.custom_checkpoint_path:
                    self.log("错误：请先选择自定义检查点文件")
                    if self.loading_overlay:
                        self.loading_overlay.hide_loading()
                    return
                checkpoint = self.custom_checkpoint_path
            elif "UCF101" in checkpoint_text:
                # 动态获取最新的默认模型路径
                from core.config import get_default_model_path
                checkpoint = get_default_model_path('ucf101')
            else:  # HMDB51
                # 动态获取最新的默认模型路径
                from core.config import get_default_model_path
                checkpoint = get_default_model_path('hmdb51')

            # 验证检查点是否存在
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

            # 获取输出路径
            output_path = None
            if self.save_video_check.isChecked():
                if mode == "video":
                    base_name = os.path.splitext(os.path.basename(self.selected_video_path))[0]
                    output_path = f"outputs/videos/{base_name}_output.mp4"
                else:
                    output_path = f"outputs/videos/webcam_output.mp4"

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 创建流水线
            self.log("初始化检测管道...")
            self.log(f"使用检查点：{checkpoint}")

            # 获取标签文件路径
            label_file = None
            if "自定义模型" in checkpoint_text and self.label_file_path:
                label_file = self.label_file_path

            # 从配置获取YOLO模型路径
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
                enable_result_collection=self.record_results_check.isChecked(),
                label_file=label_file
            )

            # 创建视频线程
            self.video_thread = VideoProcessingThread(
                pipeline=self.pipeline,
                mode=mode,
                video_path=self.selected_video_path,
                camera_index=camera_index
            )

            # 连接信号
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.processing_finished.connect(self.on_processing_finished)
            self.video_thread.error_occurred.connect(self.on_error)

            # 连接结果收集信号
            if self.record_results_check.isChecked():
                self.video_thread.result_ready.connect(self.on_result_ready)
                # 同时为结果收集器设置视频源
                if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
                    source_desc = f"{mode}" + (f": {os.path.basename(self.selected_video_path)}" if mode == "video" else "")
                    self.pipeline.result_collector.set_video_source(source_desc)

            # 更新UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.mode_combo.setEnabled(False)
            self.camera_spin.setEnabled(False)
            self.video_path_edit.setEnabled(False)
            self.checkpoint_combo.setEnabled(False)
            self.yolo_combo.setEnabled(False)

            # 启动线程
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
        """停止检测"""
        if self.video_thread and self.video_thread.isRunning():
            self.log("停止检测...")
            self.video_thread.stop()
        self.reset_ui_state()

    def reset_ui_state(self):
        """将UI重置为初始状态"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.camera_spin.setEnabled(True)
        # 视频按钮状态现在由on_mode_changed信号处理
        self.on_mode_changed(self.mode_combo.currentIndex())
        self.checkpoint_combo.setEnabled(True)
        self.yolo_combo.setEnabled(True)
        # 检查点按钮状态由on_checkpoint_changed处理
        self.on_checkpoint_changed(self.checkpoint_combo.currentIndex())
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
        self.video_label.setText("无视频")

    def update_frame(self, frame, info):
        """更新视频显示"""
        # 第一帧到达时隐藏加载叠加
        if self.loading_overlay and self.loading_overlay.isVisible():
            self.loading_overlay.hide_loading()

        # 将BGR转换为RGB以进行显示
        if frame is not None and len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建QImage
            h, w, c = rgb_frame.shape
            bytes_per_line = c * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 缩放以适应标签，同时保持纵横比
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            # 更新信息标签
            self.status_label.setText("处理中...")
            self.fps_label.setText(f"帧率：{info.get('fps', 0):.1f}")
            self.action_label.setText(f"动作：{info.get('action', '-')}")
            self.confidence_label.setText(f"置信度：{info.get('confidence', 0)*100:.1f}%")

            # 更新分辨率和设备信息
            self.resolution_label.setText(f"分辨率：{info.get('resolution', f'{w}x{h}')}")
            self.device_label.setText(f"设备：{info.get('device', '-')}")

            # 更新进度条（用于视频文件模式）
            if 'progress' in info:
                self.progress_bar.setValue(int(info['progress'] * 100))

    def on_processing_finished(self, stats):
        """处理处理完成"""
        self.log("检测完成！")
        self.log(f"处理帧数：{stats.get('frames_processed', 0)}")
        self.log(f"平均帧率：{stats.get('average_fps', 0):.1f}")

        # 如果启用了收集，则发送最终结果
        if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
            final_stats = self.pipeline.result_collector.get_statistics()
            self.on_result_ready(final_stats)

        self.reset_ui_state()

    def on_result_ready(self, stats):
        """处理检测结果更新"""
        # 更新结果选项卡（如果存在）
        parent = self.parent()
        while parent and not hasattr(parent, 'results_tab'):
            parent = parent.parent()

        if parent and hasattr(parent, 'results_tab'):
            parent.results_tab.update_results(stats)

    def on_error(self, error_message):
        """处理错误"""
        self.log(f"错误：{error_message}")
        self.status_label.setText("错误")
        self.reset_ui_state()

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()
