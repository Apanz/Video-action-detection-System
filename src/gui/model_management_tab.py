"""
Model Management Tab for GUI
Manages model files: list, upload, delete, view details
"""

import os
import shutil
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QListWidget, QListWidgetItem,
                             QFileDialog, QMessageBox, QGridLayout, QTextEdit,
                             QSplitter, QCheckBox, QTabWidget)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor
from pathlib import Path

from detection.model_metadata import ModelMetadata
from .training_curves_widget import TrainingCurvesWidget
from .tensorboard_reader import check_tensorboard_available, get_training_curves, validate_log_directory


class ModelScanThread(QThread):
    """Background thread for scanning model files"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, models_dir: str):
        super().__init__()
        self.models_dir = models_dir

    def run(self):
        """Run model scan in background thread"""
        try:
            categories = ModelMetadata.scan_models_directory(self.models_dir)
            self.finished.emit(categories)
        except Exception as e:
            self.error.emit(str(e))


class ModelManagementTab(QWidget):
    """
    Tab for managing detection models
    Allows viewing, uploading, deleting, and configuring models
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.models_dir = Path("outputs/models")
        self.current_models = {}
        self.current_model_metadata = None
        self.init_ui()
        self.refresh_model_list()

    def init_ui(self):
        """Initialize user interface"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 8, 12, 10)  # MODIFIED: Adjusted top margin to 8px for better spacing
        main_layout.setSpacing(10)

        # Title
        title = QLabel("⚙️ 模型管理 Model Management")
        title.setFont(QFont("Hiragino Sans GB", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        # MODIFIED: Set fixed height to reduce vertical space (16px font + ~10px padding = ~26px minimum, 32px for comfort)
        title.setMinimumHeight(32)
        title.setMaximumHeight(32)
        main_layout.addWidget(title)
        # MODIFIED: Use addSpacing for precise control (increased to 6px for better spacing)
        main_layout.addSpacing(6)

        # Splitter for list and details
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Model list (simplified, more focused)
        left_panel = QWidget()
        left_panel.setMaximumWidth(500)  # Increased left panel width for better list display
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)  # Reduced spacing for more compact layout

        # List header with buttons
        list_header = QHBoxLayout()
        list_header.setSpacing(8)

        list_title = QLabel("模型列表 Models")
        list_title.setFont(QFont("Hiragino Sans GB", 11, QFont.Bold))  # Reduced font size to match results tab
        list_header.addWidget(list_title)

        list_header.addStretch()

        # Define secondary button style (refresh, upload, set default)
        secondary_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #A8D4B8, stop:1 #95C5AC);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #B8DFC8, stop:1 #A5D5BC);
            }
            QPushButton:disabled {
                background: #c0c0c0;
                color: #808080;
            }
        """

        # Define danger button style (delete)
        danger_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f44336, stop:1 #da190b);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6b6b, stop:1 #f44336);
            }
            QPushButton:disabled {
                background: #c0c0c0;
                color: #808080;
            }
        """

        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_model_list)
        self.refresh_button.setMinimumWidth(70)
        self.refresh_button.setStyleSheet(secondary_button_style)
        list_header.addWidget(self.refresh_button)

        self.upload_button = QPushButton("上传")
        self.upload_button.clicked.connect(self.upload_model)
        self.upload_button.setMinimumWidth(70)
        self.upload_button.setStyleSheet(secondary_button_style)
        list_header.addWidget(self.upload_button)

        left_layout.addLayout(list_header)

        # Model list widget
        self.model_list_widget = QListWidget()
        self.model_list_widget.setIconSize(QSize(16, 16))
        self.model_list_widget.itemClicked.connect(self.on_model_selected)
        self.model_list_widget.setStyleSheet("""
            QListWidget {
                border: 2px solid rgba(189, 227, 195, 150);
                border-radius: 8px;
                background: rgba(255, 255, 255, 230);
                padding: 6px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 4px;
                margin-bottom: 1px;
            }
            QListWidget::item:selected {
                background: rgba(163, 204, 218, 120);
                color: #1E3A8A;
            }
            QListWidget::item:hover {
                background: rgba(163, 204, 218, 60);
            }
        """)
        left_layout.addWidget(self.model_list_widget)

        # Action buttons for selected model - full width layout
        action_layout = QHBoxLayout()
        action_layout.setSpacing(6)

        self.delete_button = QPushButton("删除 Delete")
        self.delete_button.setEnabled(False)
        self.delete_button.setMinimumHeight(36)
        self.delete_button.clicked.connect(self.delete_model)
        self.delete_button.setStyleSheet(danger_button_style)
        action_layout.addWidget(self.delete_button)

        self.set_default_button = QPushButton("设为默认 Set Default")
        self.set_default_button.setEnabled(False)
        self.set_default_button.setMinimumHeight(36)
        self.set_default_button.clicked.connect(self.set_as_default)
        self.set_default_button.setStyleSheet(secondary_button_style)
        action_layout.addWidget(self.set_default_button)

        left_layout.addLayout(action_layout)

        # Load model button - moved to left panel for better accessibility
        self.load_model_button = QPushButton("加载此模型 Load Model")
        self.load_model_button.setEnabled(False)
        self.load_model_button.clicked.connect(self.load_model_for_detection)
        self.load_model_button.setMinimumHeight(38)
        self.load_model_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CB85C, stop:1 #4CAF50);
            }
            QPushButton:disabled {
                background: #c0c0c0;
                color: #808080;
            }
        """)
        left_layout.addWidget(self.load_model_button)

        # Model details - compact view
        details_title = QLabel("模型详情 Details")
        details_title.setFont(QFont("Hiragino Sans GB", 11, QFont.Bold))
        left_layout.addWidget(details_title)

        # Compact details view - reduced height
        details_group = QGroupBox()
        details_group.setMaximumHeight(280)  # Limit height to save space
        details_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid rgba(189, 227, 195, 150);
                border-radius: 8px;
                background: rgba(255, 255, 255, 220);
                margin-top: 4px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        details_layout = QGridLayout()
        details_layout.setSpacing(4)
        details_layout.setContentsMargins(6, 6, 6, 6)

        # Create detail labels - compact format (2 columns)
        self.filename_label = QLabel("文件: -")
        self.path_label = QLabel("路径: -")
        self.size_label = QLabel("大小: -")
        self.modified_label = QLabel("修改: -")

        # Model architecture details - compact (2 columns)
        self.backbone_label = QLabel("骨干: -")
        self.dataset_label = QLabel("数据集: -")
        self.segments_label = QLabel("段数: -")
        self.frames_label = QLabel("每段帧数: -")
        self.total_frames_label = QLabel("总帧数: -")

        # Training info - compact (2 columns)
        self.epochs_label = QLabel("轮数: -")
        self.accuracy_label = QLabel("精度: -")

        # Status - full width
        self.status_label = QLabel("状态: -")
        self.status_label.setFont(QFont("Hiragino Sans GB", 9, QFont.Bold))
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 4px 8px;
                border-radius: 4px;
                background: rgba(163, 204, 218, 100);
            }
        """)

        # Add labels to layout in 2 columns for compact display
        labels = [self.filename_label, self.path_label, self.size_label,
                 self.modified_label, self.backbone_label, self.dataset_label,
                 self.segments_label, self.frames_label, self.total_frames_label,
                 self.epochs_label, self.accuracy_label]

        for idx, label in enumerate(labels):
            row = idx // 2
            col = idx % 2
            details_layout.addWidget(label, row, col)

        # Status label on its own row
        details_layout.addWidget(self.status_label, 6, 0, 1, 2)

        details_group.setLayout(details_layout)
        left_layout.addWidget(details_group)

        splitter.addWidget(left_panel)

        # Right panel - Description and curves tabbed view (expanded)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)  # Reduced spacing for more compact layout

        # Description and curves tabbed view
        tab_label = QLabel("详情与曲线 Details & Curves:")
        tab_label.setFont(QFont("Hiragino Sans GB", 11, QFont.Bold))  # Reduced font size
        right_layout.addWidget(tab_label)

        self.details_tab_widget = QTabWidget()

        # Description tab
        desc_tab = QWidget()
        desc_layout = QVBoxLayout(desc_tab)
        desc_layout.setContentsMargins(5, 5, 5, 5)
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        desc_layout.addWidget(self.description_text)
        self.details_tab_widget.addTab(desc_tab, "描述")

        # Training curves tab
        curves_tab = QWidget()
        curves_layout = QVBoxLayout(curves_tab)
        curves_layout.setContentsMargins(5, 5, 5, 5)

        # Create curves widget
        self.training_curves = TrainingCurvesWidget()
        curves_layout.addWidget(self.training_curves)

        # Add button to select log directory - full width
        self.select_log_button = QPushButton("选择TensorBoard日志 Select Log")
        self.select_log_button.setMinimumHeight(36)
        self.select_log_button.clicked.connect(self.select_tensorboard_log)
        self.select_log_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5dade2, stop:1 #3498db);
            }
        """)
        curves_layout.addWidget(self.select_log_button)

        self.details_tab_widget.addTab(curves_tab, "训练曲线 Curves")
        right_layout.addWidget(self.details_tab_widget)

        splitter.addWidget(right_panel)

        # Set splitter sizes - give more space to details & curves
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([380, 800])  # Initial sizes: left 380px, right 800px

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def refresh_model_list(self):
        """Refresh the model list from disk (runs in background thread)"""
        # Clear current list
        self.model_list_widget.clear()
        self.current_models = {}

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Update status to show loading
        self.status_label.setText("状态 Status: 扫描中... Scanning...")

        # Start background scan thread
        self.scan_thread = ModelScanThread(str(self.models_dir))
        self.scan_thread.finished.connect(self._on_scan_complete)
        self.scan_thread.error.connect(self._on_scan_error)
        self.scan_thread.start()

    def _on_scan_complete(self, categories: dict):
        """Handle completion of model scan"""
        # Add models to list
        for category, models in categories.items():
            if not models:
                continue

            # Add category header with distinct background colors
            category_item = QListWidgetItem(f"[{category.upper()}]")
            category_item.setFlags(Qt.NoItemFlags)
            font = category_item.font()
            font.setBold(True)
            category_item.setFont(font)

            # Add color-coded backgrounds for better visual separation
            if category == 'ucf101':
                category_item.setBackground(QColor(220, 235, 220))  # Light green
            elif category == 'custom':
                category_item.setBackground(QColor(220, 230, 250))  # Light blue
            else:
                category_item.setBackground(Qt.lightGray)

            self.model_list_widget.addItem(category_item)

            # Add models in this category
            for model_metadata in models:
                filename = model_metadata['filename']
                display_name = f"  {filename}"

                # Add file size and description
                size_mb = model_metadata.get('file_size_mb', 0)
                display_name += f" ({size_mb:.1f} MB)"

                # Add description
                description = ModelMetadata.get_model_description(model_metadata)
                display_name += f"\n  {description}"

                # Add default marker if applicable
                if category == 'ucf101' and filename == 'ucf101_best.pth':
                    display_name += " [默认 Default]"

                # Create list item
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, model_metadata)
                self.model_list_widget.addItem(item)

                # Store in dictionary
                self.current_models[filename] = model_metadata

        # Update status
        total_models = sum(len(models) for models in categories.values())
        self.status_label.setText(f"状态 Status: 找到 {total_models} 个模型 Found {total_models} models")

    def _on_scan_error(self, error_msg: str):
        """Handle error during model scan"""
        self.model_list_widget.clear()
        self.status_label.setText(f"状态 Status: 扫描失败 Scan failed: {error_msg}")
        QMessageBox.critical(
            self,
            "错误 Error",
            f"扫描模型目录失败 Failed to scan models directory:\n{error_msg}"
        )

    def on_model_selected(self, item: QListWidgetItem):
        """
        Handle model selection from list

        Args:
            item: Selected list item
        """
        try:
            metadata = item.data(Qt.UserRole)
            if not metadata:
                return

            self.current_model_metadata = metadata

            # Enable action buttons
            self.delete_button.setEnabled(True)
            self.set_default_button.setEnabled(True)
            self.load_model_button.setEnabled(True)

            # Update details (compact format)
            self.filename_label.setText(f"文件: {metadata.get('filename', 'Unknown')}")

            # Fold path to show only parent directory and filename
            full_path = metadata.get('path', 'Unknown')
            if full_path != 'Unknown':
                path_obj = Path(full_path)
                # Only show parent directory and filename
                # e.g., E:\...\models\ucf101\ucf101_best.pth -> ucf101\ucf101_best.pth
                if path_obj.parent and path_obj.parent.name:
                    short_path = f"{path_obj.parent.name}/{path_obj.name}"
                else:
                    short_path = path_obj.name
                self.path_label.setText(f"路径: {short_path}")
            else:
                self.path_label.setText("路径: -")

            file_size_mb = metadata.get('file_size_mb', 0)
            self.size_label.setText(f"大小: {file_size_mb:.2f} MB")

            # Get modification time
            file_path = Path(metadata.get('path', ''))
            if file_path.exists():
                import datetime
                mtime = file_path.stat().st_mtime
                modified_time = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                self.modified_label.setText(f"修改: {modified_time}")
            else:
                self.modified_label.setText("修改: -")

            # Architecture details (compact)
            backbone = metadata.get('backbone', 'Unknown')
            if isinstance(backbone, str):
                self.backbone_label.setText(f"骨干: {backbone.upper()}")
            else:
                self.backbone_label.setText(f"骨干: {str(backbone)}")

            dataset = metadata.get('dataset', 'Unknown')
            if isinstance(dataset, str):
                self.dataset_label.setText(f"数据集: {dataset.upper()}")
            else:
                self.dataset_label.setText(f"数据集: {str(dataset)}")

            num_segments = metadata.get('num_segments', 'Unknown')
            self.segments_label.setText(f"段数: {num_segments}")

            frames_per_segment = metadata.get('frames_per_segment', 'Unknown')
            self.frames_label.setText(f"每段帧数: {frames_per_segment}")

            # Calculate total frames if both are known integers
            try:
                if (isinstance(num_segments, int) and isinstance(frames_per_segment, int)):
                    total = num_segments * frames_per_segment
                    self.total_frames_label.setText(f"总帧数: {total}")
                else:
                    self.total_frames_label.setText("总帧数: Unknown")
            except Exception:
                self.total_frames_label.setText("总帧数: Unknown")

            # Training info (compact)
            if 'trained_epochs' in metadata:
                self.epochs_label.setText(f"轮数: {metadata['trained_epochs']}")
            else:
                self.epochs_label.setText("轮数: -")

            if 'best_accuracy' in metadata:
                acc = metadata['best_accuracy']
                self.accuracy_label.setText(f"精度: {acc:.2f}%")
            else:
                self.accuracy_label.setText("精度: -")

            # Status
            is_valid = metadata.get('is_valid', False)
            if is_valid:
                self.status_label.setText("状态: ✅ 有效")
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText(f"状态: ❌ 无效 - {metadata.get('error', 'Unknown')}")
                self.status_label.setStyleSheet("color: red;")

            # Description
            description = ModelMetadata.get_model_description(metadata)
            self.description_text.setText(description)

        except Exception as e:
            QMessageBox.critical(
                self,
                "错误 Error",
                f"加载模型详情失败 Failed to load model details:\n{str(e)}"
            )

    def upload_model(self):
        """Upload a new model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件 Select Model File",
            "",
            "Model Files (*.pth *.pt);;All Files (*)"
        )

        if not file_path:
            return

        # Ask for category
        from PyQt5.QtWidgets import QDialog, QComboBox, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("选择模型类别 Select Model Category")

        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("将此模型添加到哪个类别？Which category to add this model to?"))

        category_combo = QComboBox()
        category_combo.addItems(["UCF101", "自定义 Custom"])
        layout.addWidget(category_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        category_idx = category_combo.currentIndex()
        if category_idx == 0:
            category = "ucf101"
        else:
            category = "custom"

        # Copy model file
        try:
            # Create category directory
            category_dir = self.models_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Copy file
            filename = Path(file_path).name
            dest_path = category_dir / filename

            # Handle duplicate filenames
            counter = 1
            while dest_path.exists():
                name_without_ext = Path(file_path).stem
                ext = Path(file_path).suffix
                dest_path = category_dir / f"{name_without_ext}_{counter}{ext}"
                counter += 1

            shutil.copy2(file_path, dest_path)

            QMessageBox.information(
                self,
                "成功 Success",
                f"模型已上传到 Model uploaded to:\n{dest_path}"
            )

            # Refresh list
            self.refresh_model_list()

        except Exception as e:
            QMessageBox.critical(
                self,
                "错误 Error",
                f"上传模型失败 Failed to upload model:\n{str(e)}"
            )

    def delete_model(self):
        """Delete the selected model"""
        if not self.current_model_metadata:
            return

        metadata = self.current_model_metadata
        file_path = metadata.get('path', '')

        # Check if it's a default model
        filename = metadata.get('filename', '')
        if filename == 'ucf101_best.pth':
            reply = QMessageBox.warning(
                self,
                "警告 Warning",
                f"{filename} 是默认模型，确定要删除吗？\n"
                f"{filename} is a default model, are you sure you want to delete it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        else:
            reply = QMessageBox.question(
                self,
                "确认 Confirm",
                f"确定要删除此模型吗？\n"
                f"Are you sure you want to delete this model?\n\n{filename}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            if os.path.exists(file_path):
                os.remove(file_path)

                QMessageBox.information(
                    self,
                    "成功 Success",
                    f"模型已删除 Model deleted:\n{filename}"
                )

                # Clear selection
                self.current_model_metadata = None
                self.delete_button.setEnabled(False)
                self.set_default_button.setEnabled(False)
                self.load_model_button.setEnabled(False)

                # Clear details
                self.clear_details()

                # Refresh list
                self.refresh_model_list()

        except Exception as e:
            QMessageBox.critical(
                self,
                "错误 Error",
                f"删除模型失败 Failed to delete model:\n{str(e)}"
            )

    def set_as_default(self):
        """Set the selected model as default for its dataset"""
        if not self.current_model_metadata:
            return

        metadata = self.current_model_metadata
        category = metadata.get('category', 'custom')

        # Can only set default for ucf101
        if category != 'ucf101':
            QMessageBox.warning(
                self,
                "警告 Warning",
                "无法为自定义模型设置默认\nCannot set custom models as default"
            )
            return

        # Determine default filename
        default_name = 'ucf101_best.pth'

        try:
            # Copy current model to default name
            current_path = Path(metadata['path'])
            category_dir = current_path.parent
            default_path = category_dir / default_name

            # Remove old default if it exists and is different
            if default_path.exists() and default_path != current_path:
                os.remove(default_path)

            # Copy
            shutil.copy2(current_path, default_path)

            QMessageBox.information(
                self,
                "成功 Success",
                f"已设为默认模型 Set as default model:\n{metadata['filename']}\n→ {default_name}"
            )

            # Refresh list
            self.refresh_model_list()

        except Exception as e:
            QMessageBox.critical(
                self,
                "错误 Error",
                f"设置默认模型失败 Failed to set default model:\n{str(e)}"
            )

    def load_model_for_detection(self):
        """
        Load the selected model for detection
        This will update the detection tab's model selection
        """
        if not self.current_model_metadata:
            return

        metadata = self.current_model_metadata
        file_path = metadata.get('path', '')

        # Get parent MainWindow to access detection tab
        parent = self.parent()
        while parent and not hasattr(parent, 'detection_tab'):
            parent = parent.parent()

        if not parent or not hasattr(parent, 'detection_tab'):
            QMessageBox.warning(
                self,
                "警告 Warning",
                "无法访问检测标签页 Cannot access detection tab"
            )
            return

        try:
            # Update detection tab's custom checkpoint
            detection_tab = parent.detection_tab

            # Set to custom model mode
            detection_tab.checkpoint_combo.setCurrentIndex(2)  # "自定义模型"
            detection_tab.custom_checkpoint_path = file_path
            detection_tab.custom_checkpoint_button.setText(Path(file_path).name)

            # Switch to detection tab
            parent.tab_widget.setCurrentWidget(detection_tab)

            QMessageBox.information(
                self,
                "成功 Success",
                f"已加载模型，切换到检测标签页\nModel loaded, switched to detection tab\n\n{metadata['filename']}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "错误 Error",
                f"加载模型失败 Failed to load model:\n{str(e)}"
            )

    def select_tensorboard_log(self):
        """Open dialog to select TensorBoard log directory."""
        if not self.current_model_metadata:
            QMessageBox.warning(
                self,
                "警告 Warning",
                "请先选择一个模型\nPlease select a model first"
            )
            return

        # Start in logs directory
        default_log_dir = Path("outputs/logs")
        if default_log_dir.exists():
            start_dir = str(default_log_dir.absolute())
        else:
            start_dir = str(Path.cwd())

        log_dir = QFileDialog.getExistingDirectory(
            self,
            "选择TensorBoard日志目录 Select TensorBoard Log Directory",
            start_dir
        )

        if not log_dir:
            return

        # Validate the directory
        is_valid, error_msg = validate_log_directory(log_dir)
        if not is_valid:
            QMessageBox.warning(
                self,
                "错误 Error",
                f"无效的日志目录 Invalid log directory:\n{error_msg}"
            )
            return

        # Load and plot curves
        self.load_training_curves(log_dir)

    def load_training_curves(self, log_dir: str):
        """Load and display training curves from log directory."""
        # First check if TensorBoard is available
        available, message = check_tensorboard_available()

        if not available:
            self.training_curves.show_placeholder("TensorBoard 未安装\nTensorBoard not available")
            QMessageBox.critical(
                self,
                "依赖缺失 Missing Dependency",
                f"无法加载 TensorBoard 日志，因为 TensorBoard 未安装。\n\n"
                f"Cannot load TensorBoard logs because TensorBoard is not installed.\n\n"
                f"{message}\n\n"
                f"请安装 TensorBoard 后重试：\n"
                f"Please install TensorBoard and try again:\n"
                f"  pip install tensorboard"
            )
            return

        try:
            curves_data = get_training_curves(log_dir)

            if curves_data is None:
                self.training_curves.show_placeholder("无法加载训练曲线\nFailed to load training curves")
                QMessageBox.warning(
                    self,
                    "错误 Error",
                    "无法从日志文件中提取训练曲线\n"
                    "Failed to extract training curves from log files\n\n"
                    f"日志目录 Log directory:\n{log_dir}\n\n"
                    "请检查控制台输出的详细错误信息\n"
                    "Check console for detailed error messages"
                )
                return

            self.training_curves.plot_curves(curves_data, log_dir)

            # Switch to curves tab
            self.details_tab_widget.setCurrentIndex(1)

        except ImportError as e:
            # TensorBoard not available - already handled above, but just in case
            self.training_curves.show_placeholder("TensorBoard 未安装\nTensorBoard not available")
            QMessageBox.critical(
                self,
                "依赖缺失 Missing Dependency",
                f"无法加载 TensorBoard 日志\n\n"
                f"Cannot load TensorBoard logs\n\n"
                f"Error: {str(e)}\n\n"
                f"请安装 TensorBoard: pip install tensorboard"
            )
        except Exception as e:
            import traceback
            self.training_curves.show_placeholder()
            error_msg = (
                f"加载训练曲线时出错\n"
                f"Error loading training curves:\n\n"
                f"{str(e)}\n\n"
                f"日志目录 Log directory:\n{log_dir}"
            )
            QMessageBox.critical(self, "错误 Error", error_msg)
            print(f"Error in load_training_curves:\n{traceback.format_exc()}")

    def clear_details(self):
        """Clear model details panel"""
        self.filename_label.setText("文件: -")
        self.path_label.setText("路径: -")
        self.size_label.setText("大小: -")
        self.modified_label.setText("修改: -")
        self.backbone_label.setText("骨干: -")
        self.dataset_label.setText("数据集: -")
        self.segments_label.setText("段数: -")
        self.frames_label.setText("每段帧数: -")
        self.total_frames_label.setText("总帧数: -")
        self.epochs_label.setText("轮数: -")
        self.accuracy_label.setText("精度: -")
        self.status_label.setText("状态: -")
        self.description_text.clear()
        self.training_curves.clear()
