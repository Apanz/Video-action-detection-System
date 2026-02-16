"""
GUI的结果标签页
按动作类别显示检测结果
"""

import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFileDialog, QScrollArea, QGridLayout,
                             QSizePolicy, QMessageBox, QSplitter, QTextEdit, QDialog)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont, QImage
import cv2
import numpy as np
from pathlib import Path


class ResultsTab(QWidget):
    """
    用于显示检测结果的标签页
    显示动作统计、帧预览并允许导出结果
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_stats = None
        self.current_action_detail = None
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 标题
        title = QLabel("📊 检测结果 Detection Results")
        title.setFont(QFont("Hiragino Sans GB", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # 会话信息组
        session_group = QGroupBox("会话信息 Session Info")
        session_layout = QGridLayout()
        session_layout.setSpacing(10)
        session_layout.setContentsMargins(12, 12, 12, 12)

        self.session_id_label = QLabel("会话ID: -")
        self.video_source_label = QLabel("视频源: -")
        self.time_range_label = QLabel("时间: -")
        self.total_frames_label = QLabel("总帧数: 0")

        session_layout.addWidget(self.session_id_label, 0, 0)
        session_layout.addWidget(self.video_source_label, 0, 1)
        session_layout.addWidget(self.time_range_label, 1, 0)
        session_layout.addWidget(self.total_frames_label, 1, 1)

        session_group.setLayout(session_layout)
        main_layout.addWidget(session_group)

        # 用于表格和详情视图的分隔器（水平布局）
        splitter = QSplitter(Qt.Horizontal)

        # 动作统计表格
        table_group = QGroupBox("动作类别统计 Action Statistics (按帧数排序)")
        table_group.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(12, 12, 12, 12)
        table_layout.setSpacing(10)

        # 创建表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels([
            "动作名称 Action", "帧数 Frames", "占比 Percentage",
            "平均置信度 Avg Conf", "操作 Actions"
        ])

        # 配置表格 - 优化列宽和高度
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # 动作名称 - 响应式
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        # 优化表格外观
        self.stats_table.verticalHeader().setDefaultSectionSize(52)  # 更大的行高度以适应按钮
        self.stats_table.setMinimumWidth(350)  # 确保最小宽度以防止文本截断
        self.stats_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid rgba(189, 227, 195, 200);
                border-radius: 8px;
                gridline-color: rgba(189, 227, 195, 100);
                background: rgba(255, 255, 255, 230);
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid rgba(189, 227, 195, 80);
            }
            QTableWidget::item:selected {
                background: rgba(163, 204, 218, 120);
                color: #1E3A8A;
            }
            QTableWidget::item:hover {
                background: rgba(163, 204, 218, 60);
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(168, 212, 184, 200),
                    stop:1 rgba(149, 197, 172, 200));
                color: #2c5f4e;
                padding: 8px 6px;
                border: 1px solid rgba(189, 227, 195, 200);
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
        """)

        self.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.cellClicked.connect(self.on_table_row_clicked)

        table_layout.addWidget(self.stats_table)

        # 导出按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)

        self.export_json_button = QPushButton("导出JSON")
        self.export_json_button.setMaximumHeight(36)
        self.export_json_button.clicked.connect(self.export_json)
        button_layout.addWidget(self.export_json_button)

        self.export_csv_button = QPushButton("导出CSV")
        self.export_csv_button.setMaximumHeight(36)
        self.export_csv_button.clicked.connect(self.export_csv)
        button_layout.addWidget(self.export_csv_button)

        self.clear_button = QPushButton("清除")
        self.clear_button.setMaximumHeight(36)
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()
        table_layout.addLayout(button_layout)

        table_group.setLayout(table_layout)
        splitter.addWidget(table_group)

        # 动作详情视图
        detail_group = QGroupBox("动作详情 Action Details")
        detail_group.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(12, 12, 12, 12)
        detail_layout.setSpacing(10)

        self.detail_label = QLabel("选择动作查看详情 Select an action")
        self.detail_label.setFont(QFont("Hiragino Sans GB", 11, QFont.Bold))
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet("""
            QLabel {
                background: rgba(163, 204, 218, 150);
                color: #1E3A8A;
                padding: 10px 15px;
                border-radius: 8px;
                font-weight: bold;
            }
        """)
        detail_layout.addWidget(self.detail_label)

        # 帧预览滚动区域 - 优化为4列显示
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid rgba(189, 227, 195, 150);
                border-radius: 10px;
                background: rgba(255, 255, 255, 150);
            }
        """)

        self.frame_preview_widget = QWidget()
        self.frame_preview_layout = QGridLayout(self.frame_preview_widget)
        self.frame_preview_layout.setSpacing(10)
        self.frame_preview_layout.setContentsMargins(10, 10, 10, 10)
        self.frame_preview_layout.setColumnStretch(0, 1)
        self.frame_preview_layout.setColumnStretch(1, 1)
        self.frame_preview_layout.setColumnStretch(2, 1)
        self.frame_preview_layout.setColumnStretch(3, 1)
        self.frame_preview_layout.setAlignment(Qt.AlignTop)

        self.scroll_area.setWidget(self.frame_preview_widget)
        detail_layout.addWidget(self.scroll_area)

        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)

        # 设置分隔器大小 - 为详情区域分配更多空间以便更好地预览帧
        splitter.setStretchFactor(0, 5)  # 表格区域 ~45%
        splitter.setStretchFactor(1, 6)  # 详情区域 ~55%

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def update_results(self, stats: dict):
        """
        Update results display with new statistics

        Args:
            stats: Statistics dictionary from ResultCollector
        """
        self.current_stats = stats

        # 更新会话信息
        self.session_id_label.setText(f"会话ID: {stats.get('session_id', '-')}")
        self.video_source_label.setText(f"视频源: {stats.get('video_source', '-')}")
        start_time = stats.get('start_time', '-')
        end_time = stats.get('end_time', '-')
        self.time_range_label.setText(f"时间: {start_time} - {end_time}")
        self.total_frames_label.setText(
            f"总帧数: {stats.get('total_frames', 0)} "
            f"(检测到: {stats.get('total_detected_frames', 0)})"
        )

        # 更新表格
        self.update_table(stats.get('actions', {}))

    def update_table(self, actions: dict):
        """
        Update action statistics table

        Args:
            actions: Dictionary of action statistics
        """
        self.stats_table.setRowCount(len(actions))

        for row, (action_name, data) in enumerate(actions.items()):
            # 动作名称
            name_item = QTableWidgetItem(action_name)
            name_item.setFont(QFont("Hiragino Sans GB", 10))
            self.stats_table.setItem(row, 0, name_item)

            # 帧数
            count_item = QTableWidgetItem(str(data['count']))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.stats_table.setItem(row, 1, count_item)

            # 占比
            percentage_item = QTableWidgetItem(f"{data['percentage']:.1f}%")
            percentage_item.setTextAlignment(Qt.AlignCenter)
            self.stats_table.setItem(row, 2, percentage_item)

            # 平均置信度
            conf_item = QTableWidgetItem(f"{data['confidence_avg']:.2f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.stats_table.setItem(row, 3, conf_item)

            # 操作按钮
            btn_widget = QWidget()
            btn_widget.setStyleSheet("background: transparent;")
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setSpacing(4)

            view_btn = QPushButton("查看")
            view_btn.setMinimumHeight(28)
            view_btn.setMaximumHeight(32)
            view_btn.setFont(QFont("Hiragino Sans GB", 9))
            view_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #A8D4B8, stop:1 #95C5AC);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #B8DFC8, stop:1 #A5D5BC);
                }
            """)
            view_btn.clicked.connect(lambda checked, a=action_name: self.show_action_detail(a))
            btn_layout.addWidget(view_btn)

            export_btn = QPushButton("导出")
            export_btn.setMinimumHeight(28)
            export_btn.setMaximumHeight(32)
            export_btn.setFont(QFont("Hiragino Sans GB", 9))
            export_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #A8D4B8, stop:1 #95C5AC);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #B8DFC8, stop:1 #A5D5BC);
                }
            """)
            export_btn.clicked.connect(lambda checked, a=action_name: self.export_action_frames(a))
            btn_layout.addWidget(export_btn)

            self.stats_table.setCellWidget(row, 4, btn_widget)

    def on_table_row_clicked(self, row: int, column: int):
        """
        Handle table row click

        Args:
            row: Row index
            column: Column index
        """
        action_name = self.stats_table.item(row, 0).text()
        self.show_action_detail(action_name)

    def show_action_detail(self, action_name: str):
        """
        Show detailed view for specific action

        Args:
            action_name: Name of the action to display
        """
        if not self.current_stats or action_name not in self.current_stats.get('actions', {}):
            return

        action_data = self.current_stats['actions'][action_name]
        self.current_action_detail = action_name

        # 更新详情标签
        self.detail_label.setText(
            f"{action_name} | 帧:{action_data['count']} | 占比:{action_data['percentage']:.1f}% | "
            f"置信度:{action_data['confidence_avg']:.2f} | 已保存:{action_data['saved_frames']}"
        )

        # 清除之前的预览
        # 从布局中移除所有小部件
        while self.frame_preview_layout.count():
            item = self.frame_preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 添加帧预览
        frames = action_data.get('frames', [])
        if not frames:
            no_frames_label = QLabel("无保存的帧 No saved frames")
            no_frames_label.setAlignment(Qt.AlignCenter)
            no_frames_label.setStyleSheet("color: #888; font-size: 11px;")
            self.frame_preview_layout.addWidget(no_frames_label, 0, 0)
        else:
            # 以网格形式显示帧（4列以获得最佳显示效果）
            cols = 4
            for idx, frame_info in enumerate(frames):
                row_idx = idx // cols
                col_idx = idx % cols

                # 创建帧小部件
                frame_widget = self.create_frame_preview(frame_info)
                # 设置列跨度以确保正确的布局
                self.frame_preview_layout.addWidget(frame_widget, row_idx, col_idx)

    def create_frame_preview(self, frame_info: dict) -> QWidget:
        """
        Create a frame preview widget

        Args:
            frame_info: Frame information dictionary

        Returns:
            Widget containing frame preview
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # 帧图像 - 调整大小以获得更好的4列布局
        frame_path = frame_info.get('frame_path', '')
        if frame_path and os.path.exists(frame_path):
            pixmap = QPixmap(frame_path)
            # 根据滚动区域宽度计算最佳大小
            # 这确保帧在4列中正确适配而无需水平滚动
            scaled_pixmap = pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            image_label = QLabel()
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
        else:
            placeholder = QLabel("无图像")
            placeholder.setFixedSize(160, 160)
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background: #f0f0f0; border: 1px solid #ccc; font-size: 9px;")
            layout.addWidget(placeholder)

        # 帧信息
        frame_idx = frame_info.get('frame_idx', 0)
        timestamp = frame_info.get('timestamp', 0)
        confidence = frame_info.get('confidence', 0)

        info_label = QLabel(f"#{frame_idx}\n{timestamp:.1f}s\n{confidence:.2f}")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 9))
        layout.addWidget(info_label)

        # 查看按钮
        view_btn = QPushButton("查看")
        view_btn.setMinimumHeight(30)
        view_btn.setMinimumWidth(60)  # Ensure text fits
        view_btn.setFont(QFont("Arial", 10, QFont.Bold))
        view_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #A8D4B8, stop:1 #95C5AC);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #B8DFC8, stop:1 #A5D5BC);
            }
        """)
        view_btn.clicked.connect(lambda checked, p=frame_path: self.show_full_frame(p))
        layout.addWidget(view_btn)

        # 卡片样式
        widget.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 200);
                border: 2px solid rgba(189, 227, 195, 150);
                border-radius: 10px;
            }
            QWidget:hover {
                background: rgba(255, 255, 255, 230);
                border: 2px solid rgba(163, 204, 218, 200);
            }
        """)

        return widget

    def show_full_frame(self, frame_path: str):
        """
        Show full-size frame image

        Args:
            frame_path: Path to frame image
        """
        if not frame_path or not os.path.exists(frame_path):
            QMessageBox.warning(self, "错误 Error", "帧文件不存在 Frame file does not exist")
            return

        # 创建对话框以显示完整帧
        dialog = QDialog(self)
        dialog.setWindowTitle(f"帧预览 Frame Preview - {os.path.basename(frame_path)}")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        # 为大图像创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # 加载并显示图像
        pixmap = QPixmap(frame_path)
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

        scroll_area.setWidget(label)
        layout.addWidget(scroll_area)

        # 添加关闭按钮
        close_btn = QPushButton("关闭 Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()

    def export_json(self):
        """Export results to JSON file"""
        if not self.current_stats:
            QMessageBox.warning(self, "警告 Warning", "没有结果可导出 No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出JSON Export JSON",
            f"outputs/results/{self.current_stats.get('session_id', 'results')}.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_stats, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "成功 Success", f"结果已导出到 Results exported to\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误 Error", f"导出失败 Export failed:\n{str(e)}")

    def export_csv(self):
        """Export results to CSV file"""
        if not self.current_stats:
            QMessageBox.warning(self, "警告 Warning", "没有结果可导出 No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出CSV Export CSV",
            f"outputs/results/{self.current_stats.get('session_id', 'results')}.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                import csv

                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Action', 'Count', 'Percentage', 'Avg Confidence',
                                   'Saved Frames', 'Frame Indices'])

                    for action, data in self.current_stats.get('actions', {}).items():
                        frame_indices = [f['frame_idx'] for f in data.get('frames', [])]
                        writer.writerow([
                            action,
                            data['count'],
                            f"{data['percentage']:.2f}%",
                            f"{data['confidence_avg']:.4f}",
                            data['saved_frames'],
                            ', '.join(map(str, frame_indices))
                        ])

                QMessageBox.information(self, "成功 Success", f"结果已导出到 Results exported to\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误 Error", f"导出失败 Export failed:\n{str(e)}")

    def export_action_frames(self, action_name: str):
        """
        Export frames for a specific action

        Args:
            action_name: Name of the action
        """
        if not self.current_stats or action_name not in self.current_stats.get('actions', {}):
            return

        action_data = self.current_stats['actions'][action_name]
        frames = action_data.get('frames', [])

        if not frames:
            QMessageBox.information(self, "信息 Info", f"{action_name}\n无保存的帧 No saved frames")
            return

        # Select directory
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录 Select Export Directory",
            "outputs/results"
        )

        if dir_path:
            try:
                # 创建动作子目录
                action_dir = os.path.join(dir_path, action_name.replace('/', '_'))
                os.makedirs(action_dir, exist_ok=True)

                # 复制帧
                import shutil
                for frame_info in frames:
                    frame_path = frame_info.get('frame_path', '')
                    if frame_path and os.path.exists(frame_path):
                        filename = os.path.basename(frame_path)
                        dest_path = os.path.join(action_dir, filename)
                        shutil.copy2(frame_path, dest_path)

                QMessageBox.information(
                    self,
                    "成功 Success",
                    f"已导出 {len(frames)} 帧到\nExported {len(frames)} frames to\n{action_dir}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误 Error", f"导出失败 Export failed:\n{str(e)}")

    def clear_results(self):
        """Clear all results"""
        reply = QMessageBox.question(
            self,
            "确认 Confirm",
            "确定要清除所有结果吗？\nAre you sure you want to clear all results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.current_stats = None
            self.current_action_detail = None
            self.stats_table.setRowCount(0)
            self.session_id_label.setText("会话ID: -")
            self.video_source_label.setText("视频源: -")
            self.time_range_label.setText("时间: -")
            self.total_frames_label.setText("总帧数: 0")
            self.detail_label.setText("选择动作查看详情 Select an action")

            # 清除帧预览
            while self.frame_preview_layout.count():
                item = self.frame_preview_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            no_results_label = QLabel("无结果 No Results")
            no_results_label.setAlignment(Qt.AlignCenter)
            self.frame_preview_layout.addWidget(no_results_label, 0, 0)
