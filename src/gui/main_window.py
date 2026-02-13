"""
Main Application Window
Video Action Detection GUI with glassmorphism design
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QStatusBar,
                             QAction, QMenuBar, QToolBar, QMessageBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QFontDatabase

from .detection_tab import DetectionTab
from .results_tab import ResultsTab
from .model_management_tab import ModelManagementTab


# Load custom font
def load_custom_font():
    """Load custom font from resources"""
    font_path = os.path.join(os.path.dirname(__file__), 'resources', 'font', 'AaFengKuangYuanShiRen', 'AaFengKuangYuanShiRen-2.ttf')
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id < 0:
        print(f"Warning: Failed to load font from {font_path}")
        return None
    return QFontDatabase.applicationFontFamilies(font_id)[0]


class IconProvider:
    """Provider for UI icons with emoji fallbacks"""

    def __init__(self):
        self.icons = {}
        self._init_icons()

    def _init_icons(self):
        """Initialize icons using emoji as fallback"""
        # Using emojis as icons (could be replaced with actual icon files)
        self.emoji_map = {
            'camera': '📷',
            'video': '🎥',
            'play': '▶️',
            'pause': '⏸️',
            'stop': '⏹️',
            'settings': '⚙️',
            'save': '💾',
            'check': '✓',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌'
        }

    def get_emoji(self, name):
        """Get emoji for icon name"""
        return self.emoji_map.get(name, '')


class MainWindow(QMainWindow):
    """
    Main application window for Video Action Detection
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频行为识别系统 - Video Action Detection")
        self.setMinimumSize(1280, 850)  # Increased minimum size to accommodate larger control panel
        self.resize(1550, 1050)

        # Load custom font
        self.custom_font_family = load_custom_font()

        # Initialize icon provider
        self.icon_provider = IconProvider()

        # Apply glassmorphism style
        self._apply_glassmorphism_style()

        # Initialize UI
        self.init_ui()

        # Initialize status bar
        self.init_status_bar()

    def init_ui(self):
        """Initialize user interface"""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create header
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("main_tab_widget")

        # Add tabs
        self.detection_tab = DetectionTab()
        self.tab_widget.addTab(self.detection_tab, "实时检测 Real-time Detection")

        self.results_tab = ResultsTab()
        self.tab_widget.addTab(self.results_tab, "检测结果 Detection Results")

        self.model_management_tab = ModelManagementTab()
        self.tab_widget.addTab(self.model_management_tab, "模型管理 Model Management")

        main_layout.addWidget(self.tab_widget)

    def _create_header(self):
        """Create application header"""
        header = QWidget()
        header.setObjectName("header_widget")
        header_layout = QHBoxLayout(header)

        # Title with icon
        title_label = QLabel(f"{self.icon_provider.get_emoji('video')} 视频行为识别系统")
        title_label.setObjectName("header_title")
        if self.custom_font_family:
            title_font = QFont(self.custom_font_family, 32)
            title_font.setLetterSpacing(QFont.PercentageSpacing, 2)
        else:
            title_font = QFont("Arial", 32)
            title_font.setLetterSpacing(QFont.PercentageSpacing, 2)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Version info
        version_label = QLabel(f"v2.1 - Optimized {self.icon_provider.get_emoji('check')}")
        version_label.setObjectName("header_version")
        version_label.setFont(QFont("Arial", 10))
        header_layout.addWidget(version_label)

        header.setFixedHeight(75)

        return header

    def init_status_bar(self):
        """Initialize status bar"""
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("main_status_bar")
        self.status_bar.showMessage("就绪 Ready")
        self.setStatusBar(self.status_bar)

    def _apply_glassmorphism_style(self):
        """Apply glassmorphism style sheet"""
        style = """
        /* 全局背景色 - #BDE3C3渐变 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #BDE3C3,
                                      stop:0.5 #A8D4B8,
                                      stop:1 #7FA89A);
        }

        /* 标签页容器 */
        QTabWidget#main_tab_widget::pane {
            background: rgba(255, 255, 255, 190);
            border: 1px solid rgba(189, 227, 195, 150);
            border-radius: 12px;
            margin: 10px;
        }

        /* 标签页标签 */
        QTabWidget#main_tab_widget::tab-bar {
            alignment: left;
        }

        QTabBar::tab {
            background: rgba(255, 255, 255, 200);
            color: #1E3A8A;
            padding: 12px 30px;
            margin-right: 5px;
            margin-top: 10px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            min-width: 120px;
            font-weight: 600;
            font-size: 14px;
            border: none;
        }

        QTabBar::tab:selected {
            background: rgba(255, 255, 255, 250);
            color: #1E90FF;
            font-weight: bold;
        }

        QTabBar::tab:hover:!selected {
            background: rgba(255, 255, 255, 230);
        }

        /* 标题栏样式 */
        QWidget#header_widget {
            background: #F8F7BA;
            border: none;
            border-bottom: 2px solid rgba(163, 204, 218, 150);
        }

        QLabel#header_title {
            color: #0BA6DF;
            background: transparent;
            font-size: 32px;
            font-weight: normal;
            letter-spacing: 2px;
        }

        QLabel#header_version {
            color: white;
            background: transparent;
            padding: 5px 15px;
            border-radius: 15px;
            background: rgba(163, 204, 218, 150);
            font-size: 11px;
        }

        /* 分组框样式 */
        QGroupBox {
            background: rgba(255, 255, 255, 190);
            color: #2c5f4e;
            border: 2px solid rgba(189, 227, 195, 150);
            border-radius: 14px;
            margin-top: 18px;
            margin-bottom: 10px;
            padding: 20px;
            font-weight: bold;
            font-size: 15px;
            font-family: "Hiragino Sans GB";
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 6px 18px;
            background: rgba(255, 255, 255, 220);
            border-radius: 12px;
            color: #A3CCDA;
            font-weight: bold;
            font-size: 16px;
            font-family: "Hiragino Sans GB";
            margin-top: 2px;
        }

        /* 按钮样式 - 绿色渐变 */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #A8D4B8, stop:1 #95C5AC);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            font-size: 14px;
        }

        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #B8DFC8, stop:1 #A5D5BC);
        }

        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #95C5AC, stop:1 #82B69C);
        }

        QPushButton:disabled {
            background: rgba(150, 150, 150, 150);
            color: rgba(255, 255, 255, 150);
        }

        /* 下拉框样式 */
        QComboBox {
            background: rgba(255, 255, 255, 220);
            color: #2c3e50;
            border: 2px solid rgba(189, 227, 195, 200);
            border-radius: 6px;
            padding: 10px 15px;
            font-size: 14px;
        }

        QComboBox:hover {
            border: 2px solid #A3CCDA;
        }

        QComboBox::drop-down {
            border: none;
            width: 30px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #A3CCDA;
        }

        QComboBox QAbstractItemView {
            background: rgba(255, 255, 255, 250);
            border: 2px solid rgba(189, 227, 195, 150);
            border-radius: 6px;
            selection-background-color: rgba(163, 204, 218, 150);
            selection-color: white;
            font-size: 14px;
        }

        /* 数值输入框样式 */
        QSpinBox, QDoubleSpinBox {
            background: rgba(255, 255, 255, 220);
            color: #2c3e50;
            border: 2px solid rgba(189, 227, 195, 200);
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 14px;
        }

        QSpinBox:hover, QDoubleSpinBox:hover {
            border: 2px solid #A3CCDA;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 20px;
            border: none;
            border-top-right-radius: 4px;
            background: rgba(163, 204, 218, 150);
        }

        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 20px;
            border: none;
            border-bottom-right-radius: 4px;
            background: rgba(163, 204, 218, 150);
        }

        /* 标签样式 */
        QLabel {
            color: #2c5f4e;
            font-size: 14px;
            background: transparent;
            padding: 4px;
        }

        /* 视频显示标签 */
        QLabel#video_label {
            background: #2b2b2b;
            border: 3px solid rgba(255, 255, 255, 200);
            border-radius: 14px;
            color: #ffffff;
        }

        /* 复选框样式 */
        QCheckBox {
            color: #2c5f4e;
            font-size: 14px;
            background: transparent;
            padding: 5px;
        }

        QCheckBox::indicator {
            width: 22px;
            height: 22px;
            border: 2px solid rgba(189, 227, 195, 200);
            border-radius: 4px;
            background: rgba(255, 255, 255, 220);
        }

        QCheckBox::indicator:checked {
            background: #A3CCDA;
            border-color: #A3CCDA;
            image: none;
        }

        QCheckBox::indicator:hover {
            border-color: #A3CCDA;
        }

        /* 进度条样式 */
        QProgressBar {
            background: rgba(255, 255, 255, 150);
            border: 2px solid rgba(189, 227, 195, 150);
            border-radius: 8px;
            text-align: center;
            color: #2c5f4e;
            font-weight: bold;
            font-size: 13px;
        }

        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #A8D4B8, stop:1 #95C5AC);
            border-radius: 6px;
        }

        /* 文本编辑框样式 */
        QTextEdit {
            background: rgba(255, 255, 255, 220);
            color: #2c3e50;
            border: 2px solid rgba(189, 227, 195, 200);
            border-radius: 8px;
            font-family: "Consolas", monospace;
            font-size: 13px;
        }

        /* 状态栏样式 */
        QStatusBar {
            background: rgba(255, 255, 255, 190);
            color: #2c5f4e;
            border-top: 1px solid rgba(189, 227, 195, 150);
            font-size: 14px;
        }

        /* 滚动条样式 */
        QScrollBar:vertical {
            background: rgba(255, 255, 255, 100);
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background: rgba(163, 204, 218, 200);
            min-height: 30px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical:hover {
            background: rgba(163, 204, 218, 250);
        }

        QScrollBar:horizontal {
            background: rgba(255, 255, 255, 100);
            height: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal {
            background: rgba(163, 204, 218, 200);
            min-width: 30px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal:hover {
            background: rgba(163, 204, 218, 250);
        }
        """

        self.setStyleSheet(style)

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop video thread if running
        if hasattr(self.detection_tab, 'video_thread'):
            if self.detection_tab.video_thread and self.detection_tab.video_thread.isRunning():
                self.detection_tab.stop_detection()

        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set app style
    app.setStyle('Fusion')

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
