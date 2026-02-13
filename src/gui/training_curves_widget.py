"""
Training curves visualization widget
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import platform


def get_chinese_font():
    """Get FontProperties for Chinese font."""
    if platform.system() == 'Windows':
        # Windows: Try common Chinese fonts
        chinese_fonts = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong',             # 仿宋
        ]
    elif platform.system() == 'Darwin':
        # macOS
        chinese_fonts = [
            'PingFang SC',          # 苹方
            'Heiti SC',             # 黑体
            'STHeiti',              # 华文黑体
            'Songti SC',            # 宋体
        ]
    else:
        # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿
            'Noto Sans CJK SC',     # 思源黑体
            'Droid Sans Fallback',
        ]

    # Find available font
    from matplotlib.font_manager import fontManager
    available_fonts = set([f.name for f in fontManager.ttflist])

    for font in chinese_fonts:
        if font in available_fonts:
            try:
                return FontProperties(family=font)
            except Exception:
                continue

    # Fallback to default
    return FontProperties()


# Global font properties for Chinese text
CHINESE_FONT = get_chinese_font()


class TrainingCurvesWidget(QWidget):
    """
    Widget for displaying training loss and accuracy curves.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.current_log_dir = None

    def init_ui(self):
        """Initialize UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Info label
        self.info_label = QLabel("点击下方按钮选择TensorBoard日志目录\nClick button below to select TensorBoard log directory")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                background: #f0f0f0;
                border-radius: 8px;
                color: #666;
            }
        """)
        layout.addWidget(self.info_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)
        layout.addWidget(self.canvas)

        # Empty plot area placeholder
        self.placeholder_label = QLabel("暂无训练曲线 No training curves")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("""
            QLabel {
                padding: 40px;
                background: #fafafa;
                border: 2px dashed #ddd;
                border-radius: 8px;
                color: #999;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.placeholder_label)

    def plot_curves(self, curves_data: dict, log_dir: str = None):
        """
        Plot training curves from extracted data.

        Args:
            curves_data: Dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
            log_dir: Optional log directory path for display
        """
        if not curves_data:
            self.show_placeholder("No data to display")
            return

        self.figure.clear()
        self.canvas.setVisible(True)
        self.placeholder_label.setVisible(False)

        # Determine which subplots to create
        has_loss = 'train_loss' in curves_data or 'val_loss' in curves_data
        has_acc = 'train_acc' in curves_data or 'val_acc' in curves_data

        if has_loss and has_acc:
            self.figure.set_size_inches(10, 4)
            ax1, ax2 = self.figure.subplots(1, 2)
        elif has_loss:
            self.figure.set_size_inches(8, 5)
            ax1 = self.figure.subplots(1, 1)
            ax2 = None
        elif has_acc:
            self.figure.set_size_inches(8, 5)
            ax2 = self.figure.subplots(1, 1)
            ax1 = None
        else:
            self.show_placeholder("No valid metrics found")
            return

        # Plot loss curves
        if ax1 is not None:
            if 'train_loss' in curves_data:
                steps, values = curves_data['train_loss']
                ax1.plot(steps, values, label='训练 Train', color='#2ecc71', linewidth=2)
            if 'val_loss' in curves_data:
                steps, values = curves_data['val_loss']
                ax1.plot(steps, values, label='验证 Val', color='#e74c3c', linewidth=2)

            ax1.set_xlabel('轮次 Epoch', fontproperties=CHINESE_FONT, fontsize=11)
            ax1.set_ylabel('损失 Loss', fontproperties=CHINESE_FONT, fontsize=11)
            ax1.set_title('损失曲线 Loss Curve', fontproperties=CHINESE_FONT, fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', prop=CHINESE_FONT)
            ax1.grid(True, alpha=0.3)

        # Plot accuracy curves
        if ax2 is not None:
            if 'train_acc' in curves_data:
                steps, values = curves_data['train_acc']
                ax2.plot(steps, values, label='训练 Train', color='#2ecc71', linewidth=2)
            if 'val_acc' in curves_data:
                steps, values = curves_data['val_acc']
                ax2.plot(steps, values, label='验证 Val', color='#e74c3c', linewidth=2)

            ax2.set_xlabel('轮次 Epoch', fontproperties=CHINESE_FONT, fontsize=11)
            ax2.set_ylabel('精度 Accuracy (%)', fontproperties=CHINESE_FONT, fontsize=11)
            ax2.set_title('精度曲线 Accuracy Curve', fontproperties=CHINESE_FONT, fontsize=12, fontweight='bold')
            ax2.legend(loc='lower right', prop=CHINESE_FONT)
            ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

        # Update info label
        if log_dir:
            import os
            log_name = os.path.basename(log_dir)
            self.info_label.setText(f"日志目录 Log Directory: {log_name}")
            self.info_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background: #e8f5e9;
                    border-radius: 6px;
                    color: #2e7d32;
                }
            """)

    def show_placeholder(self, message: str = "暂无训练曲线 No training curves"):
        """Show placeholder message instead of plot."""
        self.canvas.setVisible(False)
        self.placeholder_label.setVisible(True)
        self.placeholder_label.setText(message)
        self.info_label.setText("点击下方按钮选择TensorBoard日志目录\nClick button below to select TensorBoard log directory")
        self.info_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                background: #f0f0f0;
                border-radius: 8px;
                color: #666;
            }
        """)
        self.figure.clear()

    def clear(self):
        """Clear the plot."""
        self.show_placeholder()
