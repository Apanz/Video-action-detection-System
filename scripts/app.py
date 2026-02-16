#!/usr/bin/env python3
"""
GUI应用入口点
运行此文件以启动视频动作识别系统GUI
"""

import sys
from pathlib import Path

# 在任何其他导入之前将项目根目录和src目录添加到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui.main_window import MainWindow

# 设置图标路径
ICON_PATH = ROOT_DIR / "icon" / "camara.svg"


def main():
    """主入口点"""
    app = QApplication(sys.argv)
    app.setApplicationName("视频动作检测系统")

    # 为任务栏/窗口设置应用图标
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
