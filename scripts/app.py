#!/usr/bin/env python3
"""
GUI Application Entry Point
Run this to start Video Action Detection System GUI
"""

import sys
from pathlib import Path

# Add project root and src directory to path BEFORE any other imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui.main_window import MainWindow

# Set icon path
ICON_PATH = ROOT_DIR / "icon" / "camara.svg"


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("视频动作检测系统")

    # Set application icon for taskbar/window
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
