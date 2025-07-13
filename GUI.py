from pathlib import Path

from PySide6.QtWidgets import QApplication
from src.GUI.windows_objects import MainWindow

pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")

def main():
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()

if __name__ == "__main__":
    main()