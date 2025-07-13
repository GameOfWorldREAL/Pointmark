import sys

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QPushButton,
    QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy,
    QMenuBar, QMenu, QListWidgetItem, QListWidget,
    QFormLayout, QCheckBox, QComboBox, QSpinBox,
    QSlider, QSplitter, QPlainTextEdit
)
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from src.GUI.gui_handler import (
    handle_open_project,
    handle_project_selected,
    handle_display_parameters_changed,
    PyVistaUpdater
)


#=========================
#chatGPT 4.o generated gui
#=========================


# Falls Du die Colormap-Liste direkt hier haben möchtest:
MATPLOTLIB_COLORMAPS = [
    "jet", "viridis", "plasma", "inferno", "magma",
    "cividis", "coolwarm", "terrain",
    "Greys", "spring",
]

METRICS_LIST = [
    "Combined", "Camera Distance", "Observation Count",
    "Triangulation Uncertainty", "Projection Error",
    "Reprojection Error", "Density",
    "Angle of Incidence", "Distance to Edge", "Edge Points",
    "Precision", "Brightness", "Darkness", "Image Quality", "OOI"
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        if not hasattr(sys, "_stdout_original"):
            sys._stdout_original = sys.stdout
            from src.GUI.windows_objects import ConsoleOutput
            sys.stdout = ConsoleOutput(sys._stdout_original)

        self.setWindowTitle("Pointmark")
        self.initial_camera_position = ((5, 5, 5), (0, 0, 0), (0, 0, 1))
        self.setGeometry(100, 100, 1000, 640)

        # State für Kamera pro Projekt
        self._camera_states = {}
        self.current_project = None

        # Default-Anzeigeparameter
        self.display_params = {
            "metric": "Combined",
            "color_scale": "jet",
            "point_size": 3,
            "show_cameras": "Show Frustum",
            "display_grid": False,
            "frustum_scale": 0.05,  # neu
        }

        # 3D-Viewer & Updater
        self.viewer     = PointCloudViewer()
        self._pv_updater = PyVistaUpdater(self.viewer)
        self.setCentralWidget(self.viewer)

        # Menüs & Docks
        self.add_menu_bar()
        self.add_side_panels()
        for splitter in self.findChildren(QSplitter):
            splitter.setHandleWidth(1)

    def add_menu_bar(self):
        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar.menu_bar)
        self.menu_bar.open_project_requested.connect(self.on_open_project)

    def add_side_panels(self):
        self.left_dock = LeftSidePane(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.left_dock.project_list.project_selected.connect(
            lambda proj: handle_project_selected(self, proj)
        )

        self.right_dock = RightSidePane(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        self.right_dock.parameters_changed.connect(
            lambda params: handle_display_parameters_changed(self, params)
        )

        self._ratio_left  = self.left_dock.width() / self.width()
        self._ratio_right = self.right_dock.width() / self.width()
        self._resizing    = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resizing = True
        new_w = self.width()
        self.resizeDocks(
            [self.left_dock], [int(new_w * self._ratio_left)], Qt.Horizontal
        )
        self.resizeDocks(
            [self.right_dock], [int(new_w * self._ratio_right)], Qt.Horizontal
        )
        self._resizing = False

    def on_open_project(self):
        handle_open_project(self)


class MenuBar(QObject):
    open_project_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.menu_bar = QMenuBar(parent)
        file_menu     = QMenu("File", self.menu_bar)
        self.menu_bar.addMenu(file_menu)
        open_action   = QAction("Open Project", self.menu_bar)
        file_menu.addAction(open_action)
        open_action.triggered.connect(self.open_project_requested.emit)


class PointCloudViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout       = QVBoxLayout()
        self.plotter = QtInteractor(self)
        # sehr dunkles Grau
        bg = (0.1, 0.1, 0.1)
        self.plotter.set_background(bg)
        # Hintergrund abspeichern
        self.background_color = bg
        layout.addWidget(self.plotter.interactor)
        self.setLayout(layout)
        self.current_actor = None
        self.grid_actor = None

    def load_point_cloud(self,
                         points,
                         scalars=None,
                         point_size=5,
                         cmap="jet",
                         brightness=0.8,
                         reset_camera=False,
                         alpha_array=None):  # ⬅️ NEU
        old_cam = None
        if not reset_camera:
            old_cam = self.plotter.camera_position

        self.plotter.clear()
        cloud = pv.PolyData(points)

        if scalars is not None:
            cloud["value"] = np.asarray(scalars, dtype=np.float32)

        if alpha_array is not None:
            cloud["alpha"] = np.asarray(alpha_array, dtype=np.float32)

        self.current_actor = self.plotter.add_mesh(
            cloud,
            scalars="value" if scalars is not None else None,
            cmap=cmap if scalars is not None else None,
            rgb=False,
            clim=[0.0, 1.0] if scalars is not None else None,
            point_size=point_size,
            render_points_as_spheres=True,
            ambient=0.1,
            diffuse=brightness,
            specular=0.0,
            opacity="alpha" if alpha_array is not None else 1.0,
            #show_scalar_bar = False #TODO
        )

        if reset_camera:
            self.plotter.reset_camera()
        elif old_cam is not None:
            self.plotter.camera_position = old_cam

        self.plotter.render()

    def add_camera_points(self, camera_poses: np.ndarray,
                          color: tuple[float,float,float] = (1,1,1),
                          point_size: int = 10):
        # Extrahiere C aus jedem 12-er-Vektor [R(9), C(3)]
        centers = np.vstack([cam[9:].reshape(3) for cam in camera_poses])
        cloud   = pv.PolyData(centers)
        self.plotter.add_mesh(
            cloud,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True
        )

    def add_camera_frustums(self,
                            camera_poses: np.ndarray,
                            scale: float = 0.05,
                            color: tuple[float,float,float] = (1,1,1),
                            line_width: int = 2):
        """
        Zeichnet alle Kamerafrusta in einem einzigen PolyData-Draw-Call.
        camera_poses: (N,12)-Array mit [R(9), C(3)] pro Kamera.
        """

        # 1) Basis-Frustum in Lokalkoordinaten
        local_pts = np.array([
            [0, 0, 0],
            [0.5,  0.5, 2],
            [-0.5, 0.5, 2],
            [-0.5,-0.5, 2],
            [0.5, -0.5, 2]
        ]) * scale

        # 2) Segmente (jeweils 2 Punkte)
        segments = np.array([
            [0,1],[0,2],[0,3],[0,4],
            [1,2],[2,3],[3,4],[4,1]
        ], dtype=np.int64)

        # 3) Alle Welt-Punkte und Linien-Indices sammeln
        all_pts = []
        line_cells = []
        offset = 0
        for cam in camera_poses:
            R = cam[:9].reshape(3,3)
            C = cam[9:].reshape(3)
            world = (local_pts @ R.T) + C   # (5,3)
            all_pts.append(world)

            # für jedes Segment die globalen Indizes
            for i0, i1 in segments:
                line_cells.append([offset + i0, offset + i1])
            offset += local_pts.shape[0]

        all_pts = np.vstack(all_pts)              # (N*5,3)
        line_cells = np.array(line_cells)         # (N*8,2)

        # 4) Ein PolyData mit allen Linien
        frusta = pv.PolyData()
        frusta.points = all_pts
        # VTK erwartet ein flaches array: [npts, i0, i1, npts, j0, j1, ...]
        cells = np.hstack([
            np.hstack(([2], pair)) for pair in line_cells
        ]).astype(np.int64)
        frusta.lines = cells

        # 5) Einmaliges Add-Mesh
        self.plotter.add_mesh(
            frusta,
            color      = color,
            line_width = line_width,
            render_lines_as_tubes = True
        )

    def update_grid(self, show: bool):
        if self.grid_actor:
            self.plotter.remove_actor(self.grid_actor)
            self.grid_actor = None

        if show:
            grid = pv.Plane(i_size=10, j_size=10, i_resolution=10, j_resolution=10)
            self.grid_actor = self.plotter.add_mesh(
                grid,
                color="gray",
                style="wireframe",
                line_width=1
            )


class LeftSidePane(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Projects", parent)
        self.setTitleBarWidget(QWidget())
        self.parent_mainwindow = parent
        self.minimal_width     = 100
        self.expanded_width    = 250
        self.collapsed_width   = 40
        self.is_collapsed      = False

        # Projektliste & Terminal Output
        self.project_list = ProjectListPanel()
        self.terminal = TerminalOutputWidget()

        # Redirect stdout only once if not already done
        if not isinstance(sys.stdout, ConsoleOutput):
            self._original_stdout = sys.stdout  # ← speichere Original
            sys.stdout = ConsoleOutput(self._original_stdout)

        # Verbinde neuen Text mit Terminalanzeige
        if isinstance(sys.stdout, ConsoleOutput):
            sys.stdout.new_text.connect(self.terminal.append_text)

        # Vertikaler Splitter (oben Projektliste, unten Terminal)
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(self.project_list)
        v_splitter.addWidget(self.terminal)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)

        # Toggle-Button zum Ein-/Ausklappen
        self.toggle_button = QPushButton("⮜")
        self.toggle_button.setFixedSize(15, 40)
        self.toggle_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toggle_button.clicked.connect(self.toggle_panel)

        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(0)
        btn_layout.addWidget(self.toggle_button, alignment=Qt.AlignCenter)
        btn_container.setFixedWidth(self.toggle_button.width())

        # Hauptlayout mit Splitter + Button nebeneinander
        h_layout = QHBoxLayout()
        h_layout.addWidget(v_splitter)
        h_layout.addWidget(btn_container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setLayout(h_layout)
        container.setContentsMargins(0, 0, 0, 0)

        # Dock-Eigenschaften setzen
        self.setWidget(container)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setMinimumWidth(self.minimal_width)
        self.resize(self.expanded_width, self.height())

    def toggle_panel(self):
        if self.is_collapsed:
            self.setMinimumWidth(self.minimal_width)
            self.setMaximumWidth(16777215)
            self.parent_mainwindow.resizeDocks(
                [self], [self.expanded_width], Qt.Horizontal
            )
            self.project_list.show()
            self.toggle_button.setText("⮜")
        else:
            self.expanded_width = self.width()
            self.project_list.hide()
            self.setFixedWidth(self.collapsed_width)
            self.toggle_button.setText("⮞")
        self.is_collapsed = not self.is_collapsed

class TerminalOutputWidget(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("background-color: black; color: white; font-family: Consolas;")
        self.setMaximumBlockCount(1000)  # Nur letzte 1000 Zeilen behalten

    def append_text(self, text: str):
        self.insertPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class ConsoleOutput(QObject):
    new_text = Signal(str)

    def __init__(self, original_stdout):
        super().__init__()
        self._stdout = original_stdout

    def write(self, text):
        if text.strip():
            self.new_text.emit(text)
        self._stdout.write(text)

    def flush(self):
        self._stdout.flush()

class RightSidePane(QDockWidget):
    parameters_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self.setTitleBarWidget(QWidget())
        self.parent_mainwindow = parent
        self.minimal_width     = 100
        self.expanded_width    = 250
        self.collapsed_width   = 25
        self.is_collapsed      = False

        self.toggle_button = QPushButton("⮞")
        self.toggle_button.setFixedSize(15, 40)
        self.toggle_button.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding
        )
        self.toggle_button.clicked.connect(self.toggle_panel)

        btn_container = QWidget()
        b_layout      = QVBoxLayout(btn_container)
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.setSpacing(0)
        b_layout.addWidget(self.toggle_button, alignment=Qt.AlignCenter)
        btn_container.setFixedWidth(self.toggle_button.width())

        self.controls = self.create_controls()

        main_layout = QHBoxLayout()
        main_layout.addWidget(btn_container)
        main_layout.addWidget(self.controls)
        main_layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setLayout(main_layout)
        container.setAutoFillBackground(True)
        container.setPalette(self.palette())
        self.setWidget(container)

        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.setAllowedAreas(Qt.RightDockWidgetArea)
        self.setMinimumWidth(self.minimal_width)
        self.resize(self.expanded_width, self.height())

        self.connect_signals()

    def toggle_panel(self):
        if self.is_collapsed:
            self.setMinimumWidth(self.minimal_width)
            self.setMaximumWidth(16777215)
            self.parent_mainwindow.resizeDocks(
                [self], [self.expanded_width], Qt.Horizontal
            )
            self.controls.show()
            self.toggle_button.setText("⮞")
        else:
            self.expanded_width = self.width()
            self.controls.hide()
            self.setFixedWidth(self.collapsed_width)
            self.toggle_button.setText("⮜")
        self.is_collapsed = not self.is_collapsed

    def connect_signals(self):
        self.metric_combo.currentTextChanged.connect(self.emit_parameter_change)
        self.color_scale_combo.currentTextChanged.connect(self.emit_parameter_change)
        self.point_size_slider.valueChanged.connect(self.emit_parameter_change)
        self.point_size_spinbox.valueChanged.connect(self.emit_parameter_change)
        self.frustum_scale_slider.valueChanged.connect(self.emit_parameter_change)
        self.frustum_scale_spinbox.valueChanged.connect(self.emit_parameter_change)
        self.show_cameras_combo.currentIndexChanged.connect(self.emit_parameter_change)
        self.display_grid_checkbox.stateChanged.connect(self.emit_parameter_change)
        self.aggregation_combo.currentTextChanged.connect(self.emit_parameter_change)
        self.ooi_alpha_slider.valueChanged.connect(self.emit_parameter_change)

    def emit_parameter_change(self):
        params = {
            "metric":        self.metric_combo.currentText(),
            "color_scale":   self.color_scale_combo.currentText(),
            "point_size":    self.point_size_slider.value(),
            "frustum_scale": self.frustum_scale_slider.value() / 100.0,
            "show_cameras":  self.show_cameras_combo.currentText(),
            "display_grid":  self.display_grid_checkbox.isChecked(),
            "ooi_alpha": self.ooi_alpha_slider.value(),
            "aggregation_mode": self.aggregation_combo.currentText()
        }
        self.parameters_changed.emit(params)

    def create_controls(self):
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(METRICS_LIST)

        self.color_scale_combo = QComboBox()
        self.color_scale_combo.addItems(MATPLOTLIB_COLORMAPS)

        # Point-Size Slider + SpinBox (wie gehabt)
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(3)
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 10)
        self.point_size_spinbox.setValue(3)
        self.point_size_slider.valueChanged.connect(self.point_size_spinbox.setValue)
        self.point_size_spinbox.valueChanged.connect(self.point_size_slider.setValue)
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(self.point_size_slider)
        point_size_layout.addWidget(self.point_size_spinbox)
        point_size_widget = QWidget()
        point_size_widget.setLayout(point_size_layout)

        # Frustum-Scale Slider + SpinBox (jetzt eigene Zeile)
        self.frustum_scale_slider = QSlider(Qt.Horizontal)
        self.frustum_scale_slider.setRange(1, 50)  # ergibt 0.01–0.50
        self.frustum_scale_slider.setValue(5)
        self.frustum_scale_spinbox = QSpinBox()
        self.frustum_scale_spinbox.setRange(1, 50)
        self.frustum_scale_spinbox.setValue(5)
        self.frustum_scale_slider.valueChanged.connect(self.frustum_scale_spinbox.setValue)
        self.frustum_scale_spinbox.valueChanged.connect(self.frustum_scale_slider.setValue)
        frustum_layout = QHBoxLayout()
        frustum_layout.addWidget(self.frustum_scale_slider)
        frustum_layout.addWidget(self.frustum_scale_spinbox)
        frustum_widget = QWidget()
        frustum_widget.setLayout(frustum_layout)


        # Show Cameras Combo und Grid Checkbox bleiben
        self.show_cameras_combo = QComboBox()
        self.show_cameras_combo.addItems([ "Show Frustum", "Show Cameras", "None"])
        self.display_grid_checkbox = QCheckBox("Display Grid")

        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItems(["none", "min", "max", "mean"])

        # OOI Visibility Slider
        self.ooi_alpha_slider = QSlider(Qt.Horizontal)
        self.ooi_alpha_slider.setRange(0, 100)
        self.ooi_alpha_slider.setValue(10)  # Initialwert




        # Formular zusammenbauen
        form = QFormLayout()
        form.addRow("Metric", self.metric_combo)
        form.addRow("Color Scale", self.color_scale_combo)
        form.addRow("Point Size", point_size_widget)
        form.addRow("Frustum Scale", frustum_widget)  # jetzt in eigener Zeile
        form.addRow("Show Cameras", self.show_cameras_combo)
        form.addRow(self.display_grid_checkbox)
        form.addRow("Aggregation Mode", self.aggregation_combo)
        form.addRow("OOI Alpha", self.ooi_alpha_slider)

        container = QWidget()
        container.setLayout(form)
        return container

    def set_aggregation_mode(self, mode: str):
        idx = self.aggregation_combo.findText(mode)
        if idx != -1:
            self.aggregation_combo.setCurrentIndex(idx)


class ProjectListPanel(QListWidget):
    project_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.projects = []
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemClicked.connect(self.on_item_clicked)

    def add_project(self, project):
        if any(p.path == project.path for p in self.projects):
            return
        item = QListWidgetItem(f"{project.name}\n{project.path}")
        item.setData(1000, project)
        self.addItem(item)
        self.projects.append(project)

    def on_item_clicked(self, item):
        self.setCurrentItem(item)
        project = item.data(1000)
        self.project_selected.emit(project)

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        save_action = menu.addAction("Save metrics")
        action = menu.exec(self.mapToGlobal(pos))
        if action == save_action:
            project = item.data(1000)
            project.save_metric_data()

    def clear_projects(self):
        self.clear()
        self.projects = []
