import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import uproot
from matplotlib import colors, colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import PolygonSelector, RectangleSelector
from PyQt5 import QtCore, QtWidgets, QtGui


def rebin_2d(values: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, fx: int, fy: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebin a 2D histogram by integer factors along x and y."""
    if fx <= 1 and fy <= 1:
        return values, xedges, yedges

    ny, nx = values.shape
    nx_new = (nx // fx)
    ny_new = (ny // fy)
    if nx_new < 1 or ny_new < 1:
        raise ValueError("Rebin factors are too large for the histogram dimensions")

    trimmed_vals = values[: ny_new * fy, : nx_new * fx]
    rebinned = trimmed_vals.reshape(ny_new, fy, nx_new, fx).sum(axis=(1, 3))
    xedges_new = xedges[: nx_new * fx + 1 : fx]
    yedges_new = yedges[: ny_new * fy + 1 : fy]
    return rebinned, xedges_new, yedges_new


class PIDGui(QtWidgets.QMainWindow):
    def __init__(self, root_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("PID Viewer")
        self.root_path: Path | None = root_path
        self.root_file = None
        self.hist_map: dict[str, str] = {}

        self.current_hist_name: str | None = None
        self.current_hist_title: str | None = None
        self.values: np.ndarray | None = None
        self.xedges: np.ndarray | None = None
        self.yedges: np.ndarray | None = None
        self.xcenters: np.ndarray | None = None
        self.ycenters: np.ndarray | None = None

        self.display_values: np.ndarray | None = None
        self.display_xedges: np.ndarray | None = None
        self.display_yedges: np.ndarray | None = None
        self.display_xcenters: np.ndarray | None = None
        self.display_ycenters: np.ndarray | None = None
        self.current_rebin = (1, 1)
        self.axis_titles = ("X", "Y")
        self.axis_titles_base = ("X", "Y")
        self.full_xlim: Tuple[float, float] | None = None
        self.full_ylim: Tuple[float, float] | None = None

        self.polygons: List[dict] = []
        self.draw_mode = False
        self.hover_timer = QtCore.QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._show_hover_tooltip)
        self._hover_pending = False
        self._hover_event = None
        self.poly_patches: List[MplPolygon] = []
        self.current_color = "red"
        self.zoom_selector = None
        self.entries_text = None

        self._build_ui()
        self._connect_events()
        self._init_selector()
        self._set_hist_controls_enabled(False)
        self._update_root_label()
        if root_path is not None:
            self._load_root_file(root_path)

    @staticmethod
    def _axis_title(axis, default: str) -> str:
        # uproot TAxis may expose fTitle or title depending on version
        return getattr(axis, "fTitle", None) or getattr(axis, "title", None) or default

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: controls
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        splitter.addWidget(control_widget)

        self.open_root_btn = QtWidgets.QPushButton("Open ROOT file")
        control_layout.addWidget(self.open_root_btn)

        self.root_label = QtWidgets.QLabel("No ROOT file loaded")
        self.root_label.setWordWrap(True)
        control_layout.addWidget(self.root_label)

        self.list_widget = QtWidgets.QListWidget()
        control_layout.addWidget(self.list_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load selected")
        self.clear_polys_btn = QtWidgets.QPushButton("Clear polygons")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.clear_polys_btn)
        control_layout.addLayout(btn_layout)

        self.log_checkbox = QtWidgets.QCheckBox("Log scale (Z)")
        control_layout.addWidget(self.log_checkbox)

        self.swap_checkbox = QtWidgets.QCheckBox()
        self.swap_checkbox.setChecked(False)
        self._update_swap_label()
        control_layout.addWidget(self.swap_checkbox)

        rebin_form = QtWidgets.QFormLayout()
        self.rebin_x = QtWidgets.QSpinBox()
        self.rebin_y = QtWidgets.QSpinBox()
        self.rebin_z = QtWidgets.QSpinBox()
        for spin in (self.rebin_x, self.rebin_y):
            spin.setMinimum(1)
            spin.setMaximum(1000)
            spin.setValue(1)
        rebin_form.addRow("Rebin X", self.rebin_x)
        rebin_form.addRow("Rebin Y", self.rebin_y)
        self.rebin_z.setMinimum(2)
        self.rebin_z.setMaximum(2048)
        self.rebin_z.setValue(256)
        rebin_form.addRow("Bins Z", self.rebin_z)
        control_layout.addLayout(rebin_form)

        self.apply_rebin_btn = QtWidgets.QPushButton("Apply binning")
        control_layout.addWidget(self.apply_rebin_btn)

        self.export_btn = QtWidgets.QPushButton("Export polygons to TXT")
        control_layout.addWidget(self.export_btn)

        mode_layout = QtWidgets.QHBoxLayout()
        self.mode_select_radio = QtWidgets.QRadioButton("Select/zoom")
        self.mode_draw_radio = QtWidgets.QRadioButton("Draw polygons")
        self.mode_select_radio.setChecked(True)
        mode_layout.addWidget(self.mode_select_radio)
        mode_layout.addWidget(self.mode_draw_radio)
        control_layout.addLayout(mode_layout)

        control_layout.addWidget(QtWidgets.QLabel("Polygons"))
        self.poly_list = QtWidgets.QListWidget()
        self.poly_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        control_layout.addWidget(self.poly_list)

        name_layout = QtWidgets.QHBoxLayout()
        self.poly_name_edit = QtWidgets.QLineEdit()
        self.rename_poly_btn = QtWidgets.QPushButton("Rename")
        name_layout.addWidget(self.poly_name_edit)
        name_layout.addWidget(self.rename_poly_btn)
        control_layout.addLayout(name_layout)

        color_layout = QtWidgets.QHBoxLayout()
        self.set_color_btn = QtWidgets.QPushButton("Set color")
        self.new_poly_btn = QtWidgets.QPushButton("Start new polygon")
        color_layout.addWidget(self.set_color_btn)
        color_layout.addWidget(self.new_poly_btn)
        control_layout.addLayout(color_layout)

        clear_layout = QtWidgets.QHBoxLayout()
        self.clear_selected_btn = QtWidgets.QPushButton("Clear selected")
        self.clear_all_btn = QtWidgets.QPushButton("Clear all")
        clear_layout.addWidget(self.clear_selected_btn)
        clear_layout.addWidget(self.clear_all_btn)
        control_layout.addLayout(clear_layout)

        control_layout.addStretch(1)

        # Right panel: matplotlib canvas
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        splitter.addWidget(plot_widget)

        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        splitter.setSizes([250, 800])

    def _connect_events(self) -> None:
        self.open_root_btn.clicked.connect(self.choose_root_file)
        self.load_btn.clicked.connect(self.load_selected_histogram)
        self.apply_rebin_btn.clicked.connect(self.redraw_histogram)
        self.log_checkbox.stateChanged.connect(self.redraw_histogram)
        self.swap_checkbox.stateChanged.connect(self.redraw_histogram)
        self.export_btn.clicked.connect(self.export_polygons)
        self.clear_polys_btn.clicked.connect(self.clear_polygons)
        self.list_widget.itemDoubleClicked.connect(lambda _: self.load_selected_histogram())
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_canvas_menu)

        undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_polygon)

        self.poly_list.currentRowChanged.connect(self._select_polygon_from_list)
        self.rename_poly_btn.clicked.connect(self.rename_selected_polygon)
        self.set_color_btn.clicked.connect(self.change_selected_polygon_color)
        self.new_poly_btn.clicked.connect(self.start_new_polygon)
        self.clear_selected_btn.clicked.connect(self.clear_selected_polygon)
        self.clear_all_btn.clicked.connect(self.clear_polygons)
        self.mode_draw_radio.toggled.connect(lambda checked: self.set_draw_mode(checked))

    def _update_swap_label(self) -> None:
        x_label, y_label = self.axis_titles_base
        self.swap_checkbox.setText(f"Swap axes ({x_label} <-> {y_label})")

    def _update_root_label(self) -> None:
        if self.root_path is None:
            self.root_label.setText("No ROOT file loaded")
        else:
            self.root_label.setText(f"ROOT file:\n{self.root_path}")

    def _set_hist_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.list_widget,
            self.load_btn,
            self.clear_polys_btn,
            self.log_checkbox,
            self.swap_checkbox,
            self.rebin_x,
            self.rebin_y,
            self.rebin_z,
            self.apply_rebin_btn,
            self.export_btn,
            self.mode_select_radio,
            self.mode_draw_radio,
            self.poly_list,
            self.poly_name_edit,
            self.rename_poly_btn,
            self.set_color_btn,
            self.new_poly_btn,
            self.clear_selected_btn,
            self.clear_all_btn,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def choose_root_file(self) -> None:
        start_dir = str(self.root_path.parent) if self.root_path else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open ROOT file",
            start_dir,
            "ROOT files (*.root);;All files (*.*)",
        )
        if not path:
            return
        self._load_root_file(Path(path))

    def _reset_histogram_state(self) -> None:
        self.current_hist_name = None
        self.current_hist_title = None
        self.values = None
        self.xedges = None
        self.yedges = None
        self.xcenters = None
        self.ycenters = None
        self.display_values = None
        self.display_xedges = None
        self.display_yedges = None
        self.display_xcenters = None
        self.display_ycenters = None
        self.axis_titles = ("X", "Y")
        self.axis_titles_base = ("X", "Y")
        self.full_xlim = None
        self.full_ylim = None
        self.rebin_x.setValue(1)
        self.rebin_y.setValue(1)
        self.rebin_z.setValue(256)
        self.clear_polygons()
        self.list_widget.clear()
        self.canvas.draw_idle()

    def _load_root_file(self, root_path: Path) -> None:
        if not root_path.exists():
            self.status.showMessage(f"File not found: {root_path}", 5000)
            return
        try:
            root_file = uproot.open(root_path)
        except Exception as exc:
            self.status.showMessage(f"Failed to open ROOT file: {exc}", 6000)
            return

        if self.root_file is not None:
            try:
                self.root_file.close()
            except Exception:
                pass

        self.root_file = root_file
        self.root_path = root_path
        self.hist_map = {key.split(";")[0]: key for key in self.root_file.keys()}
        self._reset_histogram_state()
        self.list_widget.addItems(sorted(self.hist_map.keys()))
        self._update_root_label()
        self._update_swap_label()
        self._set_hist_controls_enabled(True)
        self.status.showMessage(f"Opened {root_path.name} ({len(self.hist_map)} histograms)", 5000)

    def _init_selector(self) -> None:
        self.selector = PolygonSelector(
            self.ax,
            self._on_polygon_complete,
            useblit=True,
            props={"color": self.current_color, "linewidth": 1.5, "alpha": 0.9},
            handle_props={"marker": "o", "markersize": 4, "mec": "red", "mfc": "white"},
        )
        self.selector.set_active(self.draw_mode)

    def _init_zoom_selector(self) -> None:
        self.zoom_selector = RectangleSelector(
            self.ax,
            self._on_zoom,
            useblit=True,
            button=[1],
            interactive=False,
            minspanx=1,
            minspany=1,
            spancoords="data",
        )
        self.zoom_selector.set_active(not self.draw_mode)

    def _ensure_main_axes(self) -> None:
        # Keep a single main axes; drop any extras and reattach selector if needed.
        if self.ax not in self.fig.axes:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.selector = None
        else:
            for ax in list(self.fig.axes):
                if ax is not self.ax:
                    self.fig.delaxes(ax)
        if self.selector is None or self.selector.ax is not self.ax:
            try:
                if self.selector is not None:
                    self.selector.disconnect_events()
            except Exception:
                pass
            self._init_selector()

    def load_selected_histogram(self) -> None:
        if self.root_file is None:
            self.status.showMessage("Open a ROOT file first", 4000)
            return
        item = self.list_widget.currentItem()
        if not item:
            self.status.showMessage("Select a histogram first", 4000)
            return
        name = item.text()
        self.current_hist_name = name
        full_key = self.hist_map[name]
        hist = self.root_file[full_key]
        self.current_hist_title = getattr(hist, "title", None)

        # uproot returns TH2 values shaped (axis0_bins, axis1_bins). Transpose so the first
        # dimension (Y on the plot) follows axis1—whose minimum is 0 here—and the second
        # dimension (X) follows axis0.
        vals = hist.values(flow=False).T
        xedges = hist.axes[0].edges()
        yedges = hist.axes[1].edges()
        self.values, self.xedges, self.yedges = vals, xedges, yedges
        self.xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        self.ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        self.axis_titles_base = (
            self._axis_title(hist.axes[0], "X"),
            self._axis_title(hist.axes[1], "Y"),
        )
        self._update_swap_label()

        self.rebin_x.setValue(1)
        self.rebin_y.setValue(1)
        self.clear_polygons()
        self.redraw_histogram()
        axis_msg = f"X={self.axis_titles_base[0]} | Y={self.axis_titles_base[1]}"
        title_msg = self.current_hist_title or name
        self.status.showMessage(f"Loaded {title_msg} ({axis_msg})", 4000)

    def redraw_histogram(self) -> None:
        if self.values is None:
            return

        fx = max(1, self.rebin_x.value())
        fy = max(1, self.rebin_y.value())
        if (fx, fy) != self.current_rebin:
            self.clear_polygons()
            self.current_rebin = (fx, fy)
        lost_x = (self.values.shape[1] % fx) if self.values is not None else 0
        lost_y = (self.values.shape[0] % fy) if self.values is not None else 0
        try:
            vals, xedges, yedges = rebin_2d(self.values, self.xedges, self.yedges, fx, fy)
        except ValueError as exc:
            self.status.showMessage(str(exc), 5000)
            return
        if lost_x or lost_y:
            self.status.showMessage(
                f"Rebin dropped {lost_x} x-bins and {lost_y} y-bins (not divisible)",
                5000,
            )

        swap_axes = self.swap_checkbox.isChecked()
        if swap_axes:
            vals_disp = vals.T
            xedges_disp, yedges_disp = yedges, xedges
            axis_titles_disp = (self.axis_titles_base[1], self.axis_titles_base[0])
        else:
            vals_disp = vals
            xedges_disp, yedges_disp = xedges, yedges
            axis_titles_disp = self.axis_titles_base

        vals_disp, xedges_disp, yedges_disp = self._trim_empty_edges(vals_disp, xedges_disp, yedges_disp)

        self.display_values = vals_disp
        self.display_xedges = xedges_disp
        self.display_yedges = yedges_disp
        self.display_xcenters = 0.5 * (xedges_disp[:-1] + xedges_disp[1:])
        self.display_ycenters = 0.5 * (yedges_disp[:-1] + yedges_disp[1:])
        self.axis_titles = axis_titles_disp
        self.full_xlim = (xedges_disp[0], xedges_disp[-1])
        self.full_ylim = (yedges_disp[0], yedges_disp[-1])

        # Hard reset figure to avoid accumulating extra axes/colorbars from previous loads
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None
        self.selector = None
        self.zoom_selector = None
        self.poly_patches.clear()
        self._init_selector()
        self._init_zoom_selector()
        norm = None
        cmap = colormaps.get_cmap("viridis").copy()
        z_bins = max(0, self.rebin_z.value() if hasattr(self, "rebin_z") else 0)
        if self.log_checkbox.isChecked():
            data_for_norm = vals_disp[vals_disp > 0]
            vmin = data_for_norm.min() if data_for_norm.size else 1
            norm = colors.LogNorm(vmin=vmin, vmax=vals_disp.max() if vals_disp.size else None)
            image_data = np.ma.masked_less_equal(vals_disp, 0)
            if z_bins >= 2 and norm.vmax and norm.vmin:
                levels = np.geomspace(norm.vmin, norm.vmax, z_bins + 1)
                norm = colors.BoundaryNorm(levels, cmap.N)
        else:
            image_data = np.ma.masked_equal(vals_disp, 0)
            if z_bins >= 2 and vals_disp.size:
                vmin_lin = float(np.nanmin(image_data.filled(0))) if np.ma.is_masked(image_data) else float(vals_disp.min())
                vmax_lin = float(np.nanmax(image_data.filled(0))) if np.ma.is_masked(image_data) else float(vals_disp.max())
                if vmax_lin > vmin_lin:
                    levels = np.linspace(vmin_lin, vmax_lin, z_bins + 1)
                    norm = colors.BoundaryNorm(levels, cmap.N)
        cmap.set_bad("white")

        extent = [xedges_disp[0], xedges_disp[-1], yedges_disp[0], yedges_disp[-1]]
        im = self.ax.imshow(
            image_data,
            origin="lower",
            aspect="auto",
            extent=extent,
            norm=norm,
            interpolation="nearest",
            cmap=cmap,
        )

        # reset limits to current full extents (needed after swaps/zooms)
        if self.full_xlim:
            self.ax.set_xlim(*self.full_xlim)
        if self.full_ylim:
            self.ax.set_ylim(*self.full_ylim)

        if self.colorbar:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            try:
                if self.colorbar.ax in self.fig.axes:
                    self.fig.delaxes(self.colorbar.ax)
            except Exception:
                pass
            self.colorbar = None
        self.colorbar = self.fig.colorbar(im, ax=self.ax, label="Counts")

        # entries label (top-right, figure coords)
        if self.entries_text is not None:
            try:
                self.entries_text.remove()
            except Exception:
                pass
            self.entries_text = None
        total_counts = float(np.nansum(self.display_values)) if self.display_values is not None else 0.0
        self.entries_text = self.fig.text(0.98, 0.98, f"Entries: {total_counts:,.0f}", ha="right", va="top")

        self.ax.set_xlabel(self.axis_titles[0])
        self.ax.set_ylabel(self.axis_titles[1])
        self.ax.set_title(self.current_hist_title or self.current_hist_name or "")

        # ensure mode toggles respect selectors
        self.set_draw_mode(self.draw_mode, quiet=True)

        # redraw existing polygon patches
        for patch in self.poly_patches:
            patch.remove()
        self.poly_patches.clear()
        for poly in self.polygons:
            patch = MplPolygon(poly["vertices"], closed=True, fill=False, edgecolor=poly.get("color", "red"), linewidth=1.5)
            self.ax.add_patch(patch)
            self.poly_patches.append(patch)

        self._update_polygon_highlight()
        self.canvas.draw_idle()

    def clear_polygons(self) -> None:
        if hasattr(self, "selector") and self.selector is not None:
            try:
                self.selector.clear()
            except Exception:
                pass
        for patch in self.poly_patches:
            patch.remove()
        self.poly_patches.clear()
        self.polygons.clear()
        self.poly_list.clear()
        self.poly_name_edit.clear()
        if self.display_values is not None:
            # redraw to ensure any remaining artists are cleared
            self.redraw_histogram()
        else:
            self.canvas.draw_idle()

    def _refresh_polygon_list(self, select_index: int | None = None) -> None:
        self.poly_list.blockSignals(True)
        self.poly_list.clear()
        for idx, poly in enumerate(self.polygons, 1):
            name = poly.get("name", f"Polygon {idx}")
            item = QtWidgets.QListWidgetItem(f"{idx}: {name}")
            self.poly_list.addItem(item)
        if select_index is not None and 0 <= select_index < self.poly_list.count():
            self.poly_list.setCurrentRow(select_index)
        self.poly_list.blockSignals(False)
        self._update_polygon_highlight()

    def _selected_polygon_index(self) -> int | None:
        row = self.poly_list.currentRow()
        if 0 <= row < len(self.polygons):
            return row
        if self.polygons:
            return len(self.polygons) - 1
        return None

    def _update_polygon_highlight(self) -> None:
        selected = self._selected_polygon_index()
        for idx, patch in enumerate(self.poly_patches):
            if patch is None:
                continue
            patch.set_linewidth(2.0 if selected == idx else 1.5)
        if selected is not None and selected < len(self.polygons):
            self.poly_name_edit.setText(self.polygons[selected].get("name", ""))
        else:
            self.poly_name_edit.clear()
        self.canvas.draw_idle()

    def _on_polygon_complete(self, verts: List[Tuple[float, float]]) -> None:
        if not self.draw_mode:
            return
        if self.display_values is None or self.display_xedges is None or self.display_yedges is None:
            return
        verts = np.asarray(verts)
        path = MplPath(verts)
        xx, yy = np.meshgrid(self.display_xcenters, self.display_ycenters)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        mask = path.contains_points(points).reshape(self.display_values.shape)
        total = float(self.display_values[mask].sum())
        nbins = int(mask.sum())

        patch = MplPolygon(verts, closed=True, fill=False, edgecolor=self.current_color, linewidth=1.5)
        self.ax.add_patch(patch)
        self.poly_patches.append(patch)

        poly_info = {
            "hist": self.current_hist_name,
            "vertices": verts.tolist(),
            "sum_counts": total,
            "bins": nbins,
            "rebin": (self.rebin_x.value(), self.rebin_y.value()),
            "log": self.log_checkbox.isChecked(),
            "color": self.current_color,
            "name": f"Polygon {len(self.polygons) + 1}",
        }
        self.polygons.append(poly_info)
        self._refresh_polygon_list(select_index=len(self.polygons) - 1)
        self.canvas.draw_idle()
        self.status.showMessage(f"Polygon added: {nbins} bins, sum={total:.3g}", 5000)

    def _on_button_press(self, event) -> None:
        if event.button == 3:
            if self.colorbar and event.inaxes == getattr(self.colorbar, "ax", None):
                self._show_colorbar_menu()
                return
            # enlarge hit area around axes: use bbox with small margins
            if self.ax is not None and event.x is not None and event.y is not None:
                bbox = self.ax.get_window_extent()
                margin = 20  # pixels
                within_xaxis = bbox.xmin <= event.x <= bbox.xmax and (bbox.ymin - margin) <= event.y <= (bbox.ymin + margin)
                within_yaxis = (bbox.xmin - margin) <= event.x <= (bbox.xmin + margin) and bbox.ymin <= event.y <= bbox.ymax
                if within_xaxis:
                    self._show_axis_menu(axis="x")
                    return
                if within_yaxis:
                    self._show_axis_menu(axis="y")
                    return
            return
        if event.button != 1:
            return
        self._on_click(event)

    def _on_click(self, event) -> None:
        if self.display_values is None or self.display_xedges is None or self.display_yedges is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        xi = np.searchsorted(self.display_xedges, event.xdata) - 1
        yi = np.searchsorted(self.display_yedges, event.ydata) - 1
        if xi < 0 or yi < 0 or xi >= self.display_values.shape[1] or yi >= self.display_values.shape[0]:
            return
        val = float(self.display_values[yi, xi])
        xc = 0.5 * (self.display_xedges[xi] + self.display_xedges[xi + 1])
        yc = 0.5 * (self.display_yedges[yi] + self.display_yedges[yi + 1])
        # Show actual axis coords from the plot (mouse position) and the bin center for context
        self.status.showMessage(
            f"Bin ({xi}, {yi}) x={event.xdata:.3g}, y={event.ydata:.3g} (center=({xc:.3g}, {yc:.3g})) value={val:.3g}",
            4000,
        )

    def _on_zoom(self, eclick, erelease) -> None:
        if self.draw_mode:
            return
        if eclick is None or erelease is None:
            return
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        if abs(x0 - x1) < 1e-12 or abs(y0 - y1) < 1e-12:
            return
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
        self.status.showMessage(f"Zoom to X=[{xmin:.3g}, {xmax:.3g}], Y=[{ymin:.3g}, {ymax:.3g}]", 3000)

    def _set_axis_scale(self, axis: str, mode: str) -> None:
        if axis not in ("x", "y"):
            return
        edges = self.display_xedges if axis == "x" else self.display_yedges
        if mode in ("log10", "ln"):
            if edges is None or np.any(edges <= 0):
                self.status.showMessage(f"Cannot set log scale on {axis.upper()} (non-positive values)", 4000)
                return
            base = 10.0 if mode == "log10" else np.e
            try:
                getattr(self.ax, f"set_{axis}scale")("log", base=base)
            except TypeError:
                getattr(self.ax, f"set_{axis}scale")("log")
        else:
            getattr(self.ax, f"set_{axis}scale")("linear")
        self.canvas.draw_idle()
        self.status.showMessage(f"Set {axis.upper()} scale to {mode}", 2000)

    def _show_canvas_menu(self, pos) -> None:
        menu = QtWidgets.QMenu(self)
        new_poly_action = menu.addAction("Start new polygon")
        color_action = menu.addAction("Change current polygon color")
        undo_action = menu.addAction("Undo last polygon")
        unzoom_action = menu.addAction("Unzoom")
        action = menu.exec_(self.canvas.mapToGlobal(pos))
        if action == new_poly_action:
            self.start_new_polygon()
        elif action == color_action:
            self.change_current_polygon_color()
        elif action == undo_action:
            self.undo_polygon()
        elif action == unzoom_action:
            self.reset_zoom()

    def _show_axis_menu(self, axis: str) -> None:
        menu = QtWidgets.QMenu(self)
        lin = menu.addAction(f"{axis.upper()}: linear")
        log10 = menu.addAction(f"{axis.upper()}: log10")
        ln = menu.addAction(f"{axis.upper()}: log")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen == lin:
            self._set_axis_scale(axis=axis, mode="linear")
        elif chosen == log10:
            self._set_axis_scale(axis=axis, mode="log10")
        elif chosen == ln:
            self._set_axis_scale(axis=axis, mode="ln")

    def _show_colorbar_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        z_lin = menu.addAction("Z: linear")
        z_log = menu.addAction("Z: log10")
        z_ln = menu.addAction("Z: log")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen == z_lin:
            self.log_checkbox.setChecked(False)
            self.redraw_histogram()
        elif chosen == z_log:
            self.log_checkbox.setChecked(True)
            self.redraw_histogram()
        elif chosen == z_ln:
            self.log_checkbox.setChecked(True)
            self.redraw_histogram()

    def start_new_polygon(self) -> None:
        self.set_draw_mode(True, quiet=True)
        if self.selector is not None:
            try:
                self.selector.clear()
            except Exception:
                pass
            self.selector.set_active(True)
        self.status.showMessage("Polygon drawing reset", 2000)

    def change_current_polygon_color(self) -> None:
        idx = self._selected_polygon_index()
        if idx is None:
            self.status.showMessage("No polygon to recolor", 3000)
            return
        color = QtWidgets.QColorDialog.getColor(parent=self, title="Choose polygon color")
        if not color.isValid():
            return
        hex_color = color.name()
        self._apply_polygon_color(idx, hex_color)

    def undo_polygon(self) -> None:
        if not self.polygons or not self.poly_patches:
            self.status.showMessage("No polygons to undo", 3000)
            return
        # Remove last polygon and corresponding patch; if patches are out of sync, rebuild
        if self.poly_patches:
            patch = self.poly_patches.pop()
            try:
                patch.remove()
            except Exception:
                pass
        self.polygons.pop()
        if len(self.poly_patches) < len(self.polygons):
            self.redraw_histogram()
        else:
            self._refresh_polygon_list(select_index=len(self.polygons) - 1 if self.polygons else None)
            self.canvas.draw_idle()
        self.status.showMessage("Last polygon removed", 3000)

    def reset_zoom(self) -> None:
        if self.full_xlim and self.full_ylim:
            self.ax.set_xlim(*self.full_xlim)
            self.ax.set_ylim(*self.full_ylim)
            self.canvas.draw_idle()
            self.status.showMessage("Zoom reset", 2000)
        else:
            self.ax.autoscale()
            self.canvas.draw_idle()
            self.status.showMessage("Zoom reset (auto)", 2000)

    def set_draw_mode(self, draw: bool, quiet: bool = False) -> None:
        self.draw_mode = draw
        if self.mode_draw_radio.isChecked() != draw:
            self.mode_draw_radio.setChecked(draw)
            self.mode_select_radio.setChecked(not draw)
        if self.selector is not None:
            self.selector.set_active(draw)
        if self.zoom_selector is not None:
            self.zoom_selector.set_active(not draw)
        if not quiet:
            self.status.showMessage("Draw mode" if draw else "Select/zoom mode", 2000)

    def _apply_polygon_color(self, idx: int, hex_color: str) -> None:
        if idx < 0 or idx >= len(self.polygons):
            return
        self.polygons[idx]["color"] = hex_color
        self.current_color = hex_color

        # Preserve current zoom when recoloring
        xlim = None
        ylim = None
        if self.ax is not None:
            try:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
            except Exception:
                xlim = ylim = None

        # Rebuild so the patch definitely picks up the new color
        self.redraw_histogram()

        # Restore zoom if we captured it and still have valid edges
        if xlim and ylim and self.display_xedges is not None and self.display_yedges is not None:
            xmin_full, xmax_full = self.display_xedges[0], self.display_xedges[-1]
            ymin_full, ymax_full = self.display_yedges[0], self.display_yedges[-1]
            new_xlim = (
                max(xmin_full, min(xlim)),
                min(xmax_full, max(xlim)),
            )
            new_ylim = (
                max(ymin_full, min(ylim)),
                min(ymax_full, max(ylim)),
            )
            try:
                self.ax.set_xlim(*new_xlim)
                self.ax.set_ylim(*new_ylim)
            except Exception:
                pass

        self._refresh_polygon_list(select_index=idx)
        self.status.showMessage(f"Updated polygon color to {hex_color}", 3000)

    def _compute_bin_info(self, x: float, y: float) -> Tuple[int, int, float, float, float] | None:
        if (
            self.display_values is None
            or self.display_xedges is None
            or self.display_yedges is None
            or x is None
            or y is None
        ):
            return None
        xi = np.searchsorted(self.display_xedges, x) - 1
        yi = np.searchsorted(self.display_yedges, y) - 1
        if xi < 0 or yi < 0 or xi >= self.display_values.shape[1] or yi >= self.display_values.shape[0]:
            return None
        val = float(self.display_values[yi, xi])
        xc = 0.5 * (self.display_xedges[xi] + self.display_xedges[xi + 1])
        yc = 0.5 * (self.display_yedges[yi] + self.display_yedges[yi + 1])
        return xi, yi, xc, yc, val

    def _on_motion(self, event) -> None:
        # Reset hover timer if moving out of axes or missing data
        if (
            event is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
            or self.display_values is None
        ):
            self.hover_timer.stop()
            QtWidgets.QToolTip.hideText()
            self._hover_pending = False
            return
        self._hover_event = (event.xdata, event.ydata, QtGui.QCursor.pos())
        self._hover_pending = True
        self.hover_timer.start(1000)

    def _show_hover_tooltip(self) -> None:
        if not self._hover_pending or self._hover_event is None:
            return
        x, y, pos = self._hover_event
        info = self._compute_bin_info(x, y)
        if info is None:
            return
        xi, yi, xc, yc, val = info
        # Use the actual axis coordinates under the cursor, not just bin centers
        text = f"Bin ({xi}, {yi})\nX={x:.3g}\nY={y:.3g}\nCenter=({xc:.3g}, {yc:.3g})\nZ={val:.3g}"
        QtWidgets.QToolTip.showText(pos, text, self)
        self._hover_pending = False

    def _trim_empty_edges(
        self, values: np.ndarray, xedges: np.ndarray, yedges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vals = np.asarray(values)
        if vals.size == 0:
            return values, xedges, yedges
        if np.ma.isMaskedArray(vals):
            vals = vals.filled(0)

        x_sum = vals.sum(axis=0)
        y_sum = vals.sum(axis=1)

        def _bounds(sums: np.ndarray, max_len: int) -> Tuple[int, int]:
            nonzero = np.nonzero(sums)[0]
            if nonzero.size == 0:
                return 0, max_len - 1
            return int(nonzero.min()), int(nonzero.max())

        x0, x1 = _bounds(x_sum, vals.shape[1])
        y0, y1 = _bounds(y_sum, vals.shape[0])

        vals_trim = vals[y0 : y1 + 1, x0 : x1 + 1]
        xedges_trim = xedges[x0 : x1 + 2]
        yedges_trim = yedges[y0 : y1 + 2]
        return vals_trim, xedges_trim, yedges_trim

    def _select_polygon_from_list(self, row: int) -> None:
        self._update_polygon_highlight()

    def rename_selected_polygon(self) -> None:
        idx = self._selected_polygon_index()
        if idx is None:
            self.status.showMessage("No polygon selected", 3000)
            return
        new_name = self.poly_name_edit.text().strip() or f"Polygon {idx + 1}"
        self.polygons[idx]["name"] = new_name
        self._refresh_polygon_list(select_index=idx)
        self.status.showMessage(f"Renamed polygon to {new_name}", 2000)

    def change_selected_polygon_color(self) -> None:
        idx = self._selected_polygon_index()
        if idx is None:
            self.status.showMessage("No polygon selected", 3000)
            return
        color = QtWidgets.QColorDialog.getColor(parent=self, title="Choose polygon color")
        if not color.isValid():
            return
        hex_color = color.name()
        self._apply_polygon_color(idx, hex_color)

    def clear_selected_polygon(self) -> None:
        idx = self._selected_polygon_index()
        if idx is None:
            self.status.showMessage("No polygon selected", 3000)
            return
        if idx < len(self.poly_patches):
            try:
                self.poly_patches[idx].remove()
            except Exception:
                pass
            del self.poly_patches[idx]
        del self.polygons[idx]
        self._refresh_polygon_list(select_index=min(idx, len(self.polygons) - 1))
        self.status.showMessage("Polygon removed", 3000)

    @staticmethod
    def _sanitize_poly_name(name: str, fallback: str) -> str:
        # Keep simple ASCII-safe identifiers for the export key
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip())
        safe = safe.strip("_")
        return safe or fallback

    @staticmethod
    def _format_polygon_key(hist_name: str | None, poly_name: str, idx: int) -> str:
        # Export key should just be the polygon name (sanitized)
        return PIDGui._sanitize_poly_name(poly_name or f"polygon_{idx}", f"polygon_{idx}")

    def export_polygons(self) -> None:
        if not self.polygons:
            self.status.showMessage("No polygons to export", 4000)
            return
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        default_name = outputs_dir / f"{self.current_hist_name or 'pid'}_polygons.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save polygons",
            str(default_name),
            "Text Files (*.txt)",
        )
        if not path:
            return
        header = [
            "# =============================================",
            "# Initial polygons for cut seeding (rigidity, dE/dx)",
            "# Stored as semicolon-separated x,y point pairs (closed polygons repeat first point)",
            "# =============================================",
            f"# ROOT file: {self.root_path}",
            f"# Histogram: {self.current_hist_name or ''}",
            "",
        ]

        lines = list(header)
        for idx, poly in enumerate(self.polygons, 1):
            name = poly.get("name", f"Polygon {idx}")
            key = self._format_polygon_key(self.current_hist_name, name, idx)
            vertices = list(poly.get("vertices", []))
            if vertices and (vertices[0] != vertices[-1]):
                vertices = vertices + [vertices[0]]
            coord_str = ";".join(f"{x},{y}" for x, y in vertices)
            lines.append(f"# {name}")
            lines.append(f"{key} = {coord_str}")
            lines.append("")

        Path(path).write_text("\n".join(lines), encoding="ascii")
        self.status.showMessage(f"Saved {len(self.polygons)} polygons to {path}", 5000)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GUI PID viewer for ROOT histograms")
    parser.add_argument("root_file", nargs="?", help="Path to ROOT file")
    args = parser.parse_args()

    root_path = Path(args.root_file) if args.root_file else None
    if root_path is not None and not root_path.exists():
        raise SystemExit(f"File not found: {root_path}")

    app = QtWidgets.QApplication(sys.argv)
    gui = PIDGui(root_path)
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
