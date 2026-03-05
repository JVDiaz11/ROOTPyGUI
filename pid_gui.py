import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import uproot
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import PolygonSelector
from PyQt5 import QtCore, QtWidgets


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
    def __init__(self, root_path: Path):
        super().__init__()
        self.setWindowTitle("PID Viewer")
        self.root_path = root_path
        self.root_file = uproot.open(root_path)
        self.hist_map = {key.split(";")[0]: key for key in self.root_file.keys()}

        self.current_hist_name: str | None = None
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

        self.polygons: List[dict] = []
        self.poly_patches: List[MplPolygon] = []

        self._build_ui()
        self._connect_events()
        self._init_selector()

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

        control_layout.addWidget(QtWidgets.QLabel(f"ROOT file:\n{self.root_path}"))

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.addItems(sorted(self.hist_map.keys()))
        control_layout.addWidget(self.list_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load selected")
        self.clear_polys_btn = QtWidgets.QPushButton("Clear polygons")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.clear_polys_btn)
        control_layout.addLayout(btn_layout)

        self.log_checkbox = QtWidgets.QCheckBox("Log scale (Z)")
        control_layout.addWidget(self.log_checkbox)

        self.swap_checkbox = QtWidgets.QCheckBox("Swap X/Y")
        self.swap_checkbox.setChecked(True)
        control_layout.addWidget(self.swap_checkbox)

        rebin_form = QtWidgets.QFormLayout()
        self.rebin_x = QtWidgets.QSpinBox()
        self.rebin_y = QtWidgets.QSpinBox()
        for spin in (self.rebin_x, self.rebin_y):
            spin.setMinimum(1)
            spin.setMaximum(1000)
            spin.setValue(1)
        rebin_form.addRow("Rebin X", self.rebin_x)
        rebin_form.addRow("Rebin Y", self.rebin_y)
        control_layout.addLayout(rebin_form)

        self.apply_rebin_btn = QtWidgets.QPushButton("Apply binning")
        control_layout.addWidget(self.apply_rebin_btn)

        self.export_btn = QtWidgets.QPushButton("Export polygons to TXT")
        control_layout.addWidget(self.export_btn)

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
        self.load_btn.clicked.connect(self.load_selected_histogram)
        self.apply_rebin_btn.clicked.connect(self.redraw_histogram)
        self.log_checkbox.stateChanged.connect(self.redraw_histogram)
        self.swap_checkbox.stateChanged.connect(self.redraw_histogram)
        self.export_btn.clicked.connect(self.export_polygons)
        self.clear_polys_btn.clicked.connect(self.clear_polygons)
        self.list_widget.itemDoubleClicked.connect(lambda _: self.load_selected_histogram())
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _init_selector(self) -> None:
        self.selector = PolygonSelector(
            self.ax,
            self._on_polygon_complete,
            useblit=True,
            props={"color": "red", "linewidth": 1.5, "alpha": 0.9},
            handle_props={"marker": "o", "markersize": 4, "mec": "red", "mfc": "white"},
        )
        self.selector.set_active(True)

    def _remove_extra_axes(self) -> None:
        # Ensure we only keep the main plotting axes; drop leftover colorbar axes.
        for ax in list(self.fig.axes):
            if ax is not self.ax:
                self.fig.delaxes(ax)
        self.colorbar = None

    def load_selected_histogram(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            self.status.showMessage("Select a histogram first", 4000)
            return
        name = item.text()
        self.current_hist_name = name
        full_key = self.hist_map[name]
        hist = self.root_file[full_key]

        vals = hist.values(flow=False)
        xedges = hist.axes[0].edges()
        yedges = hist.axes[1].edges()
        self.values, self.xedges, self.yedges = vals, xedges, yedges
        self.xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        self.ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        self.axis_titles_base = (
            self._axis_title(hist.axes[0], "X"),
            self._axis_title(hist.axes[1], "Y"),
        )

        self.rebin_x.setValue(1)
        self.rebin_y.setValue(1)
        self.clear_polygons()
        self.redraw_histogram()
        self.status.showMessage(f"Loaded {name}", 3000)

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

        self.display_values = vals_disp
        self.display_xedges = xedges_disp
        self.display_yedges = yedges_disp
        self.display_xcenters = 0.5 * (xedges_disp[:-1] + xedges_disp[1:])
        self.display_ycenters = 0.5 * (yedges_disp[:-1] + yedges_disp[1:])
        self.axis_titles = axis_titles_disp

        self._remove_extra_axes()
        self.ax.clear()
        norm = None
        if self.log_checkbox.isChecked():
            positive = vals_disp[vals_disp > 0]
            vmin = positive.min() if positive.size else 1
            norm = colors.LogNorm(vmin=vmin, vmax=vals_disp.max() if vals_disp.size else None)

        extent = [xedges_disp[0], xedges_disp[-1], yedges_disp[0], yedges_disp[-1]]
        im = self.ax.imshow(
            vals_disp,
            origin="lower",
            aspect="auto",
            extent=extent,
            norm=norm,
            interpolation="nearest",
        )

        if self.colorbar:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            try:
                self.colorbar.ax.remove()
            except Exception:
                pass
            self.colorbar = None
        self.colorbar = self.fig.colorbar(im, ax=self.ax, label="Counts")

        self.ax.set_xlabel(self.axis_titles[0])
        self.ax.set_ylabel(self.axis_titles[1])
        self.ax.set_title(self.current_hist_name or "")

        # redraw existing polygon patches
        for patch in self.poly_patches:
            patch.remove()
        self.poly_patches.clear()
        for poly in self.polygons:
            patch = MplPolygon(poly["vertices"], closed=True, fill=False, edgecolor="red", linewidth=1.5)
            self.ax.add_patch(patch)
            self.poly_patches.append(patch)

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
        self.canvas.draw_idle()

    def _on_polygon_complete(self, verts: List[Tuple[float, float]]) -> None:
        if self.display_values is None or self.display_xedges is None or self.display_yedges is None:
            return
        verts = np.asarray(verts)
        path = MplPath(verts)
        xx, yy = np.meshgrid(self.display_xcenters, self.display_ycenters)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        mask = path.contains_points(points).reshape(self.display_values.shape)
        total = float(self.display_values[mask].sum())
        nbins = int(mask.sum())

        patch = MplPolygon(verts, closed=True, fill=False, edgecolor="red", linewidth=1.5)
        self.ax.add_patch(patch)
        self.poly_patches.append(patch)

        poly_info = {
            "hist": self.current_hist_name,
            "vertices": verts.tolist(),
            "sum_counts": total,
            "bins": nbins,
            "rebin": (self.rebin_x.value(), self.rebin_y.value()),
            "log": self.log_checkbox.isChecked(),
        }
        self.polygons.append(poly_info)
        self.canvas.draw_idle()
        self.status.showMessage(f"Polygon added: {nbins} bins, sum={total:.3g}", 5000)

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
        self.status.showMessage(f"Bin ({xi}, {yi}) center=({xc:.3g}, {yc:.3g}) value={val:.3g}", 4000)

    def export_polygons(self) -> None:
        if not self.polygons:
            self.status.showMessage("No polygons to export", 4000)
            return
        default_name = f"{self.current_hist_name or 'pid'}_polygons.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save polygons", default_name, "Text Files (*.txt)")
        if not path:
            return
        lines = ["# PID GUI polygon export", f"# ROOT file: {self.root_path}"]
        for idx, poly in enumerate(self.polygons, 1):
            lines.append(f"# Polygon {idx} | hist={poly['hist']} | rebin={poly['rebin']} | log={poly['log']} | bins={poly['bins']} | sum={poly['sum_counts']}")
            lines.append("x y")
            for x, y in poly["vertices"]:
                lines.append(f"{x} {y}")
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="ascii")
        self.status.showMessage(f"Saved {len(self.polygons)} polygons to {path}", 5000)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GUI PID viewer for ROOT histograms")
    parser.add_argument("root_file", nargs="?", default="combined_124Xe_all_runs_PID.root", help="Path to ROOT file")
    args = parser.parse_args()

    root_path = Path(args.root_file)
    if not root_path.exists():
        raise SystemExit(f"File not found: {root_path}")

    app = QtWidgets.QApplication(sys.argv)
    gui = PIDGui(root_path)
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
