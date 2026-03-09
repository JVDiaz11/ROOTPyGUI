# PID GUI for ROOT histograms

Interactive PyQt5 + matplotlib GUI to browse 2D histograms from a ROOT file: rebin, toggle per-axis scales, swap axes, zoom, draw polygons, inspect counts, and export regions.

## Requirements
- Python 3.9+ recommended
- See requirements.txt (PyQt5, matplotlib, numpy, uproot)

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python pid_gui.py /path/to/your.root  # defaults to combined_124Xe_all_runs_PID.root
```

## Controls
- Load selected: load highlighted histogram from the ROOT file list
- Swap X/Y: transpose the histogram axes
- Log scale (Z): apply log color normalization; colorbar context menu to toggle linear/log/log10
- Axis menus: right-click near X or Y axis to choose linear/log/log10 per axis
- Rebin X/Y: set integer factors and click "Apply binning" (drops edge bins if not divisible)
- Rebin Z: choose number of color bins for linear/log scales
- Zoom: rectangle zoom in Select mode; reset zoom from context menu
- Draw polygons: freehand selection adds a polygon; hover/status shows live axis coordinates and bin center
- Polygon colors: recolor polygons without resetting zoom
- Clear polygons: remove selected or all polygons; undo last polygon
- Export polygons: saves vertex lists to `outputs/` (folder auto-created). Headers only include the polygon name.

## Notes
- The ROOT file path is required; place it alongside the script or pass an explicit path.
- Polygon export writes ASCII text to `outputs/` using the currently loaded axes coordinates.
