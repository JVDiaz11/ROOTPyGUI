# PID GUI for ROOT histograms

Interactive PyQt5 + matplotlib GUI to browse 2D histograms from a ROOT file, rebin, toggle log scale, swap axes, and draw polygons to inspect counts and export regions.

## Requirements
- Python 3.9+ recommended
- Packages: PyQt5, matplotlib, numpy, uproot

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install PyQt5 matplotlib numpy uproot
python pid_gui.py /path/to/your.root  # defaults to combined_124Xe_all_runs_PID.root
```

## Controls
- Load selected: load highlighted histogram from the ROOT file list
- Swap X/Y: transpose the histogram axes
- Log scale (Z): apply log color normalization
- Rebin X/Y: set integer factors and click "Apply binning" (drops edge bins if not divisible)
- Draw polygons: freehand selection adds a red polygon, shows bin count and sum in the status bar
- Clear polygons: remove all polygons
- Export polygons: save vertex lists and metadata to a text file

## Notes
- The ROOT file path is required; place it alongside the script or pass an explicit path.
- Polygon export writes ASCII text with vertices and summary stats for each polygon.
