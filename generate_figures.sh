#!/bin/bash


# Run postanalysis_v2.py
python postanalysis_v2.py

# Run normalize_celloutlines_v2.py
python normalize_celloutlines_v2.py

# Run make_figures
python make_figures.py

# Optional to run postanalysis --- not anymore since last figure is needed
# python postanalysis.py


