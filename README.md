# Curvature-based shape sensing of a soft continuum arm (simulation study)

Code and results for "Twist, Interpolation and Resolution Limits of
Curvature-Based Shape Sensing for Soft Continuum Arms".

## Files
- `rod.py`: static Kirchhoff rod model of a three-segment arm (shooting solver)
- `validate_elastica.py`, `validate_twist.py`: cross-checks against PyElastica
- `make_dataset.py`: equilibrium shape data sets
- `recon.py`: sensor model (gauge averaging, noise, quantization) and shape reconstruction
- `sweep.py`, `analysis.py`, `torsion.py`, `extra.py`, `blend_check.py`: experiments
- `make_figs.py`: figures; `paper/`: LaTeX source and PDF
- `results/`: text outputs of every experiment

## Reproduce
```
pip install numpy matplotlib pyelastica
python make_dataset.py 1 200 data/shapes_a.npz
python make_dataset.py 2 200 data/shapes_b.npz
python make_dataset.py 3 60 data/shapes_gj2_a.npz 2.0
python make_dataset.py 4 60 data/shapes_gj02_a.npz 0.2
python make_dataset.py 5 100 data/shapes_blend_a.npz 0.6667 0.02
python make_dataset.py 6 100 data/shapes_blend_b.npz 0.6667 0.02
python validate_elastica.py > results/validate_elastica.txt
python validate_twist.py > results/validate_twist.txt
python sweep.py > results/sweep.txt
python analysis.py > results/analysis.txt
python torsion.py > results/torsion.txt
python extra.py > results/extra.txt
python blend_check.py > results/blend_check.txt
python make_figs.py
```
