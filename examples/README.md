# DDACS Examples

This directory contains examples and tutorials for using the DDACS dataset.

## Files

- `dataset_demo.ipynb` - Comprehensive Jupyter notebook demonstrating:
  - Dataset loading and exploration
  - Different access patterns (PyTorch, Iterator, Generator)
  - Performance comparisons
  - Data visualizations with matplotlib
  - ML workflow examples

- `quick_start.py` - Simple Python script for basic usage
- `performance_comparison.py` - Benchmarking different access methods
- `visualization_examples.py` - Matplotlib/seaborn plotting examples

## Installation

Install with examples dependencies:

### Using uv (recommended)
[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.
```bash
uv pip install ".[examples]"
```

### Using pip
```bash
pip install ".[examples]"
```

## Running Examples

### Run Jupyter notebook
```bash
jupyter notebook examples/dataset_demo.ipynb
```

### Run Python examples
```bash
python examples/quick_start.py
```