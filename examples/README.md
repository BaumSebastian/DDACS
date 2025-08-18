# DDACS Examples

Practical examples and tutorials for working with the Deep Drawing and Cutting Simulations dataset.

## Installation

Install DDACS with example dependencies:

```bash
# Using pip
pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[examples]"

# Using uv (faster)
uv pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[examples]"
```

**Includes:** PyTorch, Jupyter, matplotlib, seaborn, plotly for full functionality.

## Quick Start

Before running examples, ensure you have downloaded the dataset:

```bash
# Download dataset (requires ~1TB storage)
darus-download --url "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801" --path "./data"
```

## Files

### `quick_start.py`
Basic command-line script for dataset exploration:
- Count available simulations
- Display HDF5 file structure
- Show metadata examples
- Simple data extraction

**Usage:**
```bash
python examples/quick_start.py [data_directory]
```

### `dataset_demo.ipynb` 
Comprehensive Jupyter notebook covering:
- **Data Loading**: Multiple access patterns (PyTorch, generators, direct HDF5)
- **Visualization**: Stress/strain plots, geometry visualization, thickness maps
- **Analysis Examples**: Parameter correlation, failure detection, springback analysis
- **ML Workflows**: Feature extraction, data preprocessing, PyTorch integration
- **Performance**: Memory usage optimization, batch processing

**Usage:**
```bash
jupyter notebook examples/dataset_demo.ipynb
```

## Example Workflows

### Basic Data Exploration
```bash
# Quick overview of your dataset
python examples/quick_start.py ./data

# Expected output:
# DDACS Dataset Quick Start
# Available simulations: 32071
# Examining first few simulations...
```

### Advanced Analysis (Jupyter)
1. **Launch notebook**: `jupyter notebook examples/dataset_demo.ipynb`
2. **Follow sections**:
   - Dataset overview and structure
   - Stress/strain visualization
   - Parameter sensitivity analysis
   - ML model training examples

### Custom Analysis
Use the examples as templates for your own research:
- Modify parameter filters in `quick_start.py`
- Adapt visualization code from the notebook
- Extend ML workflows for your specific use case

## Requirements

- **Python**: 3.8+
- **Storage**: ~1TB for full dataset
- **Memory**: 8GB+ RAM recommended for large-scale analysis
- **Optional**: GPU for PyTorch acceleration

## Need Help?

- Check the [main documentation](../doc/) for complete data field reference
- Review [API documentation](../README.md) for all available functions
- Open an issue for specific questions