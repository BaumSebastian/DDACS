# Tutorials

Step-by-step guides for working with the DDACS dataset.

## Available Tutorials

### [Getting Started](getting-started.md)
Learn the basics of loading and exploring DDACS simulation data:

- Counting available simulations
- Iterating over the dataset
- Exploring HDF5 file structure
- Accessing metadata

### [Visualization](visualization.md)
Create visualizations of simulation results:

- 3D mesh visualization
- Thickness distribution plots
- Springback analysis
- Point clouds and vector fields

## Prerequisites

Before starting, ensure you have installed DDACS (see [Installation](../installation.md)) and downloaded the dataset:

```bash
ddacs download --small
```

## Jupyter Notebook

A comprehensive example notebook is available at
[`notebooks/dataset_demo.ipynb`](https://github.com/BaumSebastian/DDACS/tree/main/notebooks/dataset_demo.ipynb)
covering data loading, visualization, and analysis using the `ddacs` package.
