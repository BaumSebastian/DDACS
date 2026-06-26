# Installation

Requires Python 3.10 or newer. PyTorch is optional.

```bash
pip install ddacs
```

Verify the install:

```python
import ddacs
print(ddacs.__version__)
```

## PyTorch

`DDACSDataset` and the rest of `ddacs.pytorch` require PyTorch.

```bash
pip install ddacs[torch]
```

This pulls the default PyTorch wheel from PyPI (CPU on Windows and macOS, CUDA 12.x on Linux). For a different build (CUDA, ROCm, MPS) install PyTorch first from the [PyTorch selector](https://pytorch.org/get-started/locally/), then `pip install ddacs` without the extra. The package does not pin PyTorch, so an existing wheel is preserved.

## Advanced

### Documentation build

```bash
pip install ddacs[docs]
```

Then `mkdocs serve` from the repository root for a live preview at `http://127.0.0.1:8000`.

### Development install

```bash
git clone https://github.com/BaumSebastian/DDACS.git
cd DDACS
pip install -e .[dev,torch]
pre-commit install
```

`[dev]` adds `pytest`, `black`, `ruff`, `bumpver`, `pre-commit`, and `jupyter`.
