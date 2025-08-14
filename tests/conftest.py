"""Test configuration and fixtures for DDACS tests."""

import pytest
from pathlib import Path

# Configure this to point to your real dataset location
REAL_DATASET_PATH = Path("data")


@pytest.fixture
def real_data_dir():
    """Fixture for real DDACS dataset."""
    if not REAL_DATASET_PATH.exists():
        pytest.skip(f"Real dataset not found at {REAL_DATASET_PATH}")
    return REAL_DATASET_PATH