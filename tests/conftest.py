"""Test configuration and fixtures for DDACS tests."""

import pytest
import os
from pathlib import Path

# Configure this to point to your real dataset location
# Can be overridden with DDACS_TEST_DATA environment variable
REAL_DATASET_PATH = Path(os.environ.get("DDACS_TEST_DATA", "data"))


@pytest.fixture
def real_data_dir():
    """Fixture for real DDACS dataset."""
    if not REAL_DATASET_PATH.exists():
        pytest.skip(f"Real dataset not found at {REAL_DATASET_PATH}")
    return REAL_DATASET_PATH