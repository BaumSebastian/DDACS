#!/usr/bin/env python3
"""
Quick start example for DDACS dataset.

This script demonstrates basic usage of the DDACS dataset without heavy dependencies.
"""

from ddacs import iter_ddacs, count_available_simulations
from ddacs.utils import display_structure
import sys
from pathlib import Path


def main():
    """Main function demonstrating basic DDACS usage."""
    # Default data directory
    data_dir = Path("data")

    # Allow user to specify different data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Usage: python quick_start.py [data_directory]")
        return

    print(f"DDACS Dataset Quick Start")
    print(f"Data directory: {data_dir}")
    print("-" * 50)

    # Count available simulations
    try:
        count = count_available_simulations(data_dir)
        print(f"Available simulations: {count}")
    except Exception as e:
        print(f"Error counting simulations: {e}")
        return

    if count == 0:
        print("No simulations found!")
        return

    print("\nExamining first few simulations:")

    # Iterate through first few simulations
    for i, (sim_id, metadata, h5_path) in enumerate(iter_ddacs(data_dir)):
        print(f"\nSimulation {i+1}:")
        print(f"  ID: {sim_id}")
        print(f"  File: {h5_path.name}")
        print(f"  Metadata shape: {metadata.shape}")
        print(f"  Metadata: {metadata}")

        # Show structure of first file
        if i == 0:
            print(f"\nStructure of {h5_path.name}:")
            display_structure(h5_path, max_depth=2)

        # Only show first 3 simulations
        if i >= 2:
            break

    print(f"\nBasic exploration complete!")
    print(f"See dataset_demo.ipynb for advanced examples with visualization.")


if __name__ == "__main__":
    main()
