# Changelog

All notable changes to the `ddacs` package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the
`bumpver` tags. The dataset itself is versioned on DaRUS (doi:10.18419/DARUS-4801)
independently of the package; the package major version tracks the dataset major.

## [Unreleased]

### Changed
- Runtime dependencies reduced: `seaborn`, `scipy` and `pyvista` were never imported by
  the package and are no longer installed; the tutorial notebooks use the matplotlib
  `magma` colormap instead of the seaborn `rocket` palette.
- Documentation build fixed on current `pygments` (`pymdown-extensions>=11` instead of a
  `pygments` pin); the CLI reference tables are generated from the argparse definitions,
  and counts such as the number of zip archives come from the bundled manifest.
- README trimmed to the essentials (installation, download, basic usage) with links to
  the documentation; API pages for streaming (all four entry points) and the dataset
  spec (`DatasetSpec`, `DDACS_SPEC`, `MissingDataWarning`).
- Publishing: Zenodo documented alongside Kaggle and Hugging Face; the teaser card states
  where the notebooks live per platform; the Zenodo record no longer claims the DaRUS DOI
  and gets its publication date at upload time.

### Added
- CI workflow (pre-commit hooks, tests on Python 3.10 to 3.12, strict documentation build).
- This changelog.

## [3.2.2] - 2026-07-07

### Added
- Dataset license badge; workflow jobs that refresh the Kaggle and Hugging Face teasers
  on releases.

### Fixed
- Wording: OP20 is cutting, not trimming.

## [3.2.1] - 2026-07-07

### Changed
- CLI built on the `DatasetSpec` class so sibling datasets can reuse it; notebooks
  re-executed (point-cloud shading adjusted).

## [3.2.0] - 2026-07-06

### Added
- `DatasetSpec` as the single source of truth for dataset identity (DOI, manifest file,
  default data directory, id format, small test files).

### Changed
- Tests repaired for the current CLI and streaming API; documentation no longer refers
  to the unreleased `ddacs.augment`.

## [3.1.5] - 2026-07-02

### Added
- Publishing to PyPI on tag push via trusted publishing, with a TestPyPI dry run.

## [3.1.4] - 2026-07-02

### Added
- `ddacs download --quiet` (no output, implies `--yes`).

## [3.1.3] - 2026-07-01

### Changed
- Documentation front page rewritten as a research pitch; geometry parameter diagram.

## [3.1.2] - 2026-06-30

### Fixed
- `visualization` uses `plt.colormaps.get_cmap` (matplotlib >= 3.9).
- Documentation builds on Read the Docs with a bundled copy of the manifest.

### Changed
- Notebooks 01 to 06 polished for reviewers; cross-platform paths.

## [3.1.1] - 2026-06-29

### Fixed
- Documentation build with `pygments` 2.20 (pin, replaced in the next releases).

## [3.1.0] - 2026-06-29

### Added
- `ddacs.streaming.export_to_numpy_per_sim`: one `.npz` per simulation for views with
  variable record shapes.
- `ddacs.add_view` accepts mixed sources (HDF5 fields and CSV columns) and accumulates
  across calls; streamed records carry `_sim_id`.

## [3.0.0] - 2026-06-28

### Changed
- Rewrite around the Croissant 1.1 manifest shipped with the dataset: `ddacs.load`,
  `ddacs.open_h5`, `ddacs.add_view`, `DDACSDataset` over any view, streaming and numpy
  export; the CLI keeps zips intact by default (`--extract`, `--remove-zip` opt in).
- The v2 helpers (`iter_ddacs`, `count_available_simulations`, extraction utilities)
  were removed.

Earlier releases (1.0.x, 2.x) predate this changelog.

[Unreleased]: https://github.com/BaumSebastian/DDACS/compare/3.2.2...HEAD
[3.2.2]: https://github.com/BaumSebastian/DDACS/compare/3.2.1...3.2.2
[3.2.1]: https://github.com/BaumSebastian/DDACS/compare/3.2.0...3.2.1
[3.2.0]: https://github.com/BaumSebastian/DDACS/compare/3.1.5...3.2.0
[3.1.5]: https://github.com/BaumSebastian/DDACS/compare/3.1.4...3.1.5
[3.1.4]: https://github.com/BaumSebastian/DDACS/compare/3.1.3...3.1.4
[3.1.3]: https://github.com/BaumSebastian/DDACS/compare/3.1.2...3.1.3
[3.1.2]: https://github.com/BaumSebastian/DDACS/compare/3.1.1...3.1.2
[3.1.1]: https://github.com/BaumSebastian/DDACS/compare/3.1.0...3.1.1
[3.1.0]: https://github.com/BaumSebastian/DDACS/compare/v3.0.0...3.1.0
[3.0.0]: https://github.com/BaumSebastian/DDACS/releases/tag/v3.0.0
