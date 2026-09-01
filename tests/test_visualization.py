"""Tests for the plotting helpers (ddacs.visualization), rendered off-screen."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ddacs import visualization as vis  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.5]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces


def _points(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.random(n)


class TestReturnContract:
    """With values: (axis, colorbar); without values: the axis alone."""

    def test_plot_mesh(self):
        vertices, faces = _mesh()
        ax = vis.plot_mesh(vertices, faces)
        assert ax.name == "3d"
        ax2, cbar = vis.plot_mesh(vertices, faces, values=np.array([0.1, 0.9]), colorbar_label="t")
        assert ax2.name == "3d" and cbar.ax.get_ylabel() == "t"

    def test_plot_point_cloud(self):
        coords, values = _points()
        assert vis.plot_point_cloud(coords).name == "3d"
        ax, cbar = vis.plot_point_cloud(coords, values=values, title="pc")
        assert ax.get_title() == "pc" and cbar is not None

    def test_plot_vectors(self):
        coords, values = _points()
        vectors = np.ones_like(coords) * 0.1
        assert vis.plot_vectors(coords, vectors, step=10).name == "3d"
        ax, cbar = vis.plot_vectors(coords, vectors, values=values, step=10)
        assert cbar is not None

    @pytest.mark.parametrize("projection", ["xy", "xz", "yz"])
    def test_plot_2d_projection(self, projection):
        coords, values = _points()
        ax = vis.plot_2d_projection(coords, projection=projection)
        assert ax.name == "rectilinear"
        ax, cbar = vis.plot_2d_projection(coords, values=values, projection=projection)
        assert cbar is not None


class TestComposition:
    """An existing axis is drawn into instead of creating a new figure."""

    def test_existing_axis_is_reused(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        coords, _ = _points()
        vertices, faces = _mesh()
        assert vis.plot_point_cloud(coords, ax=ax) is ax
        assert vis.plot_mesh(vertices, faces, ax=ax) is ax
        assert len(fig.axes) == 1

    def test_palette_constants(self):
        assert vis.FALSE_COLOR_CMAP.N > 0
        assert isinstance(vis.COMPONENT_COLORS, dict) and vis.COMPONENT_COLORS
