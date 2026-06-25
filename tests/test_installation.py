"""Smoke test that the public surface imports cleanly."""


class TestInstallation:
    def test_package_import(self):
        import ddacs

        assert ddacs is not None
        assert ddacs.__version__

    def test_top_level_entries(self):
        from ddacs import add_view, inspect_h5, load, open_h5

        for fn in (load, add_view, open_h5, inspect_h5):
            assert callable(fn)

    def test_visualization_imports(self):
        from ddacs import plot_2d_projection, plot_mesh, plot_point_cloud, plot_vectors

        for fn in (plot_mesh, plot_point_cloud, plot_vectors, plot_2d_projection):
            assert callable(fn)

    def test_pytorch_import_handling(self):
        try:
            from ddacs.pytorch import DDACSDataset

            assert DDACSDataset is not None
        except ImportError:
            # OK — torch isn't installed in this environment
            pass
