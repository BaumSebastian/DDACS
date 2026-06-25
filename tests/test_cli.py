"""Tests for the DDACS CLI."""

import argparse
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from ddacs.cli import (
    __version__,
    _compute_changes,
    _get_file_info,
    cmd_download,
    cmd_info,
    main,
)


class TestVersion:
    """Test CLI version."""

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])


class TestGetFileInfo:
    """Tests for _get_file_info helper function."""

    def test_get_file_info_basic(self):
        """Test basic file info extraction."""
        file_meta = {
            "dataFile": {
                "filename": "test.csv",
                "filesize": 1024,
            }
        }
        name, size = _get_file_info(file_meta)
        assert name == "test.csv"
        assert size == 1024

    def test_get_file_info_with_original(self):
        """Test file info with original filename."""
        file_meta = {
            "dataFile": {
                "filename": "test.tab",
                "filesize": 500,
                "originalFileName": "test.csv",
                "originalFileSize": 1024,
            }
        }
        # With original=True (default)
        name, size = _get_file_info(file_meta, original=True)
        assert name == "test.csv"
        assert size == 1024

        # With original=False
        name, size = _get_file_info(file_meta, original=False)
        assert name == "test.tab"
        assert size == 500

    def test_get_file_info_original_without_size(self):
        """Test file info when original size is missing."""
        file_meta = {
            "dataFile": {
                "filename": "test.tab",
                "filesize": 500,
                "originalFileName": "test.csv",
            }
        }
        name, size = _get_file_info(file_meta, original=True)
        assert name == "test.csv"
        assert size == 500  # Falls back to filesize


class TestComputeChanges:
    """Tests for _compute_changes helper function."""

    def test_compute_changes_added_files(self):
        """Test detection of added files."""
        current = {"a.csv", "b.csv", "c.csv"}
        previous = {"a.csv", "b.csv"}
        changes = _compute_changes(current, previous)
        assert len(changes) == 1
        assert "[green]+ c.csv[/green]" in changes

    def test_compute_changes_removed_files(self):
        """Test detection of removed files."""
        current = {"a.csv", "b.csv"}
        previous = {"a.csv", "b.csv", "c.csv"}
        changes = _compute_changes(current, previous)
        assert len(changes) == 1
        assert "[red]- c.csv[/red]" in changes

    def test_compute_changes_both(self):
        """Test detection of both added and removed files."""
        current = {"a.csv", "c.csv"}
        previous = {"a.csv", "b.csv"}
        changes = _compute_changes(current, previous)
        assert len(changes) == 2
        assert "[green]+ c.csv[/green]" in changes
        assert "[red]- b.csv[/red]" in changes

    def test_compute_changes_no_changes(self):
        """Test when there are no changes."""
        current = {"a.csv", "b.csv"}
        previous = {"a.csv", "b.csv"}
        changes = _compute_changes(current, previous)
        assert changes == ["no changes"]

    def test_compute_changes_empty_sets(self):
        """Test with empty sets."""
        changes = _compute_changes(set(), set())
        assert changes == ["no changes"]


class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_main_no_args_shows_help(self, capsys):
        """Test that running without args shows help."""
        with patch("sys.argv", ["ddacs"]):
            main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "ddacs" in captured.out

    def test_version_flag(self):
        """Test --version flag."""
        with patch("sys.argv", ["ddacs", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_info_command_parsing(self):
        """Test info command is recognized."""
        with patch("sys.argv", ["ddacs", "info"]):
            with patch("ddacs.cli.cmd_info") as mock_info:
                main()
                mock_info.assert_called_once()

    def test_download_command_parsing(self):
        """Test download command is recognized."""
        with patch("sys.argv", ["ddacs", "download"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                mock_download.assert_called_once()

    def test_download_with_version(self):
        """Test download with specific version."""
        with patch("sys.argv", ["ddacs", "download", "1.0"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.version == "1.0"

    def test_download_default_version(self):
        """Test download uses default version 2.0."""
        with patch("sys.argv", ["ddacs", "download"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.version == "2.0"

    def test_download_small_flag(self):
        """Test --small flag."""
        with patch("sys.argv", ["ddacs", "download", "--small"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.small is True

    def test_download_output_dir(self):
        """Test --out flag."""
        with patch("sys.argv", ["ddacs", "download", "--out", "/custom/path"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.out == "/custom/path"

    def test_download_yes_flag(self):
        """Test -y/--yes flag."""
        with patch("sys.argv", ["ddacs", "download", "-y"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.yes is True

    def test_download_extract_flag(self):
        """Test --extract opt-in flag."""
        with patch("sys.argv", ["ddacs", "download", "--extract"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.extract is True

    def test_download_remove_zip_flag(self):
        """Test --remove-zip flag."""
        with patch("sys.argv", ["ddacs", "download", "--extract", "--remove-zip"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.remove_zip is True

    def test_download_specific_files(self):
        """Test --files flag."""
        with patch("sys.argv", ["ddacs", "download", "--files", "a.csv", "b.zip"]):
            with patch("ddacs.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.files == ["a.csv", "b.zip"]

    def test_token_flag(self):
        """Test --token flag."""
        with patch("sys.argv", ["ddacs", "--token", "my-secret-token", "info"]):
            with patch("ddacs.cli.cmd_info") as mock_info:
                main()
                args = mock_info.call_args[0][0]
                assert args.token == "my-secret-token"


class TestCmdInfo:
    """Tests for cmd_info command."""

    @patch("ddacs.cli._api_get")
    def test_cmd_info_api_failure(self, mock_api_get, capsys):
        """Test info command handles API failure gracefully."""
        mock_api_get.return_value = None

        args = argparse.Namespace(token=None)
        cmd_info(args)

        # Should not raise, just return early
        mock_api_get.assert_called_once()

    @patch("ddacs.cli._api_get")
    def test_cmd_info_success(self, mock_api_get, capsys):
        """Test info command displays version information."""
        mock_api_get.return_value = [
            {
                "versionNumber": 2,
                "versionMinorNumber": 0,
                "versionState": "RELEASED",
                "releaseTime": "2024-01-01T00:00:00Z",
                "fileCount": 10,
                "versionNote": "Major update",
                "license": {"name": "CC BY 4.0"},
                "files": [{"dataFile": {"filename": "test.csv"}}],
            }
        ]

        args = argparse.Namespace(token=None)
        cmd_info(args)

        captured = capsys.readouterr()
        assert "DDACS" in captured.out


class TestCmdDownload:
    """Tests for cmd_download command."""

    @patch("ddacs.cli._api_get")
    def test_cmd_download_api_failure(self, mock_api_get):
        """Test download command handles API failure gracefully."""
        mock_api_get.return_value = None

        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=None,
            small=False,
            out="./data",
            yes=True,
            extract=False,
            remove_zip=False,
        )
        cmd_download(args)

        mock_api_get.assert_called_once()

    @patch("ddacs.cli._api_get")
    def test_cmd_download_no_matching_files(self, mock_api_get, capsys):
        """Test download when no files match criteria."""
        mock_api_get.return_value = {
            "versionNumber": 2,
            "versionMinorNumber": 0,
            "versionState": "RELEASED",
            "lastUpdateTime": "2024-01-01T00:00:00Z",
            "license": {"name": "CC BY 4.0"},
            "files": [{"dataFile": {"filename": "other.csv", "filesize": 100}}],
        }

        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=["nonexistent.csv"],
            small=False,
            out="./data",
            yes=True,
            extract=False,
            remove_zip=False,
        )
        cmd_download(args)

        captured = capsys.readouterr()
        assert "No files found" in captured.out

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_success(self, mock_api_get, mock_requests_get, tmp_path):
        """Test successful file download."""
        mock_api_get.return_value = {
            "versionNumber": 2,
            "versionMinorNumber": 0,
            "versionState": "RELEASED",
            "lastUpdateTime": "2024-01-01T00:00:00Z",
            "license": {"name": "CC BY 4.0"},
            "files": [
                {
                    "dataFile": {
                        "id": 123,
                        "filename": "test.csv",
                        "filesize": 100,
                    },
                    "description": "Test file",
                }
            ],
        }

        # Mock the download request
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(return_value=[b"test content"])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=None,
            small=False,
            out=str(tmp_path),
            yes=True,
            extract=False,
            remove_zip=False,
        )
        cmd_download(args)

        # Check file was created
        assert (tmp_path / "test.csv").exists()

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_with_extraction(self, mock_api_get, mock_requests_get, tmp_path):
        """Test download with zip extraction."""
        # Create a real zip file for testing
        zip_content_path = tmp_path / "content"
        zip_content_path.mkdir()
        (zip_content_path / "inner.txt").write_text("test")

        zip_path = tmp_path / "source.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(zip_content_path / "inner.txt", "inner.txt")

        zip_bytes = zip_path.read_bytes()

        mock_api_get.return_value = {
            "versionNumber": 2,
            "versionMinorNumber": 0,
            "versionState": "RELEASED",
            "lastUpdateTime": "2024-01-01T00:00:00Z",
            "license": {"name": "CC BY 4.0"},
            "files": [
                {
                    "dataFile": {
                        "id": 123,
                        "filename": "test.zip",
                        "filesize": len(zip_bytes),
                    },
                    "description": "Test zip",
                }
            ],
        }

        # Mock the download to return our zip file
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(zip_bytes))}
        mock_response.iter_content = MagicMock(return_value=[zip_bytes])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        output_dir = tmp_path / "output"
        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=None,
            small=False,
            out=str(output_dir),
            yes=True,
            extract=True,
            remove_zip=True,
        )
        cmd_download(args)

        # Check extraction worked
        assert (output_dir / "inner.txt").exists()
        # Check zip was removed (remove_zip=True)
        assert not (output_dir / "test.zip").exists()

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_keep_zip(self, mock_api_get, mock_requests_get, tmp_path):
        """Test download with extract but no remove-zip — zip is kept."""
        # Create a real zip file
        zip_content_path = tmp_path / "content"
        zip_content_path.mkdir()
        (zip_content_path / "inner.txt").write_text("test")

        zip_path = tmp_path / "source.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(zip_content_path / "inner.txt", "inner.txt")

        zip_bytes = zip_path.read_bytes()

        mock_api_get.return_value = {
            "versionNumber": 2,
            "versionMinorNumber": 0,
            "versionState": "RELEASED",
            "lastUpdateTime": "2024-01-01T00:00:00Z",
            "license": {"name": "CC BY 4.0"},
            "files": [
                {
                    "dataFile": {
                        "id": 123,
                        "filename": "test.zip",
                        "filesize": len(zip_bytes),
                    },
                    "description": "Test zip",
                }
            ],
        }

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(zip_bytes))}
        mock_response.iter_content = MagicMock(return_value=[zip_bytes])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        output_dir = tmp_path / "output"
        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=None,
            small=False,
            out=str(output_dir),
            yes=True,
            extract=True,
            remove_zip=False,  # Keep the zip
        )
        cmd_download(args)

        # Check both extracted file and zip exist
        assert (output_dir / "inner.txt").exists()
        assert (output_dir / "test.zip").exists()

    @patch("ddacs.cli.Confirm.ask")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_user_cancels(self, mock_api_get, mock_confirm, capsys):
        """Test download cancelled by user."""
        mock_api_get.return_value = {
            "versionNumber": 2,
            "versionMinorNumber": 0,
            "versionState": "RELEASED",
            "lastUpdateTime": "2024-01-01T00:00:00Z",
            "license": {"name": "CC BY 4.0"},
            "files": [{"dataFile": {"id": 123, "filename": "test.csv", "filesize": 100}}],
        }
        mock_confirm.return_value = False

        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=None,
            small=False,
            out="./data",
            yes=False,  # Will prompt
            extract=False,
            remove_zip=False,
        )
        cmd_download(args)

        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()


class TestSmallTestSet:
    """Tests for --small flag behavior."""

    @patch("ddacs.cli._api_get")
    def test_small_filters_files_correctly(self, mock_api_get, capsys):
        """Test that --small only selects specific files."""
        mock_api_get.return_value = {
            "versionNumber": 2,
            "versionMinorNumber": 0,
            "versionState": "RELEASED",
            "lastUpdateTime": "2024-01-01T00:00:00Z",
            "license": {"name": "CC BY 4.0"},
            "files": [
                {"dataFile": {"filename": "metadata.csv", "filesize": 100, "id": 1}},
                {"dataFile": {"filename": "403926_406296.zip", "filesize": 1000, "id": 2}},
                {"dataFile": {"filename": "other_large.zip", "filesize": 10000, "id": 3}},
            ],
        }

        args = argparse.Namespace(
            version="2.0",
            token=None,
            files=None,
            small=True,
            out="./data",
            yes=False,  # Will show table but not download
            extract=False,
            remove_zip=False,
        )

        # Mock Confirm to cancel
        with patch("ddacs.cli.Confirm.ask", return_value=False):
            cmd_download(args)

        captured = capsys.readouterr()
        # Should show metadata.csv and the small test zip
        assert "metadata.csv" in captured.out
        assert "403926_406296.zip" in captured.out
        # Should NOT show the large file
        assert "other_large.zip" not in captured.out
