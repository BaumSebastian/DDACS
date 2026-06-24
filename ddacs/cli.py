"""
DDACS dataset CLI for downloading data from DaRUS.

Provides commands to view dataset information and download files from the
DDACS (Deep Drawing and Cutting Simulations) dataset hosted on DaRUS.

Usage:
    ddacs info                      # Show dataset info and versions
    ddacs download                  # Download dataset v2.0, extract, ready to use
    ddacs download --small          # Download small test set for demos
    ddacs download --files a.zip    # Download specific files
    ddacs download --no-extract     # Download without extracting
    ddacs download --keep-zip       # Keep zip files after extraction
"""

__version__ = "2.1.1"

import argparse
import os
import zipfile

import requests
from humanfriendly import format_size
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm
from rich.table import Table

from .config import DARUS_BASE_URL, DATASET_DOI, DEFAULT_VERSION, SMALL_TEST_FILES

console = Console()


def _dataset_title(version_data: dict) -> str:
    """Extract the dataset title from a DaRUS version metadata response."""
    fields = version_data.get("metadataBlocks", {}).get("citation", {}).get("fields", [])
    for f in fields:
        if f.get("typeName") == "title":
            return f.get("value", "")
    return ""


def _api_get(endpoint: str, headers: dict[str, str]) -> dict | None:
    """
    Make GET request to DaRUS API.

    Args:
        endpoint: API endpoint path (without base URL).
        headers: HTTP headers including optional API token.

    Returns:
        Response data dict if successful, None on error.
    """
    try:
        r = requests.get(f"{DARUS_BASE_URL}/api/{endpoint}", headers=headers)
        r.raise_for_status()
        return r.json()["data"]
    except Exception as e:
        console.print(f"[red]API Error:[/red] {e}")
        return None


def _get_version_files(version: str, headers: dict[str, str]) -> set[str]:
    """
    Fetch filenames for a specific dataset version.

    Args:
        version: Dataset version string (e.g., "2.0").
        headers: HTTP headers for API authentication.

    Returns:
        Set of filenames in the specified version.
    """
    data = _api_get(
        f"datasets/:persistentId/versions/{version}?persistentId={DATASET_DOI}", headers
    )
    if data:
        return {f["dataFile"]["filename"] for f in data.get("files", [])}
    return set()


def _get_file_info(file_meta: dict, original: bool = True) -> tuple[str, int]:
    """
    Extract filename and size from DaRUS file metadata.

    Args:
        file_meta: File metadata dict from DaRUS API.
        original: If True, prefer original filename over transformed.

    Returns:
        Tuple of (filename, file_size_bytes).
    """
    df = file_meta["dataFile"]
    if original and "originalFileName" in df:
        return df["originalFileName"], df.get("originalFileSize", df["filesize"])
    return df["filename"], df["filesize"]


def _compute_changes(current_files: set[str], previous_files: set[str]) -> list[str]:
    """
    Compute file changes between two dataset versions.

    Args:
        current_files: Set of filenames in current version.
        previous_files: Set of filenames in previous version.

    Returns:
        List of change descriptions with Rich markup for display.
    """
    added = current_files - previous_files
    removed = previous_files - current_files

    changes = []
    for f in sorted(added):
        changes.append(f"[green]+ {f}[/green]")
    for f in sorted(removed):
        changes.append(f"[red]- {f}[/red]")

    return changes if changes else ["no changes"]


def cmd_info(args: argparse.Namespace) -> None:
    """
    Display dataset information and available versions.

    Args:
        args: Parsed command-line arguments containing optional token.
    """
    headers = {"X-Dataverse-key": args.token} if args.token else {}

    with console.status("[bold blue]Fetching dataset information..."):
        versions = _api_get(f"datasets/:persistentId/versions?persistentId={DATASET_DOI}", headers)
        if not versions:
            return

    dataset_url = f"{DARUS_BASE_URL}/dataset.xhtml?persistentId={DATASET_DOI}"
    latest = versions[0] if versions else {}
    license_name = latest.get("license", {}).get("name", "Unknown")

    info = (
        f"[bold]URL:[/bold] [link={dataset_url}]{dataset_url}[/link]\n"
        f"[bold]Persistent ID:[/bold] {DATASET_DOI}\n"
        f"[bold]Authors:[/bold] Sebastian Baum, Pascal Heinzelmann\n"
        f"[bold]License:[/bold] {license_name}"
    )
    console.print()
    console.print(Panel(info, title=_dataset_title(latest), border_style="cyan"))

    version_files = {}
    with console.status("[bold blue]Fetching version details..."):
        for v in versions:
            ver_num = v.get("versionNumber", "?")
            ver_minor = v.get("versionMinorNumber", "0")
            ver_str = f"{ver_num}.{ver_minor}"
            if "files" in v:
                version_files[ver_str] = {f["dataFile"]["filename"] for f in v.get("files", [])}
            else:
                version_files[ver_str] = _get_version_files(ver_str, headers)

    table = Table(
        title="Available Versions", show_header=True, header_style="bold cyan", show_lines=True
    )
    table.add_column("Version", style="white")
    table.add_column("State", style="yellow")
    table.add_column("Release Date", style="green")
    table.add_column("Files", justify="right")
    table.add_column("Changes", max_width=40)
    table.add_column("Notes", max_width=40)

    version_list = list(version_files.keys())
    for i, v in enumerate(versions):
        version_num = f"{v.get('versionNumber', '?')}.{v.get('versionMinorNumber', '0')}"
        state = v.get("versionState", "Unknown")
        release_date = v.get("releaseTime", v.get("createTime", "Unknown"))
        if release_date and release_date != "Unknown":
            release_date = release_date[:10]

        file_count = v.get("fileCount", len(v.get("files", [])))
        version_note = v.get("versionNote", "-")

        if state == "RELEASED":
            state_str = "[green]Released[/green]"
        elif state == "DRAFT":
            state_str = "[yellow]Draft[/yellow]"
        else:
            state_str = state

        if i < len(version_list) - 1:
            current_files = version_files.get(version_num, set())
            previous_files = version_files.get(version_list[i + 1], set())
            changes = _compute_changes(current_files, previous_files)
        else:
            changes = ["initial release"]

        table.add_row(
            version_num, state_str, release_date, str(file_count), "\n".join(changes), version_note
        )

    console.print()
    console.print(table)
    console.print()
    console.print("Use 'ddacs download <version>' to download a specific version of the dataset.")


def cmd_download(args: argparse.Namespace) -> None:
    """
    Download dataset files from DaRUS repository.

    Handles file selection, download with progress display, and optional
    extraction of zip archives.

    Args:
        args: Parsed command-line arguments containing version, output path,
              and download options (files, small, no_extract, keep_zip, yes).
    """
    headers = {"X-Dataverse-key": args.token} if args.token else {}

    with console.status("[bold blue]Fetching dataset metadata..."):
        data = _api_get(
            f"datasets/:persistentId/versions/{args.version}?persistentId={DATASET_DOI}",
            headers,
        )
        if not data:
            return
        all_files = data["files"]

    version_number = data.get("versionNumber", "?")
    version_minor = data.get("versionMinorNumber", "0")
    version_state = data.get("versionState", "")
    last_update = data.get("lastUpdateTime", "Unknown")[:10]
    license_name = data.get("license", {}).get("name", "Unknown")
    dataset_url = f"{DARUS_BASE_URL}/dataset.xhtml?persistentId={DATASET_DOI}"

    version_str = f"{version_number}.{version_minor}"
    if version_state == "DRAFT":
        version_str += " [yellow](Draft)[/yellow]"

    dataset_info = (
        f"[bold]URL:[/bold] [link={dataset_url}]{dataset_url}[/link]\n"
        f"[bold]Persistent ID:[/bold] {DATASET_DOI}\n"
        f"[bold]Version:[/bold] {version_str}\n"
        f"[bold]Last Update:[/bold] {last_update}\n"
        f"[bold]License:[/bold] {license_name}"
    )
    console.print()
    console.print(Panel(dataset_info, title=_dataset_title(data), border_style="cyan"))

    original = True  # Always download original format

    # Determine which files to download
    if args.files:
        # User specified specific files
        selected_files = [
            f
            for f in all_files
            if _get_file_info(f, original)[0] in args.files
            or f["dataFile"]["filename"] in args.files
        ]
    elif args.small:
        # Download small test set for demos/testing
        selected_files = [
            f
            for f in all_files
            if _get_file_info(f, original)[0] in SMALL_TEST_FILES
            or f["dataFile"]["filename"] in SMALL_TEST_FILES
        ]
    else:
        # Default: download all files
        selected_files = all_files

    if not selected_files:
        console.print("[yellow]No files found matching criteria.[/yellow]")
        return

    selected_files.sort(key=lambda f: f["dataFile"]["filesize"])

    table = Table(title="Files to Download", header_style="bold cyan")
    table.add_column("Filename", max_width=40)
    table.add_column("Size", justify="right", style="green")
    table.add_column("Description", max_width=50)

    total_bytes = 0
    for f in selected_files:
        name, size = _get_file_info(f, original)
        desc = f.get("description", "No description")
        total_bytes += size
        table.add_row(name, format_size(size), desc)

    console.print()
    console.print(table)

    summary = (
        f"[bold]Files:[/bold] {len(selected_files)}\n"
        f"[bold]Total size:[/bold] {format_size(total_bytes)}\n"
        f"[bold]Destination:[/bold] {os.path.abspath(args.out)}"
    )
    console.print()
    console.print(Panel(summary, title="Download Summary", border_style="blue"))

    console.print()
    if not args.yes and not Confirm.ask("Proceed with download?"):
        console.print("[yellow]Download cancelled.[/yellow]")
        return

    os.makedirs(args.out, exist_ok=True)

    downloaded_files = []
    failed_downloads = []
    failed_extractions = []

    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for f in selected_files:
            file_name, _ = _get_file_info(f, original)
            file_id = f["dataFile"]["id"]
            directory = f.get("directoryLabel", "")

            dl_url = f"{DARUS_BASE_URL}/api/access/datafile/{file_id}"
            if original:
                dl_url += "?format=original"

            target_dir = os.path.join(args.out, directory)
            os.makedirs(target_dir, exist_ok=True)
            local_path = os.path.join(target_dir, file_name)

            try:
                with requests.get(dl_url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    total_file_size = int(r.headers.get("content-length", 0))

                    task = progress.add_task(
                        "download",
                        filename=file_name[:30],
                        total=total_file_size,
                    )

                    with open(local_path, "wb") as out_f:
                        for chunk in r.iter_content(chunk_size=8192):
                            written = out_f.write(chunk)
                            progress.update(task, advance=written)

                downloaded_files.append(local_path)
            except Exception as e:
                failed_downloads.append((file_name, str(e)))
                console.print(f"[red]Failed to download {file_name}:[/red] {e}")

    if not args.no_extract and downloaded_files:
        console.print()
        zip_files = [p for p in downloaded_files if p.endswith(".zip")]
        if zip_files:
            with Progress(
                TextColumn("[bold blue]{task.description}", justify="right"),
                BarColumn(bar_width=40),
                "[progress.percentage]{task.percentage:>3.1f}%",
                TimeRemainingColumn(),
                TextColumn("{task.fields[current_file]}"),
                console=console,
            ) as progress:
                for local_path in zip_files:
                    zip_name = os.path.basename(local_path)
                    try:
                        extract_dir = os.path.dirname(local_path)
                        with zipfile.ZipFile(local_path, "r") as zf:
                            members = zf.namelist()
                            task = progress.add_task(
                                f"Extracting {zip_name[:20]}",
                                total=len(members),
                                current_file="",
                            )
                            for member in members:
                                progress.update(task, current_file=os.path.basename(member)[:30])
                                zf.extract(member, extract_dir)
                                progress.advance(task)
                        console.print(f"[green]Extracted:[/green] {zip_name}")
                        if not args.keep_zip:
                            os.remove(local_path)
                            console.print(f"[dim]Removed:[/dim] {zip_name}")
                    except Exception as e:
                        failed_extractions.append((zip_name, str(e)))
                        console.print(f"[red]Failed to extract {zip_name}:[/red] {e}")

    console.print()
    success_count = len(downloaded_files)
    total_count = len(selected_files)

    if failed_downloads or failed_extractions:
        summary_lines = [f"[green]Downloaded {success_count}/{total_count} file(s)[/green]"]
        if failed_downloads:
            summary_lines.append(f"[red]Failed downloads: {len(failed_downloads)}[/red]")
        if failed_extractions:
            summary_lines.append(f"[red]Failed extractions: {len(failed_extractions)}[/red]")
        console.print(
            Panel("\n".join(summary_lines), title="Complete with errors", border_style="yellow")
        )
    else:
        console.print(
            Panel(
                f"[green]Successfully downloaded {success_count} file(s)[/green]",
                title="Complete",
                border_style="green",
            )
        )


def main() -> None:
    """CLI entry point for DDACS dataset commands."""
    parser = argparse.ArgumentParser(
        prog="ddacs",
        description="DDACS Dataset CLI - Download simulation data from DaRUS",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--token", help="DaRUS API token (for draft access)")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    subparsers.add_parser(
        "info",
        help="Show dataset info and versions",
        description="Display dataset metadata, available versions, and changelog.",
    )

    dl_parser = subparsers.add_parser(
        "download",
        help="Download dataset files",
        description="Download DDACS dataset files from DaRUS repository.",
    )
    dl_parser.add_argument(
        "version",
        nargs="?",
        default=DEFAULT_VERSION,
        help=f"Dataset version (default: {DEFAULT_VERSION})",
    )
    dl_parser.add_argument("--files", nargs="+", help="Specific filenames to download")
    dl_parser.add_argument(
        "--small",
        action="store_true",
        help="Download small test set for demos/testing",
    )
    dl_parser.add_argument("--out", default="./data", help="Output directory (default: ./data)")
    dl_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    dl_parser.add_argument("--no-extract", action="store_true", help="Skip extraction of zip files")
    dl_parser.add_argument(
        "--keep-zip", action="store_true", help="Keep zip files after extraction"
    )

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "download":
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
