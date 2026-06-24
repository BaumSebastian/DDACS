# CLI Reference

The `ddacs` command-line interface provides tools for downloading and managing the DDACS dataset.

## Commands

### `ddacs info`

Display dataset information and available versions.

```bash
ddacs info
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/cli_info.png" width="700">

### `ddacs download`

Download dataset files from DaRUS.

```bash
ddacs download [VERSION] [OPTIONS]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `VERSION` | `2.0` | Dataset version to download |

**Options:**

| Option | Description |
|--------|-------------|
| `--small` | Download small test set ({{ small_download_size() }}) |
| `--out PATH` | Output directory (default: `./data`) |
| `--files FILE...` | Download specific files only |
| `--no-extract` | Skip extraction of zip files |
| `--keep-zip` | Keep zip files after extraction |
| `-y, --yes` | Skip confirmation prompt |

!!! note "Download Size"
    The displayed download size refers to the compressed zip files. By default, files are extracted after download, resulting in larger disk usage than shown.

**Examples:**

```bash
# Download full dataset
ddacs download
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/cli_download.png" width="700">

```bash
# Download small test set
ddacs download --small
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/cli_download_small.png" width="700">

```bash
# Download to custom directory
ddacs download --out /path/to/data

# Download specific files
ddacs download --files {{ small_test_files() }}
```

## Global Options

| Option | Description |
|--------|-------------|
| `--token TOKEN` | DaRUS API token (for draft access) |
| `-V, --version` | Show version and exit |
