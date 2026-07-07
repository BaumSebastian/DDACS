# Publishing the DDACS teaser

Publishes the 22 MB sample (manifest + CSV + one simulation + docs) to **Kaggle**
and **Hugging Face**, plus the six tutorials as **Kaggle notebooks**. The full
~640 GB dataset stays on DaRUS; this is the discovery teaser.

The scripts here are committed and run in CI (`.github/workflows/publish.yml`);
only `.staging/` is git-ignored. Releases keep the public surfaces fresh
automatically (secrets live in the repo's `kaggle` / `huggingface` environments):

| Release tag | PyPI | Kaggle kernels | Kaggle dataset | Hugging Face |
|---|---|---|---|---|
| patch/minor (e.g. `3.2.2`) | ✓ | ✓ (regenerated) | — | — |
| major (e.g. `4.0.0`) | ✓ | ✓ | ✓ (one DaRUS fetch, shared) | ✓ |

Order on a major: publish the new version on **DaRUS first**, then push the tag —
the CI fetch reads DaRUS anonymously at tag time.

## Manual upload commands (ad hoc)

```bash
# from the repo root; SRC points at the data source dir
SRC=<data source dir>

# 1. Kaggle dataset  (first time: use `create` instead of `version`)
DDACS_TEASER_SRC=$SRC ./publish/kaggle/upload.sh version

# 2. Kaggle notebooks -> the dataset's Code tab
./publish/kaggle/kernels/push.sh

# 3. Hugging Face dataset
HF_REPO=BaumSebastian/ddacs-teaser DDACS_TEASER_SRC=$SRC ./publish/huggingface/upload.sh
```

That's it. Re-run any of the three to update that target.

## One-time setup

- **Kaggle CLI**: `.venv/bin/kaggle` (put `.venv/bin` on PATH, or run via `uv`).
  Auth is ambient via `~/.kaggle/access_token`.
- **Hugging Face CLI**: `hf auth login` with a **write** token
  (huggingface.co/settings/tokens).
- **Data**: `DDACS_TEASER_SRC=<data source dir>` (the repo's `./data`
  symlink points at a missing mount).

## How each upload works

**`stage_teaser.sh`** builds `publish/.staging/` from `$DDACS_TEASER_SRC`:
`data/{metadata.json, process_parameters.csv, h5/258864.zip}` + `ddacs_documentation.pdf`
+ `README.md`. All three uploaders call it first.

| Command | What it uploads | Notes |
|---|---|---|
| `kaggle/upload.sh {create,version}` | staged files + `dataset-metadata.json` | `--dir-mode zip`; Kaggle **auto-extracts** the h5 zip. Page **description + column docs come from `dataset-metadata.json`**, not the README. `version` first pulls current metadata to avoid the "non current" error. |
| `kaggle/kernels/push.sh` | 6 notebooks as kernels | `build.py` adapts each (pip install, `ddacs download --small`, 600 GB CTA) and attaches the dataset. |
| `huggingface/upload.sh` | staged files + `notebooks/` + card | Card = `card-header.md` (YAML) + `teaser/README.md`. HF does **not** extract zips, so `ddacs.load` works off the repo. |

## Editing content (single source)

- **README body** (both cards + the bundled file): `teaser/README.md` — edit once,
  all surfaces update on the next push.
- **HF card YAML** (license, tags): `huggingface/card-header.md`.
- **Kaggle page text** (title, subtitle, description, tags, CSV column docs):
  `kaggle/dataset-metadata.json`.
- **Notebook adaptations**: `kaggle/kernels/build.py` (notebooks themselves stay
  the source of truth in `../notebooks/`).

## Visibility

Kaggle dataset + kernels are made public on the **site** (a public kernel needs a
public dataset first). HF dataset repos are public by default.
