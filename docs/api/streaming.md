# Streaming

`ddacs.streaming` is the offline-iteration namespace with three torch-free entry points:

- `iter_view` walks a Croissant view record by record. Shares a unified index that recognises loose `.h5` files (`ddacs download --extract --remove-zip`) and zipped `*.zip` archives interchangeably.
- `export_to_numpy` materialises a view as flat `.npy` memmap shards, with optional per-field and whole-record transforms.
- `load_export` opens those shards back as a `len + getitem + iter` protocol object that plugs into `torch.utils.data.DataLoader`, `tf.data.Dataset.from_generator`, JAX, or plain Python without any adapter.

## `ddacs.streaming.iter_view`

::: ddacs.streaming.iter_view
    options:
      show_signature: true
      show_signature_annotations: true

## `ddacs.streaming.export_to_numpy`

::: ddacs.streaming.export_to_numpy
    options:
      show_signature: true
      show_signature_annotations: true

## `ddacs.streaming.load_export`

::: ddacs.streaming.load_export
    options:
      show_signature: true
      show_signature_annotations: true
