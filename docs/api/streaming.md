# Streaming

`ddacs.streaming` is the offline-iteration namespace: a plain-Python iterator over a Croissant view and a one-shot exporter that materialises that view as flat numpy memmap files. Both are torch-free and share a unified index that recognises loose `.h5` files (`ddacs download --extract --remove-zip`) and zipped `*.zip` archives interchangeably.

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
