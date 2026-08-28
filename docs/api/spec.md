# Dataset spec

`DatasetSpec` bundles the identity of a dataset (DaRUS DOI, manifest file name, default data directory, id format, small test files) so that sibling packages such as `rddac` reuse the same machinery with different values. `DDACS_SPEC` is the instance for this dataset. `MissingDataWarning` is raised by the streaming functions when requested simulations cannot be served locally.

::: ddacs.spec.DatasetSpec
    options:
      show_signature: true
      show_signature_annotations: true

::: ddacs.spec.DDACS_SPEC

::: ddacs.streaming.MissingDataWarning
