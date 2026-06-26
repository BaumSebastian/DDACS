# Croissant

`ddacs.load` and `ddacs.add_view` are the two entry points to the Croissant manifest. Both are re-exported at the top level (`ddacs.load`, `ddacs.add_view`); the implementation lives in `ddacs.croissant`.

## `ddacs.load`

::: ddacs.croissant.load
    options:
      show_signature: true
      show_signature_annotations: true

## `ddacs.add_view`

::: ddacs.croissant.add_view
    options:
      show_signature: true
      show_signature_annotations: true

## Module reference

::: ddacs.croissant
    options:
      members:
        - METADATA_URL
        - resolve_source
        - field_map
        - process_parameters_descriptions
        - dataset_name
      show_root_heading: false
      heading_level: 3
