# Repository Guide for Agents

## Project

`dataset-loader` is a Rust 2024 library crate for laser-image datasets. It
loads source records from an `index.csv`, resolves image paths, maps records
through Burn dataset adapters, splits datasets, and provides CPU image
augmentations.

## Layout

- `src/lib.rs`: crate docs, public module exports, shared error and split
  re-exports.
- `src/dataset.rs`: CSV loading, item filtering, lazy mapper integration, and
  dataset split adapter.
- `src/item.rs`, `src/item/`: shared item traits, intensity wrapper, and image
  path type.
- `src/sources.rs`, `src/sources/`: source-specific item schemas and loaders.
- `src/splitting.rs`: weighted train/validation/test partitioning.
- `src/augmentation.rs`, `src/augmentation/`: composable CPU image transforms
  and validated transform parameters.
- `src/error.rs`: crate result alias and error variants.
- `docs/`: contribution notes and focused design plans.

## Development

- The toolchain is pinned to nightly in `rust-toolchain.toml`.
- Keep Rust indentation consistent with the repository's tab-based style.
- Preserve existing public signatures; prefer additive APIs and document
  behavior and compatibility implications.
- Add unit tests in the owning module's `#[cfg(test)]` module. Use generated
  temporary fixtures rather than assuming sibling dataset repositories exist.
- Rustdoc examples are executable tests; run `cargo test --doc` when editing
  public documentation.
- Validate changes with `cargo test`, followed by `cargo clippy`. The manifest
  enables strict pedantic and restriction lints; explain narrow exceptions
  with `#[expect(..., reason = "...")]`.
- Do not use unsafe code; the manifest denies it.
- Do not run `cargo fmt` unless specifically requested.

## Design Notes

- `LaserDataset` reads `index.csv`, applies filters before resolving retained
  item paths, and uses a fixed shuffle seed of `42`.
- Burn's `Mapper` is infallible. Avoid hiding fallible image decoding inside a
  mapper with `unwrap`; a fallible loading design is tracked in
  `docs/plan-dataset-loading.md`.
- CSV source-schema enforcement and root path containment need explicit policy
  decisions; see `docs/plan-source-validation.md`.
- Augmentation parameters should be validated by their dedicated value types;
  keep image processing at runtime and preserve seeded RNG behavior.
