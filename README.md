# Dataset Loader

Rust crate for loading laser-image datasets from CSV indices, mapping records
into training data, splitting datasets, and composing image augmentations.

## Dataset layout

Each dataset root contains an `index.csv` file and the image files referenced
by its `relative_path` column. CSV fields are deserialised into the selected
source item type; `position` is optional. Source-specific metadata fields are
defined by the item structs in `src/sources/`.

Use a source module's `get_dataset` function with a Burn `Mapper`, the dataset
root, and optional predicates. The dataset adapter canonicalises retained
image paths and exposes Burn's `Dataset` interface. Image decoding happens
when an item is loaded by a mapper.

The `augmentation` module provides CPU image transforms that compose with
`ImageStep::then`; stochastic transforms accept a seeded `StdRng` when applied.
The `splitting` module provides weighted train/validation/test partitions.

## Validation

Run unit and documentation tests with `cargo test`, then run the strict lint
configuration with `cargo clippy`.

To enable the pre-commit hook, run `git config core.hooksPath .githooks`.
