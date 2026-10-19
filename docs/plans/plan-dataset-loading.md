# Plan: Reproducible and Error-Aware Dataset Loading

## Goal

Make dataset ordering configurable and let callers handle image decoding
failures without panics inside infallible Burn mapper closures.

## Scope

- Add a non-breaking constructor or builder that accepts a shuffle seed while
  preserving seed-42 behavior in existing constructors.
- Offer an error-aware loading/mapping API that can return `LoaderError` for
  image decode failures without changing Burn's `Mapper` contract.
- Keep the existing lazy `LaserDataset` API available for compatibility.
- Document eager versus lazy loading, error timing, and reproducibility.

## Steps

1. Finish schema and root-containment decisions in
   [the source validation plan](plan-source-validation.md).
2. Review Burn dataset and mapper extension points to choose whether the new
   API should be an eager loader, a separate fallible dataset trait, or a
   constructor returning materialised results.
3. Specify seed defaults and whether shuffling occurs before or after filtering.
4. Add deterministic ordering tests for equal and different seeds.
5. Add decode-failure tests proving errors are returned to callers.
6. Document migration choices without deprecating the existing API until a
   separately reviewed release decision.

## Acceptance Criteria

- Existing constructor signatures and default ordering remain unchanged.
- Identical source data and seed produce identical order.
- Image decode failures can be observed as errors through the new API.
- Existing and new usage paths have runnable documentation examples.
