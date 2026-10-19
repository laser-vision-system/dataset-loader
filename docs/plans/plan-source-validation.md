# Plan: Validate Source Rows and Dataset Paths

## Goal

Ensure records loaded from `index.csv` obey documented source-specific intensity
and metadata constraints, and define whether image paths may escape the dataset
root.

## Scope

- Specify CSV columns and accepted value domains for every source item.
- Enforce synthetic source intensity bounds for CSV rows as well as filename
  conversions.
- Decide whether analogous and in-lab metadata ranges are validation rules or
  descriptive observations; do not silently reject previously accepted data.
- Define treatment of absolute paths, `..` components, symlinks, and paths that
  resolve outside the root.
- Return actionable `LoaderError` messages while preserving public signatures
  where possible.

## Steps

1. Document representative `index.csv` headers and valid rows for each source.
2. Add source item validation after CSV deserialization and before filtering or
   path access.
3. Add CSV-backed tests for range boundaries, malformed values, optional fields,
   and rejected records using temporary directories.
4. Canonicalise the root once and verify resolved image paths satisfy the
   selected containment policy, including symlink cases.
5. Update the README and Rustdoc error sections with the settled schema and
   path policy.

## Acceptance Criteria

- CSV and filename construction reject the same documented synthetic ranges.
- Tests cover every accepted source constraint and path boundary policy.
- No path-policy change ships without documenting compatibility impact and
  symlink behavior.

## Dependencies

Complete this plan before implementing error-aware lazy image loading from
[the dataset loading ergonomics plan](plan-dataset-loading.md), because that
work relies on a stable validation and error contract.
