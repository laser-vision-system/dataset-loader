//! Simple interface for loading and using a dataset from one of the sources.
//!
//! Information about the datasets:
//!
//! - A dataset consists of a collection of dataset items.
//!    - Each item has a path to an image.
//!    - Each item has an intensity.
//!       - The intensity can be determined with the path.
//!       - The way the intensity is determined is different for each source.
//!    - Some items have additional fields.
//!       - The additional fields can also be determined with the path.

#![cfg_attr(
	test,
	allow(
		clippy::print_stdout,
		clippy::print_stderr,
		clippy::use_debug,
		clippy::panic,
		clippy::panic_in_result_fn,
		clippy::unwrap_used,
		clippy::unwrap_in_result,
		clippy::expect_used,
		clippy::indexing_slicing,
		reason = "Tests are allowed to perform all of these operations"
	)
)]

pub mod dataset;
pub mod error;
pub mod item;
pub mod sources;
pub mod splitting;

pub use error::{
	LoaderError,
	Result,
};
pub use splitting::{
	Split,
	SplitOutput,
	SplitType,
	Splitable,
};
