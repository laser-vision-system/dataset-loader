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

#![expect(
	clippy::module_name_repetitions,
	reason = "Naming in this crate follows a standard pattern for ease of use"
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
