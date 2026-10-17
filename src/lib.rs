//! Load, inspect, split, and augment laser-image datasets.
//!
//! Information about the datasets:
//!
//! - A dataset consists of a collection of dataset items.
//!    - Each item has a path to an image.
//!    - Each item has an intensity.
//!       - Source item types can parse labels from filenames where supported.
//!       - CSV indices provide the canonical item metadata for dataset loaders.
//!    - Some items have additional fields.
//!       - The additional fields can also be determined with the path.
//!
//! Dataset source modules expose item types and `get_dataset` constructors.
//! The [`augmentation`] module provides composable image transforms, and
//! [`splitting`] provides weighted train/validation/test partitions.
//!
//! # Examples
//!
//! ```
//! use dataset_loader::augmentation::{
//! 	ImageStep as _,
//! 	NoOp,
//! };
//! use image::RgbImage;
//! use rand::SeedableRng as _;
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let image = NoOp.apply(RgbImage::new(1, 1), &mut rng);
//! assert_eq!(image.dimensions(), (1, 1));
//! ```

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

pub mod augmentation;
pub mod dataset;
pub mod error;
pub mod item;
pub mod sources;
pub mod splitting;

pub use self::{
	error::{
		LoaderError,
		Result,
	},
	splitting::{
		PositiveFinite,
		Split,
		SplitOutput,
		SplitType,
		SplitWeights,
		Splitable,
		StrictlyPositiveFinite,
	},
};
