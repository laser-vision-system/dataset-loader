//! Chainable image augmentations.
//!
//! The central type is the [`ImageStep`] trait. Any type that implements it can
//! be used as an image augmentation.
//!
//! Implementers can use [`ImageStep::then`] to chain successive image
//! augmentations. This method returns a [`Sequence`] struct, which itself
//! implements [`ImageStep`], allowing repeatedly linking implementers to build
//! the augmentation chain.
//!
//! The linking is lazy, so nothing is executed until [`ImageStep::apply`] is
//! called.
//!
//! Notably, this module includes the [`NoOp`] type, to use when image
//! augmentation is not needed, but must be supplied.
//!
//! # Examples
//!
//! ```
//! # use dataset_loader::augmentation::{
//! # 	ImageStep as _,
//! # 	Sequence,
//! # 	flip::{
//! # 		RandomHorizontalFlip,
//! # 		RandomVerticalFlip,
//! # 	},
//! # 	hue::HueRotate,
//! # };
//! #
//! // A builder function for a pre-defined pipeline.
//! fn random_flip_augmentation_pipeline()
//! -> Option<Sequence<RandomHorizontalFlip, RandomVerticalFlip>> {
//! 	let pipeline =
//! 		RandomHorizontalFlip::new(0.5)?.then(RandomVerticalFlip::new(0.5)?);
//! 	Some(pipeline)
//! }
//!
//! # fn run() -> Option<()> {
//! // Get the pipeline, then modify it further.
//! let flip_pipeline = random_flip_augmentation_pipeline()?;
//! let full_pipeline = flip_pipeline.then(HueRotate::new(-180, 180)?);
//! # Some(())
//! # }
//! #
//! # fn main() -> Result<(), ()> {
//! # 	run().ok_or(())
//! # }
//! ```

pub mod blur;
pub mod flip;
pub mod hue;
pub mod zoom;

use rand::rngs::StdRng;

use crate::item::ImageType;

/// An image augmentation step that runs on the CPU.
///
/// Implementations can be deterministic or use the provided random generator
/// for seeded stochastic behavior.
pub trait ImageStep: Clone + Send + Sync {
	/// Transform an owned RGB image and return the result.
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType;

	/// Append a step to the augmentation pipeline.
	fn then<N>(
		self,
		next: N,
	) -> Sequence<Self, N>
	where
		Self: Sized,
		N: ImageStep,
	{
		Sequence {
			first: self,
			next,
		}
	}
}

/// Two sequential augmentation steps.
#[derive(Clone, Debug)]
pub struct Sequence<F, N>
where
	F: ImageStep,
	N: ImageStep,
{
	first: F,
	next: N,
}

impl<F, N> ImageStep for Sequence<F, N>
where
	F: ImageStep,
	N: ImageStep,
{
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		let image = self.first.apply(image, rng);
		self.next.apply(image, rng)
	}
}

/// An augmentation pipeline step which leaves the image unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOp;

impl ImageStep for NoOp {
	fn apply(
		&self,
		image: ImageType,
		_rng: &mut StdRng,
	) -> ImageType {
		image
	}
}

#[cfg(test)]
mod tests {
	use image::Rgb;
	use rand::{
		RngExt as _,
		SeedableRng as _,
	};

	use super::*;

	#[derive(Clone)]
	struct SetRed(u8);

	impl ImageStep for SetRed {
		fn apply(
			&self,
			mut image: ImageType,
			_rng: &mut StdRng,
		) -> ImageType {
			for pixel in image.pixels_mut() {
				pixel.0[0] = self.0;
			}
			image
		}
	}

	#[derive(Clone)]
	struct RandomRed;

	impl ImageStep for RandomRed {
		fn apply(
			&self,
			mut image: ImageType,
			rng: &mut StdRng,
		) -> ImageType {
			let red = rng.random_range(0..=u8::MAX);
			for pixel in image.pixels_mut() {
				pixel.0[0] = red;
			}
			image
		}
	}

	#[test]
	fn sequence_applies_steps_in_order() {
		let pipeline = SetRed(10).then(SetRed(20));
		let image = ImageType::from_pixel(1, 1, Rgb([0, 0, 0]));
		let mut rng = StdRng::seed_from_u64(42);

		let image = pipeline.apply(image, &mut rng);

		assert_eq!(image.get_pixel(0, 0).0[0], 20);
	}

	#[test]
	fn no_augmentation_preserves_image() {
		let original = ImageType::from_pixel(1, 1, Rgb([1, 2, 3]));
		let mut rng = StdRng::seed_from_u64(42);

		let image = NoOp.apply(original.clone(), &mut rng);

		assert_eq!(image, original);
	}

	#[test]
	fn random_steps_are_reproducible_from_seed() {
		let pipeline = RandomRed;
		let first_image = ImageType::from_pixel(1, 1, Rgb([0, 0, 0]));
		let second_image = first_image.clone();
		let mut first_rng = StdRng::seed_from_u64(42);
		let mut second_rng = StdRng::seed_from_u64(42);

		let first_image = pipeline.apply(first_image, &mut first_rng);
		let second_image = pipeline.apply(second_image, &mut second_rng);

		assert_eq!(first_image, second_image);
	}
}
