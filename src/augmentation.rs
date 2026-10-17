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
//! Raw parameter values are validated by dedicated value types. Their
//! constructors return [`std::result::Result`] and can be evaluated in a
//! constant expression; matching the result there makes invalid constants a
//! compile-time error. Augmentation constructors then accept only validated
//! values and are infallible.
//!
//! Image processing and [`ImageStep`] methods run at runtime. Pipelines can
//! still be assembled in constants with [`Sequence::new`], but applying a
//! pipeline at compile time is not supported.
//!
//! # Examples
//!
//! ```
//! # use dataset_loader::augmentation::{
//! #     ImageStep as _,
//! #     flip::{FlipProbability, RandomHorizontalFlip, RandomVerticalFlip},
//! #     hue::{HueRange, HueRotate},
//! #     NoOp,
//! #     Sequence,
//! # };
//! # use rand::SeedableRng as _;
//! #
//! const PROBABILITY: FlipProbability = match FlipProbability::new(0.5) {
//! 	Ok(probability) => probability,
//! 	Err(_) => panic!("invalid constant flip probability"),
//! };
//!
//! const FLIPS: Sequence<RandomHorizontalFlip, RandomVerticalFlip> =
//! 	Sequence::new(
//! 		RandomHorizontalFlip::new(PROBABILITY),
//! 		RandomVerticalFlip::new(PROBABILITY),
//! 	);
//!
//! const HUE_RANGE: HueRange = match HueRange::strict(-180, 180) {
//! 	Ok(range) => range,
//! 	Err(_) => panic!("invalid constant hue range"),
//! };
//!
//! // `then` and image processing execute at runtime.
//! let full_pipeline = FLIPS.then(HueRotate::new(HUE_RANGE)).then(NoOp);
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let _image = full_pipeline.apply(image::RgbImage::new(1, 1), &mut rng);
//! ```

pub mod blur;
pub mod flip;
pub mod hue;
pub mod zoom;

use image::{
	DynamicImage,
	ImageBuffer,
	Pixel,
};
use rand::rngs::StdRng;
use thiserror::Error;

/// A validation error returned when augmentation parameters are invalid.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum AugmentationError {
	/// Hue bounds are not in strictly increasing order.
	#[error("hue range minimum must be less than its maximum")]
	InvalidHueRange,
	/// A flip probability is non-finite or outside `0.0..=1.0`.
	#[error("flip probability must be finite and in 0.0..=1.0")]
	InvalidFlipProbability,
	/// Zoom-in bounds are non-finite, reversed, or less than one.
	#[error("zoom-in factors must be finite, ordered, and at least 1.0")]
	InvalidZoomInFactors,
	/// Zoom-out bounds are non-finite, reversed, or less than one.
	#[error("zoom-out divisors must be finite, ordered, and at least 1.0")]
	InvalidZoomOutDivisors,
	/// A blur sigma is negative or non-finite.
	#[error("blur sigma must be finite and non-negative")]
	InvalidBlurSigma,
	/// Blur range bounds are non-finite, negative, or not strictly increasing.
	#[error("blur sigma range must be finite, non-negative, and increasing")]
	InvalidBlurSigmaRange,
	/// A box-filter kernel contains a non-finite coefficient.
	#[error("box-filter kernel coefficients must all be finite")]
	InvalidBoxFilterKernel,
}

/// An image augmentation step that runs on the CPU.
///
/// Implementations can be deterministic or use the provided random generator
/// for seeded stochastic behavior.
///
/// # Examples
///
/// ```
/// use dataset_loader::augmentation::{
/// 	ImageStep,
/// 	NoOp,
/// };
/// use image::RgbImage;
/// use rand::SeedableRng as _;
///
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let image = NoOp.apply(RgbImage::new(1, 1), &mut rng);
/// assert_eq!(image.dimensions(), (1, 1));
/// ```
pub trait ImageStep: Clone + Send + Sync {
	/// Transforms an owned image buffer and returns the result at runtime.
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>;

	/// Appends a step to the augmentation pipeline for runtime execution.
	///
	/// Use [`Sequence::new`] when a pipeline must be assembled in a constant
	/// context.
	fn then<N>(
		self,
		next: N,
	) -> Sequence<Self, N>
	where
		Self: Sized,
		N: ImageStep,
	{
		Sequence::new(self, next)
	}
}

/// Two sequential augmentation steps.
///
/// `Sequence::new` is `const`, so validated steps can be composed in a
/// constant. Applying the sequence remains a runtime operation.
///
/// # Examples
///
/// ```
/// use dataset_loader::augmentation::{
/// 	ImageStep,
/// 	NoOp,
/// 	Sequence,
/// };
/// use image::RgbImage;
/// use rand::SeedableRng as _;
///
/// const PIPELINE: Sequence<NoOp, NoOp> = Sequence::new(NoOp, NoOp);
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let image = PIPELINE.apply(RgbImage::new(1, 1), &mut rng);
/// assert_eq!(image.dimensions(), (1, 1));
/// ```
#[derive(Clone, Debug)]
pub struct Sequence<F, N>
where
	F: ImageStep,
	N: ImageStep,
{
	first: F,
	next: N,
}

impl<F, N> Sequence<F, N>
where
	F: ImageStep,
	N: ImageStep,
{
	/// Creates a sequence from two validated augmentation steps.
	///
	/// This `const` constructor stores the steps without running them.
	pub const fn new(
		first: F,
		next: N,
	) -> Self {
		Self {
			first,
			next,
		}
	}
}

impl<F, N> ImageStep for Sequence<F, N>
where
	F: ImageStep,
	N: ImageStep,
{
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		let image = self.first.apply(image, rng);
		self.next.apply(image, rng)
	}
}

/// An augmentation pipeline step which leaves the image unchanged.
///
/// # Examples
///
/// ```
/// use dataset_loader::augmentation::{
/// 	ImageStep,
/// 	NoOp,
/// };
/// use image::RgbImage;
/// use rand::SeedableRng as _;
///
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let image = NoOp.apply(RgbImage::new(2, 3), &mut rng);
/// assert_eq!(image.dimensions(), (2, 3));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOp;

impl ImageStep for NoOp {
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		_rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		image
	}
}

#[cfg(test)]
mod tests {
	use image::{
		Rgb,
		RgbImage,
	};
	use rand::{
		RngExt as _,
		SeedableRng as _,
	};

	use super::*;

	#[derive(Clone)]
	struct SetFirstSubPixelValue(u8);

	impl ImageStep for SetFirstSubPixelValue {
		fn apply<P>(
			&self,
			image: ImageBuffer<P, Vec<P::Subpixel>>,
			_rng: &mut StdRng,
		) -> ImageBuffer<P, Vec<P::Subpixel>>
		where
			P: Pixel + 'static,
			ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
		{
			for pixel in image.pixels_mut() {
				pixel.channels_mut()[0] = self.0.into();
			}
			image
		}
	}

	#[derive(Clone)]
	struct RandomFirstSubPixelValue;

	impl ImageStep for RandomFirstSubPixelValue {
		fn apply<P>(
			&self,
			image: ImageBuffer<P, Vec<P::Subpixel>>,
			rng: &mut StdRng,
		) -> ImageBuffer<P, Vec<P::Subpixel>>
		where
			P: Pixel + 'static,
			ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
		{
			let red = rng.random_range(0..=u8::MAX);
			for pixel in image.pixels_mut() {
				pixel.channels_mut()[0] = red.into();
			}
			image
		}
	}

	#[test]
	fn sequence_applies_steps_in_order() {
		let pipeline =
			SetFirstSubPixelValue(10).then(SetFirstSubPixelValue(20));
		let image = RgbImage::from_pixel(1, 1, Rgb([0, 0, 0]));
		let mut rng = StdRng::seed_from_u64(42);

		let image = pipeline.apply(image, &mut rng);

		assert_eq!(image.get_pixel(0, 0).0[0], 20);
	}

	#[test]
	fn no_augmentation_preserves_image() {
		let original = RgbImage::from_pixel(1, 1, Rgb([1, 2, 3]));
		let mut rng = StdRng::seed_from_u64(42);

		let image = NoOp.apply(original.clone(), &mut rng);

		assert_eq!(image, original);
	}

	#[test]
	fn random_steps_are_reproducible_from_seed() {
		let pipeline = RandomFirstSubPixelValue;
		let first_image = RgbImage::from_pixel(1, 1, Rgb([0, 0, 0]));
		let second_image = first_image.clone();
		let mut first_rng = StdRng::seed_from_u64(42);
		let mut second_rng = StdRng::seed_from_u64(42);

		let first_image = pipeline.apply(first_image, &mut first_rng);
		let second_image = pipeline.apply(second_image, &mut second_rng);

		assert_eq!(first_image, second_image);
	}
}
