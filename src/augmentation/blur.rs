//! Blur augmentations and shared validation for blur radii.

/// Gaussian blurring with enhanced control.
mod advanced_gaussian;
/// 3x3 convolution filter.
mod box_filter;
/// Faster gaussian blurring.
mod fast_gaussian;
/// Normal gaussian blurring.
mod gaussian;

use image::{
	DynamicImage,
	Pixel,
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};

pub use self::{
	advanced_gaussian::GaussianAdvanced,
	box_filter::{
		BoxFilter,
		BoxFilterKernel,
	},
	fast_gaussian::{
		GaussianFast,
		RandomGaussianFast,
	},
	gaussian::{
		Gaussian,
		RandomGaussian,
	},
};
use super::{
	AugmentationError,
	GenericImageBuffer,
};

/// A validated finite, non-negative blur radius.
///
/// The default radius is zero, which disables Gaussian blur transforms.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// # 	AugmentationError,
/// # 	blur::BlurSigma,
/// # };
/// #
/// const SIGMA: BlurSigma = match BlurSigma::new(1.25) {
/// 	Ok(sigma) => sigma,
/// 	Err(_) => panic!("invalid constant blur sigma"),
/// };
///
/// assert_eq!(SIGMA.value(), 1.25);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct BlurSigma(f32);

impl BlurSigma {
	/// Validates a finite, non-negative blur radius.
	///
	/// This `const fn` can be evaluated during compilation. Matching its result
	/// in a constant initialiser turns invalid input into a compile-time error.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigma`] for a negative or
	/// non-finite value.
	pub const fn new(value: f32) -> Result<Self, AugmentationError> {
		if value.is_finite() && value >= 0.0 {
			Ok(Self(value))
		} else {
			Err(AugmentationError::InvalidBlurSigma)
		}
	}

	/// Returns this validated radius.
	pub const fn value(self) -> f32 {
		self.0
	}
}

/// A half-open range of finite, non-negative blur radii.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// # 	AugmentationError,
/// # 	blur::BlurSigmaRange,
/// # };
/// #
/// const RANGE: BlurSigmaRange = match BlurSigmaRange::new(0.0, 2.0) {
/// 	Ok(range) => range,
/// 	Err(_) => panic!("invalid constant blur range"),
/// };
///
/// assert_eq!(RANGE.bounds(), (0.0, 2.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BlurSigmaRange {
	min: f32,
	max: f32,
}

impl BlurSigmaRange {
	/// Validates a half-open range of finite, non-negative radii.
	///
	/// Requires `min < max`. Matching the result in a constant initialiser
	/// makes invalid constant input a compile-time error.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigmaRange`] when either bound
	/// is non-finite, `min` is negative, or `min >= max`.
	pub const fn new(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		if min.is_finite() && max.is_finite() && min >= 0.0 && min < max {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidBlurSigmaRange)
		}
	}

	/// Returns the lower-inclusive and upper-exclusive bounds.
	pub const fn bounds(self) -> (f32, f32) {
		(self.min, self.max)
	}

	/// Samples one radius from the half-open range.
	fn sample(
		self,
		rng: &mut StdRng,
	) -> BlurSigma {
		BlurSigma(rng.random_range(self.min..self.max))
	}
}

/// Applies a blur unless its validated radius is zero.
fn apply_blur<P>(
	image: GenericImageBuffer<P>,
	sigma: BlurSigma,
	blur: fn(&GenericImageBuffer<P>, f32) -> GenericImageBuffer<P>,
) -> GenericImageBuffer<P>
where
	P: Pixel,
	GenericImageBuffer<P>: Into<DynamicImage>,
{
	if sigma.0 == 0.0 {
		return image;
	}
	blur(&image, sigma.0)
}
