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
use typed_floats::tf32;

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

const ONE_POSITIVE_FINITE: tf32::PositiveFinite =
	match tf32::PositiveFinite::new(1.0) {
		Ok(value) => value,
		Err(_) => panic!("1.0 is a positive finite value"),
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
/// const SIGMA: BlurSigma = match BlurSigma::try_new(1.25) {
/// 	Ok(sigma) => sigma,
/// 	Err(_) => panic!("invalid constant blur sigma"),
/// };
///
/// assert_eq!(SIGMA.value(), 1.25);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct BlurSigma(tf32::PositiveFinite);

impl BlurSigma {
	/// Validates a finite, non-negative blur radius.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigma`] for a negative or
	/// non-finite value.
	pub const fn try_new(value: f32) -> Result<Self, AugmentationError> {
		match tf32::PositiveFinite::new(value) {
			Ok(sigma) => Ok(Self(sigma)),
			Err(_) => Err(AugmentationError::InvalidBlurSigma),
		}
	}

	/// Returns this validated radius.
	pub const fn value(self) -> f32 {
		self.0.get()
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
/// const RANGE: BlurSigmaRange = match BlurSigmaRange::try_new(0.0, 2.0) {
/// 	Ok(range) => range,
/// 	Err(_) => panic!("invalid constant blur range"),
/// };
///
/// assert_eq!(RANGE.bounds(), (0.0, 2.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BlurSigmaRange {
	min: tf32::PositiveFinite,
	max: tf32::PositiveFinite,
}

impl BlurSigmaRange {
	/// Validates a half-open range of finite, non-negative radii.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigmaRange`] when either bound
	/// is non-finite, `min` is negative, or `min >= max`.
	pub const fn try_new(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		Self::strict(min, max)
	}

	/// Creates a range after sorting the supplied bounds.
	///
	/// Equal bounds are allowed and represent a fixed sampled sigma.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigmaRange`] if either bound is
	/// negative or non-finite.
	pub const fn relaxed(
		first: f32,
		second: f32,
	) -> Result<Self, AugmentationError> {
		let Ok(first) = tf32::PositiveFinite::new(first) else {
			return Err(AugmentationError::InvalidBlurSigmaRange);
		};
		let Ok(second) = tf32::PositiveFinite::new(second) else {
			return Err(AugmentationError::InvalidBlurSigmaRange);
		};
		let (min, max) = if first.get() <= second.get() {
			(first, second)
		} else {
			(second, first)
		};
		Ok(Self {
			min,
			max,
		})
	}

	/// Creates an ordered range while allowing equal bounds.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigmaRange`] for invalid bounds
	/// or when `min > max`.
	pub const fn ordered(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		let Ok(min) = tf32::PositiveFinite::new(min) else {
			return Err(AugmentationError::InvalidBlurSigmaRange);
		};
		let Ok(max) = tf32::PositiveFinite::new(max) else {
			return Err(AugmentationError::InvalidBlurSigmaRange);
		};
		if min.get() <= max.get() {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidBlurSigmaRange)
		}
	}

	/// Creates a strictly increasing half-open blur range.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBlurSigmaRange`] for invalid bounds
	/// or when `min >= max`.
	pub const fn strict(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		let Ok(min) = tf32::PositiveFinite::new(min) else {
			return Err(AugmentationError::InvalidBlurSigmaRange);
		};
		let Ok(max) = tf32::PositiveFinite::new(max) else {
			return Err(AugmentationError::InvalidBlurSigmaRange);
		};
		if min.get() < max.get() {
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
		(self.min.get(), self.max.get())
	}

	/// Samples one radius from the half-open range.
	fn sample(
		self,
		rng: &mut StdRng,
	) -> BlurSigma {
		BlurSigma(
			tf32::PositiveFinite::new(
				rng.random_range(self.min.get()..self.max.get()),
			)
			.unwrap_or_default(),
		)
	}
}

impl Default for BlurSigmaRange {
	/// Returns the half-open range `0.0..1.0`.
	fn default() -> Self {
		Self {
			min: tf32::PositiveFinite::default(),
			max: ONE_POSITIVE_FINITE,
		}
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
	if sigma.0.get() == 0.0 {
		return image;
	}
	blur(&image, sigma.0.get())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn relaxed_range_sorts_bounds_and_allows_equal_bounds() {
		assert_eq!(
			BlurSigmaRange::relaxed(2.0, 0.5).unwrap().bounds(),
			(0.5, 2.0)
		);
		assert_eq!(
			BlurSigmaRange::relaxed(1.0, 1.0).unwrap().bounds(),
			(1.0, 1.0)
		);
	}

	#[test]
	fn ordered_range_allows_equal_bounds_and_rejects_reversed_bounds() {
		assert_eq!(
			BlurSigmaRange::ordered(1.0, 1.0).unwrap().bounds(),
			(1.0, 1.0)
		);
		assert_eq!(
			BlurSigmaRange::ordered(2.0, 1.0).unwrap_err(),
			AugmentationError::InvalidBlurSigmaRange
		);
	}

	#[test]
	fn strict_range_requires_increasing_finite_non_negative_bounds() {
		for bounds in [
			(1.0, 1.0),
			(2.0, 1.0),
			(-1.0, 1.0),
			(0.0, f32::INFINITY),
			(0.0, f32::NAN),
		] {
			assert_eq!(
				BlurSigmaRange::strict(bounds.0, bounds.1).unwrap_err(),
				AugmentationError::InvalidBlurSigmaRange
			);
		}
	}
}
