//! Random horizontal and vertical flips with validated probabilities.

use image::{
	DynamicImage,
	Pixel,
	imageops::{
		flip_horizontal_in_place,
		flip_vertical_in_place,
	},
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};
use typed_floats::tf64;

use super::{
	AugmentationError,
	GenericImageBuffer,
	ImageStep,
};

const HALF_PROBABILITY: tf64::PositiveFinite =
	match tf64::PositiveFinite::new(0.5) {
		Ok(probability) => probability,
		Err(_) => panic!("0.5 is a positive finite value"),
	};

/// A finite flip probability in the inclusive range `0.0..=1.0`.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// #     AugmentationError,
/// #     flip::FlipProbability,
/// # };
/// #
/// const PROBABILITY: FlipProbability = match FlipProbability::new(0.5) {
/// 	Ok(probability) => probability,
/// 	Err(_) => panic!("invalid constant probability"),
/// };
///
/// assert_eq!(PROBABILITY.value(), 0.5);
/// ```
///
/// A bad value in the same `const match` fails compilation:
///
/// ```compile_fail
/// # use dataset_loader::augmentation::flip::FlipProbability;
/// #
/// // This will raise a compile-time error.
/// const INVALID: FlipProbability = match FlipProbability::new(1.5) {
///     Ok(probability) => probability,
///     Err(_) => panic!("invalid constant probability"),
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipProbability(tf64::PositiveFinite);

impl FlipProbability {
	/// Creates a validated flip probability.
	///
	/// Returns [`AugmentationError::InvalidFlipProbability`] if `value` is
	/// non-finite or outside `0.0..=1.0`. This function is `const`.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidFlipProbability`] when the value is
	/// not finite or outside the inclusive range `0.0..=1.0`.
	pub const fn try_new(value: f64) -> Result<Self, AugmentationError> {
		match tf64::PositiveFinite::new(value) {
			Ok(probability) if probability.get() <= 1.0 => {
				Ok(Self(probability))
			},
			_ => Err(AugmentationError::InvalidFlipProbability),
		}
	}

	/// Returns the probability as a floating-point value.
	pub const fn value(self) -> f64 {
		self.0.get()
	}
}

/// Randomly mirrors an image left-to-right.
///
/// Its default probability is 50%.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::flip::{
/// # 	FlipProbability,
/// # 	RandomHorizontalFlip,
/// # };
/// #
/// let probability =
/// 	FlipProbability::new(0.75).expect("the probability is in range");
/// let _flip = RandomHorizontalFlip::new(probability);
/// ```
#[derive(Clone, Debug)]
pub struct RandomHorizontalFlip {
	probability: FlipProbability,
}

impl RandomHorizontalFlip {
	/// Creates a horizontal flip with probability in `0.0..=1.0`.
	///
	/// The probability must first be validated with [`FlipProbability::new`].
	/// This constructor is infallible and `const`.
	pub const fn new(probability: FlipProbability) -> Self {
		Self {
			probability,
		}
	}
}

impl Default for RandomHorizontalFlip {
	/// Returns a transform with a 50% flip probability.
	fn default() -> Self {
		Self {
			probability: FlipProbability(HALF_PROBABILITY),
		}
	}
}

impl ImageStep for RandomHorizontalFlip {
	fn apply<P>(
		&self,
		mut image: GenericImageBuffer<P>,
		rng: &mut StdRng,
	) -> GenericImageBuffer<P>
	where
		P: Pixel + 'static,
		GenericImageBuffer<P>: Into<DynamicImage>,
	{
		if rng.random_bool(self.probability.0.get()) {
			flip_horizontal_in_place(&mut image);
		}
		image
	}
}

/// Randomly mirrors an image top-to-bottom.
///
/// Its default probability is 50%.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::flip::{
/// # 	FlipProbability,
/// # 	RandomVerticalFlip,
/// # };
/// #
/// let probability =
/// 	FlipProbability::new(0.25).expect("the probability is in range");
/// let _flip = RandomVerticalFlip::new(probability);
/// ```
#[derive(Clone, Debug)]
pub struct RandomVerticalFlip {
	probability: FlipProbability,
}

impl RandomVerticalFlip {
	/// Creates a vertical flip with probability in `0.0..=1.0`.
	///
	/// The probability must first be validated with [`FlipProbability::new`].
	/// This constructor is infallible and `const`.
	pub const fn new(probability: FlipProbability) -> Self {
		Self {
			probability,
		}
	}
}

impl Default for RandomVerticalFlip {
	/// Returns a transform with a 50% flip probability.
	fn default() -> Self {
		Self {
			probability: FlipProbability(HALF_PROBABILITY),
		}
	}
}

impl ImageStep for RandomVerticalFlip {
	fn apply<P>(
		&self,
		mut image: GenericImageBuffer<P>,
		rng: &mut StdRng,
	) -> GenericImageBuffer<P>
	where
		P: Pixel + 'static,
		GenericImageBuffer<P>: Into<DynamicImage>,
	{
		if rng.random_bool(self.probability.0.get()) {
			flip_vertical_in_place(&mut image);
		}
		image
	}
}

#[cfg(test)]
mod tests {
	use image::{
		Rgb,
		RgbImage,
	};
	use rand::SeedableRng as _;

	use super::*;

	#[test]
	fn horizontal_flip_reverses_each_row() {
		let original = RgbImage::from_fn(2, 2, |x, y| {
			Rgb([u8::from(x == 1), u8::from(y == 1), 0])
		});
		let mut rng = StdRng::seed_from_u64(42);
		let prob =
			FlipProbability::try_new(1.0).expect("probability one is valid");
		let step = RandomHorizontalFlip::new(prob);

		let image = step.apply(original, &mut rng);

		assert_eq!(image.get_pixel(0, 0).0, [1, 0, 0]);
		assert_eq!(image.get_pixel(1, 0).0, [0, 0, 0]);
		assert_eq!(image.get_pixel(0, 1).0, [1, 1, 0]);
	}

	#[test]
	fn vertical_flip_reverses_each_column() {
		let original = RgbImage::from_fn(2, 2, |x, y| {
			Rgb([u8::from(x == 1), u8::from(y == 1), 0])
		});
		let mut rng = StdRng::seed_from_u64(42);
		let prob =
			FlipProbability::try_new(1.0).expect("probability one is valid");
		let step = RandomVerticalFlip::new(prob);

		let image = step.apply(original, &mut rng);

		assert_eq!(image.get_pixel(0, 0).0, [0, 1, 0]);
		assert_eq!(image.get_pixel(1, 0).0, [1, 1, 0]);
		assert_eq!(image.get_pixel(0, 1).0, [0, 0, 0]);
	}

	#[test]
	fn zero_probability_preserves_pixels() {
		let original = RgbImage::from_fn(2, 2, |x, y| {
			Rgb([u8::from(x == 1), u8::from(y == 1), 0])
		});
		let mut rng = StdRng::seed_from_u64(42);
		let prob =
			FlipProbability::try_new(0.0).expect("probability zero is valid");
		let image =
			RandomHorizontalFlip::new(prob).apply(original.clone(), &mut rng);
		assert_eq!(image, original);
	}

	#[test]
	fn flip_probability_rejects_out_of_range_and_non_finite_values() {
		for probability in [-0.1, 1.1, f64::INFINITY, f64::NAN] {
			assert!(matches!(
				FlipProbability::try_new(probability),
				Err(AugmentationError::InvalidFlipProbability)
			));
		}
	}

	#[test]
	fn probability_validation_is_const_evaluable() {
		const VALID: FlipProbability = match FlipProbability::try_new(0.5) {
			Ok(probability) => probability,
			Err(_) => panic!("invalid test probability"),
		};
		assert_eq!(VALID.value(), 0.5);
	}
}
