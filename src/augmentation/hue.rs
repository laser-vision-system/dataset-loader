//! Random hue rotation transforms.

use image::{
	DynamicImage,
	ImageBuffer,
	Pixel,
	imageops::huerotate,
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};

use super::{
	AugmentationError,
	ImageStep,
};

/// A validated half-open range of hue-rotation degrees.
///
/// The constructors are `const`. To require a valid range during compilation,
/// match the [`std::result::Result`] in a `const` initializer; merely storing
/// a result constant does not require its `Ok` variant.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// #     AugmentationError,
/// #     hue::HueRange,
/// # };
/// #
/// const ROTATION: HueRange = match HueRange::strict(-45, 45) {
/// 	Ok(range) => range,
/// 	Err(_) => panic!("invalid constant hue range"),
/// };
///
/// let _rotation = ROTATION;
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HueRange {
	min: i32,
	max: i32,
}

impl HueRange {
	/// Creates a range by ordering the supplied values.
	///
	/// The minimum and maximum are selected from the two input bounds.
	/// Equal values produce a no-op transform. This constructor is useful when
	/// bounds come from user input and reversed order should not be an error.
	///
	/// This function is `const` and cannot fail.
	pub const fn relaxed(
		first: i32,
		second: i32,
	) -> Self {
		let (min, max) = if first <= second {
			(first, second)
		} else {
			(second, first)
		};
		Self {
			min,
			max,
		}
	}

	/// Creates an ordered range while allowing equal bounds.
	///
	/// Returns [`AugmentationError::InvalidHueRange`] only when `min > max`.
	/// Equal bounds represent a no-op range. Use this when reversed arguments
	/// should be rejected but a zero-width range is meaningful.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidHueRange`] when `min > max`.
	pub const fn ordered(
		min: i32,
		max: i32,
	) -> Result<Self, AugmentationError> {
		if min <= max {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidHueRange)
		}
	}

	/// Creates a strict hue rotation range in the half-open degree range
	/// `min..max`.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidHueRange`] when `min >= max`.
	pub const fn strict(
		min: i32,
		max: i32,
	) -> Result<Self, AugmentationError> {
		if min < max {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidHueRange)
		}
	}
}

/// Applies a random hue rotation from a validated [`HueRange`].
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::hue::{
/// # 	HueRange,
/// # 	HueRotate,
/// # };
/// #
/// const RANGE: HueRange = HueRange::relaxed(-30, 30);
/// const ROTATION: HueRotate = HueRotate::new(RANGE);
/// ```
#[derive(Clone, Debug)]
pub struct HueRotate {
	range: HueRange,
}

impl HueRotate {
	/// Creates a hue rotation transform with a validated range.
	pub const fn new(range: HueRange) -> Self {
		Self {
			range,
		}
	}
}

impl Default for HueRotate {
	/// Returns a transform sampling rotations from `0..360` degrees.
	fn default() -> Self {
		Self {
			range: HueRange {
				min: 0,
				max: 360,
			},
		}
	}
}

impl ImageStep for HueRotate {
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		if self.range.min == self.range.max {
			return image;
		}
		let value = rng.random_range(self.range.min..self.range.max);
		// Whole-turn rotations do not change hue and need no image pass.
		if value.rem_euclid(360_i32) == 0_i32 {
			return image;
		}
		huerotate(&image, value)
	}
}

#[cfg(test)]
mod tests {
	use image::RgbImage;
	use rand::SeedableRng as _;

	use super::*;

	#[test]
	fn default_hue_rotation_changes_color_and_preserves_dimensions() {
		let hue_rotate = HueRotate::default();
		let mut rng = StdRng::seed_from_u64(42);
		let image = RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0]));
		let result = hue_rotate.apply(image.clone(), &mut rng);
		assert_eq!(result.dimensions(), image.dimensions());
		assert_ne!(result, image);
	}

	#[test]
	fn fixed_rotation_changes_color() {
		let original = RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
		let mut rng = StdRng::seed_from_u64(42);
		let range = HueRange::strict(90, 91).expect("the hue range is valid");
		let rotated = HueRotate::new(range).apply(original.clone(), &mut rng);
		assert_ne!(rotated, original);
	}

	#[test]
	fn strict_range_rejects_matching_bounds() {
		assert!(matches!(
			HueRange::strict(10, 10),
			Err(AugmentationError::InvalidHueRange)
		));
	}

	#[test]
	fn strict_range_rejects_reversed_bounds() {
		assert!(matches!(
			HueRange::strict(20, 10),
			Err(AugmentationError::InvalidHueRange)
		));
	}

	#[test]
	fn ordered_range_accepts_equal_bounds_and_rejects_reversed_bounds() {
		let no_op_range = HueRange::ordered(10, 10)
			.expect("equal bounds are a valid no-op range");
		assert_eq!((no_op_range.min, no_op_range.max), (10, 10));
		assert!(matches!(
			HueRange::ordered(20, 10),
			Err(AugmentationError::InvalidHueRange)
		));
	}

	#[test]
	fn hue_range_validation_is_const_evaluable() {
		const RANGE: HueRange = match HueRange::strict(-45, 45) {
			Ok(range) => range,
			Err(_) => panic!("invalid test hue range"),
		};
		assert_eq!((RANGE.min, RANGE.max), (-45, 45));
	}

	#[test]
	fn relaxed_equal_bounds_are_a_no_op() {
		let original = RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
		let range = HueRange::relaxed(10, 10);
		let result = HueRotate::new(range)
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));
		assert_eq!(result, original);
	}

	#[test]
	fn relaxed_bounds_are_ordered() {
		let relaxed_range = HueRange::relaxed(90, -90);
		let relaxed = HueRotate::new(relaxed_range);
		let strict_range =
			HueRange::strict(-90, 90).expect("ordered range is valid");
		let strict = HueRotate::new(strict_range);

		let original = RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
		let relaxed_result =
			relaxed.apply(original.clone(), &mut StdRng::seed_from_u64(42));
		let strict_result =
			strict.apply(original, &mut StdRng::seed_from_u64(42));
		assert_eq!(relaxed_result, strict_result);
	}
}
