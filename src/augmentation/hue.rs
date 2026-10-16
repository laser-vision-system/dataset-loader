use image::imageops::huerotate;
use rand::{
	RngExt as _,
	rngs::StdRng,
};

use super::{
	ImageStep,
	ImageType,
};

#[derive(Clone, Copy, Debug)]
struct HueRange {
	min: i32,
	max: i32,
}

impl HueRange {
	fn new(
		min: i32,
		max: i32,
	) -> Option<Self> {
		(min < max).then_some(Self {
			min,
			max,
		})
	}
}

/// Applies a random hue rotation in the half-open degree range `min..max`.
#[derive(Clone, Debug)]
pub struct HueRotate {
	range: HueRange,
}

impl HueRotate {
	/// Creates a hue rotation in the half-open degree range `min..max`.
	pub fn new(
		min: i32,
		max: i32,
	) -> Option<Self> {
		Some(Self {
			range: HueRange::new(min, max)?,
		})
	}
}

impl Default for HueRotate {
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
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		let value = rng.random_range(self.range.min..self.range.max);
		if value.rem_euclid(360_i32) == 0_i32 {
			return image;
		}
		huerotate(&image, value)
	}
}

#[cfg(test)]
mod tests {
	use rand::SeedableRng as _;

	use super::*;

	#[test]
	fn default_hue_rotation_changes_color_and_preserves_dimensions() {
		let hue_rotate = HueRotate::default();
		let mut rng = StdRng::seed_from_u64(42);
		let image = ImageType::from_pixel(10, 10, image::Rgb([255, 0, 0]));
		let result = hue_rotate.apply(image.clone(), &mut rng);
		assert_eq!(result.dimensions(), image.dimensions());
		assert_ne!(result, image);
	}

	#[test]
	fn fixed_rotation_changes_color() {
		let original = ImageType::from_pixel(1, 1, image::Rgb([255, 0, 0]));
		let mut rng = StdRng::seed_from_u64(42);
		let rotated = HueRotate::new(90, 91)
			.unwrap_or_default()
			.apply(original.clone(), &mut rng);
		assert_ne!(rotated, original);
	}

	#[test]
	fn range_rejects_matching_bounds() {
		assert!(HueRotate::new(10, 10).is_none());
	}

	#[test]
	fn range_rejects_reversed_bounds() {
		assert!(HueRotate::new(20, 10).is_none());
	}
}
