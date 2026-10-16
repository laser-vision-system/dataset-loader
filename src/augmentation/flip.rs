use image::imageops::{
	flip_horizontal_in_place,
	flip_vertical_in_place,
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};

use super::{
	ImageStep,
	ImageType,
};

#[derive(Clone, Copy, Debug)]
struct FlipProbability(f64);

impl FlipProbability {
	fn new(value: f32) -> Option<Self> {
		(value.is_finite() && (0.0..=1.0).contains(&value))
			.then_some(Self(f64::from(value)))
	}
}

/// Randomly mirrors an image left-to-right, in place.
#[derive(Clone, Debug)]
pub struct RandomHorizontalFlip {
	probability: FlipProbability,
}

impl RandomHorizontalFlip {
	/// Creates a horizontal flip with the given probability in `0.0..=1.0`.
	pub fn new(probability: f32) -> Option<Self> {
		Some(Self {
			probability: FlipProbability::new(probability)?,
		})
	}
}

impl Default for RandomHorizontalFlip {
	/// A 50% chance.
	fn default() -> Self {
		Self {
			probability: FlipProbability(0.5),
		}
	}
}

impl ImageStep for RandomHorizontalFlip {
	fn apply(
		&self,
		mut image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		if rng.random_bool(self.probability.0) {
			flip_horizontal_in_place(&mut image);
		}
		image
	}
}

/// Randomly mirrors an image top-to-bottom, in place.
#[derive(Clone, Debug)]
pub struct RandomVerticalFlip {
	probability: FlipProbability,
}

impl RandomVerticalFlip {
	/// Creates a vertical flip with the given probability in `0.0..=1.0`.
	pub fn new(probability: f32) -> Option<Self> {
		Some(Self {
			probability: FlipProbability::new(probability)?,
		})
	}
}

impl Default for RandomVerticalFlip {
	/// A 50% chance.
	fn default() -> Self {
		Self {
			probability: FlipProbability(0.5),
		}
	}
}

impl ImageStep for RandomVerticalFlip {
	fn apply(
		&self,
		mut image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		if rng.random_bool(self.probability.0) {
			flip_vertical_in_place(&mut image);
		}
		image
	}
}

#[cfg(test)]
mod tests {
	use image::Rgb;
	use rand::SeedableRng as _;

	use super::*;

	#[test]
	fn horizontal_flip_reverses_each_row() {
		let original = ImageType::from_fn(2, 2, |x, y| {
			Rgb([u8::from(x == 1), u8::from(y == 1), 0])
		});
		let mut rng = StdRng::seed_from_u64(42);
		let step = RandomHorizontalFlip::new(1.0).unwrap_or_default();

		let image = step.apply(original, &mut rng);

		assert_eq!(image.get_pixel(0, 0).0, [1, 0, 0]);
		assert_eq!(image.get_pixel(1, 0).0, [0, 0, 0]);
		assert_eq!(image.get_pixel(0, 1).0, [1, 1, 0]);
	}

	#[test]
	fn vertical_flip_reverses_each_column() {
		let original = ImageType::from_fn(2, 2, |x, y| {
			Rgb([u8::from(x == 1), u8::from(y == 1), 0])
		});
		let mut rng = StdRng::seed_from_u64(42);
		let step = RandomVerticalFlip::new(1.0).unwrap_or_default();

		let image = step.apply(original, &mut rng);

		assert_eq!(image.get_pixel(0, 0).0, [0, 1, 0]);
		assert_eq!(image.get_pixel(1, 0).0, [1, 1, 0]);
		assert_eq!(image.get_pixel(0, 1).0, [0, 0, 0]);
	}

	#[test]
	fn zero_probability_preserves_pixels() {
		let original = ImageType::from_fn(2, 2, |x, y| {
			Rgb([u8::from(x == 1), u8::from(y == 1), 0])
		});
		let mut rng = StdRng::seed_from_u64(42);
		let image = RandomHorizontalFlip::new(0.0)
			.unwrap_or_default()
			.apply(original.clone(), &mut rng);
		assert_eq!(image, original);
	}

	#[test]
	fn flip_probability_rejects_out_of_range_and_non_finite_values() {
		for probability in [-0.1, 1.1, f32::INFINITY, f32::NAN] {
			assert!(RandomHorizontalFlip::new(probability).is_none());
			assert!(RandomVerticalFlip::new(probability).is_none());
		}
	}
}
