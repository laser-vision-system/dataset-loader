use image::imageops::blur;
use rand::rngs::StdRng;

use super::{
	super::ImageStep,
	BlurSigma,
	BlurSigmaRange,
	ImageType,
	apply_blur,
};

/// Applies a Gaussian blur with a fixed standard deviation.
#[derive(Clone, Debug, Default)]
pub struct Gaussian {
	sigma: BlurSigma,
}

impl Gaussian {
	/// Creates a Gaussian blur; zero disables the transform.
	pub fn new(sigma: f32) -> Option<Self> {
		Some(Self {
			sigma: BlurSigma::new(sigma)?,
		})
	}
}

impl ImageStep for Gaussian {
	fn apply(
		&self,
		image: ImageType,
		_rng: &mut StdRng,
	) -> ImageType {
		apply_blur(image, self.sigma, blur)
	}
}

/// Samples a Gaussian blur radius for each image.
#[derive(Clone, Debug)]
pub struct RandomGaussian {
	sigma: BlurSigmaRange,
}

impl RandomGaussian {
	/// Creates a blur with a sampled radius in `sigma_min..sigma_max`.
	pub fn new(
		sigma_min: f32,
		sigma_max: f32,
	) -> Option<Self> {
		Some(Self {
			sigma: BlurSigmaRange::new(sigma_min, sigma_max)?,
		})
	}
}

impl Default for RandomGaussian {
	fn default() -> Self {
		Self {
			sigma: BlurSigmaRange {
				min: 0.0,
				max: 1.0,
			},
		}
	}
}

impl ImageStep for RandomGaussian {
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		apply_blur(image, self.sigma.sample(rng), blur)
	}
}

#[cfg(test)]
mod tests {
	use image::Rgb;
	use rand::SeedableRng as _;

	use super::*;

	fn impulse_image() -> ImageType {
		ImageType::from_fn(5, 5, |x, y| {
			if x == 2 && y == 2 {
				Rgb([255, 255, 255])
			} else {
				Rgb([0, 0, 0])
			}
		})
	}

	#[test]
	fn fixed_sigma_blur_darkens_center_and_spreads_light() {
		let original = impulse_image();
		let image = Gaussian {
			sigma: BlurSigma(1.0),
		}
		.apply(original.clone(), &mut StdRng::seed_from_u64(42));

		assert_eq!(image.dimensions(), original.dimensions());
		assert!(image.get_pixel(2, 2).0[0] < 255);
		assert!(image.get_pixel(0, 0).0[0] > 0);
	}

	#[test]
	fn zero_sigma_leaves_image_unchanged() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image = Gaussian {
			sigma: BlurSigma(0.0),
		}
		.apply(original.clone(), &mut rng);
		assert_eq!(image, original);
	}

	#[test]
	fn random_sigma_blurs_image_within_bounds() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image = RandomGaussian {
			sigma: BlurSigmaRange {
				min: 1.0,
				max: 3.0,
			},
		}
		.apply(original.clone(), &mut rng);

		assert_eq!(image.dimensions(), original.dimensions());
		assert_ne!(image, original);
		let center = image.get_pixel(2, 2).0[0];
		assert!(center < 255);
	}

	#[test]
	fn random_sigma_is_deterministic_with_seed() {
		let original = impulse_image();
		let image_a = RandomGaussian {
			sigma: BlurSigmaRange {
				min: 1.0,
				max: 3.0,
			},
		}
		.apply(original.clone(), &mut StdRng::seed_from_u64(7));
		let image_b = RandomGaussian {
			sigma: BlurSigmaRange {
				min: 1.0,
				max: 3.0,
			},
		}
		.apply(original.clone(), &mut StdRng::seed_from_u64(7));
		assert_eq!(image_a, image_b);
	}

	#[test]
	fn random_sigma_rejects_empty_range() {
		for bounds in [
			(2.0, 2.0),
			(3.0, 1.0),
			(-1.0, 1.0),
			(0.0, f32::INFINITY),
			(0.0, f32::NAN),
		] {
			assert!(RandomGaussian::new(bounds.0, bounds.1).is_none());
		}
	}

	#[test]
	fn fixed_sigma_rejects_negative_and_non_finite_values() {
		for sigma in [-1.0, f32::INFINITY, f32::NAN] {
			assert!(Gaussian::new(sigma).is_none());
		}
	}
}
