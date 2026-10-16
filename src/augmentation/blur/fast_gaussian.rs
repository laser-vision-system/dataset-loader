use image::imageops::fast_blur;
use rand::rngs::StdRng;

use super::{
	super::ImageStep,
	BlurSigma,
	BlurSigmaRange,
	ImageType,
	apply_blur,
};

/// Applies the faster approximate Gaussian blur with a fixed radius.
#[derive(Clone, Debug, Default)]
pub struct GaussianFast {
	sigma: BlurSigma,
}

impl GaussianFast {
	/// Creates a fast Gaussian blur; zero disables the transform.
	pub fn new(sigma: f32) -> Option<Self> {
		Some(Self {
			sigma: BlurSigma::new(sigma)?,
		})
	}
}

impl ImageStep for GaussianFast {
	fn apply(
		&self,
		image: ImageType,
		_rng: &mut StdRng,
	) -> ImageType {
		apply_blur(image, self.sigma, fast_blur)
	}
}

/// Samples a radius for the faster approximate Gaussian blur.
#[derive(Clone, Debug)]
pub struct RandomGaussianFast {
	sigma: BlurSigmaRange,
}

impl RandomGaussianFast {
	/// Creates a fast blur with a sampled radius in `sigma_min..sigma_max`.
	pub fn new(
		sigma_min: f32,
		sigma_max: f32,
	) -> Option<Self> {
		Some(Self {
			sigma: BlurSigmaRange::new(sigma_min, sigma_max)?,
		})
	}
}

impl Default for RandomGaussianFast {
	fn default() -> Self {
		Self {
			sigma: BlurSigmaRange {
				min: 0.0,
				max: 1.0,
			},
		}
	}
}

impl ImageStep for RandomGaussianFast {
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		apply_blur(image, self.sigma.sample(rng), fast_blur)
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
	fn fixed_sigma_blur_darkens_center_and_preserves_dimensions() {
		let original = impulse_image();
		let image = GaussianFast {
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
		let image = GaussianFast {
			sigma: BlurSigma(0.0),
		}
		.apply(original.clone(), &mut rng);
		assert_eq!(image, original);
	}

	#[test]
	fn random_sigma_blurs_image_within_bounds() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image = RandomGaussianFast {
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
		let image_a = RandomGaussianFast {
			sigma: BlurSigmaRange {
				min: 1.0,
				max: 3.0,
			},
		}
		.apply(original.clone(), &mut StdRng::seed_from_u64(7));
		let image_b = RandomGaussianFast {
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
			assert!(RandomGaussianFast::new(bounds.0, bounds.1).is_none());
		}
	}

	#[test]
	fn fixed_sigma_rejects_negative_and_non_finite_values() {
		for sigma in [-1.0, f32::INFINITY, f32::NAN] {
			assert!(GaussianFast::new(sigma).is_none());
		}
	}
}
