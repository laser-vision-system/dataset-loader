use image::{
	DynamicImage,
	ImageBuffer,
	Pixel,
	imageops::fast_blur,
};
use rand::rngs::StdRng;

use super::{
	super::ImageStep,
	BlurSigma,
	BlurSigmaRange,
	apply_blur,
};

/// Applies the faster approximate Gaussian blur with a fixed radius.
///
/// The default sigma is zero, so the default transform leaves images unchanged.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::blur::{
/// # 	BlurSigma,
/// # 	GaussianFast,
/// # };
/// #
/// const SIGMA: BlurSigma = match BlurSigma::new(1.0) {
/// 	Ok(sigma) => sigma,
/// 	Err(_) => panic!("invalid constant blur sigma"),
/// };
/// let _blur = GaussianFast::new(SIGMA);
/// ```
#[derive(Clone, Debug, Default)]
pub struct GaussianFast {
	sigma: BlurSigma,
}

impl GaussianFast {
	/// Creates a fast Gaussian blur from a validated radius; zero disables it.
	pub const fn new(sigma: BlurSigma) -> Self {
		Self {
			sigma,
		}
	}
}

impl ImageStep for GaussianFast {
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		_rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		apply_blur(image, self.sigma, fast_blur)
	}
}

/// Samples a radius for the faster approximate Gaussian blur.
///
/// # Examples
///
/// ```
/// use dataset_loader::augmentation::blur::{
/// 	BlurSigmaRange,
/// 	RandomGaussianFast,
/// };
///
/// const RANGE: BlurSigmaRange = match BlurSigmaRange::new(0.25, 1.5) {
/// 	Ok(range) => range,
/// 	Err(_) => panic!("invalid constant blur range"),
/// };
/// let _blur = RandomGaussianFast::new(RANGE);
/// ```
#[derive(Clone, Debug)]
pub struct RandomGaussianFast {
	sigma: BlurSigmaRange,
}

impl RandomGaussianFast {
	/// Creates a fast blur with a validated radius range.
	pub const fn new(sigma: BlurSigmaRange) -> Self {
		Self {
			sigma,
		}
	}
}

impl Default for RandomGaussianFast {
	/// Uses a random sigma from the half-open range `0.0..1.0`.
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
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		apply_blur(image, self.sigma.sample(rng), fast_blur)
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
	use crate::augmentation::AugmentationError;

	fn impulse_image() -> RgbImage {
		RgbImage::from_fn(5, 5, |x, y| {
			if x == 2 && y == 2 {
				Rgb([255, 255, 255])
			} else {
				Rgb([0, 0, 0])
			}
		})
	}

	fn sigma(value: f32) -> BlurSigma {
		BlurSigma::new(value).expect("the test sigma is valid")
	}

	fn range(
		min: f32,
		max: f32,
	) -> BlurSigmaRange {
		BlurSigmaRange::new(min, max).expect("the test range is valid")
	}

	#[test]
	fn fixed_sigma_blur_darkens_center_and_preserves_dimensions() {
		let original = impulse_image();
		let image = GaussianFast::new(sigma(1.0))
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));

		assert_eq!(image.dimensions(), original.dimensions());
		assert!(image.get_pixel(2, 2).0[0] < 255);
		assert!(image.get_pixel(0, 0).0[0] > 0);
	}

	#[test]
	fn zero_sigma_leaves_image_unchanged() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image =
			GaussianFast::new(sigma(0.0)).apply(original.clone(), &mut rng);
		assert_eq!(image, original);
	}

	#[test]
	fn random_sigma_blurs_image_within_bounds() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image = RandomGaussianFast::new(range(1.0, 3.0))
			.apply(original.clone(), &mut rng);

		assert_eq!(image.dimensions(), original.dimensions());
		assert_ne!(image, original);
		let center = image.get_pixel(2, 2).0[0];
		assert!(center < 255);
	}

	#[test]
	fn random_sigma_is_deterministic_with_seed() {
		let original = impulse_image();
		let image_a = RandomGaussianFast::new(range(1.0, 3.0))
			.apply(original.clone(), &mut StdRng::seed_from_u64(7));
		let image_b = RandomGaussianFast::new(range(1.0, 3.0))
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
			assert!(matches!(
				BlurSigmaRange::new(bounds.0, bounds.1),
				Err(AugmentationError::InvalidBlurSigmaRange)
			));
		}
	}

	#[test]
	fn fixed_sigma_rejects_negative_and_non_finite_values() {
		for sigma in [-1.0, f32::INFINITY, f32::NAN] {
			assert!(matches!(
				BlurSigma::new(sigma),
				Err(AugmentationError::InvalidBlurSigma)
			));
		}
	}

	#[test]
	fn sigma_validation_is_const_evaluable() {
		const FIXED: BlurSigma = match BlurSigma::new(1.0) {
			Ok(sigma) => sigma,
			Err(_) => panic!("invalid test sigma"),
		};
		const RANDOM: BlurSigmaRange = match BlurSigmaRange::new(0.0, 1.0) {
			Ok(range) => range,
			Err(_) => panic!("invalid test range"),
		};
		assert_eq!(FIXED.value(), 1.0);
		assert_eq!(RANDOM.bounds(), (0.0, 1.0));
	}
}
