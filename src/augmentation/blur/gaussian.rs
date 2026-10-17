use image::{
	DynamicImage,
	Pixel,
	imageops::blur,
};
use rand::rngs::StdRng;

use super::{
	super::{
		GenericImageBuffer,
		ImageStep,
	},
	BlurSigma,
	BlurSigmaRange,
	apply_blur,
};

/// Applies a Gaussian blur with a fixed standard deviation.
///
/// The default sigma is zero, so the default transform leaves images unchanged.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::blur::{
/// # 	BlurSigma,
/// # 	Gaussian,
/// # };
/// #
/// const SIGMA: BlurSigma = match BlurSigma::new(1.0) {
/// 	Ok(sigma) => sigma,
/// 	Err(_) => panic!("invalid constant blur sigma"),
/// };
/// let _blur = Gaussian::new(SIGMA);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Gaussian {
	sigma: BlurSigma,
}

impl Gaussian {
	/// Creates a Gaussian blur from a validated radius; zero disables it.
	pub const fn new(sigma: BlurSigma) -> Self {
		Self {
			sigma,
		}
	}
}

impl ImageStep for Gaussian {
	fn apply<P>(
		&self,
		image: GenericImageBuffer<P>,
		_rng: &mut StdRng,
	) -> GenericImageBuffer<P>
	where
		P: Pixel + 'static,
		GenericImageBuffer<P>: Into<DynamicImage>,
	{
		apply_blur(image, self.sigma, blur)
	}
}

/// Samples a Gaussian blur radius for each image.
///
/// # Examples
///
/// ```
/// use dataset_loader::augmentation::blur::{
/// 	BlurSigmaRange,
/// 	RandomGaussian,
/// };
///
/// const RANGE: BlurSigmaRange = match BlurSigmaRange::new(0.25, 1.5) {
/// 	Ok(range) => range,
/// 	Err(_) => panic!("invalid constant blur range"),
/// };
/// let _blur = RandomGaussian::new(RANGE);
/// ```
#[derive(Clone, Debug)]
pub struct RandomGaussian {
	sigma: BlurSigmaRange,
}

impl RandomGaussian {
	/// Creates a blur that samples a radius from a validated half-open range.
	pub const fn new(sigma: BlurSigmaRange) -> Self {
		Self {
			sigma,
		}
	}
}

impl Default for RandomGaussian {
	/// Uses a random sigma from the half-open range `0.0..1.0`.
	fn default() -> Self {
		Self {
			sigma: BlurSigmaRange::default(),
		}
	}
}

impl ImageStep for RandomGaussian {
	fn apply<P>(
		&self,
		image: GenericImageBuffer<P>,
		rng: &mut StdRng,
	) -> GenericImageBuffer<P>
	where
		P: Pixel + 'static,
		GenericImageBuffer<P>: Into<DynamicImage>,
	{
		apply_blur(image, self.sigma.sample(rng), blur)
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
		BlurSigma::try_new(value).expect("the test sigma is valid")
	}

	fn range(
		min: f32,
		max: f32,
	) -> BlurSigmaRange {
		BlurSigmaRange::try_new(min, max).expect("the test range is valid")
	}

	#[test]
	fn fixed_sigma_blur_darkens_center_and_spreads_light() {
		let original = impulse_image();
		let image = Gaussian::new(sigma(1.0))
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));

		assert_eq!(image.dimensions(), original.dimensions());
		assert!(image.get_pixel(2, 2).0[0] < 255);
		assert!(image.get_pixel(0, 0).0[0] > 0);
	}

	#[test]
	fn zero_sigma_leaves_image_unchanged() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image = Gaussian::new(sigma(0.0)).apply(original.clone(), &mut rng);
		assert_eq!(image, original);
	}

	#[test]
	fn random_sigma_blurs_image_within_bounds() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let image = RandomGaussian::new(range(1.0, 3.0))
			.apply(original.clone(), &mut rng);

		assert_eq!(image.dimensions(), original.dimensions());
		assert_ne!(image, original);
		let center = image.get_pixel(2, 2).0[0];
		assert!(center < 255);
	}

	#[test]
	fn random_sigma_is_deterministic_with_seed() {
		let original = impulse_image();
		let image_a = RandomGaussian::new(range(1.0, 3.0))
			.apply(original.clone(), &mut StdRng::seed_from_u64(7));
		let image_b = RandomGaussian::new(range(1.0, 3.0))
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
				BlurSigmaRange::try_new(bounds.0, bounds.1),
				Err(AugmentationError::InvalidBlurSigmaRange)
			));
		}
	}

	#[test]
	fn fixed_sigma_rejects_negative_and_non_finite_values() {
		for sigma in [-1.0, f32::INFINITY, f32::NAN] {
			assert!(matches!(
				BlurSigma::try_new(sigma),
				Err(AugmentationError::InvalidBlurSigma)
			));
		}
	}

	#[test]
	fn sigma_validation_is_const_evaluable() {
		const FIXED: BlurSigma = match BlurSigma::try_new(1.0) {
			Ok(sigma) => sigma,
			Err(_) => panic!("invalid test sigma"),
		};
		const RANDOM: BlurSigmaRange = match BlurSigmaRange::try_new(0.0, 1.0) {
			Ok(range) => range,
			Err(_) => panic!("invalid test range"),
		};
		assert_eq!(FIXED.value(), 1.0);
		assert_eq!(RANDOM.bounds(), (0.0, 1.0));
	}
}
