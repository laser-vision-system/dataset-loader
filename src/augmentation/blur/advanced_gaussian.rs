use image::{
	DynamicImage,
	ImageBuffer,
	Pixel,
	imageops::{
		GaussianBlurParameters,
		blur_advanced,
	},
};
use rand::rngs::StdRng;

use super::super::ImageStep;

/// Applies a Gaussian blur using explicit kernel parameters.
///
/// # Examples
///
/// ```
/// use dataset_loader::augmentation::blur::GaussianAdvanced;
/// use image::imageops::GaussianBlurParameters;
///
/// let _blur = GaussianAdvanced::new(GaussianBlurParameters::SMOOTHING_3);
/// ```
#[derive(Clone)]
pub struct GaussianAdvanced {
	parameters: GaussianBlurParameters,
}

impl GaussianAdvanced {
	/// Creates an advanced Gaussian blur from explicit image-crate parameters.
	pub const fn new(parameters: GaussianBlurParameters) -> Self {
		Self {
			parameters,
		}
	}
}

impl ImageStep for GaussianAdvanced {
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		_rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		blur_advanced(&image, self.parameters)
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

	fn impulse_image() -> RgbImage {
		RgbImage::from_fn(5, 5, |x, y| {
			if x == 2 && y == 2 {
				Rgb([255, 255, 255])
			} else {
				Rgb([0, 0, 0])
			}
		})
	}

	#[test]
	fn smoothing_3_darkens_center_and_preserves_dimensions() {
		let original = impulse_image();
		let image = GaussianAdvanced::new(GaussianBlurParameters::SMOOTHING_3)
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));

		assert_eq!(image.dimensions(), original.dimensions());
		assert!(image.get_pixel(2, 2).0[0] < 255);
		assert!(image.get_pixel(1, 1).0[0] > 0);
		assert_eq!(image.get_pixel(0, 0).0[0], original.get_pixel(0, 0).0[0]);
	}

	#[test]
	fn larger_smoothing_blurs_more_than_smaller() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let strong = GaussianAdvanced::new(GaussianBlurParameters::SMOOTHING_7)
			.apply(original.clone(), &mut rng);
		let weak = GaussianAdvanced::new(GaussianBlurParameters::SMOOTHING_3)
			.apply(original.clone(), &mut rng);
		// A wider kernel attenuates the center pixel more aggressively.
		assert!(strong.get_pixel(2, 2).0[0] < weak.get_pixel(2, 2).0[0]);
		// The wider kernel spreads light further to the corners.
		assert!(strong.get_pixel(0, 0).0[0] > weak.get_pixel(0, 0).0[0]);
	}

	#[test]
	fn custom_parameters_are_applied() {
		let original = impulse_image();
		let parameters = GaussianBlurParameters::new_from_radius(4.0);
		let image = GaussianAdvanced::new(parameters)
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));
		assert_eq!(image.dimensions(), original.dimensions());
		assert_ne!(image, original);
	}

	#[test]
	fn clones_parameters() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let step = GaussianAdvanced::new(GaussianBlurParameters::SMOOTHING_3);
		let clone = step.clone();
		let original_result = step.apply(original.clone(), &mut rng);
		let cloned_result = clone.apply(original, &mut rng);
		assert_eq!(original_result, cloned_result);
	}
}
