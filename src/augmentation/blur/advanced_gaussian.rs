use image::imageops::{
	GaussianBlurParameters,
	blur_advanced,
};
use rand::rngs::StdRng;

use super::{
	super::ImageStep,
	ImageType,
};

/// Applies a Gaussian blur using explicit kernel parameters.
#[derive(Clone)]
pub struct GaussianAdvanced {
	/// Kernel parameters passed to the image crate's advanced blur.
	pub parameters: GaussianBlurParameters,
}

impl ImageStep for GaussianAdvanced {
	fn apply(
		&self,
		image: ImageType,
		_rng: &mut StdRng,
	) -> ImageType {
		blur_advanced(&image, self.parameters)
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
	fn smoothing_3_darkens_center_and_preserves_dimensions() {
		let original = impulse_image();
		let image = GaussianAdvanced {
			parameters: GaussianBlurParameters::SMOOTHING_3,
		}
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
		let strong = GaussianAdvanced {
			parameters: GaussianBlurParameters::SMOOTHING_7,
		}
		.apply(original.clone(), &mut rng);
		let weak = GaussianAdvanced {
			parameters: GaussianBlurParameters::SMOOTHING_3,
		}
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
		let image = GaussianAdvanced {
			parameters,
		}
		.apply(original.clone(), &mut StdRng::seed_from_u64(42));
		assert_eq!(image.dimensions(), original.dimensions());
		assert_ne!(image, original);
	}

	#[test]
	fn clones_parameters() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		let step = GaussianAdvanced {
			parameters: GaussianBlurParameters::SMOOTHING_3,
		};
		let clone = step.clone();
		let original_result = step.apply(original.clone(), &mut rng);
		let cloned_result = clone.apply(original, &mut rng);
		assert_eq!(original_result, cloned_result);
	}
}
