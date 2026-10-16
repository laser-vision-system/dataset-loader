use image::imageops::filter3x3;
use rand::rngs::StdRng;

use super::{
	super::ImageStep,
	ImageType,
};

#[derive(Clone, Debug)]
pub struct BoxFilter {
	/// Row-major 3-by-3 convolution kernel.
	pub kernel: [f32; 9],
}

impl BoxFilter {
	/// Returns a center-weighted 3-by-3 pyramid kernel.
	pub fn pyramid() -> Self {
		Self {
			kernel: [1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0],
		}
	}
}

impl Default for BoxFilter {
	/// Returns a uniform 3-by-3 kernel.
	fn default() -> Self {
		Self {
			kernel: [1.0; 9],
		}
	}
}

impl ImageStep for BoxFilter {
	fn apply(
		&self,
		image: ImageType,
		_rng: &mut StdRng,
	) -> ImageType {
		filter3x3(&image, &self.kernel)
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
	fn filters_preserve_dimensions_and_modify_impulse() {
		let original = impulse_image();
		let mut rng = StdRng::seed_from_u64(42);
		for filter in [BoxFilter::default(), BoxFilter::pyramid()] {
			let image = filter.apply(original.clone(), &mut rng);
			assert_eq!(image.dimensions(), original.dimensions());
			assert_ne!(image, original);
		}
	}

	#[test]
	fn pyramid_kernel_is_center_weighted() {
		// The pyramid kernel centers on weight 3; the impulse becomes 3*255/15
		// = 51.
		let original = impulse_image();
		let image = BoxFilter::pyramid()
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));
		// The pyramid kernel's larger center weight retains more of the
		// impulse.
		let uniform = BoxFilter::default()
			.apply(original.clone(), &mut StdRng::seed_from_u64(42));
		assert!(image.get_pixel(2, 2).0[0] > uniform.get_pixel(2, 2).0[0]);
	}
}
