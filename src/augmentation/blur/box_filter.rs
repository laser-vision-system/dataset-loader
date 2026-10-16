use image::imageops::filter3x3;
use rand::rngs::StdRng;

use super::{
	super::{
		AugmentationError,
		ImageStep,
	},
	ImageType,
};

/// A validated 3-by-3 kernel containing only finite weights.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// # 	AugmentationError,
/// # 	blur::BoxFilterKernel,
/// # };
/// #
/// const KERNEL: BoxFilterKernel = match BoxFilterKernel::new([1.0; 9]) {
/// 	Ok(kernel) => kernel,
/// 	Err(_) => panic!("invalid constant box-filter kernel"),
/// };
///
/// assert_eq!(*KERNEL.weights(), [1.0; 9]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxFilterKernel([f32; 9]);

impl BoxFilterKernel {
	/// Creates a kernel from nine row-major finite coefficients.
	///
	/// Zero-sum kernels are allowed; the image crate applies an inverse sum of
	/// one for them.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidBoxFilterKernel`] if any coefficient
	/// is NaN or infinite.
	pub const fn new(weights: [f32; 9]) -> Result<Self, AugmentationError> {
		let [
			top_left,
			top_center,
			top_right,
			middle_left,
			center,
			middle_right,
			bottom_left,
			bottom_center,
			bottom_right,
		] = weights;
		if top_left.is_finite()
			&& top_center.is_finite()
			&& top_right.is_finite()
			&& middle_left.is_finite()
			&& center.is_finite()
			&& middle_right.is_finite()
			&& bottom_left.is_finite()
			&& bottom_center.is_finite()
			&& bottom_right.is_finite()
		{
			Ok(Self(weights))
		} else {
			Err(AugmentationError::InvalidBoxFilterKernel)
		}
	}

	/// Returns the row-major coefficients.
	pub const fn weights(&self) -> &[f32; 9] {
		&self.0
	}
}

/// Applies a validated 3-by-3 convolution kernel.
/// Images smaller than the kernel dimensions are returned unchanged.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::blur::{
/// # 	BoxFilter,
/// # 	BoxFilterKernel,
/// # };
/// #
/// const KERNEL: BoxFilterKernel = match BoxFilterKernel::new([1.0; 9]) {
/// 	Ok(kernel) => kernel,
/// 	Err(_) => panic!("invalid constant box-filter kernel"),
/// };
/// let _filter = BoxFilter::new(KERNEL);
/// ```
#[derive(Clone, Debug)]
pub struct BoxFilter {
	kernel: BoxFilterKernel,
}

impl BoxFilter {
	/// Creates a filter from a validated kernel.
	pub const fn new(kernel: BoxFilterKernel) -> Self {
		Self {
			kernel,
		}
	}

	/// Returns a center-weighted 3-by-3 pyramid filter.
	pub fn pyramid() -> Self {
		Self::new(BoxFilterKernel([
			1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0,
		]))
	}
}

impl Default for BoxFilter {
	/// Returns a uniform 3-by-3 kernel.
	fn default() -> Self {
		Self::new(BoxFilterKernel([1.0; 9]))
	}
}

impl ImageStep for BoxFilter {
	fn apply(
		&self,
		image: ImageType,
		_rng: &mut StdRng,
	) -> ImageType {
		let (width, height) = image.dimensions();
		if width < 3 || height < 3 {
			return image;
		}
		filter3x3(&image, &self.kernel.0)
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

	#[test]
	fn kernel_rejects_non_finite_weights() {
		assert!(matches!(
			BoxFilterKernel::new([
				1.0,
				1.0,
				f32::NAN,
				1.0,
				1.0,
				1.0,
				1.0,
				1.0,
				1.0,
			]),
			Err(AugmentationError::InvalidBoxFilterKernel)
		));
	}

	#[test]
	fn kernel_validation_is_const_evaluable() {
		const KERNEL: BoxFilterKernel = match BoxFilterKernel::new([1.0; 9]) {
			Ok(kernel) => kernel,
			Err(_) => panic!("invalid test kernel"),
		};
		assert_eq!(KERNEL.weights(), &[1.0; 9]);
	}

	#[test]
	fn small_images_are_unchanged() {
		for dimensions in [(0, 0), (1, 1), (2, 4), (4, 2)] {
			let image = ImageType::from_pixel(
				dimensions.0,
				dimensions.1,
				Rgb([10, 20, 30]),
			);
			let filtered = BoxFilter::default()
				.apply(image.clone(), &mut StdRng::seed_from_u64(42));
			assert_eq!(filtered, image);
		}
	}
}
