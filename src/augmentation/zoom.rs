//! Zoom-in and zoom-out transforms with validated scale ranges.

use image::{
	DynamicImage,
	GenericImageView,
	ImageBuffer,
	Pixel,
	Primitive,
	imageops::{
		FilterType,
		crop,
		overlay,
		resize,
	},
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};

use super::{
	AugmentationError,
	ImageStep,
};

/// Determines how a resized image is fitted back onto its original canvas.
#[derive(Clone, Copy)]
enum CanvasFit {
	CenterCrop,
	BlackPad,
}

/// Inclusive finite range of zoom-in scale factors, each at least one.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// #     AugmentationError,
/// #     zoom::ZoomInFactors,
/// # };
/// #
/// const FACTORS: ZoomInFactors = match ZoomInFactors::new(1.0, 3.0) {
/// 	Ok(factors) => factors,
/// 	Err(_) => panic!("invalid constant zoom-in range"),
/// };
///
/// assert_eq!(FACTORS.bounds(), (1.0, 3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomInFactors {
	min: f32,
	max: f32,
}

impl ZoomInFactors {
	/// Creates a range for probablistic zoom in operations.
	///
	/// This `const fn` can be evaluated during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidZoomInFactors`] unless both bounds
	/// are finite, `min >= 1.0`, and `min <= max`.
	pub const fn new(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		if min.is_finite() && max.is_finite() && min >= 1.0 && max >= min {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidZoomInFactors)
		}
	}

	/// Returns the inclusive minimum and maximum factors.
	pub const fn bounds(self) -> (f32, f32) {
		(self.min, self.max)
	}

	/// Samples an inclusive factor from this range.
	fn sample(
		self,
		rng: &mut StdRng,
	) -> f32 {
		rng.random_range(self.min..=self.max)
	}
}

/// Inclusive finite range of zoom-out divisors, each at least one.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// #     AugmentationError,
/// #     zoom::ZoomOutDivisors,
/// # };
/// #
/// const DIVISORS: ZoomOutDivisors = match ZoomOutDivisors::new(1.0, 3.0) {
/// 	Ok(divisors) => divisors,
/// 	Err(_) => panic!("invalid constant zoom-out range"),
/// };
///
/// assert_eq!(DIVISORS.bounds(), (1.0, 3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomOutDivisors {
	min: f32,
	max: f32,
}

impl ZoomOutDivisors {
	/// Creates a range for probablistic zoom out operations.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidZoomOutDivisors`] unless both bounds
	/// are finite, `min >= 1.0`, and `min <= max`.
	pub const fn new(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		if min.is_finite() && max.is_finite() && min >= 1.0 && max >= min {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidZoomOutDivisors)
		}
	}

	/// Returns the inclusive minimum and maximum divisors.
	pub const fn bounds(self) -> (f32, f32) {
		(self.min, self.max)
	}

	/// Samples an inclusive divisor from this range.
	fn sample(
		self,
		rng: &mut StdRng,
	) -> f32 {
		rng.random_range(self.min..=self.max)
	}
}

/// Crops the centered region of an image after scaling it up.
///
/// A factor of one leaves the image unchanged.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::zoom::{
/// # 	ZoomIn,
/// # 	ZoomInFactors,
/// # };
/// #
/// let factors =
/// 	ZoomInFactors::new(1.5, 2.0).expect("the factor range is valid");
/// let _zoom = ZoomIn::new(factors);
/// ```
#[derive(Clone, Debug)]
pub struct ZoomIn {
	factors: ZoomInFactors,
}

impl ZoomIn {
	/// Creates a zoom-in transform with a validated factor range.
	///
	/// This constructor is infallible and `const`.
	pub const fn new(factors: ZoomInFactors) -> Self {
		Self {
			factors,
		}
	}
}

impl ImageStep for ZoomIn {
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		resample_to_canvas(
			image,
			self.factors.sample(rng),
			CanvasFit::CenterCrop,
		)
	}
}

/// Scales an image down and centers it on a black canvas.
///
/// A divisor of one leaves the image unchanged.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::zoom::{
/// # 	ZoomOut,
/// # 	ZoomOutDivisors,
/// # };
/// #
/// let divisors =
/// 	ZoomOutDivisors::new(1.5, 2.0).expect("the divisor range is valid");
/// let _zoom = ZoomOut::new(divisors);
/// ```
#[derive(Clone, Debug)]
pub struct ZoomOut {
	divisors: ZoomOutDivisors,
}

impl ZoomOut {
	/// Creates a zoom-out transform with a validated divisor range.
	///
	/// This constructor is infallible and `const`.
	pub const fn new(divisors: ZoomOutDivisors) -> Self {
		Self {
			divisors,
		}
	}
}

impl ImageStep for ZoomOut {
	fn apply<P>(
		&self,
		image: ImageBuffer<P, Vec<P::Subpixel>>,
		rng: &mut StdRng,
	) -> ImageBuffer<P, Vec<P::Subpixel>>
	where
		P: Pixel + 'static,
		ImageBuffer<P, Vec<P::Subpixel>>: Into<DynamicImage>,
	{
		resample_to_canvas(
			image,
			1.0 / self.divisors.sample(rng),
			CanvasFit::BlackPad,
		)
	}
}

/// Resizes an image and fits it to its original dimensions.
#[expect(
	clippy::cast_precision_loss,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	clippy::integer_division,
	reason = "Scale factors map dimensions to pixels; fitting is centered using floor division"
)]
fn resample_to_canvas<I, P>(
	image: I,
	scale: f32,
	fit: CanvasFit,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
	I: GenericImageView<Pixel = P>
		+ Into<DynamicImage>
		+ Into<ImageBuffer<P, Vec<P::Subpixel>>>,
	P: Pixel + 'static,
{
	let (width, height) = image.dimensions();
	let scaled_width = ((width as f32 * scale).round() as u32).max(1);
	let scaled_height = ((height as f32 * scale).round() as u32).max(1);
	if scaled_width == width && scaled_height == height {
		return image.into();
	}

	let mut resized =
		resize(&image, scaled_width, scaled_height, FilterType::Nearest);
	match fit {
		CanvasFit::CenterCrop => {
			let left = scaled_width.saturating_sub(width) / 2;
			let top = scaled_height.saturating_sub(height) / 2;
			crop(&mut resized, left, top, width, height).to_image()
		},
		CanvasFit::BlackPad => {
			let channels = vec![
				P::Subpixel::DEFAULT_MIN_VALUE;
				usize::from(P::CHANNEL_COUNT)
			];
			let black = *P::from_slice(&channels);
			let mut canvas = ImageBuffer::from_pixel(width, height, black);
			let left = width.saturating_sub(scaled_width) / 2;
			let top = height.saturating_sub(scaled_height) / 2;
			overlay(&mut canvas, &resized, i64::from(left), i64::from(top));
			canvas
		},
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

	fn gradient_image(
		width: u32,
		height: u32,
	) -> RgbImage {
		RgbImage::from_fn(width, height, |x, y| {
			Rgb([
				u8::try_from(x % 256).unwrap_or_default(),
				u8::try_from(y % 256).unwrap_or_default(),
				0,
			])
		})
	}

	#[test]
	fn zoom_in_preserves_dimensions() {
		let image = gradient_image(4, 5);
		let mut rng = StdRng::seed_from_u64(42);
		let factors = ZoomInFactors::new(2.0, 2.0)
			.expect("the zoom-in factor range is valid");
		let result = ZoomIn::new(factors).apply(image, &mut rng);
		assert_eq!(result.dimensions(), (4, 5));
	}

	#[test]
	fn zoom_out_preserves_dimensions() {
		let image = gradient_image(12, 10);
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomOutDivisors::new(2.0, 2.0)
			.expect("the zoom-out divisor range is valid");
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.dimensions(), (12, 10));
	}

	#[test]
	fn zoom_in_changes_pixels_when_scaled_up() {
		let image = gradient_image(8, 8);
		let mut rng = StdRng::seed_from_u64(42);
		let factors = ZoomInFactors::new(2.0, 2.0)
			.expect("the zoom-in factor range is valid");
		let result = ZoomIn::new(factors).apply(image.clone(), &mut rng);
		assert_ne!(result, image);
	}

	#[test]
	fn zoom_out_centers_resized_content() {
		let image = RgbImage::from_pixel(8, 8, Rgb([12, 34, 56]));
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomOutDivisors::new(2.0, 2.0)
			.expect("the zoom-out divisor range is valid");
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.get_pixel(3, 3).0, [12, 34, 56]);
	}

	#[test]
	fn zoom_out_fills_padding_with_black() {
		let image = RgbImage::from_pixel(8, 8, Rgb([12, 34, 56]));
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomOutDivisors::new(2.0, 2.0)
			.expect("the zoom-out divisor range is valid");
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.get_pixel(0, 0).0, [0, 0, 0]);
	}

	#[test]
	fn zoom_in_factor_one_preserves_pixels() {
		let image = gradient_image(3, 3);
		let mut rng = StdRng::seed_from_u64(42);
		let factors =
			ZoomInFactors::new(1.0, 1.0).expect("a factor of one is valid");
		let result = ZoomIn::new(factors).apply(image.clone(), &mut rng);
		assert_eq!(result, image);
	}

	#[test]
	fn zoom_out_divisor_one_preserves_pixels() {
		let image = gradient_image(3, 3);
		let mut rng = StdRng::seed_from_u64(42);
		let divisors =
			ZoomOutDivisors::new(1.0, 1.0).expect("a divisor of one is valid");
		let result = ZoomOut::new(divisors).apply(image.clone(), &mut rng);
		assert_eq!(result, image);
	}

	#[test]
	fn zoom_in_range_rejects_values_below_one() {
		assert_eq!(
			ZoomInFactors::new(0.9, 1.2).unwrap_err(),
			AugmentationError::InvalidZoomInFactors
		);
	}

	#[test]
	fn zoom_in_range_rejects_reversed_bounds() {
		assert_eq!(
			ZoomInFactors::new(2.0, 1.0).unwrap_err(),
			AugmentationError::InvalidZoomInFactors
		);
	}

	#[test]
	fn zoom_out_range_rejects_values_below_one() {
		assert_eq!(
			ZoomOutDivisors::new(0.9, 2.0).unwrap_err(),
			AugmentationError::InvalidZoomOutDivisors
		);
	}

	#[test]
	fn zoom_out_range_rejects_non_finite_values() {
		assert_eq!(
			ZoomOutDivisors::new(1.0, f32::INFINITY).unwrap_err(),
			AugmentationError::InvalidZoomOutDivisors
		);
	}

	#[test]
	fn factor_ranges_are_const_evaluable() {
		const FACTORS: ZoomInFactors = match ZoomInFactors::new(1.0, 2.0) {
			Ok(factors) => factors,
			Err(_) => panic!("invalid test factors"),
		};
		const DIVISORS: ZoomOutDivisors = match ZoomOutDivisors::new(1.0, 2.0) {
			Ok(divisors) => divisors,
			Err(_) => panic!("invalid test divisors"),
		};
		assert_eq!(FACTORS.bounds(), (1.0, 2.0));
		assert_eq!(DIVISORS.bounds(), (1.0, 2.0));
	}
}
