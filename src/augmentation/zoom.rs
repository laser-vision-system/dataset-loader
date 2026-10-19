//! Zoom-in and zoom-out transforms with validated scale ranges.

// IDEA: Combine ZoomIn and ZoomOut.

use image::{
	DynamicImage,
	GenericImageView,
	ImageBuffer,
	Pixel,
	Primitive as _,
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
use typed_floats::tf32;

use super::{
	AugmentationError,
	GenericImageBuffer,
	ImageStep,
};

/// Determines how a resized image is fitted back onto its original canvas.
#[derive(Clone, Copy)]
enum CanvasFit {
	CenterCrop,
	BlackPad,
}

/// Inclusive range of zoom factors/divisors, each finite and at least one.
///
/// # Examples
///
/// ```
/// # use dataset_loader::augmentation::{
/// #     AugmentationError,
/// #     zoom::ZoomRange,
/// # };
/// #
/// const FACTORS: ZoomRange = match ZoomRange::try_new(1.0, 3.0) {
/// 	Ok(factors) => factors,
/// 	Err(_) => panic!("invalid constant zoom-in range"),
/// };
///
/// assert_eq!(FACTORS.bounds(), (1.0, 3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomRange {
	min: tf32::PositiveFinite,
	max: tf32::PositiveFinite,
}

impl ZoomRange {
	/// Creates a strictly increasing range of zoom factors.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidZoomRange`] if either value is
	/// non-finite, less than one, or if `min >= max`.
	pub const fn try_new(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		Self::strict(min, max)
	}

	/// Orders the supplied factors and creates a range.
	///
	/// Equal values represent a fixed zoom factor.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidZoomRange`] if either value is
	/// non-finite or less than one.
	pub const fn relaxed(
		first: f32,
		second: f32,
	) -> Result<Self, AugmentationError> {
		// Done this way to allow for const.
		let Ok(first) = tf32::PositiveFinite::new(first) else {
			return Err(AugmentationError::InvalidZoomRange);
		};
		let Ok(second) = tf32::PositiveFinite::new(second) else {
			return Err(AugmentationError::InvalidZoomRange);
		};
		if first.get() < 1.0 || second.get() < 1.0 {
			return Err(AugmentationError::InvalidZoomRange);
		}
		let (min, max) = if first.get() <= second.get() {
			(first, second)
		} else {
			(second, first)
		};
		Ok(Self {
			min,
			max,
		})
	}

	/// Creates an ordered range while allowing equal bounds.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidZoomRange`] when a value is
	/// non-finite, below one, or `min > max`.
	pub const fn ordered(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		// Done this way to allow for const.
		let Ok(min) = tf32::PositiveFinite::new(min) else {
			return Err(AugmentationError::InvalidZoomRange);
		};
		let Ok(max) = tf32::PositiveFinite::new(max) else {
			return Err(AugmentationError::InvalidZoomRange);
		};
		if min.get() >= 1.0 && max.get() >= min.get() {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidZoomRange)
		}
	}

	/// Creates a strictly increasing range of zoom factors.
	///
	/// This function is `const` and can validate during compilation.
	///
	/// # Errors
	///
	/// Returns [`AugmentationError::InvalidZoomRange`] when a value is
	/// non-finite, below one, or `min >= max`.
	pub const fn strict(
		min: f32,
		max: f32,
	) -> Result<Self, AugmentationError> {
		// Done this way to allow for const.
		let Ok(min) = tf32::PositiveFinite::new(min) else {
			return Err(AugmentationError::InvalidZoomRange);
		};
		let Ok(max) = tf32::PositiveFinite::new(max) else {
			return Err(AugmentationError::InvalidZoomRange);
		};
		if min.get() >= 1.0 && min.get() < max.get() {
			Ok(Self {
				min,
				max,
			})
		} else {
			Err(AugmentationError::InvalidZoomRange)
		}
	}

	/// Returns the inclusive minimum and maximum factors.
	pub const fn bounds(self) -> (f32, f32) {
		(self.min.get(), self.max.get())
	}

	/// Samples an inclusive factor from this range.
	fn sample(
		self,
		rng: &mut StdRng,
	) -> f32 {
		rng.random_range(self.min.get()..=self.max.get())
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
/// # 	ZoomRange,
/// # };
/// #
/// let factors =
/// 	ZoomRange::try_new(1.5, 2.0).expect("the factor range is valid");
/// let _zoom = ZoomIn::new(factors);
/// ```
#[derive(Clone, Debug)]
pub struct ZoomIn {
	factors: ZoomRange,
}

impl ZoomIn {
	/// Creates a zoom-in transform with a validated factor range.
	///
	/// This constructor is infallible and `const`.
	pub const fn new(factors: ZoomRange) -> Self {
		Self {
			factors,
		}
	}
}

impl ImageStep for ZoomIn {
	fn apply<P>(
		&self,
		image: GenericImageBuffer<P>,
		rng: &mut StdRng,
	) -> GenericImageBuffer<P>
	where
		P: Pixel + 'static,
		GenericImageBuffer<P>: Into<DynamicImage>,
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
/// # 	ZoomRange,
/// # };
/// #
/// let divisors =
/// 	ZoomRange::try_new(1.5, 2.0).expect("the divisor range is valid");
/// let _zoom = ZoomOut::new(divisors);
/// ```
#[derive(Clone, Debug)]
pub struct ZoomOut {
	divisors: ZoomRange,
}

impl ZoomOut {
	/// Creates a zoom-out transform with a validated divisor range.
	///
	/// This constructor is infallible and `const`.
	pub const fn new(divisors: ZoomRange) -> Self {
		Self {
			divisors,
		}
	}
}

impl ImageStep for ZoomOut {
	fn apply<P>(
		&self,
		image: GenericImageBuffer<P>,
		rng: &mut StdRng,
	) -> GenericImageBuffer<P>
	where
		P: Pixel + 'static,
		GenericImageBuffer<P>: Into<DynamicImage>,
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
) -> GenericImageBuffer<P>
where
	I: GenericImageView<Pixel = P>
		+ Into<DynamicImage>
		+ Into<GenericImageBuffer<P>>,
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
		let factors = ZoomRange::ordered(2.0, 2.0)
			.expect("the zoom-in factor range is valid");
		let result = ZoomIn::new(factors).apply(image, &mut rng);
		assert_eq!(result.dimensions(), (4, 5));
	}

	#[test]
	fn zoom_out_preserves_dimensions() {
		let image = gradient_image(12, 10);
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomRange::ordered(2.0, 2.0)
			.expect("the zoom-out divisor range is valid");
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.dimensions(), (12, 10));
	}

	#[test]
	fn zoom_in_changes_pixels_when_scaled_up() {
		let image = gradient_image(8, 8);
		let mut rng = StdRng::seed_from_u64(42);
		let factors = ZoomRange::ordered(2.0, 2.0)
			.expect("the zoom-in factor range is valid");
		let result = ZoomIn::new(factors).apply(image.clone(), &mut rng);
		assert_ne!(result, image);
	}

	#[test]
	fn zoom_out_centers_resized_content() {
		let image = RgbImage::from_pixel(8, 8, Rgb([12, 34, 56]));
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomRange::ordered(2.0, 2.0)
			.expect("the zoom-out divisor range is valid");
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.get_pixel(3, 3).0, [12, 34, 56]);
	}

	#[test]
	fn zoom_out_fills_padding_with_black() {
		let image = RgbImage::from_pixel(8, 8, Rgb([12, 34, 56]));
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomRange::ordered(2.0, 2.0)
			.expect("the zoom-out divisor range is valid");
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.get_pixel(0, 0).0, [0, 0, 0]);
	}

	#[test]
	fn zoom_in_factor_one_preserves_pixels() {
		let image = gradient_image(3, 3);
		let mut rng = StdRng::seed_from_u64(42);
		let factors =
			ZoomRange::ordered(1.0, 1.0).expect("a factor of one is valid");
		let result = ZoomIn::new(factors).apply(image.clone(), &mut rng);
		assert_eq!(result, image);
	}

	#[test]
	fn zoom_out_divisor_one_preserves_pixels() {
		let image = gradient_image(3, 3);
		let mut rng = StdRng::seed_from_u64(42);
		let divisors =
			ZoomRange::ordered(1.0, 1.0).expect("a divisor of one is valid");
		let result = ZoomOut::new(divisors).apply(image.clone(), &mut rng);
		assert_eq!(result, image);
	}

	#[test]
	fn zoom_in_range_rejects_values_below_one() {
		assert_eq!(
			ZoomRange::try_new(0.9, 1.2).unwrap_err(),
			AugmentationError::InvalidZoomRange
		);
	}

	#[test]
	fn zoom_in_range_rejects_reversed_bounds() {
		assert_eq!(
			ZoomRange::try_new(2.0, 1.0).unwrap_err(),
			AugmentationError::InvalidZoomRange
		);
	}

	#[test]
	fn zoom_out_range_rejects_values_below_one() {
		assert_eq!(
			ZoomRange::try_new(0.9, 2.0).unwrap_err(),
			AugmentationError::InvalidZoomRange
		);
	}

	#[test]
	fn zoom_out_range_rejects_non_finite_values() {
		assert_eq!(
			ZoomRange::try_new(1.0, f32::INFINITY).unwrap_err(),
			AugmentationError::InvalidZoomRange
		);
	}

	#[test]
	fn relaxed_zoom_range_sorts_bounds() {
		assert_eq!(ZoomRange::relaxed(3.0, 1.0).unwrap().bounds(), (1.0, 3.0));
	}

	#[test]
	fn ordered_zoom_range_allows_equal_bounds() {
		assert_eq!(ZoomRange::ordered(2.0, 2.0).unwrap().bounds(), (2.0, 2.0));
	}

	#[test]
	fn strict_zoom_range_rejects_equal_or_reversed_bounds() {
		assert_eq!(
			ZoomRange::strict(2.0, 2.0).unwrap_err(),
			AugmentationError::InvalidZoomRange
		);
		assert_eq!(
			ZoomRange::strict(3.0, 2.0).unwrap_err(),
			AugmentationError::InvalidZoomRange
		);
	}

	#[test]
	fn zoom_range_is_const_evaluable() {
		const FACTORS: ZoomRange = match ZoomRange::try_new(1.0, 2.0) {
			Ok(factors) => factors,
			Err(_) => panic!("invalid test factors"),
		};
		const DIVISORS: ZoomRange = match ZoomRange::try_new(1.0, 2.0) {
			Ok(divisors) => divisors,
			Err(_) => panic!("invalid test divisors"),
		};
		assert_eq!(FACTORS.bounds(), (1.0, 2.0));
		assert_eq!(DIVISORS.bounds(), (1.0, 2.0));
	}
}
