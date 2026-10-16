use image::imageops::{
	FilterType,
	crop,
	overlay,
	resize,
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};

use super::{
	ImageStep,
	ImageType,
};

#[derive(Clone, Copy)]
enum CanvasFit {
	CenterCrop,
	BlackPad,
}

/// Inclusive finite range of zoom-in scale factors, each at least one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomInFactors {
	min: f32,
	max: f32,
}

impl ZoomInFactors {
	/// Creates a range; returns [`None`] for non-finite, reversed, or sub-one
	/// bounds.
	pub fn new(
		min: f32,
		max: f32,
	) -> Option<Self> {
		(min.is_finite() && max.is_finite() && min >= 1.0 && max >= min)
			.then_some(Self {
				min,
				max,
			})
	}

	fn sample(
		self,
		rng: &mut StdRng,
	) -> f32 {
		rng.random_range(self.min..=self.max)
	}
}

/// Inclusive finite range of zoom-out divisors, each at least one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomOutDivisors {
	min: f32,
	max: f32,
}

impl ZoomOutDivisors {
	/// Creates a divisor range; returns [`None`] for non-finite, reversed, or
	/// sub-one bounds.
	pub fn new(
		min: f32,
		max: f32,
	) -> Option<Self> {
		(min.is_finite() && max.is_finite() && min >= 1.0 && max >= min)
			.then_some(Self {
				min,
				max,
			})
	}

	fn sample(
		self,
		rng: &mut StdRng,
	) -> f32 {
		rng.random_range(self.min..=self.max)
	}
}

#[derive(Clone, Debug)]
pub struct ZoomIn {
	factors: ZoomInFactors,
}

impl ZoomIn {
	/// Creates a zoom-in transform with a validated factor range.
	pub const fn new(factors: ZoomInFactors) -> Self {
		Self {
			factors,
		}
	}
}

impl ImageStep for ZoomIn {
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		resample_to_canvas(
			image,
			self.factors.sample(rng),
			CanvasFit::CenterCrop,
		)
	}
}

#[derive(Clone, Debug)]
pub struct ZoomOut {
	divisors: ZoomOutDivisors,
}

impl ZoomOut {
	/// Creates a zoom-out transform with a validated divisor range.
	pub const fn new(divisors: ZoomOutDivisors) -> Self {
		Self {
			divisors,
		}
	}
}

impl ImageStep for ZoomOut {
	fn apply(
		&self,
		image: ImageType,
		rng: &mut StdRng,
	) -> ImageType {
		resample_to_canvas(
			image,
			1.0 / self.divisors.sample(rng),
			CanvasFit::BlackPad,
		)
	}
}

#[expect(
	clippy::cast_precision_loss,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	clippy::integer_division,
	reason = "Scale factors map dimensions to pixels; fitting is centered using floor division"
)]
fn resample_to_canvas(
	image: ImageType,
	scale: f32,
	fit: CanvasFit,
) -> ImageType {
	let (width, height) = image.dimensions();
	let scaled_width = ((width as f32 * scale).round() as u32).max(1);
	let scaled_height = ((height as f32 * scale).round() as u32).max(1);
	if scaled_width == width && scaled_height == height {
		return image;
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
			let mut canvas = ImageType::new(width, height);
			let left = width.saturating_sub(scaled_width) / 2;
			let top = height.saturating_sub(scaled_height) / 2;
			overlay(&mut canvas, &resized, i64::from(left), i64::from(top));
			canvas
		},
	}
}

#[cfg(test)]
mod tests {
	use image::Rgb;
	use rand::SeedableRng as _;

	use super::*;

	fn gradient_image(
		width: u32,
		height: u32,
	) -> ImageType {
		ImageType::from_fn(width, height, |x, y| {
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
		let factors = ZoomInFactors {
			min: 2.0,
			max: 2.0,
		};
		let result = ZoomIn::new(factors).apply(image, &mut rng);
		assert_eq!(result.dimensions(), (4, 5));
	}

	#[test]
	fn zoom_out_preserves_dimensions() {
		let image = gradient_image(12, 10);
		let mut rng = StdRng::seed_from_u64(42);
		let divisors = ZoomOutDivisors {
			min: 2.0,
			max: 2.0,
		};
		let result = ZoomOut::new(divisors).apply(image, &mut rng);
		assert_eq!(result.dimensions(), (12, 10));
	}

	#[test]
	fn zoom_in_changes_pixels_when_scaled_up() {
		let image = gradient_image(8, 8);
		let mut rng = StdRng::seed_from_u64(42);
		let result = ZoomIn::new(ZoomInFactors {
			min: 2.0,
			max: 2.0,
		})
		.apply(image.clone(), &mut rng);
		assert_ne!(result, image);
	}

	#[test]
	fn zoom_out_centers_resized_content() {
		let image = ImageType::from_pixel(8, 8, Rgb([12, 34, 56]));
		let mut rng = StdRng::seed_from_u64(42);
		let result = ZoomOut::new(ZoomOutDivisors {
			min: 2.0,
			max: 2.0,
		})
		.apply(image, &mut rng);
		assert_eq!(result.get_pixel(3, 3).0, [12, 34, 56]);
	}

	#[test]
	fn zoom_out_fills_padding_with_black() {
		let image = ImageType::from_pixel(8, 8, Rgb([12, 34, 56]));
		let mut rng = StdRng::seed_from_u64(42);
		let result = ZoomOut::new(ZoomOutDivisors {
			min: 2.0,
			max: 2.0,
		})
		.apply(image, &mut rng);
		assert_eq!(result.get_pixel(0, 0).0, [0, 0, 0]);
	}

	#[test]
	fn zoom_in_factor_one_preserves_pixels() {
		let image = gradient_image(3, 3);
		let mut rng = StdRng::seed_from_u64(42);
		let result = ZoomIn::new(ZoomInFactors {
			min: 1.0,
			max: 1.0,
		})
		.apply(image.clone(), &mut rng);
		assert_eq!(result, image);
	}

	#[test]
	fn zoom_out_divisor_one_preserves_pixels() {
		let image = gradient_image(3, 3);
		let mut rng = StdRng::seed_from_u64(42);
		let result = ZoomOut::new(ZoomOutDivisors {
			min: 1.0,
			max: 1.0,
		})
		.apply(image.clone(), &mut rng);
		assert_eq!(result, image);
	}

	#[test]
	fn zoom_in_range_rejects_values_below_one() {
		assert!(ZoomInFactors::new(0.9, 1.2).is_none());
	}

	#[test]
	fn zoom_in_range_rejects_reversed_bounds() {
		assert!(ZoomInFactors::new(2.0, 1.0).is_none());
	}

	#[test]
	fn zoom_out_range_rejects_values_below_one() {
		assert!(ZoomOutDivisors::new(0.9, 2.0).is_none());
	}

	#[test]
	fn zoom_out_range_rejects_non_finite_values() {
		assert!(ZoomOutDivisors::new(1.0, f32::INFINITY).is_none());
	}
}
