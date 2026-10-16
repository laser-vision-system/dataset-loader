mod advanced_gaussian;
mod box_filter;
mod fast_gaussian;
mod gaussian;

pub use advanced_gaussian::GaussianAdvanced;
pub use box_filter::BoxFilter;
pub use fast_gaussian::{
	GaussianFast,
	RandomGaussianFast,
};
pub use gaussian::{
	Gaussian,
	RandomGaussian,
};
use rand::{
	RngExt as _,
	rngs::StdRng,
};

use super::ImageType;

#[derive(Clone, Copy, Debug, Default)]
struct BlurSigma(f32);

impl BlurSigma {
	fn new(value: f32) -> Option<Self> {
		(value.is_finite() && value >= 0.0).then_some(Self(value))
	}
}

#[derive(Clone, Copy, Debug)]
struct BlurSigmaRange {
	min: f32,
	max: f32,
}

impl BlurSigmaRange {
	fn new(
		min: f32,
		max: f32,
	) -> Option<Self> {
		(min.is_finite() && max.is_finite() && min >= 0.0 && min < max)
			.then_some(Self {
				min,
				max,
			})
	}

	fn sample(
		self,
		rng: &mut StdRng,
	) -> BlurSigma {
		BlurSigma(rng.random_range(self.min..self.max))
	}
}

fn apply_blur(
	image: ImageType,
	sigma: BlurSigma,
	blur: fn(&ImageType, f32) -> ImageType,
) -> ImageType {
	if sigma.0 == 0.0 {
		return image;
	}
	blur(&image, sigma.0)
}
