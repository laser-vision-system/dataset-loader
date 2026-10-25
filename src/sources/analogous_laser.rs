//! Types and loader for the analogous real-world laser dataset.

use std::{
	fmt::Display,
	path::Path,
};

use burn::data::dataset::transform::Mapper;

use crate::{
	LoaderResult,
	dataset::{
		FilterFn,
		LaserDataset,
	},
	item::{
		BaseItem,
		LaserItem as _,
		LaserItemBase,
		LaserItemType,
	},
};

/// Floating-point representation used for laser intensity in this source.
pub type IntensityValType = f32;

/// A mapped dataset whose input items are [`AnalogousLaserItem`]s.
pub type AnalogousDataset<M, MO> = LaserDataset<M, AnalogousLaserItem, MO>;

/// The first real-world dataset, built on an analogous setup.
///
/// Intensity: mV, `0 – 6.9`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct AnalogousLaserItem {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,

	/// `0/64/128`.
	#[serde(alias = "Saturation")]
	saturation: u8,

	/// `5 – 2005`, step size: `100`.
	#[serde(alias = "ExposureAbsolute")]
	exposure_absolute: u16,

	/// `75 – 400`, step size: `65`.
	#[serde(alias = "Gamma")]
	gamma: u16,
}

impl AnalogousLaserItem {
	/// Returns the `saturation` value for this item.
	///
	/// Will be either 0, 64, or 128.
	pub fn get_saturation(&self) -> u8 {
		self.saturation
	}

	/// Returns the `exposure_absolute` value for this item.
	///
	/// Will be between 5 and 2005, with a step size of 100.
	pub fn get_exposure_absolute(&self) -> u16 {
		self.exposure_absolute
	}

	/// Returns the `gamma` value for this item.
	///
	/// Will be between 75 and 400, with a step size of 65.
	pub fn get_gamma(&self) -> u16 {
		self.gamma
	}
}

impl LaserItemType for AnalogousLaserItem {
	type IntensityType = IntensityValType;

	fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
		&self.base
	}

	fn source_fields(&self) -> Vec<(&'static str, String)> {
		vec![
			("Saturation", self.saturation.to_string()),
			("ExposureAbsolute", self.exposure_absolute.to_string()),
			("Gamma", self.gamma.to_string()),
		]
	}
}

impl LaserItemBase for AnalogousLaserItem {
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
		&mut self.base
	}
}

impl Display for AnalogousLaserItem {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(
			f,
			"{} at {} (saturation {}, exposure {}, gamma {})",
			self.get_intensity(),
			self.get_position().unwrap_or("unknown position"),
			self.get_saturation(),
			self.get_exposure_absolute(),
			self.get_gamma()
		)
	}
}

/// Loads the `analogous_laser` dataset through a mapper.
///
/// # Errors
///
/// Returns a loading error if `index.csv` cannot be read, a row cannot be
/// deserialised, or an image path cannot be resolved.
pub fn get_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<AnalogousLaserItem>],
) -> LoaderResult<AnalogousDataset<M, MO>>
where
	M: Mapper<AnalogousLaserItem, MO> + Send + Sync,
	MO: Send + Sync + Clone,
	P: AsRef<Path>,
{
	LaserDataset::try_new(mapper, dataset_root_folder, filters)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn display_includes_intensity_and_position_without_image_path() {
		let csv = "relative_path,position,intensity,Saturation,ExposureAbsolute,Gamma\nimages/sample.png,front_outer_90,3.2,64,405,270\n";
		let mut reader = csv::Reader::from_reader(csv.as_bytes());
		let item: AnalogousLaserItem = reader
			.deserialize()
			.next()
			.and_then(std::result::Result::ok)
			.expect("CSV row should deserialize");
		let output = item.to_string();

		assert!(output.contains("3.2 at front_outer_90"));
		assert!(output.contains("saturation 64"));
		assert!(output.contains("exposure 405"));
		assert!(output.contains("gamma 270"));
		assert!(!output.contains("sample.png"));
	}
}
