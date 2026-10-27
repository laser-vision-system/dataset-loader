//! Types and loader for the in-lab single-laser dataset.

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

/// A mapped dataset whose input items are [`InLabSingleLaserItem`]s.
pub type InLabSingleLaserDataset<M, MO> =
	LaserDataset<M, InLabSingleLaserItem, MO>;

/// The first dataset collected in the lab.
///
/// Intensity: mW, `3.31 - 19.71`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct InLabSingleLaserItem {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,

	/// `0/64/128`.
	#[serde(alias = "Saturation")]
	saturation: u8,

	/// `75 – 400`, step size: `65`.
	#[serde(alias = "Gamma")]
	gamma: u16,

	/// `5 – 2005`, step size: `100`.
	#[serde(alias = "ExposureAbsolute")]
	exposure_absolute: u16,
}

impl InLabSingleLaserItem {
	/// Returns the `saturation` value for this item.
	///
	/// Will be either 0, 64, or 128.
	pub fn get_saturation(&self) -> u8 {
		self.saturation
	}

	/// Returns the `gamma` value for this item.
	///
	/// Will be between 75 and 400, with a step size of 65.
	pub fn get_gamma(&self) -> u16 {
		self.gamma
	}

	/// Returns the `exposure_absolute` value for this item.
	///
	/// Will be between 5 and 2005, with a step size of 100.
	pub fn get_exposure_absolute(&self) -> u16 {
		self.exposure_absolute
	}
}

impl LaserItemType for InLabSingleLaserItem {
	type IntensityType = IntensityValType;

	fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
		&self.base
	}

	fn source_fields(&self) -> Vec<(String, String)> {
		vec![
			("Saturation".to_owned(), self.saturation.to_string()),
			("Gamma".to_owned(), self.gamma.to_string()),
			(
				"ExposureAbsolute".to_owned(),
				self.exposure_absolute.to_string(),
			),
		]
	}
}

impl LaserItemBase for InLabSingleLaserItem {
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
		&mut self.base
	}
}

impl Display for InLabSingleLaserItem {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(
			f,
			"{} ({}, {}, {}): '{}'",
			self.get_intensity(),
			self.get_saturation(),
			self.get_exposure_absolute(),
			self.get_gamma(),
			self.get_image_path()
		)
	}
}

/// Loads the `in_lab_single_laser` dataset through a mapper.
///
/// # Errors
///
/// Returns a loading error if `index.csv` cannot be read, a row cannot be
/// deserialised, or an image path cannot be resolved.
pub fn get_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<InLabSingleLaserItem>],
) -> LoaderResult<InLabSingleLaserDataset<M, MO>>
where
	M: Mapper<InLabSingleLaserItem, MO> + Send + Sync,
	MO: Send + Sync + Clone,
	P: AsRef<Path>,
{
	LaserDataset::try_new(mapper, dataset_root_folder, filters)
}
