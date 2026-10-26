//! Types and loader for a legacy item dataset.

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
		Intensity,
		LaserItem as _,
		LaserItemBase,
		LaserItemType,
	},
};

/// Numeric intensity representation used by this source.
/// Integer representation used for laser intensity in this source.
pub type IntensityValType = u8;

/// A mapped dataset whose input items are [`Synthetic2Item`]s.
pub type LegacyItemDataset<M, MO> = LaserDataset<M, LegacyItem, MO>;

/// An item from the second synthetic dataset.
///
/// Both the local intensity and external intensity are measured in mW.
/// The intensity must be in `0..=100`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct LegacyItem {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,
	#[serde(alias = "externalLaserIntensity")]
	external_intensity: Option<Intensity<IntensityValType>>,
}

impl LegacyItem {
	/// Returns the external intensity measured for this item.
	///
	/// This value likely does not need to be used.
	pub fn get_external_intensity(
		&self
	) -> Option<Intensity<IntensityValType>> {
		self.external_intensity
	}
}

impl LaserItemType for LegacyItem {
	type IntensityType = IntensityValType;

	fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
		&self.base
	}

	fn source_fields(&self) -> Vec<(String, String)> {
		match self.get_external_intensity() {
			Some(external) => {
				vec![("External intensity".to_owned(), external.to_string())]
			},
			None => Vec::new(),
		}
	}
}

impl LaserItemBase for LegacyItem {
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
		&mut self.base
	}
}

impl Display for LegacyItem {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(f, "{}", self.get_intensity())?;
		match self.get_external_intensity() {
			Some(external) => {
				write!(f, " (external intensity {external})")
			},
			None => Ok(()),
		}
	}
}

/// Loads a legacy item dataset through a mapper.
///
/// # Errors
///
/// Returns a loading error if `index.csv` cannot be read, a row cannot be
/// deserialised, or a retained image path cannot be resolved.
pub fn build_legacy_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<LegacyItem>],
) -> LoaderResult<LegacyItemDataset<M, MO>>
where
	M: Mapper<LegacyItem, MO> + Send + Sync,
	MO: Send + Sync + Clone,
	P: AsRef<Path>,
{
	LaserDataset::try_new(mapper, dataset_root_folder, filters)
}
