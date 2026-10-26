//! Source-specific item types and loaders for the available laser datasets.

pub mod legacy;
pub mod standard;

// Synthetic
pub mod synthetic_1;
pub mod synthetic_2;

// Real-world
pub mod analogous_laser;
pub mod in_lab_single_laser;

pub use self::{
	legacy::{
		LegacyItem,
		LegacyItemDataset,
		build_legacy_dataset,
	},
	standard::{
		StandardItem,
		StandardLaserDataset,
		build_standard_dataset,
	},
};

#[cfg(test)]
/// Where the datasets are in relation to this crate.
const RELATIVE_DATASETS_PATH: &str = "../datasets/";
