use std::{
	fmt::Display,
	path::Path,
};

use burn::data::dataset::transform::Mapper;

use crate::{
	Result,
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

pub type IntensityValType = u8;

pub type Synthetic2Dataset<M, MO> = LaserDataset<M, Synthetic2Item, MO>;

/// The second synthetic dataset.
///
/// Intensity: mW, `0..=100`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Synthetic2Item {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,
	/// mW, varies wildly.
	external_intensity: Intensity<IntensityValType>,
}

impl Synthetic2Item {
	/// Get the external intensity.
	///
	/// This value likely does not need to be used.
	pub fn get_external_intensity(&self) -> Intensity<IntensityValType> {
		self.external_intensity
	}
}

impl LaserItemType for Synthetic2Item {
	type IntensityType = IntensityValType;

	fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
		&self.base
	}

	fn source_fields(&self) -> Vec<(&'static str, String)> {
		vec![("External intensity", self.external_intensity.to_string())]
	}
}

impl LaserItemBase for Synthetic2Item {
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
		&mut self.base
	}
}

impl Display for Synthetic2Item {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(
			f,
			"{} (external intensity {})",
			self.get_intensity(),
			self.get_external_intensity()
		)
	}
}

pub fn get_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<Synthetic2Item>],
) -> Result<Synthetic2Dataset<M, MO>>
where
	M: Mapper<Synthetic2Item, MO> + Send + Sync,
	MO: Send + Sync + Clone,
	P: AsRef<Path>,
{
	LaserDataset::try_new(mapper, dataset_root_folder, filters)
}

#[cfg(test)]
mod tests {
	use std::{
		path::PathBuf,
		process::{
			ExitCode,
			Termination,
		},
	};

	use super::*;
	const ITEM_PATH: &str = const_str::concat!(
		super::super::RELATIVE_DATASETS_PATH,
		"synthetic_2/images/"
	);

	use crate::{
		LoaderError,
		item::{
			ImagePath,
			ImageType,
			LaserItem,
			LaserItemTryFrom,
			tests::{
				create_path,
				extract_parts_from_name,
				parse_intensity_as,
			},
		},
	};

	impl TryFrom<ImagePath> for Synthetic2Item {
		type Error = LoaderError;

		/// Create a dataset item from the given path.
		///
		/// # Errors
		///
		/// [`LoaderError::Path`]:
		///
		/// - If the path is not a file
		/// - If the file name does not contain an underscore
		///
		/// [`LoaderError::Intensity`]:
		///
		/// - If the last component of the filename is not an integer 0..=255
		/// - If the path leads to an image claiming to be above an intensity of
		///   100
		fn try_from(value: ImagePath) -> Result<Self> {
			let name = value.get_filename();
			let parts = extract_parts_from_name(&name);
			if parts.len() != 2 {
				return Err(LoaderError::Intensity(
					"incorrect file name pattern".into(),
				));
			}
			let intensity = parse_intensity_as(parts[0])?;
			let external_intensity = parse_intensity_as(parts[1])?;
			let base = BaseItem::new(value, intensity);
			Ok(Self {
				base,
				external_intensity,
			})
		}
	}

	impl Termination for Synthetic2Item {
		fn report(self) -> std::process::ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[allow(dead_code)]
	struct SuccessfulImage(ImageType);

	impl From<ImageType> for SuccessfulImage {
		fn from(value: ImageType) -> Self {
			Self(value)
		}
	}

	impl Termination for SuccessfulImage {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[test]
	fn try_from_image_path_correct_min() -> Result<Synthetic2Item> {
		let path = create_path(ITEM_PATH, "render_0000_0000.png");
		let image_path = ImagePath::try_from(path)?;
		Synthetic2Item::try_from(image_path)
	}

	#[test]
	fn display_includes_intensities_without_position_or_image_path() {
		let item =
			try_from_image_path_correct_min().expect("item should be created");
		let output = item.to_string();

		assert!(output.contains("0 (external intensity 0)"));
		assert!(!output.contains("position"));
		assert!(!output.contains("render_0000_0000.png"));
	}

	#[test]
	fn try_from_image_path_correct_max() -> Result<Synthetic2Item> {
		let path = create_path(ITEM_PATH, "render_0100_0100.png");
		let image_path = ImagePath::try_from(path)?;
		Synthetic2Item::try_from(image_path)
	}

	// Cannot `try_from_image_path_incorrect_intensity` because it is impossible
	// to build an incorrect image_path.

	#[test]
	fn try_from_pathbuf_correct_min() -> Result<Synthetic2Item> {
		let path = create_path(ITEM_PATH, "render_0000_0000.png");
		Synthetic2Item::try_from_pathbuf(path)
	}

	#[test]
	fn try_from_pathbuf_correct_max() -> Result<Synthetic2Item> {
		let path = create_path(ITEM_PATH, "render_0100_0100.png");
		Synthetic2Item::try_from_pathbuf(path)
	}

	#[test]
	fn try_from_pathbuf_incorrect_path_entirely() {
		let image_path = PathBuf::from("completely random path");
		let res = Synthetic2Item::try_from_pathbuf(image_path);
		assert!(res.is_err());
	}

	#[test]
	fn try_from_pathbuf_incorrect_file() {
		let image_path = crate::item::tests::create_path(ITEM_PATH, "");
		let res = Synthetic2Item::try_from_pathbuf(image_path);
		assert!(res.is_err());
	}

	#[test]
	fn load_image_after_image_path() -> Result<SuccessfulImage> {
		let item = try_from_image_path_correct_max()?;
		item.load().map(|v| v.into())
	}

	#[test]
	fn load_image_after_pathbuf() -> Result<SuccessfulImage> {
		let item = try_from_pathbuf_correct_max()?;
		item.load().map(|v| v.into())
	}
}
