use std::{
	fmt::Display,
	path::Path,
};

use burn::data::dataset::transform::Mapper;

use crate::{
	dataset::{
		FilterFn,
		LaserDataset,
	},
	error::Result,
	item::{
		BaseItem,
		LaserItem as _,
		LaserItemBase,
		LaserItemType,
	},
};

pub type IntensityValType = u8;

/// The first synthetic dataset.
///
/// Intensity: mW, `0..=53`.
/// - 7mW has an unusually bright green spot.
/// - 9mW is unreasonably bright.
/// - 10mW has an unusually dark spot.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Synthetic1Item {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,
}

impl LaserItemType for Synthetic1Item {
	type IntensityType = IntensityValType;

	fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
		&self.base
	}
}

impl LaserItemBase for Synthetic1Item {
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
		&mut self.base
	}
}

impl Display for Synthetic1Item {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(f, "{}", self.get_intensity())
	}
}

pub type Synthetic1Dataset<M, MO> = LaserDataset<M, Synthetic1Item, MO>;

/// Create the `synthetic_1` dataset loader.
///
/// # Errors
///
/// If the dataset fails to load.
pub fn get_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<Synthetic1Item>],
) -> Result<Synthetic1Dataset<M, MO>>
where
	M: Mapper<Synthetic1Item, MO> + Send + Sync,
	MO: Send + Sync + Clone,
	P: AsRef<Path>,
{
	LaserDataset::try_new(mapper, dataset_root_folder, filters)
}

#[cfg(test)]
pub(crate) mod tests {
	use std::{
		path::PathBuf,
		process::{
			ExitCode,
			Termination,
		},
	};

	use image::DynamicImage;

	use super::*;
	const ITEM_PATH: &str = const_str::concat!(
		super::super::RELATIVE_DATASETS_PATH,
		"synthetic_1/images/"
	);

	use crate::{
		LoaderError,
		item::{
			ImagePath,
			LaserItemTryFrom as _,
			tests::{
				create_path,
				extract_parts_from_name,
				parse_intensity_as,
			},
		},
	};

	impl TryFrom<ImagePath> for Synthetic1Item {
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
		///   53
		fn try_from(value: ImagePath) -> Result<Self> {
			let name = value.get_filename();
			let parts = extract_parts_from_name(&name);
			let intensity = parse_intensity_as(parts[0])?;
			let base = BaseItem::new(value, intensity);
			Ok(Self {
				base,
			})
		}
	}

	impl Termination for Synthetic1Item {
		fn report(self) -> std::process::ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[expect(
		dead_code,
		reason = "Used to allow tests to return a type for chaining"
	)]
	struct SuccessfulImage(DynamicImage);

	impl From<DynamicImage> for SuccessfulImage {
		fn from(value: DynamicImage) -> Self {
			Self(value)
		}
	}

	impl Termination for SuccessfulImage {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[test]
	fn try_from_image_path_correct_min() -> Result<Synthetic1Item> {
		let path = create_path(ITEM_PATH, "render_0000.png");
		let image_path = ImagePath::try_from(path)?;
		Synthetic1Item::try_from(image_path)
	}

	#[test]
	fn display_includes_intensity_without_position_or_image_path() {
		let item =
			try_from_image_path_correct_min().expect("item should be created");
		let output = item.to_string();

		assert!(output.contains('0'));
		assert!(!output.contains("position"));
		assert!(!output.contains("render_0000.png"));
	}

	#[test]
	fn try_from_image_path_correct_max() -> Result<Synthetic1Item> {
		let path = create_path(ITEM_PATH, "render_0053.png");
		let image_path = ImagePath::try_from(path)?;
		Synthetic1Item::try_from(image_path)
	}

	// Cannot `try_from_image_path_incorrect_intensity` because it is impossible
	// to build an incorrect image_path.

	#[test]
	fn try_from_pathbuf_correct_min() -> Result<Synthetic1Item> {
		let path = create_path(ITEM_PATH, "render_0000.png");
		Synthetic1Item::try_from_pathbuf(path)
	}

	#[test]
	fn try_from_pathbuf_correct_max() -> Result<Synthetic1Item> {
		let path = create_path(ITEM_PATH, "render_0053.png");
		Synthetic1Item::try_from_pathbuf(path)
	}

	#[test]
	fn try_from_pathbuf_incorrect_path_entirely() {
		let image_path = PathBuf::from("completely random path");
		let res = Synthetic1Item::try_from_pathbuf(image_path);
		res.unwrap_err();
	}

	#[test]
	fn try_from_pathbuf_incorrect_file() {
		let image_path = crate::item::tests::create_path(ITEM_PATH, "");
		let res = Synthetic1Item::try_from_pathbuf(image_path);
		res.unwrap_err();
	}

	#[test]
	fn load_image_after_image_path() -> Result<SuccessfulImage> {
		let item = try_from_image_path_correct_max()?;
		item.load().map(Into::into)
	}

	#[test]
	fn load_image_after_pathbuf() -> Result<SuccessfulImage> {
		let item = try_from_pathbuf_correct_max()?;
		item.load().map(Into::into)
	}
}
