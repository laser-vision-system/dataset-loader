//! Types and loader for the first synthetic dataset.

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

/// Numeric intensity representation used by this source.
/// Integer representation used for laser intensity in this source.
pub type IntensityValType = u8;

/// An item from the first synthetic dataset.
///
/// The intensity is measured in mW and must be in `0..=53`.
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

/// A mapped dataset whose input items are [`Synthetic1Item`]s.
pub type Synthetic1Dataset<M, MO> = LaserDataset<M, Synthetic1Item, MO>;

/// Loads the `synthetic_1` dataset through a mapper.
///
/// # Errors
///
/// Returns a loading error if `index.csv` cannot be read, a row cannot be
/// deserialised, or a retained image path cannot be resolved.
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
		assert_matches,
		path::PathBuf,
		process::{
			ExitCode,
			Termination,
		},
	};

	use image::DynamicImage;

	use super::*;
	use crate::{
		LoaderError,
		item::{
			ImagePath,
			tests::{
				create_path,
				create_temporary_image,
			},
		},
	};

	const ITEM_PATH: &str = const_str::concat!(
		super::super::RELATIVE_DATASETS_PATH,
		"synthetic_1/images/"
	);

	// Production loading uses the CSV index; these filename conversions support
	// tests only.
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
			let filename = value.get_filename();
			let (_, intensity) =
				filename.rsplit_once('_').ok_or_else(|| {
					LoaderError::Path(
						"filename must contain an underscore (_)".into(),
					)
				})?;
			let intensity =
				intensity.parse::<IntensityValType>().map_err(|why| {
					LoaderError::Intensity(format!(
						"invalid filename intensity: {why}"
					))
				})?;
			if intensity > 53 {
				return Err(LoaderError::Intensity(
					"intensity must be in 0..=53".into(),
				));
			}
			Ok(Self {
				base: BaseItem::new(value, intensity.into()),
			})
		}
	}

	impl TryFrom<PathBuf> for Synthetic1Item {
		type Error = LoaderError;

		fn try_from(value: PathBuf) -> Result<Self> {
			Self::try_from(ImagePath::try_from(value)?)
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
	fn try_from_pathbuf_correct_min() -> Result<Synthetic1Item> {
		let path = create_path(ITEM_PATH, "render_0000.png");
		Synthetic1Item::try_from(path)
	}

	#[test]
	fn try_from_pathbuf_correct_max() -> Result<Synthetic1Item> {
		let path = create_path(ITEM_PATH, "render_0053.png");
		Synthetic1Item::try_from(path)
	}

	#[test]
	fn display_includes_intensity_without_position_or_image_path() {
		let item =
			try_from_pathbuf_correct_min().expect("item should be created");
		let output = item.to_string();

		assert!(output.contains('0'));
		assert!(!output.contains("position"));
		assert!(!output.contains("render_0000.png"));
	}

	#[test]
	fn try_from_rejects_intensity_above_source_limit() {
		let path = PathBuf::from("images/render_0054.png");
		assert_matches!(
			Synthetic1Item::try_from(path),
			Err(crate::LoaderError::Intensity(_))
		);
	}

	#[test]
	fn try_from_pathbuf_incorrect_path_entirely() {
		let image_path = PathBuf::from("completely random path");
		let res = Synthetic1Item::try_from(image_path);
		res.unwrap_err();
	}

	#[test]
	fn try_from_pathbuf_incorrect_file() {
		let image_path = crate::item::tests::create_path(ITEM_PATH, "");
		let res = Synthetic1Item::try_from(image_path);
		res.unwrap_err();
	}

	#[test]
	fn load_image_after_pathbuf() -> Result<SuccessfulImage> {
		let image = create_temporary_image("render_0053.png");
		let item = Synthetic1Item::try_from(image.path().clone())?;
		item.load().map(Into::into)
	}
}
