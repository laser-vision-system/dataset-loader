//! Types and loader for the second synthetic dataset.

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
pub type Synthetic2Dataset<M, MO> = LaserDataset<M, Synthetic2Item, MO>;

/// An item from the second synthetic dataset.
///
/// Both the local intensity and external intensity are measured in mW.
/// The intensity must be in `0..=100`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Synthetic2Item {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,
	/// mW, varies wildly.
	external_intensity: Intensity<IntensityValType>,
}

impl Synthetic2Item {
	/// Returns the external intensity measured for this item.
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

/// Loads the `synthetic_2` dataset through a mapper.
///
/// # Errors
///
/// Returns a loading error if `index.csv` cannot be read, a row cannot be
/// deserialised, or a retained image path cannot be resolved.
pub fn get_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<Synthetic2Item>],
) -> LoaderResult<Synthetic2Dataset<M, MO>>
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
		assert_matches,
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

	use image::DynamicImage;

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

	// Production loading uses the CSV index; these filename conversions support
	// tests only.
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
		fn try_from(value: ImagePath) -> LoaderResult<Self> {
			let filename = value.get_filename();
			let mut parts = filename.rsplit('_');
			let external_intensity = parts.next().ok_or_else(|| {
				LoaderError::Path(
					"filename is missing external intensity".into(),
				)
			})?;
			let intensity = parts.next().ok_or_else(|| {
				LoaderError::Path("filename is missing image intensity".into())
			})?;
			let prefix = parts.next().ok_or_else(|| {
				LoaderError::Path(
					"filename must contain at least two underscores".into(),
				)
			})?;
			if prefix.is_empty() {
				return Err(LoaderError::Path(
					"filename must contain a non-empty prefix".into(),
				));
			}
			let intensity = parse_source_intensity(intensity, 100)?;
			let external_intensity =
				parse_source_intensity(external_intensity, 111)?;
			Ok(Self {
				base: BaseItem::new(value, intensity.into()),
				external_intensity: external_intensity.into(),
			})
		}
	}

	impl TryFrom<PathBuf> for Synthetic2Item {
		type Error = LoaderError;

		fn try_from(value: PathBuf) -> LoaderResult<Self> {
			Self::try_from(ImagePath::try_from(value)?)
		}
	}

	fn parse_source_intensity(
		value: &str,
		max: u8,
	) -> LoaderResult<IntensityValType> {
		let intensity = value.parse::<IntensityValType>().map_err(|why| {
			LoaderError::Intensity(format!("invalid filename intensity: {why}"))
		})?;
		if intensity > max {
			return Err(LoaderError::Intensity(format!(
				"intensity must be in 0..={max}"
			)));
		}
		Ok(intensity)
	}

	impl Termination for Synthetic2Item {
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
	fn try_from_pathbuf_correct_min() -> LoaderResult<Synthetic2Item> {
		let path = create_path(ITEM_PATH, "render_0000_0000.png");
		Synthetic2Item::try_from(path)
	}

	#[test]
	fn try_from_pathbuf_correct_max() -> LoaderResult<Synthetic2Item> {
		let path = create_path(ITEM_PATH, "render_0100_0100.png");
		Synthetic2Item::try_from(path)
	}

	#[test]
	fn display_includes_intensities_without_position_or_image_path() {
		let item =
			try_from_pathbuf_correct_min().expect("item should be created");
		let output = item.to_string();

		assert!(output.contains("0 (external intensity 0)"));
		assert!(!output.contains("position"));
		assert!(!output.contains("render_0000_0000.png"));
	}

	#[test]
	fn try_from_rejects_either_intensity_above_source_limit() {
		for path in [
			PathBuf::from("images/render_0101_0000.png"),
			PathBuf::from("images/render_0000_0112.png"),
		] {
			assert_matches!(
				Synthetic2Item::try_from(path),
				Err(crate::LoaderError::Intensity(_))
			);
		}
	}

	#[test]
	fn try_from_pathbuf_incorrect_path_entirely() {
		let image_path = PathBuf::from("completely random path");
		let res = Synthetic2Item::try_from(image_path);
		res.unwrap_err();
	}

	#[test]
	fn try_from_pathbuf_incorrect_file() {
		let image_path = crate::item::tests::create_path(ITEM_PATH, "");
		let res = Synthetic2Item::try_from(image_path);
		res.unwrap_err();
	}

	#[test]
	fn load_image_after_pathbuf() -> LoaderResult<SuccessfulImage> {
		let image = create_temporary_image("render_0100_0100.png");
		let item = Synthetic2Item::try_from(image.path().clone())?;
		item.load().map(Into::into)
	}
}
