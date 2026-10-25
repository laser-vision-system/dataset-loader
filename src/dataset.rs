//! CSV-backed datasets and their mapped Burn dataset adapters.

use std::{
	fmt,
	path::Path,
};

use burn::data::dataset::{
	Dataset,
	InMemDataset,
	transform::{
		Mapper,
		MapperDataset,
		SelectionDataset,
	},
};
use serde::de::DeserializeOwned;

use crate::{
	Split,
	SplitType,
	Splitable,
	error::{
		LoaderError,
		Result,
	},
	item::{
		LaserItem,
		LaserItemBase,
	},
};

const INDEX_FILENAME: &str = "index.csv";

type DatasetStorage<MI> = InMemDataset<MI>;
type DatasetMap<M, MI> = MapperDataset<DatasetStorage<MI>, M, MI>;
type DatasetSelect<M, MI, MO> = SelectionDataset<DatasetMap<M, MI>, MO>;

type FileReader = csv::Reader<std::fs::File>;

/// A predicate selecting dataset items to retain.
///
/// Items for which the predicate returns `false` are discarded before their
/// image paths are resolved.
///
/// # Examples
///
/// ```
/// use dataset_loader::dataset::FilterFn;
///
/// let keep_positive: FilterFn<i32> = Box::new(|item| *item > 0);
/// assert!(keep_positive(&1));
/// assert!(!keep_positive(&0));
/// ```
pub type FilterFn<T> = Box<dyn Fn(&T) -> bool>;

/// A CSV-backed dataset of laser items mapped to an output type.
///
/// Rows are read from `index.csv` in the dataset root, filtered, and have their
/// image paths resolved against that root. Access order is shuffled with a
/// fixed seed, so repeated construction from the same input order produces the
/// same selection order. Mapper execution remains lazy through Burn's mapped
/// dataset adapter.
pub struct LaserDataset<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone,
{
	dataset: DatasetSelect<M, MI, MO>,
}

impl<M, MI, MO> LaserDataset<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItemBase + DeserializeOwned,
	MO: Send + Sync + Clone,
{
	/// Create a CSV reader from a root directory.
	fn create_reader(dataset_root: &Path) -> Result<FileReader> {
		let csv_file = dataset_root.join(INDEX_FILENAME);
		#[cfg(test)]
		println!("Reading: {}", csv_file.display());
		csv::Reader::from_path(csv_file).map_err(LoaderError::from)
	}

	/// Reads a CSV dataset, parsing into dataset items and filtering entries.
	fn load_dataset(
		mut reader: FileReader,
		dataset_root: &Path,
		filters: &[FilterFn<MI>],
	) -> Result<Vec<MI>> {
		reader
			.deserialize::<MI>()
			.filter_map(|res| {
				match res.map_err(LoaderError::from) {
					Err(err) => Some(Err(err)),
					Ok(mut entry) => {
						filters.iter().all(|filter| filter(&entry)).then(|| {
							entry.resolve_paths(dataset_root).map(|()| entry)
						})
					},
				}
			})
			.collect()
	}

	/// Create the chain of dataset transformations over the storage dataset.
	fn build_dataset(
		items: Vec<MI>,
		mapper: M,
	) -> DatasetSelect<M, MI, MO> {
		let dataset = InMemDataset::new(items);
		let dataset = MapperDataset::new(dataset, mapper);
		SelectionDataset::new_shuffled(dataset, 42)
	}

	/// Loads a dataset from the root folder's `index.csv` file.
	///
	/// The CSV must contain fields accepted by `MI`; the shared item path
	/// column is named `relative_path`. Filters run before path
	/// canonicalisation. The resulting dataset uses a deterministic shuffle
	/// seed of `42`.
	///
	/// # Errors
	///
	/// Returns [`LoaderError::CSV`] if the index is missing, unreadable, or has
	/// malformed records, and [`LoaderError::Path`] if a retained item's image
	/// path cannot be canonicalized.
	pub fn try_new<P>(
		mapper: M,
		dataset_root_folder: P,
		filters: &[FilterFn<MI>],
	) -> Result<Self>
	where
		P: AsRef<Path>,
	{
		let root = dataset_root_folder.as_ref();
		#[cfg(test)]
		println!("Loading from '{}'", root.display());
		let reader = Self::create_reader(root)?;
		let items = Self::load_dataset(reader, root, filters)?;
		let dataset = Self::build_dataset(items, mapper);
		Ok(Self {
			dataset,
		})
	}
}

impl<M, MI, MO> Dataset<MO> for LaserDataset<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone,
{
	fn get(
		&self,
		index: usize,
	) -> Option<MO> {
		self.dataset.get(index)
	}

	fn len(&self) -> usize {
		self.dataset.len()
	}
}

impl<M, MI, MO> Splitable<MO> for LaserDataset<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone,
{
	type DatasetType = DatasetSelect<M, MI, MO>;
	type Output = LaserDatasetSplit<M, MI, MO>;

	fn get_dataset(&self) -> &Self::DatasetType {
		&self.dataset
	}

	fn create_split(
		&self,
		start: usize,
		end: usize,
		split: SplitType,
	) -> Self::Output {
		let items = self.dataset.slice(start, end);
		LaserDatasetSplit {
			dataset: items,
			split,
		}
	}
}

impl<M, MI, MO> fmt::Display for LaserDataset<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone + LaserItem,
{
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		let items: Vec<_> = self
			.dataset
			.iter()
			.map(|item| item.simple_display().to_string())
			.collect();
		let string = items.join("\n");
		write!(f, "{string}")
	}
}

/// One train, validation, or test partition of a [`LaserDataset`].
pub struct LaserDatasetSplit<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone,
{
	dataset: DatasetSelect<M, MI, MO>,
	split: SplitType,
}

impl<M, MI, MO> Dataset<MO> for LaserDatasetSplit<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone,
{
	fn get(
		&self,
		index: usize,
	) -> Option<MO> {
		self.dataset.get(index)
	}

	fn len(&self) -> usize {
		self.dataset.len()
	}
}

impl<M, MI, MO> Split for LaserDatasetSplit<M, MI, MO>
where
	M: Mapper<MI, MO> + Send + Sync,
	MI: Send + Sync + Clone + LaserItem,
	MO: Send + Sync + Clone,
{
	fn split_type(&self) -> SplitType {
		self.split
	}
}

#[cfg(test)]
mod tests {
	//! The tests use a mirror of the Synthetic 1 dataset.

	use std::{
		fmt::Display,
		path::PathBuf,
		process::{
			ExitCode,
			Termination,
		},
		sync::atomic,
	};

	use image::RgbImage;
	use serde::Deserialize;

	use super::*;
	use crate::item::{
		BaseItem,
		ImagePath,
		Intensity,
		LaserItemBase,
		LaserItemType,
	};

	type IntensityValType = u8;

	struct TemporaryDataset(PathBuf);

	impl TemporaryDataset {
		fn root(&self) -> &Path {
			&self.0
		}
	}

	impl Drop for TemporaryDataset {
		fn drop(&mut self) {
			_ = std::fs::remove_dir_all(&self.0);
		}
	}

	fn create_temporary_dataset() -> TemporaryDataset {
		static NEXT_ID: atomic::AtomicU64 = atomic::AtomicU64::new(0);
		let id = NEXT_ID.fetch_add(1, atomic::Ordering::Relaxed);
		let root = std::env::temp_dir().join(format!(
			"dataset-loader-dataset-{}-{id}",
			std::process::id()
		));
		let images = root.join("images");
		std::fs::create_dir_all(&images)
			.expect("temporary dataset image folder should be created");
		for intensity in [9, 10] {
			let path = images.join(format!("render_{intensity:04}.png"));
			RgbImage::from_pixel(1, 1, image::Rgb([10, 20, 30]))
				.save(path)
				.expect("temporary dataset image should be written");
		}
		std::fs::write(
			root.join(INDEX_FILENAME),
			"relative_path,intensity\nimages/render_0009.png,9\nimages/render_0010.png,10\n",
		)
		.expect("temporary dataset index should be written");
		TemporaryDataset(root)
	}

	#[derive(Clone, Debug, Deserialize)]
	pub struct DummyMapperInput {
		#[serde(flatten)]
		base: BaseItem<IntensityValType>,
	}

	impl DummyMapperInput {
		fn parse_intensity(name: &str) -> Result<Intensity<IntensityValType>> {
			let intensity = name
				.rsplit('_')
				.collect::<Vec<_>>()
				.first()
				.ok_or_else(|| {
					LoaderError::Path(
						"file name must contain an underscore (_)".into(),
					)
				})?
				.to_string();
			let intensity: Intensity<IntensityValType> =
				intensity.parse().map_err(|why| {
					LoaderError::Intensity(format!(
						"last component of the filename must be an integer 0..=255: {why}"
					))
				})?;
			Ok(intensity)
		}

		fn check_intensity(
			intensity: Intensity<IntensityValType>
		) -> Result<Intensity<IntensityValType>> {
			match *intensity {
				..=53 => Ok(intensity),
				_ => {
					Err(LoaderError::Intensity(
						"No image with this intensity".into(),
					))
				},
			}
		}

		fn parse_and_check_intensity(
			name: &str
		) -> Result<Intensity<IntensityValType>> {
			let intensity = Self::parse_intensity(name)?;
			Self::check_intensity(intensity)
		}
	}

	impl LaserItemType for DummyMapperInput {
		type IntensityType = u8;

		fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
			&self.base
		}
	}

	impl LaserItemBase for DummyMapperInput {
		fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
			&mut self.base
		}
	}

	impl TryFrom<ImagePath> for DummyMapperInput {
		type Error = LoaderError;

		fn try_from(value: ImagePath) -> Result<Self> {
			let intensity =
				Self::parse_and_check_intensity(&value.get_filename())?;
			let base = BaseItem::new(value, intensity);
			Ok(Self {
				base,
			})
		}
	}

	impl Termination for DummyMapperInput {
		fn report(self) -> std::process::ExitCode {
			ExitCode::SUCCESS
		}
	}

	struct DummyMapper;

	impl Mapper<DummyMapperInput, DummyMapperInput> for DummyMapper {
		fn map(
			&self,
			item: &DummyMapperInput,
		) -> DummyMapperInput {
			item.clone()
		}
	}

	type DummyDataset =
		LaserDataset<DummyMapper, DummyMapperInput, DummyMapperInput>;

	type DummyDatasetSplit =
		LaserDatasetSplit<DummyMapper, DummyMapperInput, DummyMapperInput>;

	impl Termination for DummyDataset {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	impl Display for DummyDatasetSplit {
		fn fmt(
			&self,
			f: &mut std::fmt::Formatter<'_>,
		) -> std::fmt::Result {
			let items: Vec<_> = self
				.dataset
				.iter()
				.map(|item| item.simple_display().to_string())
				.collect();
			let string = items.join("\n");
			write!(f, "{string}")
		}
	}

	#[expect(dead_code, reason = "Dummy input for mapping")]
	#[derive(Clone, Debug)]
	struct LoadedItem {
		image: RgbImage,
		intensity: Intensity<IntensityValType>,
	}

	struct LoadImage;

	impl Mapper<DummyMapperInput, LoadedItem> for LoadImage {
		fn map(
			&self,
			item: &DummyMapperInput,
		) -> LoadedItem {
			LoadedItem {
				image: item.load().unwrap().into(),
				intensity: item.get_intensity(),
			}
		}
	}

	type LoadedDataset = LaserDataset<LoadImage, DummyMapperInput, LoadedItem>;

	type LoadedDatasetSplit =
		LaserDatasetSplit<LoadImage, DummyMapperInput, LoadedItem>;

	impl Termination for LoadedDataset {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	impl Termination for LoadedDatasetSplit {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[expect(
		dead_code,
		reason = "Used to allow tests to return a type for chaining"
	)]
	struct SuccessfulImage(RgbImage);

	impl From<RgbImage> for SuccessfulImage {
		fn from(value: RgbImage) -> Self {
			Self(value)
		}
	}

	impl Termination for SuccessfulImage {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[test]
	#[ignore = "intended for visual inspection"]
	fn try_new_dataset_dummy_correct() -> Result<DummyDataset> {
		let root = create_temporary_dataset();
		LaserDataset::try_new(DummyMapper, root.root(), &[])
			.inspect(|ds| println!("{ds}"))
	}

	#[test]
	#[ignore = "intended for visual inspection"]
	fn try_new_filtered_dataset_dummy_correct() -> Result<DummyDataset> {
		let root = create_temporary_dataset();
		LaserDataset::try_new(
			DummyMapper,
			root.root(),
			&[Box::new(|item| matches!(*item.get_intensity(), 0..10))],
		)
		.inspect(|ds| println!("{ds}"))
	}

	#[test]
	fn try_new_dataset_correct() -> Result<LoadedDataset> {
		let root = create_temporary_dataset();
		let dataset = LaserDataset::try_new(LoadImage, root.root(), &[])?;
		assert_eq!(dataset.len(), 2);
		assert_eq!(
			dataset
				.get(0)
				.expect("dataset item exists")
				.image
				.dimensions(),
			(1, 1)
		);
		Ok(dataset)
	}

	#[test]
	fn try_new_filtered_dataset_correct() -> Result<LoadedDataset> {
		let root = create_temporary_dataset();
		let dataset = LaserDataset::try_new(
			LoadImage,
			root.root(),
			&[Box::new(|item| item.get_intensity() != Intensity::from(9))],
		)?;
		assert_eq!(dataset.len(), 1);
		assert_eq!(
			dataset
				.get(0)
				.expect("filtered dataset item exists")
				.image
				.dimensions(),
			(1, 1)
		);
		Ok(dataset)
	}

	#[test]
	#[ignore = "intended for visual inspection"]
	fn split_dataset_dummy_correct() {
		let root = create_temporary_dataset();
		let dataset =
			LaserDataset::try_new(DummyMapper, root.root(), &[]).unwrap();
		let weights = crate::splitting::SplitWeights::try_new(
			typed_floats::tf32::PositiveFinite::new(80.0).unwrap(),
			typed_floats::tf32::PositiveFinite::new(15.0).unwrap(),
			typed_floats::tf32::PositiveFinite::new(5.0).unwrap(),
		)
		.unwrap();
		let splits = dataset.into_split(weights);
		let (train, validate, test) = splits.into();
		println!(
			"\n ===== Train present: {}\n ===== Validate present: {}\n===== Test present: {}",
			train.is_some(),
			validate.is_some(),
			test.is_some()
		);
	}
}
