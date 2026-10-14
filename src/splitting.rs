use std::num::NonZeroU8;

use burn::data::dataset::Dataset;

/// A subdivision of the dataset for use in specific scenarios.
#[derive(Clone, Copy, Debug)]
pub enum SplitType {
	Train,
	Validate,
	Test,
}

/// A trait that represents a split of a larger dataset.
///
/// This is either the train, validate, or test split.
pub trait Split {
	fn split_type(&self) -> SplitType;
}

/// A trait that allows you to split a dataset into train/validate/test.
pub trait Splitable<MO>
where
	Self: Sized, // To allow this to be consuming.
	MO: Send + Sync + Clone,
{
	type DatasetType: Dataset<MO>;
	type Output: Split;

	/// Helper method to get the internal dataset object.
	fn get_dataset(&self) -> &Self::DatasetType;

	/// Helper function to create a split dataset.
	fn create_split(
		&self,
		start: usize,
		end: usize,
		split: SplitType,
	) -> Self::Output;

	/// Split this dataset into train/validate/test.
	///
	/// This a consuming operation.
	#[expect(
		clippy::cast_precision_loss,
		clippy::cast_possible_truncation,
		reason = "Assume the dataset will never get large enough for this to matter"
	)]
	#[expect(
		clippy::cast_sign_loss,
		reason = "The values are checked to be positive before casting"
	)]
	fn into_split(
		self,
		train_weight: NonZeroU8,
		validate_weight: u8,
		test_weight: u8,
	) -> SplitOutput<Self::Output> {
		let sum =
			u16::from(u8::from(train_weight) + validate_weight + test_weight);
		let max = self.get_dataset().len();
		let ratio = max as f64 / f64::from(sum);

		let mut start = 0;
		let mut end = 0;

		let test = (test_weight > 0).then(|| {
			end += (f64::from(test_weight) * ratio).ceil() as usize;
			let split = Self::create_split(&self, start, end, SplitType::Test);
			start = end;
			split
		});

		let validate = (validate_weight > 0).then(|| {
			end += (f64::from(validate_weight) * ratio).ceil() as usize;
			let split =
				Self::create_split(&self, start, end, SplitType::Validate);
			start = end;
			split
		});

		end = max;

		let train = Self::create_split(&self, start, end, SplitType::Train);

		SplitOutput::new(train, validate, test)
	}
}

/// The output of splitting a dataset.
pub struct SplitOutput<T>
where
	T: Split,
{
	pub train: T,
	pub validate: Option<T>,
	pub test: Option<T>,
}

impl<T> SplitOutput<T>
where
	T: Split,
{
	pub fn new(train: T, validate: Option<T>, test: Option<T>) -> Self {
		Self {
			train,
			validate,
			test,
		}
	}
}

impl<T> From<(T, Option<T>, Option<T>)> for SplitOutput<T>
where
	T: Split,
{
	fn from(value: (T, Option<T>, Option<T>)) -> Self {
		Self::new(value.0, value.1, value.2)
	}
}

impl<T> From<SplitOutput<T>> for (T, Option<T>, Option<T>)
where
	T: Split,
{
	fn from(value: SplitOutput<T>) -> Self {
		(value.train, value.validate, value.test)
	}
}
