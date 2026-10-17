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
		let sum = u16::from(u8::from(train_weight))
			+ u16::from(validate_weight)
			+ u16::from(test_weight);
		let max = self.get_dataset().len();
		let ratio = max as f64 / f64::from(sum);

		let mut start = 0;
		let mut end = 0;

		let test = (test_weight > 0).then(|| {
			end = ((f64::from(test_weight) * ratio).ceil() as usize)
				.min(max.saturating_sub(1));
			let split = Self::create_split(&self, start, end, SplitType::Test);
			start = end;
			split
		});

		let validate = (validate_weight > 0).then(|| {
			end += ((f64::from(validate_weight) * ratio).ceil() as usize)
				.min(max.saturating_sub(end).saturating_sub(1));
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
	pub fn new(
		train: T,
		validate: Option<T>,
		test: Option<T>,
	) -> Self {
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

#[cfg(test)]
mod tests {
	use super::*;

	struct TestDataset(Vec<usize>);

	impl Dataset<usize> for TestDataset {
		fn get(
			&self,
			index: usize,
		) -> Option<usize> {
			self.0.get(index).copied()
		}

		fn len(&self) -> usize {
			self.0.len()
		}
	}

	struct TestSplit {
		len: usize,
		split: SplitType,
	}

	impl Split for TestSplit {
		fn split_type(&self) -> SplitType {
			self.split
		}
	}

	impl Splitable<usize> for TestDataset {
		type DatasetType = Self;
		type Output = TestSplit;

		fn get_dataset(&self) -> &Self::DatasetType {
			self
		}

		fn create_split(
			&self,
			start: usize,
			end: usize,
			split: SplitType,
		) -> Self::Output {
			TestSplit {
				len: end - start,
				split,
			}
		}
	}

	fn split_lengths(
		dataset_len: usize,
		train_weight: u8,
		validate_weight: u8,
		test_weight: u8,
	) -> (usize, Option<usize>, Option<usize>) {
		let output = TestDataset((0..dataset_len).collect()).into_split(
			NonZeroU8::new(train_weight).expect("train weight must be nonzero"),
			validate_weight,
			test_weight,
		);
		(
			output.train.len,
			output.validate.map(|split| split.len),
			output.test.map(|split| split.len),
		)
	}

	#[test]
	fn one_item_dataset_keeps_training_item_without_panicking() {
		assert_eq!(split_lengths(1, 1, 1, 1), (1, Some(0), Some(0)));
	}

	#[test]
	fn large_weights_do_not_overflow_and_splits_cover_dataset() {
		let lengths = split_lengths(100, u8::MAX, u8::MAX, u8::MAX);
		assert_eq!(lengths.0 + lengths.1.unwrap() + lengths.2.unwrap(), 100);
		assert!(lengths.0 > 0);
	}

	#[test]
	fn empty_dataset_produces_empty_requested_splits() {
		assert_eq!(split_lengths(0, 1, 1, 1), (0, Some(0), Some(0)));
	}
}
