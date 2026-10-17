//! Weighted train, validation, and test dataset partitions.

use std::num::NonZeroU8;

use burn::data::dataset::Dataset;

/// Identifies the intended use of a dataset subdivision.
#[derive(Clone, Copy, Debug)]
pub enum SplitType {
	/// Model fitting data.
	Train,
	/// Hyperparameter selection data.
	Validate,
	/// Final evaluation data.
	Test,
}

/// A trait that represents a split of a larger dataset.
///
/// This is either the train, validate, or test split.
pub trait Split {
	/// Returns whether this dataset represents training, validation, or test
	/// data.
	fn split_type(&self) -> SplitType;
}

/// A trait that allows you to split a dataset into train/validate/test.
pub trait Splitable<MO>
where
	Self: Sized, // To allow this to be consuming.
	MO: Send + Sync + Clone,
{
	/// Dataset adapter used as the source for this split.
	type DatasetType: Dataset<MO>;
	/// Dataset representation returned for each partition.
	type Output: Split;

	/// Returns the dataset being divided.
	fn get_dataset(&self) -> &Self::DatasetType;

	/// Creates a split from the half-open item range `start..end`.
	fn create_split(
		&self,
		start: usize,
		end: usize,
		split: SplitType,
	) -> Self::Output;

	/// Splits the dataset into training, optional validation, and optional test
	/// portions using relative weights.
	///
	/// The operation consumes the source dataset. Test items are assigned
	/// first, followed by validation items; all remaining items are
	/// training data. Each requested non-training size is rounded up, but
	/// is capped so at least one item remains for training when the dataset
	/// is non-empty. Requested validation or test splits may therefore be
	/// empty for small datasets.
	///
	/// # Examples
	///
	/// ```
	/// use dataset_loader::splitting::{Split, SplitType, Splitable};
	/// use std::num::NonZeroU8;
	/// # use burn::data::dataset::Dataset;
	/// # struct Numbers(Vec<u8>);
	/// # struct NumberSplit(usize, SplitType);
	/// # impl Dataset<u8> for Numbers {
	/// #     fn get(&self, index: usize) -> Option<u8> { self.0.get(index).copied() }
	/// #     fn len(&self) -> usize { self.0.len() }
	/// # }
	/// # impl Split for NumberSplit { fn split_type(&self) -> SplitType { self.1 } }
	/// # impl Splitable<u8> for Numbers {
	/// #     type DatasetType = Self;
	/// #     type Output = NumberSplit;
	/// #     fn get_dataset(&self) -> &Self { self }
	/// #     fn create_split(&self, start: usize, end: usize, kind: SplitType) -> NumberSplit {
	/// #         NumberSplit(end - start, kind)
	/// #     }
	/// # }
	/// let splits = Numbers((0..10).collect()).into_split(NonZeroU8::new(8).unwrap(), 1, 1);
	/// assert_eq!(splits.train.0, 8);
	/// assert_eq!(splits.validate.unwrap().0, 1);
	/// assert_eq!(splits.test.unwrap().0, 1);
	/// ```
	///
	/// # Panics
	///
	/// This method does not panic when splitting an empty or small dataset.
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

/// The three outputs of splitting a dataset.
///
/// Validation and test splits are `None` when their weights were zero.
///
/// # Examples
///
/// ```
/// use dataset_loader::splitting::{
/// 	SplitOutput,
/// 	SplitType,
/// };
/// # struct Part(SplitType);
/// # impl dataset_loader::splitting::Split for Part {
/// #     fn split_type(&self) -> SplitType { self.0 }
/// # }
/// let output = SplitOutput::new(Part(SplitType::Train), None, None);
/// assert!(output.validate.is_none());
/// ```
pub struct SplitOutput<T>
where
	T: Split,
{
	/// Training portion; always present, but may be empty for an empty source.
	pub train: T,
	/// Validation portion, omitted when its weight is zero.
	pub validate: Option<T>,
	/// Test portion, omitted when its weight is zero.
	pub test: Option<T>,
}

impl<T> SplitOutput<T>
where
	T: Split,
{
	/// Constructs a split output from its three partitions.
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
