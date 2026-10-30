//! Weighted train, validation, and test dataset partitions.

use burn::data::dataset::Dataset;
// Publically re-export the types used for split weight.
pub use typed_floats::tf32::{
	PositiveFinite,
	StrictlyPositiveFinite,
};

#[expect(
	clippy::cast_precision_loss,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "Weighted allocation uses floating-point ratios and clamps before converting to a dataset length"
)]
fn weighted_split_len(
	dataset_len: usize,
	weight: f32,
	total_weight: f32,
	allocated_len: usize,
	reserve_training_item: bool,
) -> usize {
	let available_len = dataset_len
		.saturating_sub(allocated_len)
		.saturating_sub(usize::from(reserve_training_item));
	let requested_len = (dataset_len as f64 * f64::from(weight)
		/ f64::from(total_weight))
	.ceil();
	if requested_len >= available_len as f64 {
		available_len
	} else {
		requested_len as usize
	}
}

/// A subdivision of the dataset for use in specific scenarios.
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

	/// Splits this dataset into train/validate/test partitions.
	///
	/// The operation consumes the source dataset.
	///
	/// The weights are relative, not percentages. Test items are selected
	/// first, then validation items; the remaining items form the training
	/// partition when the training weight is non-zero.
	///
	/// Requested test and validation lengths are rounded up and capped to the
	/// items still available. When the training weight is non-zero, one item is
	/// reserved for training whenever the source is non-empty. A partition with
	/// a zero weight is `None`; a non-zero weight can still produce an empty
	/// partition when the dataset is too small.
	fn into_split(
		self,
		weights: SplitWeights,
	) -> SplitOutput<Self::Output> {
		let total_weight = weights.sum.get();
		let dataset_len = self.get_dataset().len();

		let test_len = weighted_split_len(
			dataset_len,
			weights.test_weight.get(),
			total_weight,
			0,
			weights.train_weight.get() > 0.0,
		);
		let validate_len = weighted_split_len(
			dataset_len,
			weights.validate_weight.get(),
			total_weight,
			test_len,
			weights.train_weight.get() > 0.0,
		);

		let test_end = test_len;
		let validate_end = test_end + validate_len;

		let test = (weights.test_weight.get() > 0.0)
			.then(|| Self::create_split(&self, 0, test_end, SplitType::Test));
		let validate = (weights.validate_weight.get() > 0.0).then(|| {
			Self::create_split(
				&self,
				test_end,
				validate_end,
				SplitType::Validate,
			)
		});
		let train = (weights.train_weight.get() > 0.0).then(|| {
			Self::create_split(
				&self,
				validate_end,
				dataset_len,
				SplitType::Train,
			)
		});
		SplitOutput::new(train, validate, test)
	}
}

/// Relative weights for optional training, validation, and test partitions.
///
/// A zero weight omits that partition. At least one weight must be non-zero,
/// and the combined weight must remain finite.
///
/// # Examples
///
/// ```
/// use dataset_loader::splitting::SplitWeights;
/// use typed_floats::tf32::PositiveFinite;
///
/// let weights = SplitWeights::try_new(
/// 	PositiveFinite::new(0.0)?,
/// 	PositiveFinite::new(3.0)?,
/// 	PositiveFinite::new(1.0)?,
/// )
/// .expect("the total is positive and finite");
/// # let _ = weights;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SplitWeights {
	train_weight: PositiveFinite,
	validate_weight: PositiveFinite,
	test_weight: PositiveFinite,
	sum: StrictlyPositiveFinite,
}

impl SplitWeights {
	/// Creates split weights when their sum is finite and strictly positive.
	///
	/// Returns `None` when all weights are zero or when their sum overflows to
	/// infinity.
	pub const fn try_new(
		train_weight: PositiveFinite,
		validate_weight: PositiveFinite,
		test_weight: PositiveFinite,
	) -> Option<Self> {
		let sum =
			train_weight.get() + validate_weight.get() + test_weight.get();
		let Ok(sum) = StrictlyPositiveFinite::new(sum) else {
			return None;
		};
		Some(Self {
			train_weight,
			validate_weight,
			test_weight,
			sum,
		})
	}
}

/// The optional outputs of splitting a dataset.
///
/// A partition is `None` when its weight is zero. A requested partition can
/// still be empty for a small dataset.
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
/// let output = SplitOutput::new(Some(Part(SplitType::Train)), None, None);
/// assert!(output.validate.is_none());
/// ```
pub struct SplitOutput<T>
where
	T: Split,
{
	/// Training portion, absent when the training weight is zero.
	pub train: Option<T>,
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
		train: Option<T>,
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

impl<T> From<(Option<T>, Option<T>, Option<T>)> for SplitOutput<T>
where
	T: Split,
{
	fn from(value: (Option<T>, Option<T>, Option<T>)) -> Self {
		Self::new(value.0, value.1, value.2)
	}
}

impl<T> From<SplitOutput<T>> for (Option<T>, Option<T>, Option<T>)
where
	T: Split,
{
	fn from(value: SplitOutput<T>) -> Self {
		(value.train, value.validate, value.test)
	}
}

#[cfg(test)]
mod tests {
	use burn::data::dataset::{
		DatasetError,
		InMemDataset,
	};

	use super::*;

	struct TestDataset(InMemDataset<usize>);

	impl Dataset<usize> for TestDataset {
		fn get(
			&self,
			index: usize,
		) -> Result<usize, DatasetError> {
			self.0.get(index)
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
		train_weight: f32,
		validate_weight: f32,
		test_weight: f32,
	) -> (Option<usize>, Option<usize>, Option<usize>) {
		let weights = SplitWeights::try_new(
			PositiveFinite::new(train_weight)
				.expect("test train weight is valid"),
			PositiveFinite::new(validate_weight)
				.expect("test validation weight is valid"),
			PositiveFinite::new(test_weight)
				.expect("test test weight is valid"),
		)
		.expect("test weights have a valid sum");
		let output = TestDataset(InMemDataset::new((0..dataset_len).collect()))
			.into_split(weights);
		(
			output.train.map(|split| split.len),
			output.validate.map(|split| split.len),
			output.test.map(|split| split.len),
		)
	}

	#[test]
	fn one_item_dataset_reserves_training_item_when_weighted() {
		assert_eq!(
			split_lengths(1, 1.0, 1.0, 1.0),
			(Some(1), Some(0), Some(0))
		);
	}

	#[test]
	fn large_weights_do_not_overflow_and_splits_cover_dataset() {
		let lengths = split_lengths(100, 255.0, 255.0, 255.0);
		assert_eq!(
			lengths.0.unwrap() + lengths.1.unwrap() + lengths.2.unwrap(),
			100
		);
	}

	#[test]
	fn empty_dataset_produces_empty_requested_splits() {
		assert_eq!(
			split_lengths(0, 1.0, 1.0, 1.0),
			(Some(0), Some(0), Some(0))
		);
	}

	#[test]
	fn fractional_weighted_lengths_round_up_before_training_receives_remainder()
	{
		assert_eq!(
			split_lengths(10, 1.0, 1.0, 1.0),
			(Some(2), Some(4), Some(4))
		);
	}

	#[test]
	fn unweighted_validate_and_test_leaves_all_to_train() {
		assert_eq!(split_lengths(12, 1.0, 0.0, 0.0), (Some(12), None, None));
	}

	#[test]
	fn unweighted_train_and_validate_leaves_all_to_test() {
		assert_eq!(split_lengths(11, 0.0, 0.0, 1.0), (None, None, Some(11)));
	}

	#[test]
	fn zero_training_weight_omits_training_split() {
		assert_eq!(split_lengths(10, 0.0, 1.0, 1.0), (None, Some(5), Some(5)));
	}

	#[test]
	fn zero_total_weight_is_rejected() {
		assert!(
			SplitWeights::try_new(
				PositiveFinite::new(0.0).expect("zero is finite and positive"),
				PositiveFinite::new(0.0).expect("zero is finite and positive"),
				PositiveFinite::new(0.0).expect("zero is finite and positive"),
			)
			.is_none()
		);
	}

	#[test]
	fn non_finite_total_weight_is_rejected() {
		assert!(
			SplitWeights::try_new(
				PositiveFinite::new(f32::MAX).expect("maximum is finite"),
				PositiveFinite::new(f32::MAX).expect("maximum is finite"),
				PositiveFinite::new(0.0).expect("zero is allowed"),
			)
			.is_none()
		);
	}
}
