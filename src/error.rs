//! Shared error types used by dataset and item operations.

/// Specialised [`Result`] type for the data loader.
///
/// Shorthand for a result with the [`LoaderError`] error type.
///
/// # Examples
///
/// ```
/// use dataset_loader::{
/// 	LoaderError,
/// 	Result,
/// };
///
/// fn parse_intensity(value: &str) -> Result<u8> {
/// 	value.parse().map_err(|error| {
/// 		LoaderError::Intensity(format!("invalid intensity: {error}"))
/// 	})
/// }
///
/// assert_eq!(parse_intensity("12")?, 12);
/// # Ok::<(), LoaderError>(())
/// ```
///
/// [`Result`]: std::result::Result
pub type Result<T> = std::result::Result<T, LoaderError>;

/// The error type for all data loading operations.
#[derive(Clone, Debug, thiserror::Error)]
pub enum LoaderError {
	/// A dataset or image path is invalid or cannot be resolved.
	#[error("There was an issue with the path: {0}")]
	Path(String),
	/// An intensity is malformed or outside its accepted domain.
	#[error("There was an issue with the intensity: {0}")]
	Intensity(String),
	/// An image could not be decoded from its path.
	#[error("The image failed to be loaded: {0}")]
	ImageLoad(String),
	/// A dataset could not be constructed.
	#[error("An error occurred while building the dataset: {0}")]
	Dataset(String),
	/// The CSV index could not be read or parsed.
	#[error("The csv parser failed: {0}")]
	CSV(String),
	/// A CSV field could not be converted into its Rust type.
	#[error("The csv entry failed to convert: {0}")]
	Conversion(String),
	/// An operation failed for a reason not covered by a more specific variant.
	#[error("An error occurred: {0}")]
	General(String),
}

impl From<csv::Error> for LoaderError {
	fn from(err: csv::Error) -> Self {
		Self::CSV(err.to_string())
	}
}
