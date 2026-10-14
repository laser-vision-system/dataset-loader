use thiserror::Error;

/// Specialised [`Result`] type for the data loader.
///
/// Shorthand for a result with the [`LoaderError`] error type.
///
/// [`Result`]: std::result::Result
pub type Result<T> = std::result::Result<T, LoaderError>;

/// The error type for all data loading operations.
#[derive(Clone, Debug, Error)]
pub enum LoaderError {
	#[error("There was an issue with the path: {0}")]
	Path(String),
	#[error("There was an issue with the intensity: {0}")]
	Intensity(String),
	#[error("The image failed to be loaded: {0}")]
	ImageLoad(String),
	#[error("An error occured while building the dataset: {0}")]
	Dataset(String),
	#[error("The csv parser failed: {0}")]
	CSV(String),
	#[error("The csv entry failed to convert: {0}")]
	Conversion(String),
	#[error("An error occured: {0}")]
	General(String),
}

impl From<csv::Error> for LoaderError {
	fn from(err: csv::Error) -> Self {
		Self::CSV(err.to_string())
	}
}
