#![expect(
	clippy::expect_used,
	reason = "This type handles the path validation internally"
)]

use std::{
	borrow::{
		Borrow,
		Cow,
	},
	fmt,
	path::{
		Path,
		PathBuf,
	},
};

use crate::error::{
	LoaderError,
	Result,
};

/// A deconstructed type for an item's path.
///
/// Used for path validation logic.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(try_from = "PathBuf")]
pub struct ImagePath {
	inner: PathBuf,
}

impl ImagePath {
	/// Attach a root path to the relative path and canonicalise.
	///
	/// # Errors
	///
	/// If the absolute path is non-canonical.
	pub fn canonicalise_with(
		&mut self,
		root: &Path,
	) -> Result<()> {
		let absolute = root.join(&self.inner);
		self.inner = absolute.canonicalize().map_err(|why| {
			LoaderError::Path(format!(
				"could not canonicalise '{}': {}",
				absolute.display(),
				why
			))
		})?;
		Ok(())
	}

	/// Get the file extension for this item path.
	///
	/// # Panics
	///
	/// If the dataset item has an invalid path.
	pub fn get_extension(&self) -> Cow<'_, str> {
		self.inner
			.extension()
			.expect("extension should be guaranteed by this type's creation")
			.to_string_lossy()
	}

	/// Get the file name for this item path.
	///
	/// # Panics
	///
	/// If the dataset item has an invalid path.
	pub fn get_filename(&self) -> Cow<'_, str> {
		self.inner
			.file_stem()
			.expect("filename should be guaranteed by this type's creation")
			.to_string_lossy()
	}

	/// Get all parent directories for this item path.
	///
	/// The list is in ascending order
	/// (parent -> grandparent -> great-grandparent -> ...).
	pub fn get_parents(&self) -> Vec<Cow<'_, str>> {
		self.inner
			.components()
			.rev()
			.skip(1)
			.map(|comp| comp.as_os_str().to_string_lossy())
			.collect()
	}
}

impl TryFrom<PathBuf> for ImagePath {
	type Error = LoaderError;

	fn try_from(path: PathBuf) -> Result<Self> {
		// TODO: Basic non-canonical validation.
		path.extension().ok_or(LoaderError::Path(format!(
			"path is not a file or has no extension: {}",
			path.display()
		)))?;

		Ok(Self {
			inner: path,
		})
	}
}

impl fmt::Display for ImagePath {
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		write!(f, "{}", self.inner.display())
	}
}

impl From<ImagePath> for PathBuf {
	fn from(value: ImagePath) -> Self {
		value.inner
	}
}

impl<T> AsRef<T> for ImagePath
where
	T: ?Sized,
	PathBuf: AsRef<T>,
{
	fn as_ref(&self) -> &T {
		self.inner.as_ref()
	}
}

// PathBuf provides `Borrow<Path>`
impl Borrow<PathBuf> for ImagePath {
	fn borrow(&self) -> &PathBuf {
		&self.inner
	}
}

#[cfg(test)]
mod tests {
	use std::process::{
		ExitCode,
		Termination,
	};

	use super::*;

	impl Termination for ImagePath {
		fn report(self) -> std::process::ExitCode {
			ExitCode::SUCCESS
		}
	}

	#[test]
	fn create_image_path_correct() -> Result<ImagePath> {
		let path = PathBuf::from("./src/item/path.rs");
		ImagePath::try_from(path)
	}

	#[test]
	fn create_image_path_incorrect_filename() {
		let path = PathBuf::from("./src/item/");
		let res = ImagePath::try_from(path);
		res.unwrap_err();
	}
}
