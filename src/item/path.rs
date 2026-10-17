//! Validated image-path representation used by dataset items.

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

/// A path to a dataset image.
///
/// Construction verifies that the path has an extension. It does not verify
/// that the file exists or that its contents are an image; canonicalisation
/// during dataset loading performs the existence check.
///
/// # Examples
///
/// ```
/// use std::path::PathBuf;
///
/// use dataset_loader::item::ImagePath;
///
/// let path = ImagePath::try_from(PathBuf::from("images/sample.png"))?;
/// assert_eq!(path.get_extension(), "png");
/// assert_eq!(path.get_filename(), "sample");
/// # Ok::<(), dataset_loader::LoaderError>(())
/// ```
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(try_from = "PathBuf")]
pub struct ImagePath {
	inner: PathBuf,
}

impl ImagePath {
	/// Joins a dataset root to this path and stores the canonical absolute
	/// path.
	///
	/// # Errors
	///
	/// Returns [`LoaderError::Path`] if the joined path does not exist or
	/// cannot be canonicalised.
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

	/// Returns the path extension as a lossy UTF-8 string.
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

	/// Returns the final filename without its extension.
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

	/// Returns parent path components from nearest to farthest.
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

	/// Creates a path after checking that it has a file extension.
	///
	/// This check does not require the path to exist or point to a regular
	/// file.
	///
	/// # Errors
	///
	/// Returns [`LoaderError::Path`] when the path has no extension.
	fn try_from(path: PathBuf) -> Result<Self> {
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
