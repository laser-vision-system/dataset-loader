//! The common fields that all items must have.

use std::{
	fmt,
	path::Path,
};

use burn::tensor::Element;
use serde::Deserialize;

pub use super::{
	intensity::Intensity,
	path::ImagePath,
};
use crate::LoaderResult;

/// The idea is that all items have this as a field, guaranteed by a trait
/// method.
///
/// Any additional fields would be defined on the higher-level types.
///
/// This allows trait methods to be able to provide a default implementation
/// which builds this type.
///
/// # Examples
///
/// ```
/// # use dataset_loader::item::BaseItem;
/// #
/// let csv = "relative_path,position,intensity\nimages/sample.png,facing_downward,12\n";
/// let mut reader = csv::Reader::from_reader(csv.as_bytes());
/// let item: BaseItem<u8> = reader
/// 	.deserialize()
/// 	.next()
/// 	.transpose()?
/// 	.expect("CSV contains one item");
/// assert_eq!(item.get_position(), Some("facing_downward"));
/// assert_eq!(*item.get_intensity(), 12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Deserialize)]
pub struct BaseItem<T>
where
	T: Element,
{
	#[serde(rename = "relative_path")]
	image_path: ImagePath,
	#[serde(default)]
	position: Option<String>,
	intensity: Intensity<T>,
}

impl<T> BaseItem<T>
where
	T: Element,
{
	/// Attach the root to the relative item path.
	///
	/// # Errors
	///
	/// If the path does not resolve as canonical.
	pub(super) fn resolve_root(
		&mut self,
		root: &Path,
	) -> LoaderResult {
		self.image_path.canonicalise_with(root)
	}

	/// Returns the item's image path.
	pub fn get_image_path(&self) -> &ImagePath {
		&self.image_path
	}

	/// Returns the item's intensity value.
	pub fn get_intensity(&self) -> Intensity<T> {
		self.intensity
	}

	/// Returns optional position metadata for this item.
	pub fn get_position(&self) -> Option<&str> {
		self.position.as_deref()
	}
}

impl<T> fmt::Display for BaseItem<T>
where
	T: Element,
{
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		write!(f, "{}: '{}'", self.intensity, self.image_path)
	}
}

#[cfg(test)]
pub(crate) mod tests {
	use std::{
		path::PathBuf,
		process::{
			ExitCode,
			Termination,
		},
		sync::atomic,
	};

	use burn::tensor::Element;
	use image::RgbImage;

	use super::*;

	// Production items come from CSV; direct construction is only a test
	// fixture.
	impl<T> BaseItem<T>
	where
		T: Element,
	{
		pub(crate) fn new(
			image_path: ImagePath,
			intensity: Intensity<T>,
		) -> Self {
			Self {
				image_path,
				position: None,
				intensity,
			}
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

	pub(crate) fn create_path(
		base: &str,
		name: &str,
	) -> PathBuf {
		PathBuf::from(base).join(name)
	}

	pub(crate) struct TemporaryImage(PathBuf);

	impl TemporaryImage {
		pub(crate) fn path(&self) -> &PathBuf {
			&self.0
		}
	}

	impl Drop for TemporaryImage {
		fn drop(&mut self) {
			_ = std::fs::remove_file(&self.0);
		}
	}

	pub(crate) fn create_temporary_image(filename: &str) -> TemporaryImage {
		static NEXT_ID: atomic::AtomicU64 = atomic::AtomicU64::new(0);
		let id = NEXT_ID.fetch_add(1, atomic::Ordering::Relaxed);
		let path = std::env::temp_dir().join(format!(
			"dataset-loader-{}-{id}-{filename}",
			std::process::id()
		));
		RgbImage::from_pixel(1, 1, image::Rgb([10, 20, 30]))
			.save(&path)
			.expect("temporary image should be writable");
		TemporaryImage(path)
	}

	fn parse_base_item(input: &str) -> Option<BaseItem<u8>> {
		let mut reader = csv::Reader::from_reader(input.as_bytes());
		reader.deserialize::<BaseItem<u8>>().next()?.ok()
	}

	#[test]
	fn base_item_position_is_optional_in_csv() {
		let positioned = parse_base_item(
			"relative_path,position,intensity\nimages/sample.png,front_outer_90,12\n",
		);
		let without_position =
			parse_base_item("relative_path,intensity\nimages/sample.png,12\n");

		assert_eq!(
			positioned.and_then(|item| item.get_position().map(str::to_owned)),
			Some("front_outer_90".to_owned())
		);
		assert_eq!(
			without_position
				.and_then(|item| item.get_position().map(str::to_owned)),
			None
		);
	}
}
