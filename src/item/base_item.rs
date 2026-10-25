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
	use burn::tensor::Element;

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
}
