//! Common methods used across all dataset items.

use std::{
	fmt,
	path::Path,
};

use burn::tensor::Element;
use image::DynamicImage;

use super::{
	base_item::BaseItem,
	intensity::Intensity,
	path::ImagePath,
};
use crate::{
	LoaderError,
	LoaderResult,
};

/// A dataset item that exposes its common path, position, and intensity.
pub trait LaserItemType {
	/// Numeric type used to represent item intensities.
	type IntensityType: Element;

	/// Returns the shared path, position, and intensity fields.
	///
	/// You likely want [`get_intensity`], [`get_position`], or even
	/// [`get_image_path`].
	///
	/// [`get_intensity`]: LaserItem::get_intensity
	/// [`get_position`]: LaserItem::get_position
	/// [`get_image_path`]: LaserItem::get_image_path
	fn get_base_item(&self) -> &BaseItem<Self::IntensityType>;

	/// Returns source-specific field names and their display values.
	///
	/// The default implementation returns an empty list.
	fn source_fields(&self) -> Vec<(&'static str, String)> {
		Vec::new()
	}
}

/// An item that supports the mutable operations required while loading.
///
/// If you are accessing a method on this trait, you are probably doing
/// something wrong.
pub trait LaserItemBase: LaserItemType {
	/// Returns the shared fields for mutation during loading.
	///
	/// You very likely do not need to call this method.
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType>;

	/// Convert the relative path to an absolute path with the use of a root.
	///
	/// You very likely do not need to call this method.
	///
	/// # Errors
	///
	/// If the path does not resolve as canonical.
	fn resolve_paths(
		&mut self,
		root: &Path,
	) -> LoaderResult {
		self.get_base_item_mut().resolve_root(root)
	}
}

/// Common read and image-loading operations for dataset items.
pub trait LaserItem: LaserItemType {
	/// Returns the path to the image.
	///
	/// You likely don't need to use this method.
	fn get_image_path(&self) -> &ImagePath {
		self.get_base_item().get_image_path()
	}

	/// Returns the intensity value of this item.
	///
	/// Useful when a mapper needs the item's label.
	fn get_intensity(&self) -> Intensity<Self::IntensityType> {
		self.get_base_item().get_intensity()
	}

	/// Get optional position metadata for this item.
	///
	/// Sources without position data return `None`.
	fn get_position(&self) -> Option<&str> {
		self.get_base_item().get_position()
	}

	/// Load the image from the path.
	///
	/// # Errors
	///
	/// [`LoaderError::ImageLoad`]:
	///
	/// - If the image could not be loaded at the provided path
	fn load(&self) -> LoaderResult<DynamicImage> {
		let image = image::open(self.get_image_path()).map_err(|why| {
			LoaderError::ImageLoad(format!(
				"unable to load image at '{}': {}",
				self.get_image_path(),
				why
			))
		})?;

		Ok(image)
	}

	/// Returns a display adapter containing the intensity and image path.
	fn simple_display(&self) -> DatasetItemFormatter<'_, Self>
	where
		Self: Sized,
	{
		DatasetItemFormatter(self)
	}
}

impl<T> LaserItem for T where T: LaserItemBase
{
}

/// A display adapter that formats an item as intensity and image path.
///
/// # Examples
///
/// ```
/// use dataset_loader::item::LaserItem;
///
/// fn simple_display<T: LaserItem>(item: &T) -> String {
/// 	item.simple_display().to_string()
/// }
/// ```
pub struct DatasetItemFormatter<'a, T>(&'a T)
where
	T: LaserItem;

impl<T> fmt::Display for DatasetItemFormatter<'_, T>
where
	T: LaserItem,
{
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		write!(
			f,
			"{}: '{}'",
			self.0.get_intensity(),
			self.0.get_image_path()
		)
	}
}
