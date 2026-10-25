//! Shared traits and value types for dataset records.

use std::{
	fmt,
	path::Path,
};

use burn::tensor::Element;
use image::DynamicImage;
use serde::Deserialize;

pub mod intensity;
pub mod path;

pub use self::{
	intensity::Intensity,
	path::ImagePath,
};
use crate::{
	LoaderError,
	Result,
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
	) -> Result<()> {
		self.get_base_item_mut().resolve_root(root)
	}
}

/// Common read and image-loading operations for dataset items.
pub trait LaserItem: LaserItemType {
	/// Returns the path to the image.
	///
	/// You likely don't need to use this method.
	fn get_image_path(&self) -> &ImagePath {
		&self.get_base_item().image_path
	}

	/// Returns the intensity value of this item.
	///
	/// Useful when a mapper needs the item's label.
	fn get_intensity(&self) -> Intensity<Self::IntensityType> {
		self.get_base_item().intensity
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
	fn load(&self) -> Result<DynamicImage> {
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
	fn resolve_root(
		&mut self,
		root: &Path,
	) -> Result<()> {
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

	use image::RgbImage;

	use super::{
		BaseItem,
		Element,
		ImagePath,
		Intensity,
	};

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
