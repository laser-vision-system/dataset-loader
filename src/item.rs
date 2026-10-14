use std::{
	fmt,
	path::{Path, PathBuf},
};

use serde::Deserialize;

pub mod intensity;
pub mod path;

pub use self::{
	intensity::{Intensity, NumericLike},
	path::ImagePath,
};
use crate::{LoaderError, Result};

pub type ImageType = image::RgbImage;

/// A trait for a laser dataset item that defines the type the intensity is
/// stored as.
pub trait LaserItemType {
	type IntensityType: NumericLike;

	/// Get the base item type.
	///
	/// This value holds the relative filepath and the intensity value.
	fn get_base_item(&self) -> &BaseItem<Self::IntensityType>;

	/// Return source-specific field names and display values.
	fn source_fields(&self) -> Vec<(&'static str, String)> {
		Vec::new()
	}
}

/// A trait for a dataset item that is necessary for loading.
///
/// If you are accessing a method on this trait, you are probably doing
/// something wrong.
pub trait LaserItemBase: LaserItemType {
	/// Get the base item type as mutable.
	///
	/// You very likely do not need to call this method.
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType>;

	/// Convert the relative path to an absolute path with the use of a root.
	///
	/// You very likely do not need to call this method.
	fn resolve_paths(&mut self, root: &Path) -> Result<()> {
		self.get_base_item_mut().resolve_root(root)
	}
}

/// Auto trait on [`LaserItemBase`] that provides the common functions for a
/// laser dataset item.
pub trait LaserItem: LaserItemType {
	/// Get the path to the image.
	///
	/// You likely don't need to use this method.
	fn get_image_path(&self) -> &ImagePath {
		&self.get_base_item().image_path
	}

	/// Get the intensity value of this item.
	///
	/// Useful for mapping operations.
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
	fn load(&self) -> Result<ImageType> {
		let image = image::open(self.get_image_path())
			.map_err(|why| {
				LoaderError::ImageLoad(format!(
					"unable to load image at '{}': {}",
					self.get_image_path(),
					why
				))
			})?
			.into();

		Ok(image)
	}

	/// Return a struct that implements display using the path and intensity.
	fn simple_display(&self) -> DatasetItemFormatter<'_, Self>
	where
		Self: Sized,
	{
		DatasetItemFormatter(self)
	}
}

impl<T> LaserItem for T where T: LaserItemBase {}

/// Auto trait on [`LaserItemBase`] implementers that also implement
/// [`TryFrom<ImagePath>`].
pub trait LaserItemTryFrom:
	LaserItemBase + TryFrom<ImagePath, Error = LoaderError>
{
	/// Trivial conversion from [`PathBuf`] via [`ImagePath`].
	fn try_from_pathbuf(root: PathBuf) -> Result<Self> {
		let image_path = ImagePath::try_from(root)?;
		Self::try_from(image_path)
	}
}

impl<T> LaserItemTryFrom for T where
	T: LaserItemBase + TryFrom<ImagePath, Error = LoaderError>
{
}

pub struct DatasetItemFormatter<'a, T>(&'a T)
where
	T: LaserItem;

impl<T> fmt::Display for DatasetItemFormatter<'_, T>
where
	T: LaserItem,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
#[derive(Clone, Debug, Deserialize)]
pub struct BaseItem<T>
where
	T: NumericLike,
{
	#[serde(rename = "relative_path")]
	image_path: ImagePath,
	#[serde(default)]
	position: Option<String>,
	intensity: Intensity<T>,
}

impl<T> BaseItem<T>
where
	T: NumericLike,
{
	fn resolve_root(&mut self, root: &Path) -> Result<()> {
		self.image_path.canonicalise_with(root)
	}

	pub fn get_image_path(&self) -> &ImagePath {
		&self.image_path
	}

	pub fn get_intensity(&self) -> Intensity<T> {
		self.intensity
	}

	/// Get optional position metadata for this item.
	pub fn get_position(&self) -> Option<&str> {
		self.position.as_deref()
	}
}

impl<T> fmt::Display for BaseItem<T>
where
	T: NumericLike,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}: '{}'", self.intensity, self.image_path)
	}
}

#[cfg(test)]
pub(crate) mod tests {
	use std::{
		path::PathBuf,
		process::{ExitCode, Termination},
		str::FromStr,
	};

	use super::{BaseItem, ImagePath, ImageType, Intensity, NumericLike};
	use crate::{LoaderError, Result};

	impl<T> BaseItem<T>
	where
		T: NumericLike,
	{
		pub fn new(image_path: ImagePath, intensity: Intensity<T>) -> Self {
			Self {
				image_path,
				position: None,
				intensity,
			}
		}
	}

	#[allow(dead_code)]
	struct SuccessfulImage(ImageType);

	impl From<ImageType> for SuccessfulImage {
		fn from(value: ImageType) -> Self {
			Self(value)
		}
	}

	impl Termination for SuccessfulImage {
		fn report(self) -> ExitCode {
			ExitCode::SUCCESS
		}
	}

	pub(crate) fn create_path(base: &str, name: &str) -> PathBuf {
		let mut path = PathBuf::from(base);
		path.push(name);
		path
	}

	/// Get the last two parts from the name, separated by `_`.
	pub(crate) fn extract_parts_from_name(name: &str) -> Vec<&str> {
		name.rsplit('_')
			.enumerate()
			.map_while(|(i, part)| if i < 2 { Some(part) } else { None })
			.collect()
	}

	/// Parse a string into an intensity.
	pub(crate) fn parse_intensity_as<T>(input: &str) -> Result<Intensity<T>>
	where
		T: NumericLike,
		<T as FromStr>::Err: std::fmt::Display,
	{
		let intensity: Intensity<T> = input.parse().map_err(|why| {
			LoaderError::Intensity(format!(
				"could not parse intensity from filename: {why}"
			))
		})?;
		Ok(intensity)
	}

	#[test]
	fn split_parts_one() {
		let parts = extract_parts_from_name("render.png");
		assert_eq!(parts.len(), 1)
	}

	#[test]
	fn split_parts_two() {
		let parts = extract_parts_from_name("render_0012.png");
		assert_eq!(parts.len(), 2)
	}

	#[test]
	fn split_parts_three_capped() {
		let parts = extract_parts_from_name("render_0012_0012.png");
		assert_eq!(parts.len(), 2)
	}

	#[test]
	fn parse_intensity_correct() {
		let parsed = parse_intensity_as("0012").unwrap();
		let intensity = Intensity::from(12u8);
		assert_eq!(parsed, intensity);
	}

	#[test]
	fn parse_intensity_incorrect_type() {
		let parsed = parse_intensity_as::<u8>("2.1");
		assert!(parsed.is_err());
	}

	#[test]
	fn parse_intensity_too_large_for_type() {
		let parsed = parse_intensity_as::<u8>("256");
		assert!(parsed.is_err());
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
