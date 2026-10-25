//! An entry into the dataset.

mod base_item;
mod intensity;
mod path;
mod traits;

pub use self::{
	base_item::BaseItem,
	intensity::Intensity,
	path::ImagePath,
	traits::*,
};

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

	use super::base_item::BaseItem;

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
