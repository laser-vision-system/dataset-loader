//! Types and loader for a standard item dataset.
//!
//! This item type should be broadly applicable to any dataset.

use std::{
	fmt,
	path::Path,
};

use burn::data::dataset::transform::Mapper;

use crate::{
	LoaderResult,
	dataset::{
		FilterFn,
		LaserDataset,
	},
	item::{
		BaseItem,
		Intensity,
		LaserItem as _,
		LaserItemBase,
		LaserItemType,
		controls::GeneralCameraControl,
	},
};

serde_with::with_prefix!(brightness_prefix "Brightness");
serde_with::with_prefix!(contrast_prefix "Contrast");
serde_with::with_prefix!(hue_prefix "Hue");
serde_with::with_prefix!(saturation_prefix "Saturation");
serde_with::with_prefix!(sharpness_prefix "Sharpness");
serde_with::with_prefix!(gamma_prefix "Gamma");
serde_with::with_prefix!(white_balance_prefix "WhiteBalance");
serde_with::with_prefix!(backlight_comp_prefix "BacklightComp");
serde_with::with_prefix!(gain_prefix "Gain");
serde_with::with_prefix!(pan_prefix "Pan");
serde_with::with_prefix!(tilt_prefix "Tilt");
serde_with::with_prefix!(zoom_prefix "Zoom");
serde_with::with_prefix!(exposure_prefix "Exposure");
serde_with::with_prefix!(iris_prefix "Iris");
serde_with::with_prefix!(focus_prefix "Focus");
serde_with::with_prefix!(auto_exposure_prefix "AutoExposure");
serde_with::with_prefix!(exposure_absolute_prefix "ExposureAbsolute");
serde_with::with_prefix!(auto_white_balance_prefix "AutoWhiteBalance");
serde_with::with_prefix!(power_line_frequency_prefix "PowerLineFrequency");
serde_with::with_prefix!(exposure_auto_priority_prefix "ExposureAutoPriority");

/// Floating-point representation used for laser intensity in this source.
pub type IntensityValType = f32;

/// A mapped dataset whose input items are [`StandardItem`]s.
pub type StandardLaserDataset<M, MO> = LaserDataset<M, StandardItem, MO>;

/// A standard form dataset item.
///
/// Camera controls are deserialised from columns named after each control,
/// with optional metadata columns using the corresponding `-default`,
/// `-min`, `-max`, `-step`, `-possible`, and `-items` suffixes.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct StandardItem {
	#[serde(flatten)]
	base: BaseItem<IntensityValType>,
	#[serde(rename = "externalLaserIntensity")]
	external_laser_intensity: Option<Intensity<IntensityValType>>,

	#[serde(flatten, with = "brightness_prefix")]
	brightness: Option<GeneralCameraControl>,
	#[serde(flatten, with = "contrast_prefix")]
	contrast: Option<GeneralCameraControl>,
	#[serde(flatten, with = "hue_prefix")]
	hue: Option<GeneralCameraControl>,
	#[serde(flatten, with = "saturation_prefix")]
	saturation: Option<GeneralCameraControl>,
	#[serde(flatten, with = "sharpness_prefix")]
	sharpness: Option<GeneralCameraControl>,
	#[serde(flatten, with = "gamma_prefix")]
	gamma: Option<GeneralCameraControl>,
	#[serde(flatten, with = "white_balance_prefix")]
	white_balance: Option<GeneralCameraControl>,
	#[serde(flatten, with = "backlight_comp_prefix")]
	backlight_comp: Option<GeneralCameraControl>,
	#[serde(flatten, with = "gain_prefix")]
	gain: Option<GeneralCameraControl>,
	#[serde(flatten, with = "pan_prefix")]
	pan: Option<GeneralCameraControl>,
	#[serde(flatten, with = "tilt_prefix")]
	tilt: Option<GeneralCameraControl>,
	#[serde(flatten, with = "zoom_prefix")]
	zoom: Option<GeneralCameraControl>,
	#[serde(flatten, with = "exposure_prefix")]
	exposure: Option<GeneralCameraControl>,
	#[serde(flatten, with = "iris_prefix")]
	iris: Option<GeneralCameraControl>,
	#[serde(flatten, with = "focus_prefix")]
	focus: Option<GeneralCameraControl>,
	#[serde(flatten, with = "auto_exposure_prefix")]
	auto_exposure: Option<GeneralCameraControl>,
	#[serde(flatten, with = "exposure_absolute_prefix")]
	exposure_absolute: Option<GeneralCameraControl>,
	#[serde(flatten, with = "auto_white_balance_prefix")]
	auto_white_balance: Option<GeneralCameraControl<bool, bool>>,
	#[serde(flatten, with = "power_line_frequency_prefix")]
	power_line_frequency: Option<GeneralCameraControl>,
	#[serde(flatten, with = "exposure_auto_priority_prefix")]
	exposure_auto_priority: Option<GeneralCameraControl>,
}

impl StandardItem {
	/// Returns the `external_laser_intensity` value for this item.
	pub fn external_laser_intensity(
		&self
	) -> Option<Intensity<IntensityValType>> {
		self.external_laser_intensity
	}

	/// Returns the `brightness` value for this item.
	pub fn brightness(&self) -> Option<GeneralCameraControl> {
		self.brightness.clone()
	}

	/// Returns the `contrast` value for this item.
	pub fn contrast(&self) -> Option<GeneralCameraControl> {
		self.contrast.clone()
	}

	/// Returns the `hue` value for this item.
	pub fn hue(&self) -> Option<GeneralCameraControl> {
		self.hue.clone()
	}

	/// Returns the `saturation` value for this item.
	pub fn saturation(&self) -> Option<GeneralCameraControl> {
		self.saturation.clone()
	}

	/// Returns the `sharpness` value for this item.
	pub fn sharpness(&self) -> Option<GeneralCameraControl> {
		self.sharpness.clone()
	}

	/// Returns the `gamma` value for this item.
	pub fn gamma(&self) -> Option<GeneralCameraControl> {
		self.gamma.clone()
	}

	/// Returns the `white_balance` value for this item.
	pub fn white_balance(&self) -> Option<GeneralCameraControl> {
		self.white_balance.clone()
	}

	/// Returns the `backlight_comp` value for this item.
	pub fn backlight_comp(&self) -> Option<GeneralCameraControl> {
		self.backlight_comp.clone()
	}

	/// Returns the `gain` value for this item.
	pub fn gain(&self) -> Option<GeneralCameraControl> {
		self.gain.clone()
	}

	/// Returns the `pan` value for this item.
	pub fn pan(&self) -> Option<GeneralCameraControl> {
		self.pan.clone()
	}

	/// Returns the `tilt` value for this item.
	pub fn tilt(&self) -> Option<GeneralCameraControl> {
		self.tilt.clone()
	}

	/// Returns the `zoom` value for this item.
	pub fn zoom(&self) -> Option<GeneralCameraControl> {
		self.zoom.clone()
	}

	/// Returns the `exposure` value for this item.
	///
	/// You likely are intending to use
	/// [`exposure_absolute`](Self::exposure_absolute).
	pub fn exposure(&self) -> Option<GeneralCameraControl> {
		self.exposure.clone()
	}

	/// Returns the `iris` value for this item.
	pub fn iris(&self) -> Option<GeneralCameraControl> {
		self.iris.clone()
	}

	/// Returns the `focus` value for this item.
	pub fn focus(&self) -> Option<GeneralCameraControl> {
		self.focus.clone()
	}

	/// Returns the `auto_exposure` value for this item.
	pub fn auto_exposure(&self) -> Option<GeneralCameraControl> {
		self.auto_exposure.clone()
	}

	/// Returns the `exposure_absolute` value for this item.
	pub fn exposure_absolute(&self) -> Option<GeneralCameraControl> {
		self.exposure_absolute.clone()
	}

	/// Returns the `auto_white_balance` value for this item.
	pub fn auto_white_balance(
		&self
	) -> Option<GeneralCameraControl<bool, bool>> {
		self.auto_white_balance.clone()
	}

	/// Returns the `power_line_frequency` value for this item.
	pub fn power_line_frequency(&self) -> Option<GeneralCameraControl> {
		self.power_line_frequency.clone()
	}

	/// Returns the `exposure_auto_priority` value for this item.
	pub fn exposure_auto_priority(&self) -> Option<GeneralCameraControl> {
		self.exposure_auto_priority.clone()
	}
}

impl LaserItemType for StandardItem {
	type IntensityType = IntensityValType;

	fn get_base_item(&self) -> &BaseItem<Self::IntensityType> {
		&self.base
	}

	fn source_fields(&self) -> Vec<(String, String)> {
		[
			self.external_laser_intensity().map(|val| {
				vec![("externalLaserIntensity".to_owned(), val.to_string())]
			}),
			self.brightness()
				.map(|val| val.display_with_prefix("Brightness")),
			self.contrast()
				.map(|val| val.display_with_prefix("Contrast")),
			self.hue().map(|val| val.display_with_prefix("Hue")),
			self.saturation()
				.map(|val| val.display_with_prefix("Saturation")),
			self.sharpness()
				.map(|val| val.display_with_prefix("Sharpness")),
			self.gamma().map(|val| val.display_with_prefix("Gamma")),
			self.white_balance()
				.map(|val| val.display_with_prefix("WhiteBalance")),
			self.backlight_comp()
				.map(|val| val.display_with_prefix("BacklightComp")),
			self.gain().map(|val| val.display_with_prefix("Gain")),
			self.pan().map(|val| val.display_with_prefix("Pan")),
			self.tilt().map(|val| val.display_with_prefix("Tilt")),
			self.zoom().map(|val| val.display_with_prefix("Zoom")),
			self.exposure()
				.map(|val| val.display_with_prefix("Exposure")),
			self.iris().map(|val| val.display_with_prefix("Iris")),
			self.focus().map(|val| val.display_with_prefix("Focus")),
			self.auto_exposure()
				.map(|val| val.display_with_prefix("AutoExposure")),
			self.exposure_absolute()
				.map(|val| val.display_with_prefix("ExposureAbsolute")),
			self.auto_white_balance()
				.map(|val| val.display_with_prefix("AutoWhiteBalance")),
			self.power_line_frequency()
				.map(|val| val.display_with_prefix("PowerLineFrequency")),
			self.exposure_auto_priority()
				.map(|val| val.display_with_prefix("ExposureAutoPriority")),
		]
		.into_iter()
		.flatten()
		.flatten()
		.collect()
	}
}

impl LaserItemBase for StandardItem {
	fn get_base_item_mut(&mut self) -> &mut BaseItem<Self::IntensityType> {
		&mut self.base
	}
}

impl fmt::Display for StandardItem {
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		// TODO
		write!(f, "{}: '{}'", self.get_intensity(), self.get_image_path())
	}
}

/// Loads a standard item dataset through a mapper.
///
/// # Errors
///
/// Returns a loading error if `index.csv` cannot be read, a row cannot be
/// deserialised, or an image path cannot be resolved.
pub fn build_standard_dataset<M, MO, P>(
	mapper: M,
	dataset_root_folder: P,
	filters: &[FilterFn<StandardItem>],
) -> LoaderResult<StandardLaserDataset<M, MO>>
where
	M: Mapper<StandardItem, MO> + Send + Sync,
	MO: Send + Sync + Clone,
	P: AsRef<Path>,
{
	LaserDataset::try_new(mapper, dataset_root_folder, filters)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn standard_item_deserialises_prefixed_control_fields() {
		let csv = concat!(
			"relative_path,intensity,externalLaserIntensity,",
			"Brightness,Brightness-default,Brightness-min,",
			"Brightness-max,Brightness-step,WhiteBalance,",
			"WhiteBalance-default,WhiteBalance-min,WhiteBalance-max,",
			"WhiteBalance-step,AutoWhiteBalance,",
			"AutoWhiteBalance-default,PowerLineFrequency,",
			"PowerLineFrequency-default,PowerLineFrequency-items\n",
			"sample.png,9,1.5,32,32,-64,64,1,4600,4600,2800,",
			"6500,1,true,true,1,1,\"[{ Id: 0, Name: Disabled }]\"\n",
		);
		let mut reader = csv::Reader::from_reader(csv.as_bytes());
		let item: StandardItem = reader
			.deserialize()
			.next()
			.and_then(std::result::Result::ok)
			.expect("CSV row should deserialize");

		let brightness =
			item.brightness().expect("brightness should be present");
		assert_eq!(*brightness.value(), 32);
		assert_eq!(brightness.try_default(), Some(32));
		assert_eq!(brightness.try_min(), Some(-64));
		assert_eq!(brightness.try_max(), Some(64));
		assert_eq!(brightness.try_step(), Some(1));

		let white_balance = item
			.white_balance()
			.expect("white balance should be present");
		assert_eq!(*white_balance.value(), 4600);
		assert_eq!(white_balance.try_max(), Some(6500));

		let auto_white_balance = item
			.auto_white_balance()
			.expect("auto white balance should be present");
		assert!(auto_white_balance.value());
		assert_eq!(auto_white_balance.try_default(), Some(true));

		let power_line_frequency = item
			.power_line_frequency()
			.expect("power line frequency should be present");
		assert_eq!(power_line_frequency.value(), &1);
		assert_eq!(
			power_line_frequency.try_items().as_deref(),
			Some("[{ Id: 0, Name: Disabled }]")
		);

		assert!(item.contrast().is_none());
		assert_eq!(
			item.external_laser_intensity().map(|value| *value),
			Some(1.5)
		);
	}
}
