//! Generally-applicable control types.
//!
//! Ideally, the actual controls would be serialised and then could be
//! deserialised directly here.

use std::{
	borrow::Cow,
	fmt,
};

use serde::{
	Deserialize,
	Serialize,
};

/// A generally-applicable control type for deserialising control data.
///
/// When flattened, the value field uses the control name and optional metadata
/// fields use `-default`, `-min`, `-max`, `-step`, `-possible`, and `-items`
/// suffixes. A containing struct can add a control name prefix with a `Serde`
/// field-name adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneralCameraControl<V = i64, D = i64, Mi = i64, Ma = i64, S = i64>
where
	V: fmt::Display + Clone + Copy,
	D: fmt::Display + Clone,
	Mi: fmt::Display + Clone + Copy,
	Ma: fmt::Display + Clone + Copy,
	S: fmt::Display + Clone + Copy,
{
	/// The only field a control is guaranteed to have.
	///
	/// While it is possible for a control ID to not have a description, it
	/// will never be saved to a dataset.
	#[serde(rename = "")]
	value: V,
	/// Only a [`String`] type control may not have a default field.
	#[serde(rename = "-default")]
	default: Option<D>,
	#[serde(rename = "-min")]
	min: Option<Mi>,
	#[serde(rename = "-max")]
	max: Option<Ma>,
	#[serde(rename = "-step")]
	step: Option<S>,
	/// This field is only present on `Enum` control descriptions.
	#[serde(rename = "-possible")]
	possible: Option<String>,
	/// This field is only present on `Menu` control descriptions.
	#[serde(rename = "-items")]
	items: Option<String>,
}

impl<V, D, Mi, Ma, S> GeneralCameraControl<V, D, Mi, Ma, S>
where
	V: fmt::Display + Clone + Copy,
	D: fmt::Display + Clone,
	Mi: fmt::Display + Clone + Copy,
	Ma: fmt::Display + Clone + Copy,
	S: fmt::Display + Clone + Copy,
{
	/// Returns the `value` field.
	pub fn value(&self) -> &V {
		&self.value
	}

	/// Returns the `default` field.
	pub fn try_default(&self) -> Option<D> {
		self.default.clone()
	}

	/// Returns the `min` field.
	pub fn try_min(&self) -> Option<Mi> {
		self.min
	}

	/// Returns the `max` field.
	pub fn try_max(&self) -> Option<Ma> {
		self.max
	}

	/// Returns the `step` field.
	pub fn try_step(&self) -> Option<S> {
		self.step
	}

	/// Returns the `possible` field.
	pub fn try_possible(&self) -> Option<Cow<'_, str>> {
		self.possible.as_deref().map(Cow::Borrowed)
	}

	/// Returns the `items` field.
	pub fn try_items(&self) -> Option<Cow<'_, str>> {
		self.items.as_deref().map(Cow::Borrowed)
	}

	/// Expand this control into each sub-value for display.
	pub fn display_with_prefix(
		&self,
		prefix: &'static str,
	) -> Vec<(String, String)> {
		[
			(prefix.to_owned(), Some(self.value().to_string())),
			(
				format!("{prefix}-default"),
				self.try_default().map(|val| val.to_string()),
			),
			(
				format!("{prefix}-min"),
				self.try_min().map(|val| val.to_string()),
			),
			(
				format!("{prefix}-max"),
				self.try_max().map(|val| val.to_string()),
			),
			(
				format!("{prefix}-step"),
				self.try_step().map(|val| val.to_string()),
			),
			(
				format!("{prefix}-possible"),
				self.try_possible().map(|val| val.to_string()),
			),
			(
				format!("{prefix}-items"),
				self.try_items().map(|val| val.to_string()),
			),
		]
		.into_iter()
		.filter_map(|(key, value)| value.map(|val| (key, val)))
		.collect()
	}
}
