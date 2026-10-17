//! Typed wrappers for dataset intensity labels.

use std::{
	borrow::{
		Borrow,
		BorrowMut,
	},
	fmt::Display,
	ops::{
		Deref,
		DerefMut,
	},
	str::FromStr,
};

use burn::tensor::Element;
use serde::Deserialize;

/// A transparent wrapper around a dataset intensity value.
///
/// This type preserves the underlying value's representation for Serde while
/// giving intensity values a distinct API type.
///
/// # Examples
///
/// ```
/// use dataset_loader::item::Intensity;
///
/// let intensity = Intensity::from(12_u8);
/// assert_eq!(*intensity, 12);
/// assert_eq!(intensity.to_string(), "12");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialOrd, PartialEq, Deserialize)]
#[serde(transparent)]
pub struct Intensity<T>
where
	T: Element,
{
	inner: T,
}

impl<T> Display for Intensity<T>
where
	T: Element,
{
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(f, "{}", self.inner)
	}
}

impl<T> From<T> for Intensity<T>
where
	T: Element,
{
	fn from(value: T) -> Self {
		Self {
			inner: value,
		}
	}
}

impl<T> Deref for Intensity<T>
where
	T: Element,
{
	type Target = T;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<T> DerefMut for Intensity<T>
where
	T: Element,
{
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

impl<T, O> AsRef<O> for Intensity<T>
where
	T: Element,
	O: ?Sized,
	<Intensity<T> as Deref>::Target: AsRef<O>,
{
	fn as_ref(&self) -> &O {
		self.deref().as_ref()
	}
}

impl<T, O> AsMut<O> for Intensity<T>
where
	T: Element,
	O: ?Sized,
	<Intensity<T> as Deref>::Target: AsMut<O>,
{
	fn as_mut(&mut self) -> &mut O {
		self.deref_mut().as_mut()
	}
}

impl<T> Borrow<T> for Intensity<T>
where
	T: Element,
{
	fn borrow(&self) -> &T {
		&self.inner
	}
}

impl<T> BorrowMut<T> for Intensity<T>
where
	T: Element,
{
	fn borrow_mut(&mut self) -> &mut T {
		&mut self.inner
	}
}

impl<T> FromStr for Intensity<T>
where
	T: Element + FromStr,
{
	type Err = T::Err;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		T::from_str(s).map(|inner| {
			Self {
				inner,
			}
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn intensity_with_u8() {
		_ = Intensity::from(0_u8);
	}

	#[test]
	fn intensity_with_f32() {
		_ = Intensity::from(0_f32);
	}
}
