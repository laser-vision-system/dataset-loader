use std::{
	borrow::{Borrow, BorrowMut},
	fmt::{Debug, Display},
	ops::{Deref, DerefMut},
	str::FromStr,
};

use serde::Deserialize;

/// A marker trait for types that can be used in [`Intensity`].
pub trait NumericLike:
	Send
	+ Sync
	+ Clone
	+ Copy
	+ Debug
	+ Default
	+ Display
	+ PartialOrd
	+ PartialEq
	+ FromStr
{
}

/// Blanket implementation over all valid types.
impl<T> NumericLike for T where
	T: Send
		+ Sync
		+ Clone
		+ Copy
		+ Debug
		+ Default
		+ Display
		+ PartialOrd
		+ PartialEq
		+ FromStr
{
}

/// Razor thin wrapper for the intensity value.
#[derive(Clone, Copy, Debug, Default, PartialOrd, PartialEq, Deserialize)]
#[serde(transparent)]
pub struct Intensity<T>
where
	T: NumericLike,
{
	inner: T,
}

impl<T> Display for Intensity<T>
where
	T: NumericLike,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.inner)
	}
}

impl<T> From<T> for Intensity<T>
where
	T: NumericLike,
{
	fn from(value: T) -> Self {
		Self { inner: value }
	}
}

impl<T> Deref for Intensity<T>
where
	T: NumericLike,
{
	type Target = T;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<T> DerefMut for Intensity<T>
where
	T: NumericLike,
{
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

impl<T, O> AsRef<O> for Intensity<T>
where
	T: NumericLike,
	O: ?Sized,
	<Intensity<T> as Deref>::Target: AsRef<O>,
{
	fn as_ref(&self) -> &O {
		self.deref().as_ref()
	}
}

impl<T, O> AsMut<O> for Intensity<T>
where
	T: NumericLike,
	O: ?Sized,
	<Intensity<T> as Deref>::Target: AsMut<O>,
{
	fn as_mut(&mut self) -> &mut O {
		self.deref_mut().as_mut()
	}
}

impl<T> Borrow<T> for Intensity<T>
where
	T: NumericLike,
{
	fn borrow(&self) -> &T {
		&self.inner
	}
}

impl<T> BorrowMut<T> for Intensity<T>
where
	T: NumericLike,
{
	fn borrow_mut(&mut self) -> &mut T {
		&mut self.inner
	}
}

impl<T> FromStr for Intensity<T>
where
	T: NumericLike,
{
	type Err = T::Err;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		T::from_str(s).map(|inner| Self { inner })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn intensity_with_u8() {
		let _ = Intensity::from(0u8);
	}

	#[test]
	fn intensity_with_f32() {
		let _ = Intensity::from(0f32);
	}
}
