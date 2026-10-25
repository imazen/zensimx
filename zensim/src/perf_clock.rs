//! The clock behind the attribution path's always-on timing counters.
//!
//! `std::time::Instant::now` panics on `wasm32-unknown-unknown`, which has no
//! clock, so every map computed in a browser module failed. This `Instant`
//! has the same `now`/`elapsed` shape; there it reads zero, and the counters
//! it feeds only report time, never change a result.

/// A point in time, or nothing where the target has no clock.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Instant(
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))] std::time::Instant,
);

impl Instant {
    /// Now.
    #[inline]
    pub(crate) fn now() -> Self {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            Self(std::time::Instant::now())
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            Self()
        }
    }

    /// The time since [`Self::now`]; zero without a clock.
    #[inline]
    pub(crate) fn elapsed(&self) -> std::time::Duration {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            self.0.elapsed()
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            std::time::Duration::ZERO
        }
    }
}
