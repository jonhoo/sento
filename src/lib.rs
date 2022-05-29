//! A lock-free, append-only atomic pool.
//!
//! This library implements an atomic, append-only collection of items, where individual items can
//! be acquired and relased so they are always associated with at most one owner. Thus, each item
//! is at all times either free or acquired, and the library presents operations for acquiring a
//! free item or for releasing an already acquired one.
//!
//! The implementation is inspired by the one used in [folly's hazard pointer implementation],
//! originally ported into [`haphazard`]. It consists of two linked lists implemented using a
//! single shared node type, where each node holds both a pointer to the next node in the overall
//! list, and a "skip pointer" to the next _available (non-acquired) node in the list. This enables
//! acquiring to be efficient and atomic.
//!
//! [`haphazard`]: https://crates.io/crates/haphazard
//! [folly's hazard pointer implementation]: https://github.com/facebook/folly/blob/594b7e770176003d0f6b4cf725dd02a09cba533c/folly/synchronization/HazptrRec.h#L35-L36
//!
//! # Examples
//!
//! ## Basic operation
//!
//! ```rust
//! use sento::Pool;
//! type Value = i32;
//! let pool = Pool::<Value>::new();
//! let v1 = pool.acquire(); // this will allocate a new Value with Value::default()
//! let v2 = pool.acquire(); // so will this
//!
//! // we can get a long-lived shared reference to the value
//! let v1_ref = v1.into_ref();
//!
//! // by releasing v1, it is now "free" and will be used rather than allocating a new Value.
//! // note, however, that v1 will not be deallocated until pool is dropped!
//! pool.release(v1);
//! let v1_again = pool.acquire();
//!
//! // note that the semantics of acquire and release are up to you.
//! // .release does not require that you surrender the released reference,
//! // since the referenced value still lives in the same place.
//! assert_eq!(v1_ref as *const _, &*v1_again as *const _);
//!
//! // when the Pool is dropped, it also frees all the nodes,
//! // so at that point you can't access the values any more.
//! drop(pool);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links, rust_2018_idioms)]
#![cfg_attr(not(feature = "std"), no_std)]

// XXX: how do we ensure items are returned to the _same_ pool?
static DISCRIMINATOR: AtomicUsize = AtomicUsize::new(0);

extern crate alloc;

mod sync;

use crate::sync::atomic::{AtomicPtr, AtomicUsize};
use alloc::boxed::Box;
use core::{marker::PhantomData, sync::atomic::Ordering};

// Make AtomicPtr usable with loom API.
trait WithMut<T> {
    fn with_mut<R>(&mut self, f: impl FnOnce(&mut *mut T) -> R) -> R;
}
impl<T> WithMut<T> for core::sync::atomic::AtomicPtr<T> {
    fn with_mut<R>(&mut self, f: impl FnOnce(&mut *mut T) -> R) -> R {
        f(self.get_mut())
    }
}

const LOCK_BIT: usize = 1;

/// An item from a [`Pool`].
///
/// Only exposed in public interfaces as [`Acquired`].
#[derive(Debug)]
pub struct Node<T> {
    /// The value being stored in the node.
    v: T,

    /// The next `Node` in the list of all `Node`s in a `Pool`.
    /// This value never changes once set.
    next: AtomicPtr<Self>,

    /// The next non-acquired `Node` in the associatd `Pool`.
    available_next: AtomicPtr<Self>,

    /// A unique identifier for the `Pool` this `Node` was created by.
    discriminator: usize,
}

/// An item acquired from a [`Pool`].
///
/// This type dereferences to a `T`.
/// If you want a longer-lived reference to the underlying `T`, use [`Acquired::into_ref`].
///
/// To release this `T` back to the [`Pool`], use [`Pool::release`].
#[derive(Debug)]
#[repr(transparent)]
pub struct Acquired<'pool, T>(&'pool Node<T>);

impl<T> AsRef<T> for Acquired<'_, T> {
    fn as_ref(&self) -> &T {
        self.into_ref()
    }
}

impl<T> core::ops::Deref for Acquired<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.into_ref()
    }
}

impl<'pool, T> Acquired<'pool, T> {
    /// Extract a shared reference to the acquired `T`.
    ///
    /// Note that the returned reference outlives this `Acquired`. While it will remain _valid_
    /// until the associated [`Pool`] is dropped, there's no guarantee that that `T` will not be
    /// released and acquired elsewhere during that time.
    #[inline]
    pub fn into_ref(&self) -> &'pool T {
        &self.0.v
    }
}

// Macro to make new const only when not in loom.
macro_rules! new_node {
    ($($decl:tt)*) => {
        $($decl)*(discriminator: usize, v: T) -> Self {
            Self {
                v,
                next: AtomicPtr::new(core::ptr::null_mut()),
                available_next: AtomicPtr::new(core::ptr::null_mut()),
                discriminator,
            }
        }
    };
}

impl<T> Node<T> {
    #[cfg(not(loom))]
    new_node!(const fn new);
    #[cfg(loom)]
    new_node!(fn new);
}

/// A shareable pool of `T`s that can be acquired and released without locks.
///
/// Use [`Pool::acquire`] to acquire a `T`, and [`Pool::release`] to return one.
pub struct Pool<T> {
    head: AtomicPtr<Node<T>>,
    head_released: AtomicPtr<Node<T>>,
    discriminator: usize,
    count: AtomicUsize,
}

// Sharing a Pool<T> in isolation is fine.
// Sharing a Pool<T> enables sharing a T, so we require T: Sync.
// Sharing a Pool<T> does _not_ enable moving a T (without unsafe code), so no T: Send is needed.
unsafe impl<T> Sync for Pool<T> where T: Sync {}

// Sending a Pool<T> in isolation is fine.
// Sending a Pool<T> also makes &T available on that other thread, so we require T: Sync.
// Dropping a Pool<T> will drop the Ts too, so we require T: Send.
unsafe impl<T> Send for Pool<T> where T: Sync + Send {}

impl<T> core::fmt::Debug for Pool<T>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
    /// Allocate a new, empty [`Pool`].
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(core::ptr::null_mut()),
            head_released: AtomicPtr::new(core::ptr::null_mut()),
            count: AtomicUsize::new(0),
            discriminator: DISCRIMINATOR.fetch_add(1, Ordering::AcqRel),
        }
    }

    /// The number of `T`s in the pool, including both acquired and free items.
    ///
    /// Since this is a concurrent collection, this number is only a lower bound.
    pub fn size(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    // TODO: Add a pub fn try_acquire<N> that returns <= N without calling acquire_new

    fn try_acquire_available<const N: usize>(&self) -> (*const Node<T>, usize) {
        debug_assert!(N >= 1);
        debug_assert_eq!(core::ptr::null::<Node<T>>() as usize, 0);

        loop {
            let avail = self.head_released.load(Ordering::Acquire);
            if avail.is_null() {
                return (avail, 0);
            }
            debug_assert_ne!(avail, LOCK_BIT as *mut _);
            if (avail as usize & LOCK_BIT) == 0 {
                // The available list is not currently locked.
                //
                // XXX: This could be a fetch_or and allow progress even if there's a new (but
                // unlocked) head. However, `AtomicPtr` doesn't support fetch_or at the moment, so
                // we'd have to convert it to an `AtomicUsize`. This will in turn make Miri fail
                // (with -Zmiri-tag-raw-pointers, which we want enabled) to track the provenance of
                // the pointer in question through the int-to-ptr conversion. The workaround is
                // probably to mock a type that is `AtomicUsize` with `fetch_or` with
                // `#[cfg(not(miri))]`, but is `AtomicPtr` with `compare_exchange` with
                // `#[cfg(miri)]`. It ain't pretty, but should do the job. The issue is tracked in
                // https://github.com/rust-lang/miri/issues/1993.
                if self
                    .head_released
                    .compare_exchange_weak(
                        avail,
                        with_lock_bit(avail),
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // Safety: We hold the lock on the available list.
                    let (rec, n) = unsafe { self.try_acquire_available_locked::<N>(avail) };
                    debug_assert!(n >= 1, "head_available was not null");
                    debug_assert!(n <= N);
                    return (rec, n);
                } else {
                    #[cfg(not(any(loom, feature = "std")))]
                    core::hint::spin_loop();
                    #[cfg(any(loom, feature = "std"))]
                    crate::sync::yield_now();
                }
            }
        }
    }

    /// # Safety
    ///
    /// Must already hold the lock on the available list
    unsafe fn try_acquire_available_locked<const N: usize>(
        &self,
        head: *const Node<T>,
    ) -> (*const Node<T>, usize) {
        debug_assert!(N >= 1);
        debug_assert!(!head.is_null());

        let mut tail = head;
        let mut n = 1;
        let mut next = unsafe { &*tail }.available_next.load(Ordering::Relaxed);

        while !next.is_null() && n < N {
            debug_assert_eq!((next as usize) & LOCK_BIT, 0);
            tail = next;
            next = unsafe { &*tail }.available_next.load(Ordering::Relaxed);
            n += 1;
        }

        // NOTE: This releases the lock
        self.head_released.store(next, Ordering::Release);
        unsafe { &*tail }
            .available_next
            .store(core::ptr::null_mut(), Ordering::Relaxed);

        (head, n)
    }
}

impl<T> Pool<T>
where
    T: Default,
{
    /// Acquire a `T` from the pool.
    ///
    /// If no pre-existing `T` is available, allocates a new `T` using `T::default()`.
    ///
    /// Remember to return the acquired `T`s to the pool using [`Pool::release`].
    pub fn acquire(&self) -> Acquired<'_, T> {
        self.acquire_many::<1>().into_iter().next().expect("N = 1")
    }

    /// Acquire `N` `T`s from the pool.
    ///
    /// If not enough pre-existing `T`s are available, remaining `T`s are allocated using
    /// `T::default()`.
    ///
    /// Remember to return the acquired `T`s to the pool using [`Pool::release_many`].
    pub fn acquire_many<const N: usize>(&self) -> [Acquired<'_, T>; N] {
        debug_assert!(N >= 1);

        let (mut head, n) = self.try_acquire_available::<N>();
        assert!(n <= N);

        let mut tail = core::ptr::null();
        [(); N].map(|_| {
            if !head.is_null() {
                tail = head;
                // Safety: Nodes are never deallocated.
                let rec = unsafe { &*head };
                head = rec.available_next.load(Ordering::Relaxed);
                Acquired(rec)
            } else {
                let rec = self.acquire_new();
                // Make sure we also link in the newly allocated nodes.
                if !tail.is_null() {
                    unsafe { &*tail }
                        .available_next
                        .store(rec as *const _ as *mut _, Ordering::Relaxed);
                }
                tail = rec as *const _;
                Acquired(rec)
            }
        })
    }

    pub(crate) fn acquire_new(&self) -> &Node<T>
    where
        T: Default,
    {
        // No free Nodes -- need to allocate a new one
        let node = Box::into_raw(Box::new(Node::new(self.discriminator, T::default())));
        // And stick it at the head of the linked list
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            // Safety: hazptr was never shared, so &mut is ok.
            unsafe { &mut *node }.next.with_mut(|p| *p = head);
            match self.head.compare_exchange_weak(
                head,
                node,
                // NOTE: Folly uses Release, but needs to be both for the load on success.
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // NOTE: Folly uses SeqCst because it's the default, not clear if
                    // necessary.
                    self.count.fetch_add(1, Ordering::SeqCst);
                    // Safety: Nodes are never de-allocated while the domain lives.
                    break unsafe { &*node };
                }
                Err(head_now) => {
                    // Head has changed, try again with that as our next ptr.
                    head = head_now
                }
            }
        }
    }
}

impl<T> Pool<T> {
    /// Release a `T` back to the pool.
    ///
    /// This will make it available to subsequent calls to [`Pool::acquire`].
    ///
    /// # Panics
    ///
    /// If the [`Acquired`] item was obtained from a different [`Pool`].
    pub fn release(&self, rec: Acquired<'_, T>) {
        let rec = rec.0;
        assert!(rec.available_next.load(Ordering::Relaxed).is_null());
        self.push_available(rec, rec);
    }

    /// Release multiple `T`s back to the pool at once.
    ///
    /// This will make it available to subsequent calls to [`Pool::acquire`].
    ///
    /// # Panics
    ///
    /// If the [`Acquired`] item was obtained from a different [`Pool`].
    pub fn release_many<const N: usize>(&self, recs: [Acquired<'_, T>; N]) {
        if N == 0 {
            return;
        }

        let head = recs[0].0;
        let tail = recs.last().expect("N > 0").0;
        assert!(tail.available_next.load(Ordering::Relaxed).is_null());
        self.push_available(head, tail);
    }

    fn push_available(&self, head: &Node<T>, tail: &Node<T>) {
        debug_assert!(tail.available_next.load(Ordering::Relaxed).is_null());
        if cfg!(debug_assertions) {
            let mut node = head;
            loop {
                assert_eq!(
                    self.discriminator, node.discriminator,
                    "Tried to call Pool::release with Acquired object \
                    that was obtained from a different Pool instance"
                );
                let next = node.available_next.load(Ordering::Acquire);
                if next.is_null() {
                    break;
                } else {
                    // Safety: Nodes are never deallocated.
                    node = unsafe { &*next };
                }
            }
            assert_eq!(node as *const _, tail as *const _);
        }
        debug_assert_eq!(head as *const _ as usize & LOCK_BIT, 0);
        loop {
            let avail = self.head_released.load(Ordering::Acquire);
            if (avail as usize & LOCK_BIT) == 0 {
                tail.available_next
                    .store(avail as *mut _, Ordering::Relaxed);
                if self
                    .head_released
                    .compare_exchange_weak(
                        avail,
                        head as *const _ as *mut _,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    break;
                }
            } else {
                #[cfg(not(any(loom, feature = "std")))]
                core::hint::spin_loop();
                #[cfg(any(loom, feature = "std"))]
                crate::sync::yield_now();
            }
        }
    }
}

/// An iterator over all the `T`s in the pool (whether acquired or not).
pub struct PoolIter<'a, T> {
    list: PhantomData<&'a Pool<T>>,
    head: *const Node<T>,
}

impl<T> Pool<T> {
    /// Iterate over all the `T`s in the pool (whether acquired or not).
    pub fn iter(&self) -> PoolIter<'_, T> {
        PoolIter {
            list: PhantomData,
            head: self.head.load(Ordering::Acquire),
        }
    }
}

impl<'a, T> IntoIterator for &'a Pool<T> {
    type IntoIter = PoolIter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'pool, T> Iterator for PoolIter<'pool, T> {
    type Item = &'pool T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.head.is_null() {
            None
        } else {
            // Safety: Nodes are never de-allocated while the domain lives.
            let n = unsafe { &*self.head };
            let v = &n.v;
            self.head = n.next.load(Ordering::Relaxed);
            Some(v)
        }
    }
}

impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        let mut node: *mut Node<T> = self.head.with_mut(|p| *p);
        while !node.is_null() {
            // Safety: we have &mut self, so no-one holds any of our hazard pointers any more,
            // as all holders are tied to 'domain (which must have expired to create the &mut).
            let mut n: Box<Node<T>> = unsafe { Box::from_raw(node) };
            node = n.next.with_mut(|p| *p);
            drop(n);
        }
    }
}

// Helpers to set and unset the lock bit on a `*mut Node` without losing pointer
// provenance. See https://github.com/rust-lang/miri/issues/1993 for details.
fn with_lock_bit<T>(ptr: *mut Node<T>) -> *mut Node<T> {
    int_to_ptr_with_provenance(ptr as usize | LOCK_BIT, ptr)
}
#[allow(dead_code)]
fn without_lock_bit<T>(ptr: *mut Node<T>) -> *mut Node<T> {
    int_to_ptr_with_provenance(ptr as usize & !LOCK_BIT, ptr)
}
fn int_to_ptr_with_provenance<T>(addr: usize, prov: *mut T) -> *mut T {
    let ptr = prov.cast::<u8>();
    ptr.wrapping_add(addr.wrapping_sub(ptr as usize)).cast()
}

#[cfg(test)]
mod tests {
    use super::Pool;
    use core::{
        ptr::null_mut,
        sync::atomic::{AtomicU8, Ordering},
    };

    #[test]
    fn simple() {
        let list = Pool::<AtomicU8>::new();
        let rec1 = list.acquire();
        rec1.store(1, Ordering::Release);
        let rec2 = list.acquire();
        rec2.store(2, Ordering::Release);
        list.release(rec1);
        let rec3 = list.acquire();
        assert_eq!(rec3.load(Ordering::Acquire), 1);
        list.release(rec2);
    }

    #[test]
    fn acquire_many_skips_used_nodes() {
        let list = Pool::<()>::new();
        let rec1 = list.acquire();
        let rec2 = list.acquire();
        let rec3 = list.acquire();

        assert_eq!(
            rec3.0.next.load(Ordering::Relaxed),
            rec2.0 as *const _ as *mut _
        );
        assert_eq!(
            rec2.0.next.load(Ordering::Relaxed),
            rec1.0 as *const _ as *mut _
        );
        assert_eq!(rec1.0.next.load(Ordering::Relaxed), core::ptr::null_mut());
        list.release(rec1);
        list.release(rec3);

        let [one, two, three] = list.acquire_many();

        assert_eq!(
            one.0.available_next.load(Ordering::Relaxed),
            two.0 as *const _ as *mut _
        );
        assert_eq!(
            two.0.available_next.load(Ordering::Relaxed),
            three.0 as *const _ as *mut _
        );
        assert_eq!(
            three.0.available_next.load(Ordering::Relaxed),
            core::ptr::null_mut(),
        );

        // one was previously rec3
        // two was previously rec1
        // so proper ordering for next is three -> rec3/one -> rec2 -> rec1/two
        assert_eq!(
            three.0.next.load(Ordering::Relaxed),
            one.0 as *const _ as *mut _
        );
        assert_eq!(
            one.0.next.load(Ordering::Relaxed),
            rec2.0 as *const _ as *mut _
        );
        assert_eq!(
            rec2.0.next.load(Ordering::Relaxed),
            two.0 as *const _ as *mut _
        );
    }

    #[test]
    fn acquire_many_orders_nodes_between_acquires() {
        let list = Pool::<()>::new();
        let rec1 = list.acquire();
        let rec2 = list.acquire();

        assert_eq!(
            rec2.0.next.load(Ordering::Relaxed),
            rec1.0 as *const _ as *mut _
        );
        list.release(rec2);

        // one was previously rec2
        // two is a new node
        let [one, two] = list.acquire_many();

        assert_eq!(
            one.0.available_next.load(Ordering::Relaxed),
            two.0 as *const _ as *mut _
        );
        assert_eq!(
            two.0.available_next.load(Ordering::Relaxed),
            core::ptr::null_mut(),
        );

        assert_eq!(
            two.0.next.load(Ordering::Relaxed),
            one.0 as *const _ as *mut _
        );
        assert_eq!(
            one.0.next.load(Ordering::Relaxed),
            rec1.0 as *const _ as *mut _
        );
    }

    #[test]
    fn acquire_many_properly_orders_reused_nodes() {
        let list = Pool::<()>::new();
        let rec1 = list.acquire();
        let rec2 = list.acquire();
        let rec3 = list.acquire();

        // rec3 -> rec2 -> rec1
        assert_eq!(rec1.0.next.load(Ordering::Relaxed), core::ptr::null_mut(),);
        assert_eq!(
            rec2.0.next.load(Ordering::Relaxed),
            rec1.0 as *const _ as *mut _
        );
        assert_eq!(
            rec3.0.next.load(Ordering::Relaxed),
            rec2.0 as *const _ as *mut _
        );

        // rec1 available_next -> null
        list.release(rec1);
        // rec2 available_next -> rec1
        list.release(rec2);
        // rec3 available_next -> rec2
        list.release(rec3);

        // one is rec3
        // two is rec2
        // three is rec1
        let [one, two, three, four, five] = list.acquire_many();

        assert_eq!(
            one.0.available_next.load(Ordering::Relaxed),
            two.0 as *const _ as *mut _
        );
        assert_eq!(
            two.0.available_next.load(Ordering::Relaxed),
            three.0 as *const _ as *mut _
        );
        assert_eq!(
            three.0.available_next.load(Ordering::Relaxed),
            four.0 as *const _ as *mut _
        );
        assert_eq!(
            four.0.available_next.load(Ordering::Relaxed),
            five.0 as *const _ as *mut _
        );
        assert_eq!(
            five.0.available_next.load(Ordering::Relaxed),
            core::ptr::null_mut(),
        );

        assert_eq!(
            five.0.next.load(Ordering::Relaxed),
            four.0 as *const _ as *mut _
        );
        assert_eq!(
            four.0.next.load(Ordering::Relaxed),
            one.0 as *const _ as *mut _
        );
        assert_eq!(
            one.0.next.load(Ordering::Relaxed),
            two.0 as *const _ as *mut _
        );
        assert_eq!(
            two.0.next.load(Ordering::Relaxed),
            three.0 as *const _ as *mut _
        );
        assert_eq!(three.0.next.load(Ordering::Relaxed), null_mut());
    }
}
