[![Crates.io](https://img.shields.io/crates/v/sento.svg)](https://crates.io/crates/sento)
[![Documentation](https://docs.rs/sento/badge.svg)](https://docs.rs/sento/)
[![codecov](https://codecov.io/gh/jonhoo/sento/branch/main/graph/badge.svg?token=8FYF6JKJ8W)](https://codecov.io/gh/jonhoo/sento)
[![Dependency status](https://deps.rs/repo/github/jonhoo/sento/status.svg)](https://deps.rs/repo/github/jonhoo/sento)

A lock-free, append-only atomic pool.

This library implements an atomic, append-only collection of items,
where individual items can be acquired and relased so they are always
associated with at most one owner. Thus, each item is at all times
either free or acquired, and the library presents operations for
acquiring a free item or for releasing an already acquired one.

## License

Licensed under Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be licensed as above, without any additional terms or
conditions.
