# API Stability and Release Policy

This document explains how yscv handles versioning, what stability guarantees each crate provides, and the process for publishing releases.

## Versioning

All workspace crates follow [Semantic Versioning 2.0](https://semver.org/). During pre-1.0 development (the current stage), minor version bumps may include breaking changes. Once a crate reaches 1.0, breaking changes will require a major version bump.

All crates currently share workspace version `0.1.7` (defined in the root `Cargo.toml`). The public API is functional and tested, but not yet frozen.

## Stability tiers

Not all crates are equally mature. The stability tier tells you how likely the API is to change:

| Tier | What it means | Crates |
|---|---|---|
| **Stable** | API is unlikely to change before 1.0 | `yscv-tensor` (core types) |
| **Maturing** | API is solidifying, minor changes possible | `yscv-kernels`, `yscv-autograd`, `yscv-optim`, `yscv-imgproc` |
| **Evolving** | Active development, API changes expected | `yscv-model`, `yscv-onnx`, `yscv-video`, `yscv-detect`, `yscv-track`, `yscv-recognize`, `yscv-eval` |
| **Experimental** | New capabilities, expect breakage | GPU multi-device, distributed training, model zoo, TCP transport, HEVC decoder |

## What counts as public API

The following are considered public API and subject to semver rules: all `pub` items in crate root modules, all trait definitions and their required methods, all error enum variants, and public struct field types and names.

Things that are not covered by semver: benchmark performance numbers, internal module structure, CI workflow details, and documentation wording.

## Release checklist

Before any version bump, all of these must be true:

1. `cargo test` passes across the entire workspace.
2. `cargo clippy -D warnings` is clean.
3. `cargo doc --no-deps` builds without warnings.
4. A CHANGELOG.md entry has been added for the version.
5. `docs/ecosystem-capability-matrix.md` reflects the current state.
6. `context.md` and `agents.md` are up to date.

## Publishing

Crates must be published in dependency order because each crate depends on the ones before it. The order is automated via `scripts/publish.sh`:

1. yscv-tensor → 2. yscv-kernels → 3. yscv-autograd → 4. yscv-optim → 5. yscv-imgproc → 6. yscv-video → 7. yscv-video-mpp → 8. yscv-onnx → 9. yscv-pipeline → 10. yscv-detect → 11. yscv-recognize → 12. yscv-track → 13. yscv-eval → 14. yscv-model → 15. yscv-cli → 16. yscv (umbrella)

The `apps/bench` and `apps/camera-face-tool` binaries live in the workspace but are not part of the 16-crate library set and are not published to crates.io.

To bump all crate versions at once, run `scripts/bump-version.sh <version>`.

## Changelog discipline

Every merged PR that changes public API or observable behavior must include a CHANGELOG.md entry under one of: Added, Changed, Deprecated, Removed, Fixed, or Security.
