# yscv-video-mpp

Rockchip MPP (Media Process Platform) hardware H.264 / HEVC video
encoder bindings, loaded via `dlopen` so the rest of the workspace
builds cleanly on hosts without `librockchip_mpp.so`.

Targets RK3588 / RK3566 / RK3568 / RV1106 boards. On those boards
the MPP encoder taps the dedicated VPU and frees the Cortex-A
cores for inference / overlay rendering.

## What's in here

- `Encoder` — high-level wrapper around `MppCtx`: configure with
  width / height / fps / bitrate / pixel format, push `Tensor`-style
  YUV frames, pull packed H.264 or HEVC NAL units.
- `ffi` — opaque function-pointer table populated at first use via
  `dlopen("librockchip_mpp.so")`. Build never links the library;
  if the runtime host doesn't have it, `Encoder::new()` returns
  `Error::LibraryNotFound`.

## Why dlopen rather than dynamic linking

Most yscv consumers will never run on a Rockchip board. Linking
`librockchip_mpp.so` directly would force every Linux build to
ship the symbol or fail at load time. Lazy `dlopen` keeps the
crate compilable everywhere; missing-on-this-host is reported as
a recoverable error, not a link failure.

## Status

`mpp` Cargo feature gates the FFI module so the crate compiles to
an empty stub when the feature is off (which is the default).
Turn it on with `--features mpp` when targeting an RK board:

```toml
yscv-video-mpp = { version = "0.1", features = ["mpp"] }
```

Library only; no `cargo run` entry point. Pair with `yscv-video`
for the decode side.
