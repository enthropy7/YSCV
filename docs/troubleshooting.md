# Troubleshooting

The errors people actually hit, with the fix. Grouped by area; use
your browser's find for a specific symptom.

> **Not finding your symptom here?** Open an issue with the full
> error output + your `rustc --version`, `uname -a`, and
> `cargo build --features` line. We add new entries here as they
> come up.

---

## Build errors

### `error: package requires Rust 1.94`

Workspace MSRV is 1.94 (we use stabilised aarch64 NEON intrinsics).
Update:

```bash
rustup update stable
rustc --version    # should report 1.94 or newer
```

### `linker 'cc' not found`

Need a C compiler for some transitive build scripts (mostly `prost`,
`metal-rs`).

```bash
# macOS
xcode-select --install

# Debian/Ubuntu
sudo apt install build-essential

# Fedora
sudo dnf install gcc gcc-c++
```

### `error: linking with 'rust-lld' failed: unable to find library -lopenblas`

Cross-compile (or native build on Linux) wants OpenBLAS but the
sysroot doesn't have it. Two fixes:

```bash
# Native Linux: install the dev package
sudo apt install libopenblas-dev

# Cross-compile: use `cross` instead of bare `cargo build --target ...`
cargo install cross --git https://github.com/cross-rs/cross
cross build --release --target aarch64-unknown-linux-gnu
```

For native macOS this never happens — Accelerate is built into the OS.

### `protoc binary not found`

Rare — `yscv-onnx` ships a built-in fallback. If you see it anyway:

```bash
sudo apt install protobuf-compiler         # Debian/Ubuntu
brew install protobuf                      # macOS
```

### `cannot find function 'vreinterpret_f16_u16'`

Outdated Rust. Update to 1.94+:

```bash
rustup update stable
```

### `error: could not compile 'yscv-onnx': no space left on device`

Disk full from incremental compile artefacts. The workspace `target/`
balloons to 30+ GB after a few feature combos.

```bash
cargo clean             # nuke everything (~10 min full rebuild after)
cargo clean -p yscv-onnx -p yscv-kernels -p yscv-video    # selective
rm -rf target/release/incremental                          # incremental only
```

---

## Runtime errors

### `librknnrt.so not found — RKNN runtime not available`

The `rknn` feature was compiled in but the host doesn't have the
Rockchip runtime library. Two scenarios:

**On a non-Rockchip dev machine (Mac / x86 Linux)** — expected,
nothing to do. The framework loads `librknnrt.so` lazily via
`dlopen`, so the binary builds fine and runs fine for non-RKNN tasks
on dev hosts. Only RKNN-specific code paths fail loud.

**On the actual board** — you need the RKNN runtime installed:

```bash
# RK3588 / RK3576 / RV1106 — official Rockchip release
# Grab the matching librknnrt.so from:
# https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp librknnrt.so /usr/lib/aarch64-linux-gnu/
sudo ldconfig
ldconfig -p | grep librknnrt    # verify
```

Or use the Rockchip `rknn-toolkit2-lite` Debian package if available
for your distro.

### `Permission denied` on `set_sched_fifo` (SCHED_FIFO)

You don't have `CAP_SYS_NICE`. Either run as root (bad) or grant the
capability to your binary (good):

```bash
sudo setcap cap_sys_nice=eip ./your-binary
./your-binary    # now SCHED_FIFO works
```

The framework logs this as a warning and continues without
SCHED_FIFO — your pipeline still works, you just don't get hard
real-time guarantees.

### `mlockall failed: ENOMEM`

Process memory-lock limit is too low. Default is usually 64 KB; you
need at least 64 MB for a real pipeline.

```bash
ulimit -l unlimited      # interactive shell

# Or in a systemd unit:
[Service]
LimitMEMLOCK=infinity
```

### `cpu_governor write failed: Permission denied`

Need `CAP_SYS_ADMIN` or root to write `/sys/devices/system/cpu/.../scaling_governor`.

```bash
sudo setcap cap_sys_admin+ep ./your-binary
# ... or do it once at boot via a tmpfiles.d / systemd one-shot:
sudo cpupower frequency-set -g performance     # one-shot, persists until reboot
```

The framework counts how many cores succeeded (`RtAppliedState.cpu_governor_cores`)
and logs failures — pipeline still runs, just at default governor.

### `RKNN dispatch failed: TIMEOUT (-14)`

NPU is stuck. The pool's auto-recovery should handle this on the 3rd
consecutive failure, but if you see repeated timeouts:

1. Check the model isn't corrupted (re-export from rknn-toolkit2).
2. Check NPU temperature — passive-cooled boards throttle hard.
   ```bash
   cat /sys/class/thermal/thermal_zone*/temp
   ```
3. Confirm you compiled the `.rknn` for the right SoC
   (`target_platform='rk3588'` etc.).
4. Manual recovery: `handle.recover_all()?` or
   `pool.recover_failed(slot_idx)`.

### `validate_models: ModelInvalid — expected RKNN / RKNF magic, got [...]`

The file at `model_path` isn't a real `.rknn` — likely an `.onnx`
saved with the wrong extension, or a Git-LFS pointer file (8 bytes
of `version`, then a hash). Check:

```bash
file model.rknn         # should say "data" or "RKNN..."
head -c 4 model.rknn | xxd    # first 4 bytes must be `RKNN` or `RKNF`
```

If it's an `.onnx` you meant to compile to `.rknn`, just rename the
file extension to `.onnx` — the framework auto-compiles at startup.

### `validate_models: ModelInvalid — expected ONNX protobuf (first byte 0x08)`

Same as above but for ONNX — file is corrupt or wrong format. Check
with `python -c "import onnx; onnx.load('model.onnx')"`.

### `MPSGraph compile failed: unsupported op '...'`

Apple's MPSGraph supports a subset of ONNX ops. Two paths:

1. Drop to the per-op Metal backend
   (`compile_metal_plan`/`run_metal_plan`) — covers more ops, slower.
2. Fall back to the CPU runner (`run_onnx_model`) for that model.

The unsupported op name in the error is your search key. We add new
op support upstream regularly; check
[docs/onnx-inference.md](onnx-inference.md) for current coverage.

### `'libmpp.so' not found` on RK3588

The MPP hardware encoder needs Rockchip's MPP library. Same vendor
package as `librknnrt.so`:

```bash
# Usually pre-installed on Rockchip official BSP images. If not:
sudo apt install librockchip-mpp1
# or grab from rockchip-linux/mpp on GitHub
```

If `librockchip_mpp.so` isn't present, the encoder falls back to
the software path — works but ~5× slower.

### `validate_accelerators: AcceleratorUnavailable — task X requested rknn`

Compiled binary doesn't include the `rknn` feature. Rebuild:

```bash
cargo build --release --features rknn
# or for the example runner:
cargo run --release --features "rknn realtime" --example edge_pipeline_v2 -- config.toml
```

### `dispatch_frame failed: input 'X': N bytes but the model expects exactly M`

You're feeding the wrong number of bytes. Common causes:

- Wrong dtype assumption — RKNN expects u8 NCHW for vision models;
  ONNX runner via the dispatcher path expects f32 LE bytes (dispatch
  reinterprets `bytes.len() / 4` as element count).
- Wrong image dimensions — model wants 640×640, you fed 1280×720.
- NCHW vs NHWC mismatch — different ops expect different layouts.

The error message includes the expected size; compare against your
actual byte count + intended dtype.

---

## Performance issues

### "My CPU inference is much slower than the benchmarks promise"

Check, in order:

1. **`--release` flag.** Debug builds are 100× slower on SIMD code.
2. **`lto = "thin"` and `codegen-units = 1`.** Without these the
   per-op SIMD doesn't inline. Add to your `Cargo.toml`:
   ```toml
   [profile.release]
   lto = "thin"
   codegen-units = 1
   ```
3. **Apple Silicon `target-cpu`.** Add to `.cargo/config.toml`:
   ```toml
   [target.aarch64-apple-darwin]
   rustflags = ["-C", "target-cpu=apple-m1"]
   ```
4. **BLAS** is on by default but verify it didn't get
   `--no-default-features`-d:
   ```bash
   otool -L target/release/your-binary | grep -i 'accelerate\|blas'
   # macOS: should mention Accelerate.framework
   ```

### "First inference is 100× slower than subsequent ones"

Expected. JIT kernel selection + L2 warmup + (on Apple Silicon)
DVFS step-up. Discard the first 1-3 iterations as warmup in
benchmarks. In production this happens once at startup.

### "RKNN inference is fast but throughput is low"

You're running synchronous calls. Use the pipelined path:

```rust
// Bad — NPU sits idle while CPU marshals next frame
for frame in frames {
    let out = pool.run(&[("images", &frame)])?;
}

// Good — NPU and CPU overlap (3× throughput on RK3588)
let mut h = pool.submit(&[("images", &frame0)])?;
for frame in frames.skip(1) {
    let h_next = pool.submit(&[("images", &frame)])?;
    let out = pool.wait(h)?;
    h = h_next;
}
let _ = pool.wait(h)?;
```

### "p99 latency is bad even though p50 is fine"

Several common causes on edge devices:

- **No SCHED_FIFO** — kernel preempts your inference thread for
  random userspace. Set `[realtime] sched_fifo = true` and grant
  `CAP_SYS_NICE`.
- **No CPU pinning** — your thread bounces between cores, losing L1/L2
  cache each time. Set `[realtime] affinity.dispatch = [4, 5, 6]`.
- **Default cpufreq governor** — `ondemand` or `schedutil` adds 5-30 ms
  latency on the first burst after idle. Set
  `[realtime] cpu_governor = "performance"`.
- **No `mlockall`** — pages of model weights swap out under memory
  pressure, causing 50 ms+ glitches when they swap back in.
  Automatically applied when `sched_fifo = true`.
- **Watchdog overrun spikes** — check `PipelineStats5::watchdog_alarm`;
  if it's set, your stage budget is too tight for actual hardware.

### "Cross-compile to aarch64-linux works on Linux but not macOS"

`rust-lld` on macOS doesn't have a glibc sysroot. Use `cross`:

```bash
cargo install cross --git https://github.com/cross-rs/cross
cross build --release --target aarch64-unknown-linux-gnu
# Or build directly on the Rockchip board (slow but reliable).
```

---

## Test failures

### `proptest_*` tests fail intermittently

Proptest finds edge-case inputs we don't test for. Open an issue with
the fixture file proptest writes to
`crates/<crate>/proptest-regressions/<test>.txt` — it'll have the
exact seed.

### `metal-backend` tests fail with `MPSGraph compile failed`

Two causes:
- Test ONNX uses an op MPSGraph doesn't support. Skip on macOS:
  `#[cfg_attr(target_os = "macos", ignore)]`.
- macOS version too old. Need 13.0+ for full MPSGraph op coverage.

### `rknn` tests pass on macOS but fail with `dlopen` errors

Expected on dev hosts. The `rknn` tests verify error-path behaviour
when `librknnrt.so` isn't loadable; tests that need real hardware
are gated behind the `RKNN_HARDWARE_TEST` env var (not set by
default).

---

## Confusing-but-correct behaviours

### `RknnInferenceHandle` warns about being dropped

You're not calling `wait()` on a handle. The next `submit()` on the
same slot will back-pressure-wait, so the GPU work completes anyway —
you just lose visibility of the outputs. This is intentional safety;
the `#[must_use]` exists to surface accidental drops at compile
time.

### "Auto-compile created a `.rknn` next to my `.onnx`"

Working as intended. The framework caches the compiled RKNN model
next to the source ONNX so subsequent runs skip the (slow)
compilation. Delete the cache file to force re-compile.

### "TOML config validation passes but pipeline behaves unexpectedly"

Validation checks: file existence + magic bytes + accelerator
availability + cycle detection. It does **not** check:

- That tensor shapes match between connected tasks (you'll get a
  runtime error on the first mismatched dispatch).
- That `model_path` matches the actual model the dispatcher loads
  (relevant for ONNX→RKNN auto-compile — a stale cache wins over
  the source ONNX).
- That your accelerator can actually run the model fast enough for
  your `[realtime]` budgets (no static analysis here).

The first dispatch surfaces these; expect a clear error if anything
is wrong.

---

## Where to ask

| Channel | When to use |
|---|---|
| GitHub issues | Reproducible bugs, missing features, docs gaps |
| GitHub Discussions | "How do I..." questions, design feedback |
| Telegram (link on the maintainer's GitHub profile) | Direct chat for production deployments |

When opening an issue please include:

```
rustc --version
uname -a
cargo build / run command (with all features)
Full error output
Minimal repro (a Cargo.toml + main.rs + config.toml is ideal)
```
