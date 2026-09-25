//! Example: run a Kolmogorov–Arnold network (KAN) on one sensor window with `KanLinear`.
//!
//! Loads a B-spline KAN exported from `efficient-kan` / PyTorch, classifies a feature window and
//! prints the latency of a single-window forward pass (the on-device streaming case). Without
//! arguments it builds a random network of the size of a UCI HAR classifier (561 features →
//! 7 → 6 classes, grid 5) and only times it.
//!
//! Usage:
//!   cargo run --release --example kan_sensor
//!   cargo run --release --example kan_sensor -- model.kan window.f32
//!
//! `model.kan` (little-endian): `b"KAN1"`, `u32` layer count, then per layer `u32` in, `u32` out,
//! `u32` grid size, `f32` grid low, `f32` grid high, followed by `base_weight` `[out, in]`,
//! `spline_weight` `[out, in, grid + 3]` and `spline_scaler` `[out, in]` as `f32`, row-major —
//! the parameters of `efficient_kan.KANLinear` as they are. `window.f32` holds the `in` features
//! of the first layer as raw `f32`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Instant;

use yscv_kernels::KanLinear;

fn read_u32(b: &[u8], at: &mut usize) -> u32 {
    let v = u32::from_le_bytes([b[*at], b[*at + 1], b[*at + 2], b[*at + 3]]);
    *at += 4;
    v
}

fn read_f32s(b: &[u8], at: &mut usize, n: usize) -> Vec<f32> {
    let v = b[*at..*at + 4 * n]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *at += 4 * n;
    v
}

fn load(path: &str) -> Result<Vec<KanLinear>, Box<dyn std::error::Error>> {
    let b = std::fs::read(path)?;
    if b.get(..4) != Some(b"KAN1".as_slice()) {
        return Err(format!("{path}: not a KAN1 file").into());
    }
    let mut at = 4;
    let layers = read_u32(&b, &mut at);
    (0..layers)
        .map(|_| {
            let (n_in, n_out, grid) = (
                read_u32(&b, &mut at) as usize,
                read_u32(&b, &mut at) as usize,
                read_u32(&b, &mut at) as usize,
            );
            let range = read_f32s(&b, &mut at, 2);
            let base = read_f32s(&b, &mut at, n_in * n_out);
            let spline = read_f32s(&b, &mut at, n_in * n_out * (grid + 3));
            let scaler = read_f32s(&b, &mut at, n_in * n_out);
            Ok(KanLinear::new(
                n_in,
                n_out,
                grid,
                (range[0], range[1]),
                &base,
                &spline,
                &scaler,
            )?)
        })
        .collect()
}

fn random_har_sized() -> Vec<KanLinear> {
    let mut s = 1u32;
    let mut next = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
    };
    [(561, 7), (7, 6)]
        .into_iter()
        .map(|(n_in, n_out)| {
            let e = n_in * n_out;
            let (base, spline, scaler): (Vec<f32>, Vec<f32>, Vec<f32>) = (
                (0..e).map(|_| next()).collect(),
                (0..e * 8).map(|_| next()).collect(),
                (0..e).map(|_| next()).collect(),
            );
            KanLinear::new(n_in, n_out, 5, (-1.0, 1.0), &base, &spline, &scaler)
                .expect("valid layer")
        })
        .collect()
}

fn forward(layers: &[KanLinear], window: &[f32]) -> Vec<f32> {
    layers.iter().fold(window.to_vec(), |x, layer| {
        let mut y = vec![0.0; layer.out_features()];
        layer.forward(&x, 1, &mut y).expect("layer shapes chain");
        y
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let layers = match args.first() {
        Some(path) => load(path)?,
        None => random_har_sized(),
    };
    let n_in = layers[0].in_features();
    let window = match args.get(1) {
        Some(path) => {
            let b = std::fs::read(path)?;
            read_f32s(&b, &mut 0, n_in)
        }
        None => (0..n_in)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) * 2.0 - 1.0)
            .collect(),
    };

    let logits = forward(&layers, &window);
    let class = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(c, _)| c);
    println!(
        "layers: {}",
        layers
            .iter()
            .map(|l| format!("{}→{}", l.in_features(), l.out_features()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("class {class}, logits {logits:?}");

    let mut us: Vec<f64> = (0..2000)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(forward(&layers, std::hint::black_box(&window)));
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    us.sort_by(f64::total_cmp);
    println!(
        "latency per window: min {:.1} µs, p50 {:.1} µs, p90 {:.1} µs",
        us[0],
        us[us.len() / 2],
        us[us.len() * 9 / 10]
    );
    Ok(())
}
