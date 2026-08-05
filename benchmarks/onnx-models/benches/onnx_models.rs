use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use yscv_onnx::{OnnxModel, OnnxRunner, load_onnx_model_from_file, optimize_onnx_graph};
use yscv_onnx_model_bench::{asset_dir, download_assets, make_inputs, model_cases};
use yscv_tensor::Tensor;

struct PreparedCase {
    name: String,
    model: OnnxModel,
    inputs: Vec<(String, Tensor)>,
}

fn bench_onnx_models(criterion: &mut Criterion) {
    let asset_dir = asset_dir();
    download_assets(&asset_dir).expect("download model benchmark assets");
    let cases: Vec<_> = model_cases()
        .expect("load model benchmark cases")
        .into_iter()
        .map(|case| {
            let mut model = load_onnx_model_from_file(asset_dir.join(&case.model))
                .unwrap_or_else(|error| panic!("load {}: {error}", case.model));
            optimize_onnx_graph(&mut model)
                .unwrap_or_else(|error| panic!("optimize {}: {error}", case.model));
            let inputs = make_inputs(&case, &asset_dir).expect("build model benchmark inputs");
            PreparedCase {
                name: case.name,
                model,
                inputs,
            }
        })
        .collect();
    let mut group = criterion.benchmark_group("onnx_models");

    for case in &cases {
        let runner = OnnxRunner::new(&case.model).expect("initialize ONNX runner");
        let feed: Vec<_> = case
            .inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        runner.run(&feed).expect("warm up ONNX model");
        group.bench_function(BenchmarkId::from_parameter(&case.name), |bencher| {
            bencher.iter(|| {
                black_box(runner.run(black_box(&feed)).expect("run ONNX model"));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_onnx_models);
criterion_main!(benches);
