#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yscv_autograd::Graph;
use yscv_model::{SequentialModel, SupervisedLoss, train_step_sgd, train_step_sgd_with_loss};
use yscv_optim::Sgd;
use yscv_tensor::Tensor;

fn tensor_from_shape(shape: &[usize], seed: f32) -> Tensor {
    let len = shape.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(len);
    for idx in 0..len {
        data.push(((idx % 257) as f32 * 0.0037 + seed).fract());
    }
    Tensor::from_vec(shape.to_vec(), data).expect("valid tensor")
}

fn bench_forward_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_forward_modes");

    group.bench_function("linear_64x32_batch32", |b| {
        let mut graph = Graph::new();
        let mut model = SequentialModel::new(&graph);
        model
            .add_linear(
                &mut graph,
                64,
                32,
                tensor_from_shape(&[64, 32], 0.11),
                tensor_from_shape(&[32], 0.23),
            )
            .expect("add linear");

        let input = graph.constant(tensor_from_shape(&[32, 64], 0.41));
        let persistent = graph.node_count();

        b.iter(|| {
            graph.zero_grads();
            graph.truncate(persistent).expect("truncate");
            let out = model.forward(&mut graph, input).expect("forward");
            let first = graph.value(out).expect("value").data()[0];
            black_box(first);
        });
    });

    group.bench_function("linear_relu_linear_batch32", |b| {
        let mut graph = Graph::new();
        let mut model = SequentialModel::new(&graph);
        model
            .add_linear(
                &mut graph,
                64,
                64,
                tensor_from_shape(&[64, 64], 0.05),
                tensor_from_shape(&[64], 0.31),
            )
            .expect("add linear 1");
        model.add_relu();
        model
            .add_linear(
                &mut graph,
                64,
                32,
                tensor_from_shape(&[64, 32], 0.79),
                tensor_from_shape(&[32], 0.17),
            )
            .expect("add linear 2");

        let input = graph.constant(tensor_from_shape(&[32, 64], 0.52));
        let persistent = graph.node_count();

        b.iter(|| {
            graph.zero_grads();
            graph.truncate(persistent).expect("truncate");
            let out = model.forward(&mut graph, input).expect("forward");
            let first = graph.value(out).expect("value").data()[0];
            black_box(first);
        });
    });

    group.finish();
}

fn bench_train_step_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_train_step_modes");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));

    group.bench_function("sgd_step_batch16", |b| {
        let mut graph = Graph::new();
        let mut model = SequentialModel::new(&graph);
        model
            .add_linear(
                &mut graph,
                64,
                32,
                tensor_from_shape(&[64, 32], 0.09),
                tensor_from_shape(&[32], 0.13),
            )
            .expect("add linear");
        let trainable = model.trainable_nodes();

        let input = graph.constant(tensor_from_shape(&[16, 64], 0.27));
        let target = graph.constant(tensor_from_shape(&[16, 32], 0.44));
        let persistent = graph.node_count();

        let mut optimizer = Sgd::new(0.01).expect("optimizer");

        b.iter(|| {
            graph.zero_grads();
            graph.truncate(persistent).expect("truncate");
            let pred = model.forward(&mut graph, input).expect("forward");
            let loss = train_step_sgd(&mut graph, &mut optimizer, pred, target, &trainable)
                .expect("train step");
            black_box(loss);
        });
    });

    group.bench_function("sgd_step_batch64", |b| {
        let mut graph = Graph::new();
        let mut model = SequentialModel::new(&graph);
        model
            .add_linear(
                &mut graph,
                64,
                32,
                tensor_from_shape(&[64, 32], 0.61),
                tensor_from_shape(&[32], 0.19),
            )
            .expect("add linear");
        let trainable = model.trainable_nodes();

        let input = graph.constant(tensor_from_shape(&[64, 64], 0.33));
        let target = graph.constant(tensor_from_shape(&[64, 32], 0.74));
        let persistent = graph.node_count();

        let mut optimizer = Sgd::new(0.01).expect("optimizer");

        b.iter(|| {
            graph.zero_grads();
            graph.truncate(persistent).expect("truncate");
            let pred = model.forward(&mut graph, input).expect("forward");
            let loss = train_step_sgd(&mut graph, &mut optimizer, pred, target, &trainable)
                .expect("train step");
            black_box(loss);
        });
    });

    group.bench_function("sgd_step_batch16_hinge", |b| {
        let mut graph = Graph::new();
        let mut model = SequentialModel::new(&graph);
        model
            .add_linear(
                &mut graph,
                64,
                32,
                tensor_from_shape(&[64, 32], 0.47),
                tensor_from_shape(&[32], 0.29),
            )
            .expect("add linear");
        let trainable = model.trainable_nodes();

        let input = graph.constant(tensor_from_shape(&[16, 64], 0.14));
        let mut target_data = tensor_from_shape(&[16, 32], 0.73).data().to_vec();
        for value in &mut target_data {
            *value = if *value >= 0.5 { 1.0 } else { -1.0 };
        }
        let target =
            graph.constant(Tensor::from_vec(vec![16, 32], target_data).expect("target tensor"));
        let persistent = graph.node_count();

        let mut optimizer = Sgd::new(0.01).expect("optimizer");

        b.iter(|| {
            graph.zero_grads();
            graph.truncate(persistent).expect("truncate");
            let pred = model.forward(&mut graph, input).expect("forward");
            let loss = train_step_sgd_with_loss(
                &mut graph,
                &mut optimizer,
                pred,
                target,
                &trainable,
                SupervisedLoss::Hinge { margin: 1.0 },
            )
            .expect("train step hinge");
            black_box(loss);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_forward_modes, bench_train_step_modes);
criterion_main!(benches);
