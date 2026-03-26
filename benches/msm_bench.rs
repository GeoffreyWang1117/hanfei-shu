use group::Curve;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use group::{Group};
use pasta_curves::pallas;
use hanfei_shu::gpu_best_multiexp;
use rand_core::OsRng;

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("pallas_msm");

    for k in [14, 17, 19] {
        let n = 1usize << k;
        let scalars: Vec<_> = (0..n).map(|_| pallas::Scalar::random(OsRng)).collect();
        let bases: Vec<_> = (0..n)
            .map(|_| pallas::Point::random(OsRng).to_affine())
            .collect();

        group.bench_with_input(BenchmarkId::new("gpu_auto", k), &k, |b, _| {
            b.iter(|| gpu_best_multiexp(&scalars, &bases));
        });

        group.bench_with_input(BenchmarkId::new("cpu_only", k), &k, |b, _| {
            b.iter(|| hanfei_shu::cpu_best_multiexp(&scalars, &bases));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_msm);
criterion_main!(benches);
