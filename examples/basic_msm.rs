use group::Curve;
//! Basic example: compute an MSM on the Pallas curve using GPU acceleration.

use ff::Field;
use group::{Group};
use pasta_curves::pallas;
use hanfei_shu::{gpu_best_multiexp, is_gpu_available};
use rand_core::OsRng;
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("pallas-gpu-msm basic example");
    println!("GPU available: {}", is_gpu_available());

    for k in [10, 14, 17] {
        let n = 1 << k;
        println!("\n--- n = {} (k={}) ---", n, k);

        // Generate random scalars and bases
        let scalars: Vec<_> = (0..n).map(|_| pallas::Scalar::random(OsRng)).collect();
        let bases: Vec<_> = (0..n)
            .map(|_| pallas::Point::random(OsRng).to_affine())
            .collect();

        // GPU/auto MSM
        let t0 = Instant::now();
        let result = gpu_best_multiexp(&scalars, &bases);
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // CPU baseline
        let t1 = Instant::now();
        let cpu_result = hanfei_shu::cpu_best_multiexp(&scalars, &bases);
        let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let matches = result.to_affine() == cpu_result.to_affine();
        let speedup = cpu_ms / gpu_ms;

        println!(
            "  GPU: {:.1}ms | CPU: {:.1}ms | speedup: {:.2}x | match: {}",
            gpu_ms, cpu_ms, speedup, matches
        );
    }
}
