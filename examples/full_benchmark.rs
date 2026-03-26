use group::Curve;
//! Comprehensive benchmark: GPU vs CPU MSM across multiple input sizes.
//!
//! Run with: cargo run --release --example full_benchmark

use ff::Field;
use group::{Group};
use pasta_curves::pallas;
use hanfei_shu::{gpu_best_multiexp, is_gpu_available};
use rand_core::OsRng;
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("  pallas-gpu-msm: Full Benchmark Suite");
    println!("================================================================\n");

    println!("GPU available: {}", is_gpu_available());
    println!();

    // Warmup GPU
    if is_gpu_available() {
        println!("Warming up GPU...");
        let n = 1 << 14;
        let s: Vec<_> = (0..n).map(|_| pallas::Scalar::random(OsRng)).collect();
        let b: Vec<_> = (0..n).map(|_| pallas::Point::random(OsRng).to_affine()).collect();
        let _ = gpu_best_multiexp(&s, &b);
        let _ = gpu_best_multiexp(&s, &b);
        println!("Warmup complete.\n");
    }

    println!("{:<8} {:>10} {:>10} {:>10} {:>10} {:>8}",
             "k", "n", "GPU (ms)", "CPU (ms)", "Speedup", "Match");
    println!("{}", "-".repeat(62));

    let test_sizes = vec![10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];

    for k in test_sizes {
        let n: usize = 1 << k;

        // Generate random data
        let scalars: Vec<_> = (0..n).map(|_| pallas::Scalar::random(OsRng)).collect();
        let bases: Vec<_> = (0..n)
            .map(|_| pallas::Point::random(OsRng).to_affine())
            .collect();

        // Number of iterations for averaging (fewer for large inputs)
        let iters = if k <= 14 { 5 } else if k <= 17 { 3 } else { 1 };

        // GPU/auto benchmark
        let mut gpu_total = 0.0f64;
        let mut gpu_result = pallas::Point::identity();
        for _ in 0..iters {
            let t = Instant::now();
            gpu_result = gpu_best_multiexp(&scalars, &bases);
            gpu_total += t.elapsed().as_secs_f64() * 1000.0;
        }
        let gpu_ms = gpu_total / iters as f64;

        // CPU benchmark
        let mut cpu_total = 0.0f64;
        let mut cpu_result = pallas::Point::identity();
        for _ in 0..iters {
            let t = Instant::now();
            cpu_result = hanfei_shu::cpu_best_multiexp(&scalars, &bases);
            cpu_total += t.elapsed().as_secs_f64() * 1000.0;
        }
        let cpu_ms = cpu_total / iters as f64;

        let speedup = cpu_ms / gpu_ms;
        let matches = gpu_result.to_affine() == cpu_result.to_affine();

        println!("{:<8} {:>10} {:>10.1} {:>10.1} {:>9.2}x {:>8}",
                 k, n, gpu_ms, cpu_ms, speedup,
                 if matches { "OK" } else { "MISMATCH" });
    }

    println!("\n================================================================");
    println!("  Notes:");
    println!("  - k < 13: GPU fallback to CPU (below 8K threshold)");
    println!("  - GPU times include host<->device transfer");
    println!("  - CPU uses hanfei_shu::cpu_best_multiexp (multi-threaded)");
    println!("================================================================");
}
