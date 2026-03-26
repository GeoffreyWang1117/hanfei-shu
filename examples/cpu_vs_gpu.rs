// Comprehensive CPU vs GPU comparison across all scales.
//
// Compares three implementations:
// 1. hanfei-shu GPU (CUDA, auto-dispatch)
// 2. hanfei-shu CPU Pippenger (pure Rust, single-threaded)
// 3. hanfei-shu naive MSM (scalar mul + accumulate)
//
// Run: cargo run --release --example cpu_vs_gpu

use ff::Field;
use group::{Curve, Group};
use pasta_curves::pallas;
use hanfei_shu::{gpu_best_multiexp, is_gpu_available};
use hanfei_shu::cpu::{pippenger_msm, naive_msm};
use rand_core::OsRng;
use std::time::Instant;

type Scalar = pallas::Scalar;
type Affine = pallas::Affine;
type Point = pallas::Point;

fn bench(_label: &str, f: impl Fn() -> Point, iters: usize) -> (f64, Point) {
    // Warmup
    let r = f();
    let mut best = f64::INFINITY;
    let mut result = r;
    for _ in 0..iters {
        let t = Instant::now();
        result = f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best { best = ms; }
    }
    (best, result)
}

fn main() {
    println!("================================================================");
    println!("  hanfei-shu: CPU vs GPU Comprehensive Comparison");
    println!("================================================================\n");
    println!("GPU: {}\n", if is_gpu_available() { "NVIDIA CUDA (auto-detected)" } else { "not available" });

    println!("{:<8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8}",
             "k", "n", "Naive(ms)", "Pip(ms)", "GPU(ms)", "GPU/Pip", "Correct");
    println!("{}", "-".repeat(76));

    for k in [8, 10, 12, 14, 15, 16, 17, 18] {
        let n: usize = 1 << k;
        let scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let bases: Vec<Affine> = (0..n)
            .map(|_| Point::random(OsRng).to_affine())
            .collect();

        let iters = if k <= 12 { 3 } else { 1 };

        // Naive (skip for large n — too slow)
        let naive_ms = if k <= 14 {
            let (ms, _) = bench("naive", || naive_msm(&scalars, &bases), iters);
            ms
        } else {
            f64::NAN
        };

        // CPU Pippenger
        let (pip_ms, pip_result) = bench("pippenger", || pippenger_msm(&scalars, &bases), iters);

        // GPU (auto-dispatch: CPU fallback below threshold, GPU above)
        let (gpu_ms, gpu_result) = bench("gpu", || gpu_best_multiexp(&scalars, &bases), iters);

        let correct = pip_result.to_affine() == gpu_result.to_affine();
        let speedup = pip_ms / gpu_ms;

        if naive_ms.is_nan() {
            println!("{:<8} {:>10} {:>10} {:>9.1} {:>9.1} {:>11.2}x {:>8}",
                     k, n, "—", pip_ms, gpu_ms, speedup,
                     if correct { "OK" } else { "FAIL" });
        } else {
            println!("{:<8} {:>10} {:>9.1} {:>9.1} {:>9.1} {:>11.2}x {:>8}",
                     k, n, naive_ms, pip_ms, gpu_ms, speedup,
                     if correct { "OK" } else { "FAIL" });
        }
    }

    println!("\n  Legend:");
    println!("  Naive   = scalar-mul + accumulate (O(n×256) group ops)");
    println!("  Pip     = Pippenger algorithm (O(n/w + 2^w) group ops)");
    println!("  GPU     = hanfei-shu CUDA MSM (Pippenger + GPU parallelism)");
    println!("  GPU/Pip = speedup of GPU over CPU Pippenger (>1 = GPU faster)");
}
