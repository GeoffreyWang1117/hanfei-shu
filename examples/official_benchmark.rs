use group::Curve;
//! Official benchmark for comparison with nanoZkinference/ChainProve results.
//!
//! Outputs JSON compatible with experiments/results/proof/gpu_msm_final.json
//! Run with: cargo run --release --example official_benchmark

use ff::Field;
use group::{Group};
use pasta_curves::pallas;
use hanfei_shu::{gpu_best_multiexp, is_gpu_available};
use rand_core::OsRng;
use std::time::Instant;

fn main() {
    eprintln!("================================================================");
    eprintln!("  pallas-gpu-msm: Official Benchmark (nanoZkinference compatible)");
    eprintln!("================================================================\n");

    let gpu_avail = is_gpu_available();
    eprintln!("GPU available: {}", gpu_avail);

    // Warmup
    if gpu_avail {
        eprintln!("Warming up GPU (2 rounds at k=14)...");
        let n = 1 << 14;
        let s: Vec<_> = (0..n).map(|_| pallas::Scalar::random(OsRng)).collect();
        let b: Vec<_> = (0..n).map(|_| pallas::Point::random(OsRng).to_affine()).collect();
        let _ = gpu_best_multiexp(&s, &b);
        let _ = gpu_best_multiexp(&s, &b);
        eprintln!("Warmup complete.\n");
    }

    // Test sizes matching nanoZkinference experiments
    let test_ks = vec![14, 16, 17, 18, 19, 20, 21];
    let mut results = Vec::new();

    eprintln!("{:<6} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10} {:>8}",
             "k", "n", "cpu_avg", "cpu_best", "gpu_avg", "gpu_best", "speedup", "match");
    eprintln!("{}", "-".repeat(90));

    for k in &test_ks {
        let n: usize = 1 << k;
        let runs = if *k <= 17 { 5 } else if *k <= 19 { 3 } else { 3 };

        let scalars: Vec<_> = (0..n).map(|_| pallas::Scalar::random(OsRng)).collect();
        let bases: Vec<_> = (0..n)
            .map(|_| pallas::Point::random(OsRng).to_affine())
            .collect();

        // CPU benchmark
        let mut cpu_times = Vec::new();
        let mut cpu_result = pallas::Point::identity();
        for _ in 0..runs {
            let t = Instant::now();
            cpu_result = hanfei_shu::cpu_best_multiexp(&scalars, &bases);
            cpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let cpu_avg = cpu_times.iter().sum::<f64>() / runs as f64;
        let cpu_best = cpu_times.iter().cloned().fold(f64::INFINITY, f64::min);

        // GPU benchmark
        let mut gpu_times = Vec::new();
        let mut gpu_result = pallas::Point::identity();
        for _ in 0..runs {
            let t = Instant::now();
            gpu_result = gpu_best_multiexp(&scalars, &bases);
            gpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let gpu_avg = gpu_times.iter().sum::<f64>() / runs as f64;
        let gpu_best = gpu_times.iter().cloned().fold(f64::INFINITY, f64::min);

        let speedup_avg = cpu_avg / gpu_avg;
        let speedup_best = cpu_best / gpu_best;
        let correct = cpu_result.to_affine() == gpu_result.to_affine();

        eprintln!("{:<6} {:>10} {:>10.1}ms {:>10.1}ms {:>10.1}ms {:>10.1}ms {:>9.2}x {:>8}",
                 k, n, cpu_avg, cpu_best, gpu_avg, gpu_best, speedup_best,
                 if correct { "OK" } else { "FAIL" });

        results.push(serde_json::json!({
            "k": k,
            "n": n,
            "cpu_avg_ms": (cpu_avg * 10.0).round() / 10.0,
            "cpu_best_ms": (cpu_best * 10.0).round() / 10.0,
            "gpu_avg_ms": (gpu_avg * 10.0).round() / 10.0,
            "gpu_best_ms": (gpu_best * 10.0).round() / 10.0,
            "speedup": (speedup_best * 100.0).round() / 100.0,
            "runs": runs,
            "correct": correct
        }));
    }

    // Output JSON to stdout
    let output = serde_json::json!({
        "timestamp": chrono::Local::now().to_rfc3339(),
        "experiment": "hanfei_shu_v0.1.0_official",
        "description": "Official benchmark: pallas-gpu-msm v0.1.0 standalone crate. Best-of-N runs. Rust-native, no Python FFI overhead.",
        "hardware": {
            "cpu": "AMD Ryzen 9 5950X 16-Core (32 threads)",
            "gpu": "NVIDIA RTX 3090, 82 SMs, 24GB, sm_86, CUDA 13.0",
            "ram": "128 GB DDR4"
        },
        "baseline_comparison": {
            "source": "nanoZkinference/experiments/results/proof/gpu_msm_final.json",
            "note": "gpu_msm_final.json was measured on Intel Xeon 32T. This benchmark uses AMD Ryzen 9 5950X 32T."
        },
        "results": results
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
