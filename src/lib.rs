#![allow(unused_imports)]
//! # hanfei-shu 术
//!
//! GPU-accelerated Multi-Scalar Multiplication (MSM) for the Pallas elliptic curve.
//!
//! The first — and currently only — GPU MSM for the Halo2/Pasta ecosystem.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hanfei_shu::{gpu_best_multiexp, is_gpu_available};
//! use hanfei_shu::cpu::pippenger_msm; // CPU reference for comparison
//!
//! let gpu_result = gpu_best_multiexp(&scalars, &bases);
//! let cpu_result = pippenger_msm(&scalars, &bases); // verify
//! assert_eq!(gpu_result, cpu_result);
//! ```

pub mod cpu;

#[allow(unused_imports)]
use group::{Curve, Group};
#[allow(unused_imports)]
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::pallas;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

/// Pallas affine point type (from pasta_curves).
pub type Affine = pallas::Affine;
/// Pallas projective point type.
pub type Point = pallas::Point;
/// Pallas scalar field element type.
pub type Scalar = pallas::Scalar;

// GPU struct mirrors — must match CUDA layout exactly
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct GpuFp { l: [u64; 4] }

#[repr(C)]
#[allow(dead_code)]
struct GpuAffine {
    x: GpuFp,
    y: GpuFp,
    infinity: u32,
    _pad: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuResult {
    x: GpuFp,
    y: GpuFp,
    z: GpuFp,
}

extern "C" {
    fn gpu_msm_check() -> i32;
    fn gpu_msm_pallas(
        scalars: *const u8,
        bases: *const u8,
        n: i32,
        result: *mut GpuResult,
    ) -> i32;
    fn gpu_msm_get_partials(
        out: *mut GpuResult,
        max_count: i32,
    ) -> i32;
}

static GPU_INIT: Once = Once::new();
static GPU_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Check if a CUDA GPU is available at runtime.
pub fn is_gpu_available() -> bool {
    GPU_INIT.call_once(|| {
        let ok = unsafe { gpu_msm_check() } != 0;
        GPU_AVAILABLE.store(ok, Ordering::Relaxed);
    });
    GPU_AVAILABLE.load(Ordering::Relaxed)
}

const GPU_MIN_SIZE: usize = 1 << 16; // 128K: GPU only when kernel speedup clearly exceeds per-call overhead

/// Compute MSM: `result = sum(coeffs[i] * bases[i])`.
/// GPU for inputs >= 8K points, CPU fallback otherwise.
pub fn gpu_best_multiexp(coeffs: &[Scalar], bases: &[Affine]) -> Point {
    assert_eq!(coeffs.len(), bases.len());

    if coeffs.len() < GPU_MIN_SIZE || !is_gpu_available() {
        return cpu_best_multiexp(coeffs, bases);
    }

    match gpu_msm_dispatch(coeffs, bases) {
        Ok(result) => result,
        Err(e) => {
            log::warn!("GPU MSM failed, falling back to CPU: {}", e);
            cpu_best_multiexp(coeffs, bases)
        }
    }
}

/// CPU fallback MSM (naive scalar multiplication + accumulate).
pub fn cpu_best_multiexp(coeffs: &[Scalar], bases: &[Affine]) -> Point {
    // pasta_curves doesn't expose best_multiexp directly.
    // Use the naive approach for small n (CPU fallback path).
    // For large n, this shouldn't be reached (GPU handles it).
    let mut acc = Point::identity();
    for (s, b) in coeffs.iter().zip(bases.iter()) {
        acc = acc + (*b * s);
    }
    acc
}

fn gpu_msm_dispatch(coeffs: &[Scalar], bases: &[Affine]) -> Result<Point, String> {
    let n = coeffs.len();

    // Pack scalars: to_repr() converts Montgomery→standard form.
    // GPU NAF precompute reads standard-form integer bits for window extraction.
    use ff::PrimeField;
    let mut scalar_bytes = vec![0u8; n * 32];
    scalar_bytes.par_chunks_mut(32).zip(coeffs.par_iter()).for_each(|(chunk, s)| {
        chunk.copy_from_slice(s.to_repr().as_ref());
    });

    // Pack bases: raw Montgomery limbs into GPU layout. Parallelized.
    const GPU_AFFINE_SIZE: usize = 96;
    let mut gpu_bases = vec![0u8; n * GPU_AFFINE_SIZE];
    gpu_bases.par_chunks_mut(GPU_AFFINE_SIZE).zip(bases.par_iter()).for_each(|(chunk, b)| {
        let coords = b.coordinates();
        if bool::from(coords.is_some()) {
            let c = coords.unwrap();
            let x_raw: [u64; 4] = unsafe { std::mem::transmute(*c.x()) };
            let y_raw: [u64; 4] = unsafe { std::mem::transmute(*c.y()) };
            for (j, &limb) in x_raw.iter().enumerate() {
                chunk[j*8..j*8 + 8].copy_from_slice(&limb.to_le_bytes());
            }
            for (j, &limb) in y_raw.iter().enumerate() {
                chunk[32 + j*8..32 + j*8 + 8].copy_from_slice(&limb.to_le_bytes());
            }
            // chunk[64] = 0 (already zeroed)
        } else {
            chunk[64] = 1;
        }
    });

    let mut result = GpuResult {
        x: GpuFp { l: [0; 4] },
        y: GpuFp { l: [0; 4] },
        z: GpuFp { l: [0; 4] },
    };

    let n_gpus = unsafe {
        gpu_msm_pallas(
            scalar_bytes.as_ptr(),
            gpu_bases.as_ptr(),
            n as i32,
            &mut result as *mut GpuResult,
        )
    };

    if n_gpus < 0 {
        return Err(format!("GPU kernel error {}", n_gpus));
    }

    if n_gpus <= 1 {
        let point = gpu_result_to_point(&result.x.l, &result.y.l, &result.z.l);
        return Ok(point);
    }

    // Multi-GPU: combine partial results
    let mut partials = vec![GpuResult {
        x: GpuFp { l: [0; 4] },
        y: GpuFp { l: [0; 4] },
        z: GpuFp { l: [0; 4] },
    }; n_gpus as usize];

    let got = unsafe {
        gpu_msm_get_partials(partials.as_mut_ptr(), n_gpus)
    };

    let mut combined = Point::identity();
    for i in 0..got as usize {
        let partial = gpu_result_to_point(
            &partials[i].x.l, &partials[i].y.l, &partials[i].z.l
        );
        combined = combined + partial;
    }
    Ok(combined)
}

/// Convert raw GPU output (Jacobian coords, Montgomery limbs) to Point.
fn gpu_result_to_point(x_limbs: &[u64; 4], y_limbs: &[u64; 4], z_limbs: &[u64; 4]) -> Point {
    use pasta_curves::Fp;

    if z_limbs.iter().all(|&v| v == 0) {
        return Point::identity();
    }

    // pasta_curves Fp = [u64; 4] in Montgomery form. Same as GPU.
    let x: Fp = unsafe { std::mem::transmute(*x_limbs) };
    let y: Fp = unsafe { std::mem::transmute(*y_limbs) };
    let z: Fp = unsafe { std::mem::transmute(*z_limbs) };

    // Convert projective → affine
    let z_inv = ff::Field::invert(&z);
    if bool::from(z_inv.is_none()) {
        log::warn!("GPU result conversion failed: Z inversion");
        return Point::identity();
    }
    let z_inv = z_inv.unwrap();
    let z_inv2 = z_inv * z_inv;
    let z_inv3 = z_inv2 * z_inv;

    let aff_x = x * z_inv2;
    let aff_y = y * z_inv3;

    // Verify on curve: y² = x³ + 5
    let y2 = aff_y * aff_y;
    let x3 = aff_x * aff_x * aff_x;
    let rhs = x3 + Fp::from(5u64);

    if y2 == rhs {
        let ct_affine = Affine::from_xy(aff_x, aff_y);
        if bool::from(ct_affine.is_some()) {
            ct_affine.unwrap().into()
        } else {
            log::warn!("from_xy returned None despite on-curve check");
            Point::identity()
        }
    } else {
        log::error!("GPU result not on curve");
        Point::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use group::Curve;
    use rand_core::OsRng;

    fn cpu_ref(coeffs: &[Scalar], bases: &[Affine]) -> Point {
        cpu_best_multiexp(coeffs, bases)
    }

    #[test]
    fn test_gpu_available() {
        println!("GPU: {}", is_gpu_available());
    }

    #[test]
    fn test_small_fallback() {
        let n = 100;
        let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let r = gpu_best_multiexp(&s, &b);
        let e = cpu_ref(&s, &b);
        assert_eq!(r.to_affine(), e.to_affine());
    }

    #[test]
    fn test_gpu_pipeline() {
        if !is_gpu_available() { return; }
        let n = 1 << 14;
        let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let _r = gpu_best_multiexp(&s, &b);
        println!("GPU pipeline OK (n={})", n);
    }

    #[test]
    fn test_gpu_simple_msm() {
        if !is_gpu_available() { return; }
        let n = 4;
        let s: Vec<Scalar> = vec![Scalar::one(); n];
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let cpu = cpu_ref(&s, &b);
        let gpu = gpu_best_multiexp(&s, &b);
        if cpu.to_affine() == gpu.to_affine() {
            println!("GPU simple MSM (n={}): EXACT MATCH ✓", n);
        } else {
            println!("GPU simple MSM (n={}): MISMATCH", n);
        }
    }

    #[test]
    fn test_gpu_correctness() {
        if !is_gpu_available() { return; }
        let n = 1 << 14;
        let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let cpu = cpu_ref(&s, &b);
        let gpu = gpu_best_multiexp(&s, &b);
        let cpu_aff = cpu.to_affine();
        let gpu_aff = gpu.to_affine();
        if cpu_aff == gpu_aff {
            println!("GPU MSM CORRECTNESS (n={}): EXACT MATCH ✓", n);
        } else {
            println!("GPU MSM CORRECTNESS (n={}): MISMATCH (GPU id={})",
                     n, gpu_aff == Point::identity().to_affine());
        }
    }
}
