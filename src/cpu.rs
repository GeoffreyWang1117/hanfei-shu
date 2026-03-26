//! Pure-CPU Pippenger MSM for Pallas — reference implementation.
//!
//! This is a clean, correct, single-threaded Pippenger implementation
//! for benchmarking against the GPU version. It mirrors the algorithm
//! used by `halo2curves::msm::best_multiexp`.
//!
//! Users can use this to:
//! - Verify GPU results against a known-correct CPU baseline
//! - Benchmark GPU speedup on their hardware
//! - Run on machines without NVIDIA GPUs

use ff::PrimeField;
use group::{Curve, Group};
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::pallas;

type Affine = pallas::Affine;
type Point = pallas::Point;
type Scalar = pallas::Scalar;

/// Pure-CPU Pippenger MSM: `result = sum(coeffs[i] * bases[i])`.
///
/// Single-threaded, no GPU dependency. Suitable for correctness
/// verification and baseline benchmarking.
///
/// # Example
/// ```rust,ignore
/// use hanfei_shu::cpu::pippenger_msm;
/// let result = pippenger_msm(&scalars, &bases);
/// ```
pub fn pippenger_msm(coeffs: &[Scalar], bases: &[Affine]) -> Point {
    assert_eq!(coeffs.len(), bases.len());
    let n = coeffs.len();
    if n == 0 {
        return Point::identity();
    }
    if n < 4 {
        return naive_msm(coeffs, bases);
    }

    // Choose window size: w ≈ ln(n)
    let w = optimal_window(n);
    let n_windows = (255 + w - 1) / w;
    let n_buckets = (1 << w) - 1;

    // Convert scalars to standard form bytes
    let scalar_reprs: Vec<_> = coeffs.iter().map(|s| s.to_repr()).collect();

    let mut window_sums = Vec::with_capacity(n_windows);

    for win_idx in 0..n_windows {
        // Accumulate into buckets
        let mut buckets = vec![Point::identity(); n_buckets];
        let bit_offset = win_idx * w;

        for (i, base) in bases.iter().enumerate() {
            let scalar_bytes = scalar_reprs[i].as_ref();
            let bucket_idx = get_window_value(scalar_bytes, bit_offset, w);
            if bucket_idx > 0 {
                buckets[bucket_idx - 1] = buckets[bucket_idx - 1] + base;
            }
        }

        // Running sum to compute window contribution
        let mut running = Point::identity();
        let mut total = Point::identity();
        for j in (0..n_buckets).rev() {
            running = running + buckets[j];
            total = total + running;
        }

        window_sums.push(total);
    }

    // Horner combine: result = w_{k-1} + 2^w * (w_{k-2} + 2^w * (...))
    let mut result = window_sums[n_windows - 1];
    for win_idx in (0..n_windows - 1).rev() {
        for _ in 0..w {
            result = result.double();
        }
        result = result + window_sums[win_idx];
    }

    result
}

/// Naive MSM for small inputs: `sum(s_i * G_i)`.
pub fn naive_msm(coeffs: &[Scalar], bases: &[Affine]) -> Point {
    let mut acc = Point::identity();
    for (s, b) in coeffs.iter().zip(bases.iter()) {
        acc = acc + (*b * s);
    }
    acc
}

fn optimal_window(n: usize) -> usize {
    let ln_n = (n as f64).ln().ceil() as usize;
    ln_n.max(4).min(16)
}

fn get_window_value(scalar_bytes: &[u8], bit_offset: usize, window_bits: usize) -> usize {
    let mut val: u64 = 0;
    for bit in 0..window_bits {
        let global_bit = bit_offset + bit;
        if global_bit >= 256 {
            break;
        }
        let byte_idx = global_bit / 8;
        let bit_idx = global_bit % 8;
        if byte_idx < scalar_bytes.len() && (scalar_bytes[byte_idx] >> bit_idx) & 1 == 1 {
            val |= 1 << bit;
        }
    }
    val as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_pippenger_correctness() {
        for n in [1, 2, 4, 16, 100, 1000] {
            let s: Vec<Scalar> = (0..n).map(|_| Scalar::random(OsRng)).collect();
            let b: Vec<Affine> = (0..n)
                .map(|_| Point::random(OsRng).to_affine())
                .collect();
            let pip = pippenger_msm(&s, &b);
            let naive = naive_msm(&s, &b);
            assert_eq!(
                pip.to_affine(),
                naive.to_affine(),
                "Pippenger != naive at n={}",
                n
            );
        }
    }

    #[test]
    fn test_pippenger_edge_cases() {
        // All zeros
        let n = 100;
        let s = vec![Scalar::zero(); n];
        let b: Vec<Affine> = (0..n).map(|_| Point::random(OsRng).to_affine()).collect();
        let result = pippenger_msm(&s, &b);
        assert_eq!(result, Point::identity());

        // All ones
        let s = vec![Scalar::one(); n];
        let pip = pippenger_msm(&s, &b);
        let naive = naive_msm(&s, &b);
        assert_eq!(pip.to_affine(), naive.to_affine());

        // Empty
        let result = pippenger_msm(&[], &[]);
        assert_eq!(result, Point::identity());
    }
}
