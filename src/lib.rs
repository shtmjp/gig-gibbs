// use pyo3::prelude::*;
extern crate rand;
extern crate rand_distr;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Gamma, InverseGaussian};
use std::time;

/// Data-augmented composition Gibbs sampler for GIG(p, a, b)
#[pyfunction]
fn gig_sample(p: f64, a: f64, b: f64, nsim: usize, seed: Option<u64>) -> Vec<f64> {
    assert!(a > 0.0 && b > 0.0, "a and b must be positive");

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut x = vec![0.0; nsim];

    if p < -0.5 {
        let mut y = -(p + 0.5) * (b / a).sqrt();

        for i in 0..nsim {
            let lambda = b / (a + 2.0 * y);
            let inv_gauss = InverseGaussian::new(1.0 / lambda.sqrt(), b).unwrap();
            x[i] = inv_gauss.sample(&mut rng);

            let gamma = Gamma::new(-(p + 0.5), x[i]).unwrap();
            y = gamma.sample(&mut rng);
        }
    } else if (p - (-0.5)).abs() < f64::EPSILON {
        for i in 0..nsim {
            let inv_gauss = InverseGaussian::new((b / a).sqrt(), a).unwrap();
            x[i] = inv_gauss.sample(&mut rng);
        }
    } else {
        let mut y = (p + 0.5) * (b / a).sqrt();

        for i in 0..nsim {
            let lambda = (b + 2.0 * y) / a;
            let inv_gauss = InverseGaussian::new(lambda.sqrt(), b + 2.0 * y).unwrap();
            x[i] = inv_gauss.sample(&mut rng);

            let gamma = Gamma::new(p + 0.5, 1.0 / x[i]).unwrap();
            y = gamma.sample(&mut rng);
        }
    }

    x
}

/// A Python module implemented in Rust.
#[pymodule]
fn gig_gibbs_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gig_sample, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gig_sampler() {
        let p = -0.7;
        let a = 2.0;
        let b = 3.0;
        let nsim = 100;

        // Use a fixed seed for reproducibility
        let seed = Some(42);
        let samples = gig_sample(p, a, b, nsim, seed);

        assert_eq!(samples.len(), nsim);
        assert!(samples.iter().all(|&x| x > 0.0)); // All samples should be positive

        // Generate the same samples again to verify reproducibility
        let samples_repeated = gig_sample(p, a, b, nsim, seed);
        assert_eq!(samples, samples_repeated);
    }

    #[test]
    fn test_time() {
        let p = -0.9;
        let a = 1.3;
        let b = 0.5;
        let nsim = 1000000;
        let seed = Some(42);

        let now = time::Instant::now();
        gig_sample(p, a, b, nsim, seed);
        println!("{:?}", now.elapsed());
    }
}
