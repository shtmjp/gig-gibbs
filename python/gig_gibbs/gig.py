import gig_gibbs_rs
import numpy as np


def gig_sample(
    p: float, a: float, b: float, n: int, seed: int | None = None, n_burn_in: int = 0
) -> np.ndarray:
    """Generate samples from the Generalized Inverse Gaussian (GIG) distribution using Gibbs sampling.

    Args:
        p (float): The parameter p
        a (float): The parameter a
        b (float): The parameter b
        n (int): The number of samples to generate
        seed (int | None, optional): Random seed to be passed to a random number generator in Rust. Defaults to None.
        n_burn_in (int, optional): The Burn in period. Defaults to 0.

    Returns:
        np.ndarray: The generated samples. length = n

    Note:
    The parametrization is based on the paper below and same as Wikipedia (checked on 2025-01-17, https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution):

    Reference: Peña, V., & Jauch, M. Gibbs Sampling, Exact Sampling, and Distribution Function Evaluation for the Generalized Inverse Gaussian Distribution. arXiv preprint arXiv:2401.00749. (2021).
    https://arxiv.org/abs/2401.00749
    """
    sample = gig_gibbs_rs.gig_sample(p, a, b, n + n_burn_in, seed)
    sample = np.array(sample)
    return sample[n_burn_in:]


def convert_to_scipy_params(p: float, a: float, b: float) -> tuple:
    """Convert the GIG parameters in the paper (or Wikipedia on 2025-01-17) to the scipy GIG parameters.

    Args:
        p (float): The parameter p
        a (float): The parameter a
        b (float): The parameter b

    Returns:
        tuple: The scipy GIG parameters (p, b, scale). (except location)
    """
    # b_ / scale = a
    # b_ * scale = b
    # -> scale = sqrt(b / a)
    # -> b_ = a * sqrt(b / a) = sqrt(a * b)

    scale = np.sqrt(b / a)
    b_ = np.sqrt(a * b)
    return p, b_, scale
