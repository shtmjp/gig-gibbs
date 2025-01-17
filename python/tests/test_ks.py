import gig_gibbs
import numpy as np
import scipy.stats as stats


def test_comparison_with_scipy_KS():
    # Parameters
    p = -0.5
    a = 1
    b = 1
    nsim = 1000
    seed = 42

    # Generate samples
    gibbs_sample = gig_gibbs.gig_sample(p, a, b, nsim, seed, n_burn_in=nsim)
    scipy_params = gig_gibbs.convert_to_scipy_params(p, a, b)
    scipy_params = {
        "p": scipy_params[0],
        "b": scipy_params[1],
        "scale": scipy_params[2],
    }

    # Perform KS test
    ks_stat, ks_pval = stats.kstest(
        gibbs_sample, lambda x: stats.geninvgauss.cdf(x, **scipy_params)
    )
    print(f"KS statistic: {ks_stat}")
    print(f"KS p-value: {ks_pval}")


def test_comparison_moments():
    # Parameters
    p = -0.5
    a = 1
    b = 1
    nsim = 500000
    seed = 42

    # Generate samples
    gibbs_sample = gig_gibbs.gig_sample(p, a, b, nsim, seed, n_burn_in=nsim)
    scipy_params = gig_gibbs.convert_to_scipy_params(p, a, b)
    scipy_params = {
        "p": scipy_params[0],
        "b": scipy_params[1],
        "scale": scipy_params[2],
    }

    scipy_sample = np.array(
        stats.geninvgauss.rvs(**scipy_params, size=nsim, random_state=seed)
    )

    m, v = stats.geninvgauss.stats(**scipy_params, moments="mv")
    print(
        f"Theoretical mean: {m}, Gibbs mean: {gibbs_sample.mean()}, Scipy mean: {scipy_sample.mean()}"
    )
    print(
        f"Theoretical variance: {v}, Gibbs variance: {gibbs_sample.var()}, Scipy variance: {scipy_sample.var()}"
    )
