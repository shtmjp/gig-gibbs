import sys
import time
from contextlib import contextmanager

from scipy import stats

sys.path.append("../gig-gibbs/python")
from gig_gibbs import convert_to_scipy_params, gig_sample


@contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    print("{}: {:.3f}".format(label, end - start))


def main():
    # Parameters
    p = -0.9
    a = 1.3
    b = 0.5
    nsim = 1000000
    seed = 42

    scipy_params = convert_to_scipy_params(p, a, b)
    scipy_params = {
        "p": scipy_params[0],
        "b": scipy_params[1],
        "scale": scipy_params[2],
    }

    with timer("Gibbs sampling"):
        gig_sample(p, a, b, nsim, seed, n_burn_in=0)
    with timer("Scipy sampling"):
        stats.geninvgauss.rvs(**scipy_params, size=nsim, random_state=seed)


if __name__ == "__main__":
    main()
