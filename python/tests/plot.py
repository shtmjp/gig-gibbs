import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append("../gig-gibbs/python")
from gig_gibbs import convert_to_scipy_params, gig_sample

if __name__ == "__main__":
    params = {"p": -0.5, "a": 1, "b": 1}
    sample = gig_sample(**params, n=1000, seed=42, n_burn_in=10000)
    fig, ax = plt.subplots()
    ax.hist(sample, bins=50, density=True, alpha=0.6, color="g")
    scipy_params = convert_to_scipy_params(**params)
    scipy_params = {
        "p": scipy_params[0],
        "b": scipy_params[1],
        "scale": scipy_params[2],
    }
    x = np.linspace(0, 10, 1000)
    y = stats.geninvgauss.pdf(x, **scipy_params)
    ax.plot(x, y, "r", linewidth=2)
    plt.show()
