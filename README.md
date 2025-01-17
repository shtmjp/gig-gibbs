# gig-gibbs

Gibbs sampler for Generalized Inverse Gaussian (GIG) distribution.
See [arxiv:2401.00749](https://arxiv.org/abs/2401.00749) for details.

## Usage

```Python
import gig_gibbs

# Parameters
p, a, b = -0.7, 2, 1
nsim = 1000
n_burn_in = 500
seed = 42

# Generate samples
gibbs_sample = gig_gibbs.gig_sample(p, a, b, nsim, seed, n_burn_in)
```

## Note

-  Peña & Jauch (2024) にあるGIG分布のGibbs samplerをRustで実装し、Python wrapperを用意しました。
- `python/tests/test_ks.py`では、Gibbs samplingによって得た乱数列がscipy.stats.geninvgaussのGIGのcdfを用いたKS検定が棄却されないことを見て、確かにGIG分布からサンプリングできていることを確認しました。