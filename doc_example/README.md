This is an example of Regression with white, Gaussian noise from PyMC3 notebook (ref https://docs.pymc.io/notebooks/GP-Marginal.html).

After generating 100 data points, its latent function (true function `f`) and its observed values (observed `y`), a Gaussian Process model is implemented.
With a series new values of `x`, and the Bayesian estimate of parameters in the GP model, several visulizations are realized:

- Posterior distribution over f(x) at the observed values

- Posterior predictive distribution, y_*

- Predictive mean and 2σ interval
