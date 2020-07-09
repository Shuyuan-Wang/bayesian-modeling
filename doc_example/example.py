# Regression with white, Gaussian noise
import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# DATA PREPARATION
# set the seed
np.random.seed(1)

n = 100  # The number of data points
X = np.linspace(0, 10, n)[:, None]  # The inputs to the GP, they must be arranged as a column vector

# Define the true covariance function and its parameters
ℓ_true = 1.0
η_true = 3.0
cov_func = η_true ** 2 * pm.gp.cov.Matern52(1, ℓ_true)

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
# Note that we have to call `eval()` because PyMC3 built on top of Theano
f_true = np.random.multivariate_normal(mean_func(X).eval(),
                                       cov_func(X).eval() + 1e-8 * np.eye(n), 1).flatten()

# The observed data is the latent function plus a small amount of IID Gaussian noise
# The standard deviation of the noise is `sigma`
σ_true = 2.0
y = f_true + σ_true * np.random.randn(n)

# Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.plot(X, f_true, "dodgerblue", lw=3, label="True f")
ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
ax.set_xlabel("X")
ax.set_ylabel("The true f(x)")
plt.legend()

# BAYESIAN MODELING
with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
    η = pm.HalfCauchy("η", beta=5)

    cov = η ** 2 * pm.gp.cov.Matern52(1, ℓ)
    gp = pm.gp.Marginal(cov_func=cov)

    σ = pm.HalfCauchy("σ", beta=5)
    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=σ)

    mp = pm.find_MAP()

# collect the results into a pandas dataframe to display
# "mp" stands for marginal posterior
pd.DataFrame({"Parameter": ["ℓ", "η", "σ"],
              "Value at MAP": [float(mp["ℓ"]), float(mp["η"]), float(mp["σ"])],
              "True value": [ℓ_true, η_true, σ_true]})

''' USING .CONDITIONAL, new curves '''
# new values from x=0 to x=20
X_new = np.linspace(0, 20, 600)[:, None]

# add the GP conditional to the model, given the new X values (without noise)
with model:
    f_pred = gp.conditional("f_pred", X_new)
    # To use the MAP values, you can just replace the trace with a length-1 list with `mp`
    pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=20)

# plot the results
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

#1 plot the 2000 samples by iteration
for c in pred_samples['f_pred']:
    plt.plot(X_new,c,"gray",alpha=0.1)

#2 plot the 2000 samples by package function
# plot the samples from the gp posterior with samples and shading
# from pymc3.gp.util import plot_gp_dist
# plot_gp_dist(ax, pred_samples["f_pred"], X_new)

# plot the data and the true latent function
plt.plot(X, f_true, "dodgerblue", lw=3, label="True f")
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data")

# axis labels and title
plt.xlabel("X")
plt.ylim([-13, 13])
plt.title("Posterior distribution over $f(x)$ at the observed values")
plt.legend()

# WITH NOISE
with model:
    y_pred = gp.conditional("y_pred", X_new, pred_noise=True)
with model:
    y_samples = pm.sample_posterior_predictive([mp], vars=[y_pred], samples=15)

fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
# posterior predictive distribution

#1 plot by package function
# from pymc3.gp.util import plot_gp_dist
# plot_gp_dist(ax, y_samples["y_pred"], X_new, plot_samples=False, palette="bone_r");

#2 plot by iteration
for c in y_samples['y_pred']:
    plt.plot(X_new, c, "gray", alpha=0.1)

# overlay a scatter of one draw of random points from the
#   posterior predictive distribution
plt.plot(X_new, y_samples["y_pred"][10, :].T, "co", ms=2, label="Predicted data")

# plot original data and true function
plt.plot(X, y, 'ok', ms=3, alpha=1.0, label="observed data")
plt.plot(X, f_true, "dodgerblue", lw=3, label="true f")
plt.xlabel("x")
plt.ylim([-13, 13])
plt.title("posterior predictive distribution, y_*")
plt.legend()

''' USING .PREDICT '''
# predict without noise
mu, var = gp.predict(X_new, point=mp, diag=True, pred_noise=False)
sd = np.sqrt(var)

# draw plot
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

# plot mean and 2σ intervals
plt.plot(X_new, mu, 'r', lw=2, label="mean and 2σ region")
plt.plot(X_new, mu + 2 * sd, 'r', lw=1)
plt.plot(X_new, mu - 2 * sd, 'r', lw=1)
plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)

# plot original data and true function
plt.plot(X, y, 'ok', ms=3, alpha=1.0, label="observed data")
plt.plot(X, f_true, "dodgerblue", lw=3, label="true f")

plt.xlabel("x")
plt.ylim([-13, 13])
plt.title("predictive mean and 2σ interval without noise")
plt.legend()

# predict with noise
mu, var = gp.predict(X_new, point=mp, diag=True, pred_noise=True)
sd = np.sqrt(var)

# draw plot
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

# plot mean and 2σ intervals
plt.plot(X_new, mu, 'r', lw=2, label="mean and 2σ region")
plt.plot(X_new, mu + 2 * sd, 'r', lw=1)
plt.plot(X_new, mu - 2 * sd, 'r', lw=1)
plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)

# plot original data and true function
plt.plot(X, y, 'ok', ms=3, alpha=1.0, label="observed data")
plt.plot(X, f_true, "dodgerblue", lw=3, label="true f")

plt.xlabel("x")
plt.ylim([-13, 13])
plt.title("predictive mean and 2σ interval with noise")
plt.legend()
