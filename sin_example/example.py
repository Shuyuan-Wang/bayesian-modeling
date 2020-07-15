# initialization and generate data
import numpy as np
import scipy.stats
import scipy.special

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm

import pandas as pd
import seaborn as sns

sns.set()

import pymc3 as pm

"""generate training data"""
np.random.seed(708)
x = np.linspace(0.5, np.pi * 2, 30)
# 0.5:amplitude of noise
y = np.sin(x) + np.random.rand(30) * 0.5 - 0.25
true_y = np.sin(x)
plt.figure(figsize=(4, 3))
plt.xlim(0, np.pi * 2)
plt.scatter(x, y, marker='+', c='r')
plt.plot(x, true_y, 'b')
plt.show()

# pymc modeling
with pm.Model() as model:
    amp = pm.HalfCauchy("amp", 1)
    ls = pm.HalfCauchy("ls", 1)
    cov_func = amp ** 2 * pm.gp.cov.ExpQuad(1, ls)  # input_dim=1,ls=ls
    M = pm.gp.mean.Linear(coeffs=(y / x).mean())
    gp = pm.gp.Marginal(M, cov_func)
    noise = pm.HalfCauchy("noise", 2)
    gp.marginal_likelihood("f", X=x.reshape(-1, 1), y=y, noise=noise)
    trace = pm.sample(1000, chains=1)

map_ = pm.find_MAP(model=model)
X_new = np.linspace(0, np.pi * 2, 150).reshape(-1, 1)

# .predict method: return the mean and variance given a particular point
mu, var = gp.predict(X_new, point=map_, diag=True, pred_noise=True)
sd = np.sqrt(var)

# plot
# draw plot
plt.figure(figsize=(4, 3))
# plot mean and 2σ intervals
plt.ylim(-2, 2)
plt.xlim(0, np.pi * 2)
plt.plot(X_new, mu, lw=2, c='r', label="mean and 2σ region")
plt.plot(X_new, mu - 2 * sd, lw=1, c='r')
plt.plot(X_new, mu + 2 * sd, lw=1, c='r')
plt.fill_between(X_new.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)
# plt true y
plt.plot(x, true_y, "k", label="True y")
# plot original data and true function
plt.scatter(x, y, alpha=1, c="b", label="observed data", marker='+')
plt.title("predictive mean and 2σ interval")
plt.legend(loc='best', fontsize=8)
plt.show()
plt.savefig('./plots/predictive_mean_2sigma.png')

# generate new curves (with noise)
with model:
    y_pred = gp.conditional("y_pred", X_new, pred_noise=True)
with model:
    sample_pred = pm.sample_posterior_predictive(trace, vars=[y_pred], samples=20)

# draw plot
fig = plt.figure(figsize=(4, 3));
plt.ylim(-2, 2)
plt.xlim(0, np.pi * 2)
# plot generated curves
for c in sample_pred['y_pred']:
    plt.plot(X_new, c, "gray", alpha=0.1)
# plot original data and true function
plt.scatter(x, y, alpha=1, c="b", label="observed data", marker='+')
plt.title("predictive posterior distribution with noise")
plt.legend(loc='best', fontsize=8)
plt.savefig('./plots/new_curves_noisy.png')


# generate new curve (without noise)
with model:
    f_pred = gp.conditional("f_pred", X_new, pred_noise=False)
with model:
    sample_f_pred = pm.sample_posterior_predictive(trace, vars=[f_pred], samples=20)

# draw plot
fig = plt.figure(figsize=(4, 3));
plt.ylim(-2, 2)
plt.xlim(0, np.pi * 2)
# plot generated curves
for c in sample_f_pred['f_pred']:
    plt.plot(X_new, c, "gray", alpha=0.1)
# plot original data and true function
plt.scatter(x, y, alpha=1, c="b", label="observed data", marker='+')
plt.title("predictive posterior without noise")
plt.legend(loc='best', fontsize=8)
plt.savefig('./plots/new_curves_noise_free.png')