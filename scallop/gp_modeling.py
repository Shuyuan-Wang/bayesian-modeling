import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import pymc3 as pm
from gp_plot import two_d_heatmap
from gp_plot import sliced_mean_2sig_region
from gp_plot import plot_sampled_curves

''' load data and transform data '''
scallop = pd.read_csv("/Users/wangshuyuan/PycharmProjects/GP/scallop/data/scallop.csv", index_col=0)
# log transformation
data = scallop.copy()
data["tot.catch"] = np.log(data["tot.catch"] + 0.25)
plt.hist(data["tot.catch"], bins=25)
plt.xlabel('log(tot.catch)')
plt.ylabel('Count')
plt.title('Distribution of log(tot.catch)')
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(data["longitude"], data["latitude"], alpha=0.5, c=data["tot.catch"], cmap='plasma')
plt.colorbar()
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()

''' PyMC3 modeling '''
dim = 2
seed = 630
y_obs = np.array(data['tot.catch'])
X_obs = data[['longitude', 'latitude']].values
with pm.Model() as model_1:
    l = pm.InverseGamma("l", alpha=5, beta=5)
    sigma_f = pm.HalfNormal("sigma_f", sigma=1)

    # specify the covariance function
    cov_func = sigma_f ** 2 * pm.gp.cov.ExpQuad(dim, ls=l)
    # specify the GP with mean 0
    gp = pm.gp.Marginal(cov_func=cov_func)

    sigma_n = pm.HalfNormal("sigma_n", sigma=1)
    y = gp.marginal_likelihood("y", X=X_obs, y=y_obs, noise=sigma_n)

    mp = pm.find_MAP()

''' generate new X data '''
x1_range = np.arange(min(data['longitude']), max(data['longitude']), 0.05)
x2_range = np.arange(min(data['latitude']), max(data['latitude']), 0.05)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x_full = np.c_[x1_grid.ravel(), x2_grid.ravel()]

''' without noise '''
with model_1:
    f_pred = gp.conditional("f_pred", x_full)
    pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=50)

# 2D heat map
mu, var = gp.predict(x_full, point=mp, diag=True, pred_noise=False)
sd = np.sqrt(var)

two_d_heatmap(mu, sd, x1_grid.ravel(), x2_grid.ravel(), with_noise=False)

# sliced, mean and 2 sigma region
# fix latitude
x_new = []
for i in x_full:
    if i[1] == x_full[212][1]:
        x_new.append(i)
x_new = np.array(x_new)

plot_x = []
for i in x_new:
    plot_x.append(i[0])
plot_x = np.array(plot_x)

plot_obs_x = []
plot_obs_y = []
for i in range(len(X_obs)):
    if round(X_obs[i][1], 1) == round(x_full[212][1], 1):
        plot_obs_x.append(X_obs[i][0])
        plot_obs_y.append(y_obs[i])
plot_obs_x = np.array(plot_obs_x)
plot_obs_y = np.array(plot_obs_y)

# fix longitude
x_new_2 = []
for i in x_full:
    if i[0] == x_full[266][0]:
        x_new_2.append(i)
x_new_2 = np.array(x_new_2)

plot_x_2 = []
for i in x_new_2:
    plot_x_2.append(i[1])
plot_x_2 = np.array(plot_x_2)

plot_obs_x_2 = []
plot_obs_y_2 = []
for i in range(len(X_obs)):
    if round(X_obs[i][0], 1) == round(x_full[266][0], 1):
        plot_obs_x_2.append(X_obs[i][1])
        plot_obs_y_2.append(y_obs[i])
plot_obs_x_2 = np.array(plot_obs_x_2)
plot_obs_y_2 = np.array(plot_obs_y_2)

# sliced at latitude
mu, var = gp.predict(x_new, point=mp, diag=True, pred_noise=False)
sd = np.sqrt(var)
sliced_mean_2sig_region(plot_x, mu, sd, plot_obs_x, plot_obs_y, x_full[212][1])

# sliced at longitude
mu2, var2 = gp.predict(x_new_2, point=mp, diag=True, pred_noise=False)
sd2 = np.sqrt(var2)
sliced_mean_2sig_region(plot_x_2, mu2, sd2, plot_obs_x_2, plot_obs_y_2, x_full[266][0], sliced_dim='longitude')

# draw sampled curves
# sliced at latitude
with model_1:
    f_pred_1 = gp.conditional("f_pred_1", x_new)
    pred_samples_1 = pm.sample_posterior_predictive([mp], vars=[f_pred_1], samples=15)
plot_sampled_curves(pred_samples_1, 'f_pred_1', plot_x, plot_obs_x, plot_obs_y, x_full[212][1])

# sliced at longitude
with model_1:
    f_pred_2 = gp.conditional("f_pred_2", x_new_2)
    pred_samples_2 = pm.sample_posterior_predictive([mp], vars=[f_pred_2], samples=15)
plot_sampled_curves(pred_samples_2, 'f_pred_2', plot_x_2, plot_obs_x_2, plot_obs_y_2, x_full[266][0],
                    sliced_dim='longitude')

''' with noise '''
# 2D heatmap
mu, var = gp.predict(x_full, point=mp, diag=True, pred_noise=True)
sd = np.sqrt(var)
two_d_heatmap(mu, sd, x1_grid.ravel(), x2_grid.ravel(), with_noise=True)

# sliced, mean and 2 sigma region
# sliced at latitude
mu, var = gp.predict(x_new, point=mp, diag=True, pred_noise=True)
sd = np.sqrt(var)
sliced_mean_2sig_region(plot_x, mu, sd, plot_obs_x, plot_obs_y, x_full[212][1], with_noise=True)

# sliced at longitude
mu2, var2 = gp.predict(x_new_2, point=mp, diag=True, pred_noise=True)
sd2 = np.sqrt(var2)
sliced_mean_2sig_region(plot_x_2, mu2, sd2, plot_obs_x_2, plot_obs_y_2, x_full[266][0], sliced_dim='longitude',
                        with_noise=True)

# draw sampled curves
# sliced at latitude
with model_1:
    y_pred = gp.conditional("y_pred", x_new, pred_noise=True)
with model_1:
    y_samples = pm.sample_posterior_predictive([mp], vars=[y_pred], samples=50)

plot_sampled_curves(y_samples, 'y_pred', plot_x, plot_obs_x, plot_obs_y, x_full[123][1], plot_predicted=10)

# sliced at longitude
with model_1:
    y_pred_2 = gp.conditional("y_pred_2", x_new_2, pred_noise=True)
with model_1:
    y_samples_2 = pm.sample_posterior_predictive([mp], vars=[y_pred_2], samples=50)

plot_sampled_curves(y_samples_2, 'y_pred_2', plot_x_2, plot_obs_x_2, plot_obs_y_2, x_full[266][0],
                    plot_predicted=10, sliced_dim='longitude')
