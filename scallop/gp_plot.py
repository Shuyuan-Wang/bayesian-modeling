import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def two_d_heatmap(mu, sd, x1, x2, with_noise=False, cmap="plasma", alpha=0.5):
    """
    plot the posterior mean and sd of the GP
    input:
        - mu: posterior mean
        - sd: posterior standard deviation
        - x1, x2: points for which mean and sd are plotted, x1: longitude, x2: latitude
        - with_noise: bool, whether the GP is modeled with noise term, False by default
        - cmap: style of heat map, plasma by default
        - alpha: transparency, 0.5 by default
    output: a set of plot (2 subplots): posterior mean and posterior sd, respectively
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    fig1 = ax1.scatter(x1, x2, alpha=alpha, c=mu, cmap=cmap)
    plt.colorbar(fig1)
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    if not with_noise:
        ax1.set_title('Posterior mean without noise')
    if with_noise:
        ax1.set_title('Posterior mean with noise')

    ax2 = fig.add_subplot(1, 2, 2)
    fig2 = ax2.scatter(x1, x2, alpha=alpha, c=sd, cmap=cmap)
    plt.colorbar(fig2)
    ax2.set_xlabel("longitude")
    ax2.set_ylabel("latitude")
    if not with_noise:
        ax2.set_title('Posterior sd without noise')
    if with_noise:
        ax2.set_title('Posterior sd with noise')
    plt.show()


def sliced_mean_2sig_region(plot_x, mu, sd, plot_obs_x, plot_obs_y,
                            sliced_value, sliced_dim="latitude", with_noise=False):
    """
    plot mean and 2σ interval
    input:
        - plot_x: the data which the region is plotted across
        - mu, sd: posterior mean and sd
        - plot_obs_x, plot_obs_y: observation coordinates
        - sliced_dim: the dimension which is sliced, latitude by default
        - sliced_value: the value at which sliced_dim is sliced
        - with_noise: bool, whether the GP is modeled with noise term, False by default

    """
    fig = plt.figure(figsize=(12, 5))
    ax = fig.gca()
    plt.plot(plot_x, mu, 'r', label="mean and 2σ region")
    plt.plot(plot_x, mu + 2 * sd, 'r', lw=1)
    plt.plot(plot_x, mu - 2 * sd, 'r', lw=1)
    plt.fill_between(plot_x.flatten(), mu - 2 * sd, mu + 2 * sd, color='r', alpha=0.3)
    plt.plot(plot_obs_x, plot_obs_y, 'ok', ms=3, alpha=1.0, label="observed data")
    if not with_noise:
        plt.title('predict without noise, {}={}'.format(sliced_dim, sliced_value))
    if with_noise:
        plt.title('predict with noise, {}={}'.format(sliced_dim, sliced_value))
    plt.legend()
    plt.show()


def plot_sampled_curves(samples, pred_dist, plot_x, plot_obs_x, plot_obs_y,
                        sliced_value, sliced_dim='latitude', curve_alpha=0.1,
                        plot_predicted=None):
    """
    inputs:
        - samples: the samples drawn from predictive posterior distribution
        - pred_dist: string, name of the predictive posterior distribution
        - plot_x: the data which the curves are plotted across
        - plot_obs_x, plot_obs_y: observation coordinates
        - sliced_value: the value at which sliced_dim is sliced
        - sliced_dim: the dimension which is sliced, latitude by default
        - curve_alpha: the transparency of curves, 0.1 by default
        - plot_predicted: index, overlay a scatter of one draw of random points from the posterior predictive dist
    :return:
    the posterior predictive distribution over y (if with noise), or posterior distribution over f (if without noise)
    """
    fig = plt.figure(figsize=(12, 5))
    ax = fig.gca()
    for c in samples[pred_dist]:
        plt.plot(plot_x, c, "gray", alpha=curve_alpha)
    plt.plot(plot_obs_x, plot_obs_y, 'ok', ms=3, alpha=1.0, label="observed data")
    plt.title('sampling without noise, {}={}'.format(sliced_dim, sliced_value))
    if plot_predicted is not None:
        plt.plot(plot_x, samples[pred_dist][plot_predicted, :].T, "co", ms=2, alpha=1, label="Predicted data")
    plt.legend()
    plt.show()
