import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation

def gen_data(k, p, N, pi_0, lim = [-10, 10]):
    '''
    Generates data from a random mixture of Gaussians in a given range.
    input:
        - k: Number of Gaussian clusters
        - p: Dimension of generated points
        - N: Total number of points to generate
        - pi_0: Mixing proportions of the Gaussians
        - lim: Range of mean values
    output:
        - X: Generated points (N, p)
        - mean: Mean of the Gaussians (k, p)
        - covs: Covariance matrices of the Gaussians (k, p, p)
    '''
    X = np.zeros((N, p))
    mean = np.random.rand(k, p) * (lim[1] - lim[0]) + lim[0]
    covs = np.zeros((k, p, p))
    for j in range(k):
        cov = np.random.rand(p, p + 10)
        covs[j] = cov @ cov.T   # Ensure positive definiteness
    for i in range(N):
        j = np.random.choice(range(k), p = pi_0)
        X[i] = np.random.multivariate_normal(mean[j], covs[j])
    
    return X, mean, covs

def add_gaussian(ax, k, means, covs, n_std, color):
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        n_std = 2.0 # Set how many standard deviations to draw the ellipse

        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width = ell_radius_x * 2,
                        height = ell_radius_y * 2, facecolor = 'none',
                        edgecolor = color)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

def plot_single(X, k, title, true_param = None, estimator = None, filename = None):
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(X[:,0], X[:,1], s = 3, alpha = 0.4)
    ax.autoscale(enable=True)
    if true_param:
        true_means = true_param[0]
        true_covs = true_param[1]
        ax.scatter(true_means[:, 0], true_means[:, 1], s = 2, color = 'black')
        add_gaussian(ax, k, true_means, true_covs, 2.0, color = 'black')
    if estimator:
        est_means = estimator[0]
        est_covs = estimator[1]
        ax.scatter(est_means[:, 0], est_means[:, 1], s = 2, color = 'red')
        add_gaussian(ax, k, est_means, est_covs, 2.0, color = 'red')
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.show()

    if filename:
        fig.savefig(filename)

def plot_data(X, k, title, ax, true_param = None, estimator = None):
    ax.scatter(X[:,0], X[:,1], s = 3, alpha = 0.4)
    ax.autoscale(enable=True)
    if true_param:
        true_means = true_param[0]
        true_covs = true_param[1]
        ax.scatter(true_means[:, 0], true_means[:, 1], s = 2, color = 'black')
        add_gaussian(ax, k, true_means, true_covs, 2.0, color = 'black')
    if estimator:
        est_means = estimator[0]
        est_covs = estimator[1]
        ax.scatter(est_means[:, 0], est_means[:, 1], s = 2, color = 'red')
        add_gaussian(ax, k, est_means, est_covs, 2.0, color = 'red')
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.show()

def animate_plot(snapshots, X, k, true_means, true_covs, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    iterations = sorted(snapshots.keys())

    def update(frame):
        ax.clear()
        iter = iterations[frame]
        mu = snapshots[iter]['mu']
        sigma = snapshots[iter]['sigma']
        
        if iter == 0:
            title = "Initialization: K-Means"
        else:
            title = f"Iteration: {iter}"

        if iter == iterations[-1]:  # Last frame
            true_param = (true_means, true_covs)
        else:
            true_param = None

        plot_data(X, k, title, ax, true_param = true_param, estimator = (mu, sigma))
        fig.savefig(filename + f"_iter_{iter}.png")

    anim = FuncAnimation(fig, update, frames = len(iterations), interval = 1000, repeat = False)
    return anim