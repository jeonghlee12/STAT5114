import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
from itertools import permutations

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

def animate_plot(snapshots, X, k, true_means, true_covs, init_kmeans, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    iterations = sorted(snapshots.keys())

    def update(frame):
        ax.clear()
        iter = iterations[frame]
        mu = snapshots[iter]['mu']
        sigma = snapshots[iter]['sigma']
        
        if iter == 0:
            if init_kmeans:
                title = "Initialization: K-Means"
            else:
                title = "Initialization: Random from data"
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

def best_mse(true, predicted):
    # Generate all permutations of the true means
    perm_indices = permutations(range(len(true)))
    min_mse = float('inf')

    best_indices = [1, 2, 3]

    for idx in perm_indices:
        permuted_true = true[list(idx)]
        mse = np.mean(np.sum((predicted - permuted_true) ** 2, axis=1))
        if mse < min_mse:
            min_mse = mse
            best_indices = idx
    
    return min_mse, best_indices

def cov_comp(true, predicted, best_indices):
    aligned_true_covs = true[list(best_indices)]

    det_diffs = []
    trace_diffs = []
    frobenius_norms = []

    for pred, true in zip(predicted, aligned_true_covs):
        det_diff = abs(np.linalg.det(pred) - np.linalg.det(true))
        trace_diff = abs(np.trace(pred) - np.trace(true))
        frob_norm = np.linalg.norm(pred - true, ord='fro')

        det_diffs.append(det_diff)
        trace_diffs.append(trace_diff)
        frobenius_norms.append(frob_norm)
    
    det_diffs = np.array(det_diffs)
    trace_diffs = np.array(trace_diffs)
    frobenius_norms = np.array(frobenius_norms)

    avg_det_diff = np.mean(det_diffs)
    avg_trace_diff = np.mean(trace_diffs)
    avg_frobenius_norm = np.mean(frobenius_norms)

    return avg_det_diff, avg_trace_diff, avg_frobenius_norm

def kl_gaussian(mu0, cov0, mu1, cov1):
    d = mu0.shape[0]
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0

    term1 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0) + 1e-12)
    term2 = np.trace(cov1_inv @ cov0)
    term3 = diff.T @ cov1_inv @ diff

    return 0.5 * (term1 - d + term2 + term3)

def get_KL_div(true_means, true_covs, mus, sigmas, best_indices):
    aligned_true_means = true_means[list(best_indices)]
    aligned_true_covs = true_covs[list(best_indices)]
    kl_scores = []

    for i in range(3):
        mu_true = aligned_true_means[i]
        cov_true = aligned_true_covs[i]
        mu_pred = mus[i]
        cov_pred = sigmas[i]

        kl = kl_gaussian(mu_true, cov_true, mu_pred, cov_pred)
        kl_scores.append(kl)

    kl_scores = np.array(kl_scores)
    avg_kl = kl_scores.mean()

    return avg_kl

def get_KL_pi(true_pis, pis, best_indices):
    def safe_kl(p_true, p_pred, epsilon=1e-12):
        p_true = np.clip(p_true, epsilon, 1)
        p_pred = np.clip(p_pred, epsilon, 1)
        return np.sum(p_true * np.log(p_true / p_pred))

    aligned_true_pis = true_pis[list(best_indices)]

    kl_true_to_pred = safe_kl(aligned_true_pis, pis)
    
    return kl_true_to_pred

def compute_accuracy(true_means, true_covs, true_pis, mus, sigmas, pis):

    mse, best_perm = best_mse(true_means, mus)
    cov_results = cov_comp(true_covs, sigmas, best_perm)
    avg_kl_gauss = get_KL_div(true_means, true_covs, mus, sigmas, best_perm)
    avg_kl_pi = get_KL_pi(np.array(true_pis), pis, best_perm)

    return (mse, cov_results, avg_kl_gauss, avg_kl_pi)