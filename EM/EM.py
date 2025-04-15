import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

def EM(k, p, X, n_iter, tol = 1e-5, init_kmeans = True):
    '''
    Run the EM algorithm for a GMM with a known number of clusters and
    dimensions.
    Algorithm stops if it reaches given number of iterations or improvement
    of the log-likelihood is below the threshold.
    -------------
    input:
        - k: Number of Gaussian components
        - p: Dimension of Gaussians
        - X: Data
        - n_iter: Max number of iterations for the E-M steps
        - tol: Stopping criterion for log-likelihood
    -------------
    output:
        - mu: Means of the Gaussians
        - sigma: Covariances of the Gaussians
        - pi: Mixing proportions of the Gaussians
        - snapshots: Dictionary with snapshots of means and covariances
        - lls: List of log-likelihoods at each iteration
    '''
    # checkpoints
    chkpts = range(n_iter)
    snapshots = dict()

    ### Initialization ###
    if init_kmeans:
        # Fit K-means to get initial means and covariances
        print("Initializing under K-Means...")
        kmeans = KMeans(n_clusters = k, random_state=0, n_init="auto").fit(X)
        mu = kmeans.cluster_centers_

        # Initialize covariances as covariance of each cluster
        sigma = np.zeros((k, p, p))
        for i in range(k):
            sigma[i] = np.cov(X[kmeans.labels_ == i], rowvar=False)

        # Initialize weights to assignment probability as
        # the proportion of points assigned to each cluster
        pi = np.array([np.sum(kmeans.labels_ == i) / len(X) for i in range(k)])
    
    else:
        # Randomly select k points from X as initial means
        print("Initializing under random selection...")
        mu = X[np.random.choice(len(X), k, replace=False)]
        sigma = np.zeros((k, p, p))
        for i in range(k):
            # Initialize each covariance given each mean
            diff = X - mu[i]
            sigma[i] = diff.T @ diff / len(X)
            # Regularization to keep covariance PD
            sigma[i].flat[:: p + 1] += 1e-6
        
        # Initialize uniform weights to assignment probability
        pi = np.ones(k) / k


    # Save initialization
    snapshots[0] = {'mu': mu.copy(), 'sigma': sigma.copy()}

    # Initialize responsibility
    gamma = np.zeros((len(X), k))

    # Initialize log-likelihoods
    lls = [0]

    for t in tqdm(range(n_iter)):
        ### E-step ###
        for i in range(k):
            gamma[:, i] = pi[i] * multivariate_normal.pdf(X, mean = mu[i],
                                                          cov = sigma[i])
        gamma /= np.sum(gamma, axis = 1, keepdims = True)
        
        ### M-step ###
        N_k = np.sum(gamma, axis = 0)
        mu = gamma.T @ X
        mu /= N_k[:, None]
        pi = N_k / len(X)
        for i in range(k):
            s = 1e-6 * np.eye(p)    # regularization to keep covariance PD
            for j in range(len(X)):
                s += gamma[j, i] * \
                     (X[j, np.newaxis] - mu[i]).T @ (X[j, np.newaxis] - mu[i])
            sigma[i] = s / N_k[i]

        # Save snapshots
        if t in chkpts:
            snapshots[t + 1] = {'mu': mu.copy(), 'sigma': sigma.copy()}

        # Calculate log-likelihood
        ll_new = 0
        for j in range(k):
            ll_new += pi[j] * multivariate_normal.pdf(X, mean = mu[j],
                                                    cov = sigma[j])
        ll_new = np.sum(np.log(ll_new))
        lls.append(ll_new)

        # Stop if improvement is below threshold
        if tol is not None and np.abs(ll_new - lls[t]) < tol:
            snapshots[t + 1] = {'mu': mu.copy(), 'sigma': sigma.copy()}
            print(f"EM converged at iteration {t + 1}!")
            return mu, sigma, pi, snapshots, lls

    return mu, sigma, pi, snapshots, lls