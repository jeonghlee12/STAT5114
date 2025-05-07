import numpy as np

# Define log of target distribution
def log_posterior(mu, phi, x):
    n = len(x)
    sum_squared = np.sum((x - mu)**2)
    return (n/2 - 1) * np.log(phi) - (phi / 2) * sum_squared

def metropolis_hastings(proposal_params, initial_state, n_samples, data):
    """
    Metropolis-Hastings algorithm for MCMC sampling.

    Parameters:
    - proposal_params: the hyperparameters for the proposal distribution function
    - initial_state: initial state for the Markov chain
    - n_samples: number of samples to generate
    - data: the underlying data sample

    Returns:
    - samples: array of generated samples
    - accept_prob: acceptance probability
    """
    samples = np.zeros((n_samples, len(initial_state)))
    current_state = initial_state

    acceptance_rate = 0

    for i in range(n_samples):
        log_phi = np.log(current_state[1])

        # Generate sample from proposal distribution given current state
        proposed_state = np.array([np.random.normal(current_state[0], proposal_params[0]),
                                   np.exp(np.random.normal(log_phi, proposal_params[1]))])

        # Calculate acceptance ratio
        pi_ast = log_posterior(proposed_state[0], proposed_state[1], data)
        pi = log_posterior(current_state[0], current_state[1], data)
        log_acceptance_ratio = pi_ast - pi #+ np.log(g_phi_ast) - np.log(g_phi)

        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_state = proposed_state
            acceptance_rate += 1
        samples[i] = current_state
    
    return samples, acceptance_rate / n_samples

def metropolis_gibbs(proposal_params, initial_state, n_samples, data):
    """
    A version of the Metropolis sampling method, i.e., Gibbs sampling for MCMC sampling.

    Parameters:
    - proposal_params: the hyperparameters for the proposal distribution function
    - initial_state: initial state for the Markov chain
    - n_samples: number of samples to generate
    - data: the underlying data sample

    Returns:
    - samples: array of generated samples
    - accept_prob: a tuple of acceptance probability for each parameter
    """
    current_state = initial_state
    samples = np.zeros((n_samples, len(initial_state)))
    mu_acceptance_count = 0
    phi_acceptance_count = 0

    for i in range(n_samples):
        mu_proposed = np.random.normal(current_state[0], proposal_params[0])
        pi_ast = log_posterior(mu_proposed, current_state[1], data)
        pi = log_posterior(current_state[0], current_state[1], data)
        log_acceptance_ratio = pi_ast - pi
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_state[0] = mu_proposed
            mu_acceptance_count += 1
        
        log_phi_current = np.log(current_state[1])
        log_phi_proposed = np.random.normal(log_phi_current, proposal_params[1])
        phi_proposed = np.exp(log_phi_proposed)
        pi_ast = log_posterior(current_state[0], phi_proposed, data)
        pi = log_posterior(current_state[0], current_state[1], data)
        log_acceptance_ratio = pi_ast - pi
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current_state[1] = phi_proposed
            phi_acceptance_count += 1
        
        samples[i] = current_state

    return samples, (mu_acceptance_count / n_samples, phi_acceptance_count / n_samples)