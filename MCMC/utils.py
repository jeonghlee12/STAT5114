import matplotlib.pyplot as plt

def plot_vanilla(results, burn_T):
    results = results["vanilla"]
    accept_prob = results["accept_prob"]
    results = results["samples"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(results[:, 0], color = 'steelblue')
    axs[0].set_title(r"Trace Plot for $\mu$")
    axs[0].set_ylabel(r"$\mu$")
    axs[0].set_xlabel("Iteration")
    axs[0].axvline(burn_T, color = 'lightgray', ls = '--')

    axs[1].plot(results[:, 1], color = 'indianred')
    axs[1].set_title(r"Trace Plot for $\phi$")
    axs[1].set_ylabel(r"$\phi$")
    axs[1].set_xlabel("Iteration")
    axs[1].axvline(burn_T, color = 'lightgray', ls = '--')

    fig.suptitle(f"Trace Plots for Metropolis-Hastings Sampling (Acceptance Rate: {accept_prob:.2f})", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_gibbs(results, burn_T):
    results = results["gibbs"]
    mu_prob, phi_prob = results["accept_prob"]
    results = results["samples"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(results[:, 0], color = 'steelblue')
    axs[0].set_title(fr"Trace Plot for $\mu$ (Accept. rate:{mu_prob:.2f})")
    axs[0].set_ylabel(r"$\mu$")
    axs[0].set_xlabel("Iteration")
    axs[0].axvline(burn_T, color = 'lightgray', ls = '--')

    axs[1].plot(results[:, 1], color = 'indianred')
    axs[1].set_title(fr"Trace Plot for $\phi$ (Accept. rate:{phi_prob:.2f})")
    axs[1].set_ylabel(r"$\phi$")
    axs[1].set_xlabel("Iteration")
    axs[1].axvline(burn_T, color = 'lightgray', ls = '--')

    fig.suptitle(f"Trace Plots for Gibbs Sampling", fontsize=16)
    plt.tight_layout()
    plt.show()

def hist_results(results, burn_T, vanilla = True):
    if vanilla:
        results = results["vanilla"]["samples"]
    else:
        results = results["gibbs"]["samples"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].hist(results[burn_T:, 0], color = "steelblue")
    axs[0].set_title(r"Histogram for $\mu$")
    axs[0].set_xlabel(r"$\mu$")

    axs[1].hist(results[burn_T:, 1], color = 'indianred')
    axs[1].set_title(r"Histogram for $\phi$")
    axs[1].set_xlabel(r"$\phi$")

    if vanilla:
        fig.suptitle(f"Histograms for Metropolis-Hastings Sampling", fontsize=16)
    else:
        fig.suptitle(f"Histograms for Gibbs Sampling", fontsize=16)
    plt.tight_layout()
    plt.show()

def contour(results, burn_T, vanilla = True):
    if vanilla:
        results = results["vanilla"]["samples"]
    else:
        results = results["gibbs"]["samples"]
    
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(results[burn_T:, 0], results[burn_T:, 1])
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\phi$")
    if vanilla:
        plt.title(r"Contour plot of $\mu,\phi\,|\,X$ from Metropolis-Hastings")
    else:
        plt.title(r"Contour plot of $\mu,\phi\,|\,X$ from Gibbs sampling")
    plt.show()
