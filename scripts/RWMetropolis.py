# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2023-08-03 08:26:13
# @Last Modified by:   aaronreding
# @Last Modified time: 2023-08-03 08:57:13
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, gamma
from scipy.stats._distn_infrastructure import rv_frozen
import matplotlib
from tqdm import tqdm


class Target:
    """
    π(x), the unnormalized target density (in this case Gamma dist).

    The true target is the posterior, but the normalizing constant cancels out anyways.
    """

    def __init__(self, v: float, alpha: float, beta: float, data: list):
        """
        Args:
            v (float): likelihood shape
            alpha (float): prior shape
            beta (float): prior rate
            data (list): likelihood data
        """
        self.v = v
        self.alpha = alpha
        self.beta = beta
        self.x = data
        self.n = len(data)

    def density(self, theta):
        """Unnormalized density of the Gamma distribution.

        Args:
            theta (np.ndarray): Values to evaluate the pdf at.

        Returns:
            np.ndarray: densities
        """
        return theta ** (self.v * self.n + self.alpha - 1) * np.exp(
            -theta * (self.beta + sum(self.x))
        )


class UpdateDist:
    """
    Used with matplotlib FuncAnimation function to create Metropolis visualization.

    This currently only does Metropolis random walk w/ N(0, proposal_variance).

    Modified from the Matplotlib docs at:
        https://matplotlib.org/stable/gallery/animation/bayes_update.html
    """

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        x_0: float,
        target: Target,
        true_posterior: rv_frozen,
        proposal_sigma: float,
        lower_bound: float,
        upper_bound: float,
        bins: int,
        sample_count: int,
    ):
        """

        Args:
            ax (matplotlib.axes.Axes)
            x_0 (float): initial value of the chain.
            target (Target): target density (need not be a true pdf)
            true_posterior (rv_frozen): true posterior, use scipy.stats dist
            proposal_sigma (float): Normal proposal std dev
            lower_bound (float): plot lower x limit
            upper_bound (float): plot upper x limit
            bins (int): histogram bins
            sample_count (int): number of draws (for progress bar)
        """
        assert lower_bound < upper_bound, "Lower bound must be less than upper bound."
        assert true_posterior.pdf(x_0) > 0.0, "Initial value density is 0."

        self.samples = []
        self.pbar = tqdm(total=sample_count)

        self.init = x_0
        self.lower = lower_bound
        self.upper = upper_bound
        self.bins = bins
        self.parameter_values = np.linspace(lower_bound, upper_bound, 300)
        self.ax = ax
        self.proposal_sigma = proposal_sigma
        self.target = target
        self.true_posterior = true_posterior

        self.ax.set_xlim(lower_bound, upper_bound)
        self.ax.set_ylim(0, 1)

        self.ax.grid(True)
        (self.true_posterior_line,) = ax.plot(
            self.parameter_values,
            self.true_posterior.pdf(self.parameter_values),
            "r-",
            label="True Posterior",
        )
        self.ax.legend()

    def __call__(self, i):
        """Summary

        Args:
            i (TYPE): Description

        Returns:
            TYPE: Description
        """
        if i == 0:
            self.samples = []
            self.ax.clear()
            self.ax.grid(True)
            self.ax.set_xlim(self.lower, self.upper)
            self.ax.set_ylim(0, 1)
            (self.true_posterior_line,) = self.ax.plot(
                self.parameter_values,
                self.true_posterior.pdf(self.parameter_values),
                "r-",
                label="True Posterior",
            )
            return (self.true_posterior_line,)

        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlim(self.lower, self.upper)
        self.ax.set_ylim(0, 1)

        proposal_mu = self.samples[-1] if self.samples else self.init
        proposed_value = norm.rvs(proposal_mu, self.proposal_sigma)

        # plot
        proposal_density = norm.pdf(
            self.parameter_values, proposal_mu, self.proposal_sigma
        )
        (self.proposal_line,) = self.ax.plot(
            self.parameter_values, proposal_density, "g--", label="Proposal density"
        )

        # Metropolis acceptance criterion
        acceptance_ratio = self.target.density(proposed_value) / self.target.density(
            self.samples[-1] if self.samples else self.init,
        )

        (self.true_posterior_line,) = self.ax.plot(
            self.parameter_values,
            self.true_posterior.pdf(self.parameter_values),
            "r-",
            label="True Posterior",
        )

        self.ax.hist(
            self.samples,
            bins=self.bins,
            density=True,
            alpha=0.5,
            label="Sampled distribution",
        )

        if np.random.rand() >= acceptance_ratio:  # rejection
            self.ax.text(3.5, 0.5, "Rejection!", fontsize=12, color="red", ha="center")
            self.ax.scatter(
                [proposed_value],
                [0],
                c="r",
                s=50,
                label="Current proposal",
            )
        else:  # proposal accepted
            self.samples.append(proposed_value)
            self.ax.scatter(
                [proposed_value],
                [0],
                c="b",
                s=50,
                label="Current proposal",
            )

        self.ax.legend()
        self.pbar.update()
        return (self.true_posterior_line,)


def main():
    np.random.seed(1)

    # set up distributions
    alpha = 1
    beta = 1
    v = 1
    x = [1]
    n = len(x)
    x_0 = 1
    samples = 1000  # this can be very slow if drawing too many samples!

    target = Target(v, alpha, beta, data=x)

    true_posterior = gamma(a=alpha + n * v, scale=1.0 / (beta + sum(x)))

    fig, ax = plt.subplots()

    ud = UpdateDist(
        ax,
        x_0,
        target,
        true_posterior,
        proposal_sigma=0.4,
        lower_bound=0.0,
        upper_bound=5.0,
        bins=35,
        sample_count=samples,
    )

    # animate and save

    # this limit allows up to ~4k samples, but it's slow!
    matplotlib.rcParams["animation.embed_limit"] = 500
    print("Animating...")
    anim = FuncAnimation(fig, ud, frames=samples, interval=100, blit=True)

    plt.close(fig)

    # recommend saving mp4 over gif. easier to work with and smaller.
    anim.save("animation-1000.mp4", writer="ffmpeg", fps=30)
    # anim.save("animation.gif", writer="imagemagick", fps=30)


if __name__ == "__main__":
    main()
