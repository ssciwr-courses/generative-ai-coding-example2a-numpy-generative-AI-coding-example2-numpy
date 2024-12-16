# for example, interpolate between the points
# for example, numerical integration
# for example, matrix diagonalization
# for example, particle in a box hamiltonian
# here we will use the particle in a box example
# Quantum mechanics and the time-independent Schrodinger equation:
# The particle in a box I
#
# Reference:
# https://doi.org/10.1021/acs.jchemed.7b00003
#
# A standard problem in quantum chemistry is the particle in a box.
# This is a mostly educational
# example to illustrate fundamentals of quantum mechanics and the difference
# to classical mechanics.
# As the potential within the box is zero, it is the simplest example
# (apart from the free particle)
# in quantum mechanics.
#
# In this example, a particle is contained in a box of length $L$,
# with zero potential within the box,
# and potential walls extending to infinity at both sides of the box.

import numpy as np
import matplotlib.pyplot as plt

# define constants
# atomic units
HBAR = 1.0
MASS = 1.0
# width of the box
WIDTH = 1
# grid along x
GRID_VAL = WIDTH / 2
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]


class SetHamiltonian:
    def __init__(self, steps: int = 100, width: float = 1.0) -> None:
        """Set up the Hamiltonian for the particle in a box.

        Parameters:
            steps (int): number of grid points
            width (float): width of the box

        Returns:
            None"""
        self.steps = steps
        self.width = width
        self.grid = self.width / 2

    def set_laplacian(self):
        # get step size h for the numerical derivative
        self.xgrid, h = np.linspace(-self.grid, self.grid, self.steps, retstep=True)
        # create the second derivative matrix
        self.laplacian = (
            -2.0 * np.diag(np.ones(self.steps))
            + np.diag(np.ones(self.steps - 1), 1)
            + np.diag(np.ones(self.steps - 1), -1)
        ) / (float)(h**2)

    def set_hamiltonian(self):
        # now construct the Hamiltonian matrix
        self.hamiltonian = (-0.5 * (HBAR**2) / MASS) * self.laplacian

    def plot_matrix(self, matrix):
        # we can look at this matrix - it is a band matrix as only the j-1, j, and j+1
        # grid points are required for each second derivative
        plt.matshow(matrix)
        plt.show()

    def diagonalize(self):
        # now we diagonalize the Hamiltonian
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hamiltonian)

    def plot_eigenvalues(self):
        mf = 14
        n = 3
        _, ax = plt.subplots(3, figsize=(8, 4 * n), sharex=True)
        # plot 1
        # Column-major order from Fortran leads to the unexpected
        # indexing of the eigenvectors
        # we could also transpose the eigenvectors using transpose() or .T
        for i in range(n):
            ax[0].plot(self.xgrid, self.eigenvectors[:, i])
        ax[0].set_ylabel("$\Psi$", fontsize=mf)  # noqa
        ax[0].text(
            -self.grid, max(self.eigenvectors[:, 0]), "Wave function", fontsize=mf
        )

        # plot 2
        for i in range(n):
            ax[1].plot(self.xgrid, self.eigenvectors[:, i] ** 2)
        ax[1].set_ylabel("$|\Psi|^2$", fontsize=mf)  # noqa
        ax[1].text(
            -self.grid, max(self.eigenvectors[:, 0] ** 2), "Density", fontsize=mf
        )

        # plot 3
        for i in range(n):
            ax[2].hlines(self.eigenvalues[i], -self.grid, self.grid, color=COLORS[i])
        ax[2].set_xlabel("x (a.u.)", fontsize=mf)
        ax[2].set_ylabel("E (E$_h$)", fontsize=mf)
        ax[2].text(-self.grid, self.eigenvalues[n - 1], "Energy", fontsize=mf)

        for i in range(3):
            ax[i].xaxis.set_tick_params(labelsize=mf)
            ax[i].yaxis.set_tick_params(labelsize=mf)

        plt.subplots_adjust(hspace=0.0)
        plt.show()


# write a class that does different operations on the numpy array
class OperateArray:
    def __init__(self, array):
        self.array = array

    def sum(self):
        return self.array.sum()

    def mean(self):
        return self.array.mean()

    def std(self):
        return self.array.std()

    # multiply the array by a scalar
    def multiply(self, scalar):
        return self.array * scalar

    # multiply the array by another array element-wise (Hadamard product)
    def multiply_array(self, array):
        return self.array * array

    # perform matrix multiplication
    def matrix_multiply(self, array):
        return self.array @ array
