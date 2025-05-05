import numpy as np
from math import log2
from typing import Dict


class InterestingnessMeasures:
    def __init__(self, counts):
        """
        Initialize with an array of counts for each tuple

        Parameters:
        counts (list): List of count values for each tuple in the summary
        """
        self.counts = np.array(counts)
        self.m = len(counts)  # total number of tuples
        self.N = sum(counts)  # total count

        # Calculate probability distributions
        self.P = self.counts / self.N  # actual probability distribution
        self.q = np.array([1 / self.m] * self.m)  # uniform distribution
        self.u = self.N / self.m  # uniform count

        # Calculate combined distribution r
        self.r = (self.counts + self.u) / (2 * self.N)

    def variance(self) -> float:
        """Calculate the IVariance measure"""
        return np.sum((self.P - self.q[0]) ** 2) / (self.m - 1)

    def simpson(self) -> float:
        """Calculate the ISimpson measure"""
        return np.sum(self.P**2)

    def shannon(self) -> float:
        """Calculate the IShannon measure"""
        # Avoid log2(0) by filtering out zero probabilities
        nonzero_P = self.P[self.P > 0]
        return -np.sum(nonzero_P * np.log2(nonzero_P))

    def total(self) -> float:
        """Calculate the ITotal measure"""
        return self.m * self.shannon()

    def max(self) -> float:
        """Calculate the IMax measure"""
        return log2(self.m)

    def mcintosh(self) -> float:
        """Calculate the IMcIntosh measure"""
        return (self.N - np.sqrt(np.sum(self.counts**2))) / (
            self.N - np.sqrt(self.N)
        )  # noqa

    def lorenz(self) -> float:
        """Calculate the ILorenz measure"""
        # note we would want to reverse this for the Lorenz curve
        # but then we would also enumerate from len(sorted_P) -> 1
        # (4)(0.143) + (3)(0.143) + (2)(0.286) + (1)(0.429)
        # but instead we don't reverse and enumerate from 1 -> len(sorted_P)
        # (1)(0.143) + (2)(0.143) + (3)(0.286) + (4)(0.429)
        sorted_P = np.sort(self.P)

        result = 0
        for i, p in enumerate(sorted_P, 1):  # enumerate from 1
            result += (self.m - i + 1) * p

        # Multiply by q (which is 1/m)
        return (1 / self.m) * result

    def gini(self) -> float:
        """Calculate the IGini measure"""
        total = 0
        for pi in self.P:
            row_sum = sum(abs(pi - pj) for pj in self.P)
            total += row_sum
        return (self.q[0] * total) / 2  # q[0] is 1/m

    def berger(self) -> float:
        """Calculate the IBerger measure"""
        return np.max(self.P)

    def schutz(self) -> float:
        """Calculate the ISchutz measure"""
        return np.sum(np.abs(self.P - self.q)) / (2 * self.m * self.q[0])

    def bray(self) -> float:
        """Calculate the IBray measure"""
        return np.sum(np.minimum(self.counts, self.u)) / self.N

    def whittaker(self) -> float:
        """Calculate the IWhittaker measure"""
        return 1 - (np.sum(np.abs(self.P - self.q[0])) / 2)

    def kullback(self) -> float:
        """Calculate the IKullback measure"""
        # Avoid log2(0) by filtering out zero probabilities
        nonzero_mask = self.P > 0
        return log2(self.m) - np.sum(
            self.P[nonzero_mask]
            * np.log2(self.P[nonzero_mask] / self.q[nonzero_mask])  # noqa
        )

    def macarthur(self) -> float:
        """Calculate the IMacArthur measure"""
        nonzero_r = self.r[self.r > 0]
        nonzero_P = self.P[self.P > 0]
        term1 = -np.sum(nonzero_r * np.log2(nonzero_r))
        term2 = (-np.sum(nonzero_P * np.log2(nonzero_P)) + log2(self.m)) / 2
        return term1 - term2

    def theil(self) -> float:
        """Calculate the ITheil measure"""
        nonzero_mask = self.P > 0
        P = self.P[nonzero_mask]
        q = self.q[nonzero_mask]
        return np.sum(np.abs(P * np.log2(P) - q * np.log2(q))) / (
            self.m * self.q[0]
        )  # noqa

    def atkinson(self) -> float:
        """Calculate the IAtkinson measure"""
        return 1 - np.prod((self.P / self.q[0]) ** self.q[0])

    def calculate_all(self) -> Dict[str, float]:
        """Calculate all interestingness measures"""
        return {
            "variance": self.variance(),
            "simpson": self.simpson(),
            "shannon": self.shannon(),
            "total": self.total(),
            "max": self.max(),
            "mcintosh": self.mcintosh(),
            "lorenz": self.lorenz(),
            "gini": self.gini(),
            "berger": self.berger(),
            "schutz": self.schutz(),
            "bray": self.bray(),
            "whittaker": self.whittaker(),
            "kullback": self.kullback(),
            "macarthur": self.macarthur(),
            "theil": self.theil(),
            "atkinson": self.atkinson(),
        }
