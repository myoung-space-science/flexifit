from typing import (
    Callable,
    Iterable,
)
import numpy as np


class Var1D:

    def __init__(self, x: np.ndarray):
        self.x = x

    def linear(self, c0, c1):
        """Functional form of a straight line."""
        return c0 + self.x*c1

    def polynomial(self, *cs):
        """Functional form of a polynomial."""
        y = np.zeros_like(self.x)
        for p, c in enumerate(cs):
            y += c*np.power(self.x, p)
        return y

    def exponential(self, *c):
        """Functional form of an exponential."""
        if len(c) == 1:
            return c[0]*np.exp(self.x)
        elif len(c) == 2:
            return c[0]*np.exp(c[1]*self.x)
        elif len(c) == 3:
            return c[0]*np.exp(c[1]*self.x + c[2])
        elif len(c) == 4:
            return c[0]*np.exp(c[1]*self.x + c[2]) + c[3]
        else:
            raise ValueError(c)

    def single_power_law(self, c0, c1):
        """Functional form of a single power law."""
        return c0*np.power(self.x, c1)

