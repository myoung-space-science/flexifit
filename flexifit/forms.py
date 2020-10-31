from typing import (
    Callable,
    Iterable,
)
import numpy as np


class Functions1D:
    """A class for managing functional forms for fitters.

    This class acts as a namespace for various functional forms that may be
    useful when experimenting with fits to a 1-D vector of independent data. For
    a class that maintains its own copy of the independent data, see ``Var1D``.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def linear(x, c0, c1):
        """Functional form of a straight line."""
        return c0 + x*c1

    @staticmethod
    def polynomial(x, *cs):
        """Functional form of a polynomial."""
        y = np.zeros_like(x)
        for p, c in enumerate(cs):
            y += c*np.power(x, p)
        return y

    @staticmethod
    def exponential(x, *c):
        """Functional form of an exponential."""
        if len(c) == 1:
            return c[0]*np.exp(x)
        elif len(c) == 2:
            return c[0]*np.exp(c[1]*x)
        elif len(c) == 3:
            return c[0]*np.exp(c[1]*x + c[2])
        elif len(c) == 4:
            return c[0]*np.exp(c[1]*x + c[2]) + c[3]
        else:
            raise ValueError(c)

    @staticmethod
    def single_power_law(x, c0, c1):
        """Functional form of a single power law."""
        return c0*np.power(x, c1)


class Var1D:
    """A class for managing functional forms.

    The user creates an instance of this class by passing in a vector of
    independent values. That instance has multiple methods that model different
    functional forms, allowing the user to explore the data. For use with
    fitters like ``scipy.optimize.curve_fit()``, see the class ``Functions1D``.
    """
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

