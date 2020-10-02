from typing import Callable
import numpy as np


def linear(x, c0, c1):
    """Functional form of a straight line."""
    return c0 + x*c1


def polynomial(x, *cs):
    """Functional form of a polynomial."""
    y = np.zeros_like(x)
    for p, c in enumerate(cs):
        y += c*np.power(x, p)
    return y


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


def single_power_law(x, c0, c1):
    """Functional form of a single power law."""
    return c0*np.power(x, c1)


analytic_forms = {
    'linear': linear,
    'polynomial': polynomial,
    'exponential': exponential,
    'single power law': single_power_law,
}


def available():
    """Produces a list of available functional forms."""
    return [form for form in analytic_forms]


def list_available():
    """Prints a list of the available functional forms."""
    print("The following funcional forms are available:")
    for form in analytic_forms:
        print(f"\t{form}")


def load(name: str) -> Callable:
    """Load a functional form by name."""
    return analytic_forms[name]