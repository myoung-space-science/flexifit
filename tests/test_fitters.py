import numpy as np
import pytest

from .context import core


def function(x, c0, c1, c2):
    return c0 * np.exp(-c1 * x) + c2

free = ['c0', 'c1']
fixed = {'c2': 0.35015434}
initial = {'c0': 2, 'c1': 1}
lower = {'c0': 0, 'c1': 0}
upper = {'c0': 3, 'c1': 1}
expected_values = np.array([2.43708906, 1.])
expected_covariance = np.array(
    [
        [0.01410692, 0.00583461],
        [0.00583461, 0.0052144 ],
    ]
)

xdata = np.linspace(0, 4, 50)
y = function(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise

fit = core.fitters.FlexiFit(
    function=function,
    free=free,
    fixed=fixed,
    initial=initial,
    lower=lower,
    upper=upper,
    xdata=xdata,
    ydata=ydata,
)

def test_free():
    assert fit.context.free == free


def test_fixed():
    assert fit.context.fixed == fixed


def test_initial():
    assert fit.context.initial == initial


def test_lower():
    assert fit.context.lower == lower


def test_upper():
    assert fit.context.upper == upper


def test_p0():
    assert fit.context.p0 == list(initial.values())


def test_bounds():
    assert fit.context.bounds == (
        list(lower.values()),
        list(upper.values()),
    )


def test_xdata():
    assert (fit.dataset.xdata == xdata).all()


def test_ydata():
    assert (fit.dataset.ydata == ydata).all()


def test_values():
    assert fit.values == pytest.approx(
        expected_values, abs=1e-8
    )


def test_covariance():
    assert fit.covariance == pytest.approx(
        expected_covariance, abs=1e-8
    )

