import numpy as np
import pytest

from .context import flexifit


tolerance = 1e-8

def test_linear():
    x = np.linspace(0, 1, 5)
    f = flexifit.forms.Var1D(x)
    c = [1.1, 1.1]
    result = [1.1, 1.375, 1.65, 1.925, 2.2]
    correct = pytest.approx(np.array(result), abs=tolerance)
    assert f.linear(*c) == correct

def test_polynomial():
    x = np.linspace(0, 1, 5)
    f = flexifit.forms.Var1D(x)
    c = []
    results = [
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.375, 1.65, 1.925, 2.2],
        [1.1, 1.44375, 1.925, 2.54375, 3.3],
        [1.1, 1.4609375, 2.0625, 3.0078125, 4.4],
    ]
    for result in results:
        c.append(1.1)
        correct = pytest.approx(np.array(result), abs=tolerance)
        assert f.polynomial(*c) == correct


def test_exponential():
    x = np.linspace(0, 1, 5)
    f = flexifit.forms.Var1D(x)
    c = []
    results = [
        [1.1, 1.41242796, 1.8135934, 2.32870002, 2.99011001],
        [1.1, 1.44818374, 1.90657832, 2.51006884, 3.30458263],
        [3.30458263, 4.3505844, 5.72767781, 7.54066353, 9.92751485],
        [4.40458263, 5.4505844, 6.82767781, 8.64066353, 11.02751485],
    ]
    for result in results:
        c.append(1.1)
        correct = pytest.approx(np.array(result), abs=tolerance)
        assert f.exponential(*c) == correct


def test_exponential_exception():
    x = np.linspace(0, 1, 5)
    f = flexifit.forms.Var1D(x)
    c = [1.1] * 5
    with pytest.raises(ValueError) as err:
        assert f.exponential(*c)
    assert str(err.value) == str(tuple(c))


def test_single_power_law():
    x = np.linspace(0, 1, 5)
    f = flexifit.forms.Var1D(x)
    c = [1.1, 1.1]
    result = [0.0, 0.2394014, 0.51316815, 0.80160437, 1.1]
    correct = pytest.approx(np.array(result), abs=tolerance)
    assert f.single_power_law(*c) == correct


