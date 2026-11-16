from types import SimpleNamespace as NS

import numpy as np
import pytest

from evaluation.runner import check_finite


def fake(bad):
    w = np.array([1.0, np.nan if bad else 2.0])
    return NS(name="fake", stacks=[NS(W1=np.ones(2)), NS(W1=w)], exp3=NS(log_w=np.zeros(3)))


def test_check_finite_passes():
    check_finite(fake(False))


def test_check_finite_raises_on_nan_in_list_object():
    with pytest.raises(FloatingPointError, match=r"stacks\[1\]\.W1"):
        check_finite(fake(True))


def test_check_finite_survives_reference_cycle():
    p = fake(False)
    p.stacks[0].owner = p
    check_finite(p)
