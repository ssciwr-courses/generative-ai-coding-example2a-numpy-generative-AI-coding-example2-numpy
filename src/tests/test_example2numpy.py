import numpy as np
from pathlib import Path
import sys

# extend path for imports
filepath = Path(__file__).parents[1]
sys.path.append(str(filepath))
import example2numpy as e2n  # noqa

# Create a numpy array
lim = 100
a = np.random.rand(lim, lim)


def test_example2numpy():
    assert e2n.OperateArray(a).array is a
