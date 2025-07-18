# Adapted to tecorigin hardware
import pytest

from thinc.api import CupyOps
from thinc.compat import has_torch, has_torch_sdaa_gpu


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.skipif(not has_torch_sdaa_gpu, reason="needs a GPU")
def test_issue564():
    import torch

    if CupyOps.xp is not None:
        ops = CupyOps()
        t = torch.zeros((10, 2)).sdaa()
        a = ops.asarray(t)

        assert a.shape == t.shape
        ops.xp.testing.assert_allclose(
            a,
            ops.alloc2f(10, 2),
        )
