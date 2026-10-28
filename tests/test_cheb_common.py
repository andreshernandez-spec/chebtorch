"""The table cache in cheb_common.py builds outside inference mode."""

import torch

from chebtorch.cheb_common import Tables

CPU = torch.device("cpu")


def test_tables_build_outside_inference_mode():
    tables = Tables({"T": (2,)}, {"T": [1.0, 2.0]})
    with torch.inference_mode():
        table = tables("T", torch.float64, CPU)
    assert not table.is_inference()
    # the product saves it for its backward, which inference tensors refuse
    x = torch.ones(2, dtype=torch.float64, requires_grad=True)
    (x * table).sum().backward()
