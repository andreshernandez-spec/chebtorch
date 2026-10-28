"""The caches in cheb_common.py. Tables and Memo build outside inference
mode and keep nothing built under torch.compile; Memo keeps python numbers
only, per (value, dtype, device), never under CUDA graph capture, and at
most _KEEP of them."""

import pytest
import torch

from chebtorch import cheb_common
from chebtorch.cheb_common import Memo, Tables

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CPU = torch.device("cpu")


def _counting():
    """A Memo of x -> (2 x,), and the list of the values it ran on."""
    ran = []

    def double(x):
        ran.append(x)
        return (2 * x,)

    return Memo(double), ran


@pytest.mark.parametrize("device", DEVICES)
def test_memo_keeps_python_numbers(device):
    memo, ran = _counting()
    dev = torch.device(device)
    kept = memo(1.5, torch.float64, dev)
    assert memo(1.5, torch.float64, dev) is kept and len(ran) == 1
    assert kept[0].item() == 3.0 and kept[0].device.type == device
    memo(1.5, torch.float32, dev)
    memo(2.5, torch.float64, dev)
    assert len(ran) == 3
    # a tensor may hold another value, or need a gradient, on the next call
    x = torch.tensor(1.5, dtype=torch.float64, device=dev)
    memo(x, torch.float64, dev)
    memo(x, torch.float64, dev)
    assert len(ran) == 5


def test_memo_stops_keeping_at_its_limit(monkeypatch):
    monkeypatch.setattr(cheb_common, "_KEEP", 3)
    memo, ran = _counting()
    for v in (1.0, 2.0, 3.0, 4.0, 4.0, 1.0):
        memo(v, torch.float64, CPU)
    # what was kept stays kept (a CUDA graph may read it); 4.0 runs each time
    assert [x.item() for x in ran] == [1.0, 2.0, 3.0, 4.0, 4.0]


def test_caches_build_outside_inference_mode():
    memo, _ = _counting()
    tables = Tables({"T": (2,)}, {"T": [1.0, 2.0]})
    with torch.inference_mode():
        kept = memo(1.5, torch.float64, CPU)[0]
        table = tables("T", torch.float64, CPU)
    assert not kept.is_inference() and not table.is_inference()
    # the product saves both for its backward, which inference tensors refuse
    x = torch.ones(2, dtype=torch.float64, requires_grad=True)
    (x * kept * table).sum().backward()


def test_caches_keep_nothing_under_torch_compile():
    # what a compiled call builds is a value of its graph, and with CUDA
    # graphs (mode="reduce-overhead") the next replay overwrites it
    memo, _ = _counting()
    tables = Tables({"T": (2,)}, {"T": [1.0, 2.0]})
    f = torch.compile(lambda y: memo(1.5, torch.float64, y.device)[0]
                      * tables("T", torch.float64, y.device) * y)
    assert torch.equal(f(torch.ones(2, dtype=torch.float64)),
                       torch.tensor([3.0, 6.0], dtype=torch.float64))
    assert not memo._kept and not tables._cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memo_keeps_nothing_a_cuda_graph_captures():
    # kept tensors are complete before capture and read by the graph; a value
    # first seen in capture is computed in the graph, since waiting for the
    # GPU is not allowed there and the result would not exist until a replay
    memo, _ = _counting()
    dev = torch.device("cuda")
    memo(1.5, torch.float64, dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = memo(1.5, torch.float64, dev)[0] + memo(2.5, torch.float64, dev)[0]
    graph.replay()
    torch.cuda.synchronize()
    assert out.item() == 8.0
    assert list(memo._kept) == [(1.5, torch.float64, dev)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memo_results_are_complete_on_other_streams():
    # any stream may read a kept result from the next call on, so keeping
    # it waits until it exists: else the read below races the work queued
    # ahead of it (~20 ms of matmuls)
    memo, _ = _counting()
    dev = torch.device("cuda")
    a = torch.randn(2048, 2048, device=dev)
    for _ in range(20):
        a = torch.tanh(a @ a)
    memo(1.2345678, torch.float64, dev)
    side = torch.cuda.Stream()             # not told to wait for this stream
    with torch.cuda.stream(side):
        out = memo(1.2345678, torch.float64, dev)[0] * 1
    torch.cuda.synchronize()
    assert out.item() == 2 * 1.2345678


def test_memo_raw_passes_the_number():
    # for terms whose sizes depend on the value itself
    seen = []

    def terms(v, dtype, device):
        seen.append(v)
        return (torch.arange(int(v * 2), dtype=dtype, device=device),)

    memo = Memo(terms, raw=True)
    kept = memo(1.5, torch.float64, CPU)
    assert memo(1.5, torch.float64, CPU) is kept and seen == [1.5]
    assert kept[0].tolist() == [0.0, 1.0, 2.0]
