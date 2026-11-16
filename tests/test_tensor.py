"""Layer 1 tests, step 1: construction, basic arithmetic, simple backward.

Contract: see the conversation / ROADMAP.md. Tensor is float64, requires_grad
defaults to False, ops return new Tensors and never mutate their inputs.
"""
import numpy as np
import pytest

from autograd.tensor import Tensor


# ---------------------------------------------------------------- construction

def test_stores_data_as_float64_ndarray_from_list():
    t = Tensor([[1, 2], [3, 4]])
    assert isinstance(t.data, np.ndarray)
    assert t.data.dtype == np.float64
    np.testing.assert_array_equal(t.data, [[1.0, 2.0], [3.0, 4.0]])


def test_accepts_python_scalar():
    t = Tensor(3)
    assert t.data.dtype == np.float64
    assert t.shape == ()
    assert t.ndim == 0


def test_accepts_ndarray():
    t = Tensor(np.zeros((2, 3), dtype=np.float32))
    assert t.data.dtype == np.float64
    assert t.shape == (2, 3)
    assert t.ndim == 2


def test_defaults():
    t = Tensor([1.0])
    assert t.requires_grad is False
    assert t.grad is None


def test_requires_grad_flag_is_stored():
    assert Tensor([1.0], requires_grad=True).requires_grad is True


# ---------------------------------------------------------------- forward ops

A = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
B = np.array([[0.5, 2.0, -1.0], [3.0, -4.0, 2.0]])


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ],
    ids=["add", "sub", "mul", "div"],
)
def test_binary_forward_matches_numpy(op):
    out = op(Tensor(A), Tensor(B))
    assert isinstance(out, Tensor)
    np.testing.assert_allclose(out.data, op(A, B))


def test_neg_forward():
    np.testing.assert_allclose((-Tensor(A)).data, -A)


def test_scalar_on_either_side():
    t = Tensor(A)
    np.testing.assert_allclose((t + 2).data, A + 2)
    np.testing.assert_allclose((2 + t).data, 2 + A)
    np.testing.assert_allclose((t - 2).data, A - 2)
    np.testing.assert_allclose((2 - t).data, 2 - A)
    np.testing.assert_allclose((t * 3).data, A * 3)
    np.testing.assert_allclose((3 * t).data, 3 * A)
    np.testing.assert_allclose((t / 2).data, A / 2)
    np.testing.assert_allclose((2 / Tensor(B)).data, 2 / B)


def test_broadcast_forward():
    row = np.array([10.0, 20.0, 30.0])
    out = Tensor(A) + Tensor(row)
    np.testing.assert_allclose(out.data, A + row)


def test_ops_do_not_mutate_inputs():
    a, b = Tensor(A.copy()), Tensor(B.copy())
    _ = a + b
    _ = a * b
    _ = -a
    np.testing.assert_array_equal(a.data, A)
    np.testing.assert_array_equal(b.data, B)


# ------------------------------------------------------------ requires_grad flow

def test_result_requires_grad_if_any_input_does():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0])
    assert (a + b).requires_grad is True
    assert (b * a).requires_grad is True


def test_result_does_not_require_grad_if_no_input_does():
    assert (Tensor([1.0]) + Tensor([2.0])).requires_grad is False


# ------------------------------------------------------------------ backward

def test_backward_on_non_scalar_raises():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        (x * 2).backward()


def test_backward_add():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    (a + b).backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_backward_mul():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    (a * b).backward()
    assert a.grad == pytest.approx(3.0)
    assert b.grad == pytest.approx(2.0)


def test_backward_sub_and_neg():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    (a - b).backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(-1.0)

    c = Tensor(5.0, requires_grad=True)
    (-c).backward()
    assert c.grad == pytest.approx(-1.0)


def test_backward_div():
    a = Tensor(6.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    (a / b).backward()
    assert a.grad == pytest.approx(1.0 / 3.0)
    assert b.grad == pytest.approx(-6.0 / 9.0)


def test_grad_has_same_shape_as_data():
    # Seeded by hand: out = (x * w) summed manually via a scalar-valued chain.
    x = Tensor(4.0, requires_grad=True)
    y = x * x
    y.backward()
    assert isinstance(x.grad, np.ndarray)
    assert x.grad.shape == x.data.shape


def test_reused_node_accumulates():
    # y = x*x + x  ->  dy/dx = 2x + 1
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert x.grad == pytest.approx(7.0)


def test_chain_rule():
    # f = ((a * b) + a) * b  = a*b^2 + a*b
    # df/da = b^2 + b ; df/db = 2ab + a
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    f = ((a * b) + a) * b
    f.backward()
    assert a.grad == pytest.approx(12.0)
    assert b.grad == pytest.approx(14.0)


def test_grads_accumulate_across_backward_calls_until_zero_grad():
    x = Tensor(2.0, requires_grad=True)
    (x * 3).backward()
    (x * 3).backward()
    assert x.grad == pytest.approx(6.0)
    x.zero_grad()
    assert x.grad is None
    (x * 3).backward()
    assert x.grad == pytest.approx(3.0)


def test_leaf_without_requires_grad_gets_no_grad():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0)
    (a * b).backward()
    assert a.grad == pytest.approx(3.0)
    assert b.grad is None


def test_scalar_operand_backward():
    x = Tensor(4.0, requires_grad=True)
    (2 * x + 1).backward()
    assert x.grad == pytest.approx(2.0)
    y = Tensor(4.0, requires_grad=True)
    (1 / y).backward()
    assert y.grad == pytest.approx(-1.0 / 16.0)
