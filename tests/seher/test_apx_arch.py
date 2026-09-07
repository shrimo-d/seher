import jax
import jax.numpy as jnp
import jax.random as jr
from seher.apx_arch import GRUCell

def test_grucell_make_parameter_shapes():
    cell = GRUCell.make(in_dim=3, hidden_dim=5, key=jr.PRNGKey(0))

    assert cell.Wz.shape == (8, 5)
    assert cell.bz.shape == (5,)
    assert cell.Wr.shape == (8, 5)
    assert cell.br.shape == (5,)
    assert cell.Wh.shape == (8, 5)
    assert cell.bh.shape == (5,)


def test_grucell_make_uses_initializers_for_all_parameter_groups():
    seen_shapes = []

    def w_init(key, shape):
        seen_shapes.append(('w', shape))
        return jnp.full(shape, 2.0)

    def b_init(key, shape):
        seen_shapes.append(('b', shape))
        return jnp.full(shape, -3.0)

    cell = GRUCell.make(in_dim=2, hidden_dim=4, key=jr.PRNGKey(1), w_init=w_init, b_init=b_init)

    assert seen_shapes == [
        ('w', (6, 4)), ('b', (4,)),
        ('w', (6, 4)), ('b', (4,)),
        ('w', (6, 4)), ('b', (4,)),
    ]
    assert jnp.all(cell.Wz == 2.0)
    assert jnp.all(cell.Wr == 2.0)
    assert jnp.all(cell.Wh == 2.0)
    assert jnp.all(cell.bz == -3.0)
    assert jnp.all(cell.br == -3.0)
    assert jnp.all(cell.bh == -3.0)


def test_grucell_forward_matches_manual_equations():
    cell = GRUCell(
        Wz=jnp.array([
            [0.2, -0.1],
            [0.4, 0.3],
            [-0.5, 0.7],
            [0.1, 0.2],
            [0.6, -0.4],
        ]),
        bz=jnp.array([0.05, -0.2]),
        Wr=jnp.array([
            [-0.3, 0.8],
            [0.2, -0.6],
            [0.9, 0.1],
            [0.4, -0.2],
            [-0.7, 0.5],
        ]),
        br=jnp.array([0.3, -0.1]),
        Wh=jnp.array([
            [0.5, -0.4],
            [-0.2, 0.6],
            [0.1, 0.3],
            [0.7, -0.8],
            [-0.9, 0.2],
        ]),
        bh=jnp.array([-0.05, 0.15]),
    )
    h = jnp.array([0.25, -0.4])
    x = jnp.array([0.7, -0.1, 0.2])

    hx = jnp.concatenate([h, x], axis=-1)
    z = jax.nn.sigmoid(hx @ cell.Wz + cell.bz)
    r = jax.nn.sigmoid(hx @ cell.Wr + cell.br)
    rhx = jnp.concatenate([r * h, x], axis=-1)
    h_tilde = jnp.tanh(rhx @ cell.Wh + cell.bh)
    expected = (1.0 - z) * h + z * h_tilde

    actual = cell(h, x)
    assert jnp.allclose(actual, expected, atol=1e-7)


def test_grucell_zero_weights_and_biases_returns_half_previous_hidden_state():
    cell = GRUCell(
        Wz=jnp.zeros((5, 2)),
        bz=jnp.zeros((2,)),
        Wr=jnp.zeros((5, 2)),
        br=jnp.zeros((2,)),
        Wh=jnp.zeros((5, 2)),
        bh=jnp.zeros((2,)),
    )
    h = jnp.array([1.2, -0.8])
    x = jnp.array([0.3, 0.4, -0.2])

    actual = cell(h, x)
    expected = 0.5 * h
    assert jnp.allclose(actual, expected, atol=1e-7)


def test_grucell_output_shape_matches_hidden_dimension():
    cell = GRUCell.make(in_dim=4, hidden_dim=3, key=jr.PRNGKey(2))
    h = jnp.ones((3,))
    x = jnp.ones((4,))

    out = cell(h, x)
    assert out.shape == (3,)


def test_grucell_is_deterministic_for_same_inputs_and_parameters():
    cell = GRUCell.make(in_dim=2, hidden_dim=2, key=jr.PRNGKey(3))
    h = jnp.array([0.1, -0.2])
    x = jnp.array([0.3, 0.4])

    out1 = cell(h, x)
    out2 = cell(h, x)

    assert jnp.array_equal(out1, out2)
