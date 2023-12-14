from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve


def random_ssm(rng, n):
    """Randomly generate a state space model.

    Args:
        rng (ten): rng is a jax.random.PRNGKey
        n (_type_): n is the dimension of the state space model

    Returns:
        _type_: _description_
    """
    a_r, b_r, c_r = jax.random.split(rng, 3)
    a = jax.random.uniform(a_r, (n, n))
    b = jax.random.uniform(b_r, (n, 1))
    c = jax.random.uniform(c_r, (1, n))
    return a, b, c


def discretize(a, b, c, step):
    i = np.eye(a.shape[0])
    bl = inv(i - (step / 2.0) * a)
    ab = bl @ (i + (step / 2.0) * a)
    bb = (bl * step) @ b
    return ab, bb, c


def scan_ssm(ab, bb, cb, u, x0):
    """Scan a state space model.

    Args:
        ab (_type_): _description_
        bb (_type_): _description_
        cb (function): _description_
        u (_type_): _description_
        x0 (_type_): _description_
    """

    def step(x_k_1, u_k):
        x_k = ab @ x_k_1 + bb @ u_k
        y_k = cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_ssm(a, b, c, u):
    """Run ssm

    Args:
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_
        u (_type_): _description_

    Returns:
        _type_: _description_
    """
    l = u.shape[0]
    n = a.shape[0]
    ab, bb, cb = discretize(a, b, c, step=1.0 / l)

    # run recurrence
    return scan_ssm(ab, bb, cb, u[:, np.newaxis], np.zeros((n,)))[1]


def k_conv(ab, bb, cb, l):
    """K conolution

    Args:
        ab (_type_): _description_
        bb (_type_): _description_
        cb (function): _description_
        l (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.array(
        [(cb @ matrix_power(ab, l) @ bb).reshape() for i in range(l)]
    )


def causal_convolution(u, k, nofft=False):
    """Causal convolution

    Args:
        u (_type_): _description_
        k (_type_): _description_
        nofft (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if nofft:
        return convolve(u, k, mode="full")[:, u.shape[0]]
    else:
        assert k.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, k.shape[0])))
        kd = np.fft.rfft(np.pad(k, (0, u.shape[0])))
        out = ud * kd
        return np.fft.irfft(out)[: u.shape[0]]


def test_cnn_is_rnn(n=4, l=16, rng=None, step=1.0 / 16):
    """Test cnn is rnn

    Args:
        n (int, optional): _description_. Defaults to 4.
        l (int, optional): _description_. Defaults to 16.
        rng (_type_, optional): _description_. Defaults to None.
        step (_type_, optional): _description_. Defaults to 1.0/16.
    """
    ssm = random_ssm(rng, n)
    u = jax.random.uniform(rng, (l,))
    jax.random.split(rng, 3)

    # rnn
    rec = run_ssm(*ssm, u)

    # cnn
    ssmb = discretize(*ssm, step=step)
    conv = causal_convolution(u, k_conv(*ssmb, l))

    # check
    assert np.allclose(rec.ravel(), conv.ravel())
